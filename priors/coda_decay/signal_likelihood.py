import os, errno, sys, time, traceback
import numpy as np, scipy

from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from optparse import OptionParser

import plot
import sigvisa
import learn, sigvisa_util
import signals.SignalPrior
from utils.waveform import *
import utils.geog
import obspy.signal.util


from priors.coda_decay.train_wiggles import *
from priors.coda_decay.coda_decay_common import *
from priors.coda_decay.source_spectrum import *
from priors.coda_decay.templates import *
from priors.coda_decay.train_coda_models import CodaModel


def load_gp_params(fname, model_type):
    # file format:
    # sta is_p chan band target_str model_type params

    param_dict = NestedDict()

    f = open(fname, 'r')

    for line in f:
        entries=line.split()
        #        siteid = sta_to_siteid(entries[0], cursor)
        siteid = int(entries[0])
        phase_class = entries[1]
        chan = entries[2]
        short_band = entries[3]
        my_model_type = entries[4]
        target = entries[5]
        params =np.array([float(x) for x in entries[6:]])
        
        if my_model_type == model_type:
            param_dict[siteid][phase_class][chan][short_band][target] = params
    f.close()
    return param_dict

class TraceModel:

    def __init__(self, cursor, sigmodel, model_dict, model_type=CodaModel.MODEL_TYPE_GAUSSIAN):
        self.cursor=cursor
        self.sigmodel = sigmodel
        self.model_dict = model_dict
        self.model_type = model_type

        self.sites = read_sites(cursor)
        self.ssm = SourceSpectrumModel()

    def log_likelihood(self, segment, event, pp=None, marginalize_method = "monte_carlo", iid=False):
        """ Evaluate the marginal log-probability density of a
        segment, given a particular event. If marginalize_method is
        "monte_carlo", integrate out the template parameters. If
        "optimize", use the best template parameters (this is
        effectively fitting a template to the received signal, and
        then just using the likelihood of that template). If "mode",
        just use the most likely template predicted from the event's
        location.

        Currently this treats all channels and frequency bands as independent."""

        total_ll = 0

        set_noise_processes(self.sigmodel, segment)

        all_params = NestedDict()

        for chan in chans:
            for band in bands:
                tr = segment[chan][band]

                start_time = tr.stats.starttime_unix
                end_time = start_time + tr.stats.npts / tr.stats.sampling_rate
                siteid = tr.stats.siteid
                noise_floor = tr.stats.noise_floor

                print "ll for siteid", siteid, "event", event

                phaseids = [1,5]

                # compute p(trace | template)p(template|event)
                f = lambda params: -1 * c_cost(self.sigmodel, tr, phaseids, params, iid=iid) + self.__param_log_likelihood(phaseids, event, siteid, chan, band, params)

                if marginalize_method == "mode": #just use the mean parameters
                    params = self.__predict_params(phaseids, event, siteid, chan, band)
                    ll = f(params)
                    all_params[chan][band] = params

                elif marginalize_method == "optimize": # delegate optimziation to scipy
                    params = self.__predict_params(phaseids, event, siteid, chan, band)

                    atimes = params[:, 0]
                    timeless_params = params[:, 1:]
                    pparams = remove_peak(timeless_params)
                    pshape = pparams.shape
                    assem_params = lambda p : np.hstack([np.reshape(atimes, (-1, 1)), restore_peak(np.reshape(p, pshape))])

                    sf = lambda flat_params : -1 * f(assem_params(flat_params))
                    x = scipy.optimize.fmin(sf, pparams.flatten(), maxfun=30)
                    try:
                        params = assem_params(x)
                    except:
                        import pdb
                        pdb.set_trace()
                    ll = f(params)
                    all_params[chan][band] = params
                    print "found best value", ll

                elif marginalize_method == "monte_carlo": # do a monte carlo integral over parameters.

                    sum_ll = np.float("-inf")
                    n = 50

                    best_params = None
                    best_param_ll = np.float("-inf")

                    for i in range(n):
                        params = self.__sample_params(phaseids, event, siteid, chan, band)
                        ll = f(params)
                        sum_ll = np.logaddexp(sum_ll, ll) if not (ll < 1e-300) else sum_ll

                        if ll > best_param_ll:
                            best_params = params
                            best_param_ll = ll

                        if np.isnan(sum_ll):
                            print "sum_ll is nan!!"
                            import pdb
                            pdb.set_trace()

                    all_params[chan][band] = best_params

                    ll = sum_ll - np.log(n)
                    print "got ll", ll
                    params = None

                total_ll += ll

        return total_ll, all_params

    def __generate_params(self, phaseids, ev, siteid, chan, band, tt_f, template_f):
        """ Either predict or sample template parameters (plus arrival
        time), depending on the tt_f and template_f arguments. """

        short_band = band[16:]

        models_p = self.model_dict[siteid][band][chan][0]
        models_s = self.model_dict[siteid][band][chan][1]

        params = np.zeros((len(phaseids), NUM_PARAMS))

        for phaseidx,phaseid in enumerate(phaseids):
            if phaseid in P_PHASEIDS:
                models = models_p
            elif phaseid in S_PHASEIDS:
                models = models_s
            else:
                raise Exception("unknown phaseid %d" % phaseid)

            try:
                params[phaseidx, ARR_TIME_PARAM]  = ev[EV_TIME_COL] + tt_f(ev, siteid, phaseid)
                params[phaseidx, PEAK_OFFSET_PARAM]  = template_f(models["onset"], ev, self.model_type)
                params[phaseidx, CODA_HEIGHT_PARAM]  = self.ssm.source_logamp(ev[EV_MB_COL], phaseid, short_band) + template_f(models["amp_transfer"], ev, self.model_type)
                params[phaseidx, CODA_DECAY_PARAM] = template_f(models["decay"], ev, self.model_type)
            except:
                import pdb
                pdb.set_trace()

            if "peak_amp" in models:
                # TODO: consider actually implementing peak height
                # models (currently this block will just cause a key
                # error and fill in the default values below)
                params[phaseidx, PEAK_HEIGHT_PARAM]  = ev[EV_MB_COL] + template_f(models["peak_amp"], ev, self.model_type)
                params[phaseidx, PEAK_DECAY_PARAM] = template_f(models["peak_decay"], ev, self.model_type)
            else:
                params[phaseidx, PEAK_HEIGHT_PARAM] = params[phaseidx, CODA_HEIGHT_PARAM]
                params[phaseidx, PEAK_DECAY_PARAM] = 1

        return params

    def __predict_params(self, phaseids, ev, siteid, chan, band):

        tt_f = lambda ev, siteid, phaseid : self.sigmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid-1, phaseid-1)
        template_f = lambda cm, ev, model_type : cm.predict(ev, model_type)

        return self.__generate_params(phaseids, ev, siteid, chan, band, tt_f=tt_f, template_f=template_f)

    def __hack_sample_tt(self, ev, siteid, phaseid):

        meantt = self.sigmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid-1, phaseid-1)

        # peak of a laplace distribution is 1/2b, where b is the
        # scale param, so (HACK ALERT) we can recover b by
        # evaluating the density at the peak
        ttscale = 2.0 / np.exp(self.sigmodel.arrtime_logprob(0, 0, 0, siteid-1, phaseid-1))

        # sample from a Laplace distribution:
        U = np.random.random() - .5
        tt = meantt - ttscale * np.sign(U) * np.log(1 - 2*np.abs(U))
        return tt

    def __sample_params(self, phaseids, ev, siteid, chan, band):

        tt_f = lambda ev, siteid, phaseid : self.__hack_sample_tt(ev, siteid, phaseid)
        template_f = lambda cm, ev, model_type : cm.sample(ev, model_type)

        return self.__generate_params(phaseids, ev, siteid, chan, band, tt_f=tt_f, template_f=template_f)

    def __param_log_likelihood(self, phaseids, ev, siteid, chan, band, params):

        models_p = self.model_dict[siteid][band][chan][0]
        models_s = self.model_dict[siteid][band][chan][1]

        short_band = band[16:]

        ll = 0

        for phaseidx,phaseid in enumerate(phaseids):
            if phaseid in P_PHASEIDS:
                models = models_p
            elif phaseid in S_PHASEIDS:
                models = models_s
            else:
                raise Exception("unknown phaseid %d" % phaseid)


            predtt = self.sigmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid-1, phaseid-1)
            tt = params[phaseidx, ARR_TIME_PARAM] - ev[EV_TIME_COL]
            ll += self.sigmodel.arrtime_logprob(tt, predtt, 0, siteid-1, phaseid-1)

            onset_time = params[phaseidx, PEAK_OFFSET_PARAM]
            ll += models["onset"].log_likelihood(onset_time, ev, self.model_type)

            transfer = params[phaseidx, CODA_HEIGHT_PARAM] - self.ssm.source_logamp(ev[EV_MB_COL], phaseid, short_band)
            ll += models["amp_transfer"].log_likelihood(transfer, ev, self.model_type)

            decay = params[phaseidx, CODA_DECAY_PARAM]
            ll += models["decay"].log_likelihood(decay,  ev, self.model_type)

            if "peak_amp" in models:
                # TODO: consider actually implementing peak height
                # models (currently this block will just cause a key
                # error and fill in the default values below)

                peak_height = params[phaseidx, PEAK_HEIGHT_PARAM]
                ll += models["peak_amp"].log_likelihood(peak_height, ev, self.model_type)

                peak_decay = params[phaseidx, PEAK_DECAY_PARAM]
                ll += models["peak_decay"].log_likelihood(peak_decay, ev, self.model_type)
            else:
                pass

        print "param ll", ll
        return ll

    def __move_event_to(self, ev, lon=None, lat=None, t=None):
        ev2 = ev.copy()
        ev2[EV_LON_COL] = lon if lon is not None else ev[EV_LON_COL]
        ev2[EV_LAT_COL] = lat if lat is not None else ev[EV_LAT_COL]
        ev2[EV_TIME_COL] = t if t is not None else ev[EV_TIME_COL]
        return ev2

    # get the likelihood of an event location, if we don't know the event time.
    def event_location_likelihood(self, ev, segments, pp, marginalize_method, true_ev_loc, iid=False):

        evlon = ev[EV_LON_COL]
        evlat = ev[EV_LAT_COL]
        depth = ev[EV_DEPTH_COL]
        time = ev[EV_TIME_COL]
        mb = ev[EV_MB_COL]
        orid = ev[EV_ORID_COL]
        evid = ev[EV_EVID_COL]

        event_time_proposals = []
        for segment in segments:
            tr = segment['BHZ']['narrow_envelope_2.00_3.00']
            siteid = int(tr.stats.siteid)
            p_arrival = tr.stats.p_time
            s_arrival = tr.stats.s_time

            dist = utils.geog.dist_km((evlon, evlat), true_ev_loc)
            is_new = lambda l, x : (len(l) == 0 or np.min([np.abs(lx - x) for lx in l]) > 1.5)

            p_projection = 0
            s_projection = 0
            try:
                p_phaseid = int(tr.stats.p_phaseid)
                p_projection = p_arrival - self.sigmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid-1, p_phaseid-1)
                if is_new(event_time_proposals, p_projection):
                    event_time_proposals.append(p_projection)
            except:
                pass

            try:
                s_phaseid = int(tr.stats.s_phaseid)
                s_projection = s_arrival - self.sigmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid-1, s_phaseid-1)

                if is_new(event_time_proposals, s_projection):
                    event_time_proposals.append(s_projection)
            except:
                pass
    

            print "siteid %d, lat %f lon %f, distance from true %f, backprojecting p_error %f s_error %f" % (siteid, evlon, evlat, dist, p_projection-ev[EV_TIME_COL], s_projection-ev[EV_TIME_COL])


            

            

        # find the event time that maximizes the likelihood
        maxll = np.float("-inf")
        maxt = 0

        f = lambda t: np.sum([self.log_likelihood(s, self.__move_event_to(ev, t=t), pp=pp, marginalize_method=marginalize_method, iid=iid)[0] for s in segments])
        for proposed_t in event_time_proposals:
            ll = f(proposed_t)
            if ll > maxll:
                maxll = ll
                maxt = proposed_t

        return maxll

    def sensitivity(self, pp, segments, event, marginalize_method="optimize"):
        """ Find whether each segment contributes to the likelihood of
        the event, by comparing p(segment|noise) to p(segment|true
        event). """

        null_event = event.copy()
        null_event[EV_MB_COL] = -5
        null_event[EV_TIME_COL] = 0

        for segment in segments:
            ll2, params2 = self.log_likelihood(segment, null_event, pp=pp, marginalize_method="mode")
            ll1, params1 = self.log_likelihood(segment, event, pp=pp, marginalize_method=marginalize_method)
            print "best sampled params"
            for c in params1.keys():
                for b in params1[c].keys():
                    print c,b
                    print params1[c][b]
                    print_params(params1[c][b])

            print "segment from %d gives signal ll %f vs noise ll %f" % (segment[chans[0]][bands[0]].stats.siteid, ll1, ll2)

    def event_heat_map(self, pp, segments, base_event, map_width=3, marginalize_method="optimize", n=20, true_params = None, iid=False, run_label=None):

        evlon = base_event[EV_LON_COL]
        evlat = base_event[EV_LAT_COL]
        depth = base_event[EV_DEPTH_COL]
        time = base_event[EV_TIME_COL]
        mb = base_event[EV_MB_COL]
        orid = base_event[EV_ORID_COL]
        evid = base_event[EV_EVID_COL]

        # this is more legit -- propose the event time based on the arrival time and event location
        timed_f = lambda lon, lat: self.event_location_likelihood(self.__move_event_to(base_event, lon=lon, lat=lat), segments, pp=pp, marginalize_method=marginalize_method, true_ev_loc=(evlon, evlat), iid=iid)
        f = timed_f

        # plot the likelihood heat map
        fname = None if run_label is None else "logs/heatmap_%s_values.txt" % run_label
        bmap, max_lonlat = plot_heat(pp, f, center=(evlon, evlat), width = map_width, title=("%d" % (evid)), n=n, fname=fname)
        print "got maxlonlat", max_lonlat, "vs true", (evlon, evlat)

        # plot the true event, the likelihood peak, and the station locations
        from utils.draw_earth import draw_events
        draw_events(bmap, ((evlon, evlat),), marker="o", ms=5, mfc="none", mec="yellow", mew=2)
        draw_events(bmap, (max_lonlat,), marker="o", ms=5, mfc="none", mec="blue", mew=2)

        for s in segments:
            sid = s['BHZ']['narrow_envelope_2.00_3.00'].stats.siteid
            (slon, slat, selev, ar) = self.sites[sid - 1]

            draw_events(bmap, ((slon, slat),),  marker="x", ms=50, mfc="none", mec="red", mew=5)

        print "evaluating bestll and basell"
        print max_lonlat
        bestll = f(max_lonlat[0], max_lonlat[1])
        print "got bestll", bestll
        print evlon, evlat
        basell = f(evlon, evlat)
        plt.title("map width %d deg\n best lon %f lat %f ll %f\n true lon %f lat %f ll %f" % (map_width, max_lonlat[0], max_lonlat[1], bestll, evlon, evlat, basell))
        pp.savefig()

        # plot the signals predicted by the event location
        best_event = np.copy(np.array(base_event))
        best_event[1] = max_lonlat[0]
        best_event[2] = max_lonlat[1]
        return best_event

    def plot_predicted_signal(self, s, event, pp, iid=False, band='narrow_envelope_2.00_3.00', chan='BHZ'):

        tr = s[chan][band]
        siteid = tr.stats.siteid
        
        ll, pdict  = self.log_likelihood(s, event, pp=pp, marginalize_method="mode", iid=iid)
        params = pdict[chan][band]
        if params is not None and not isinstance(params, NestedDict):
            gen_tr = get_template(self.sigmodel, tr, [1, 5], params)
            fig = plot.plot_trace(gen_tr, title="siteid %d ll %f \n p_arr %f p_height %f decay %f" % (siteid, ll, params[0, ARR_TIME_PARAM], params[0, CODA_HEIGHT_PARAM], params[0, CODA_DECAY_PARAM]))
            pp.savefig()
            plt.close(fig)

def train_param_models(siteids, runids, evid):

    cursor, sigmodel, earthmodel, sites, dbconn = sigvisa_util.init_sigmodel()

    dad_params = {"decay": [.0235, .0158, 4.677, 0.005546, 0.00072], "onset": [1.87, 4.92, 2.233, 0., 0.0001], "amp_transfer": [1.1019, 3.017, 9.18, 0.00002589, 0.403]}


    lldda_sum_params = {"decay": [.01, .05, 1, 0.00001, 20, 0.000001, 1, .05, 300], "onset": [2, 5, 1, 0.00001, 20, 0.000001, 1, 5, 300], "amp": [.4, 0.00001, 1, 0.00001, 20, 0.00001, 1, .4, 800] , "amp_transfer": [.4, 0.00001, 1, 0.00001, 20, 0.00001, 1, .4, 800] }

    gp_params = load_gp_params("parameters/gp_hyperparams.txt", "dad")

    model_dict = NestedDict()

    for siteid in siteids:
        for band in bands:
            for chan in chans:
                print "loading/training siteid %d band %s chan %s" % (siteid, band, chan)
                for (is_s, PSids) in enumerate((P_PHASEIDS, S_PHASEIDS)):

                    short_band = band[16:]
                    fit_data = load_shape_data(cursor, chan=chan, short_band=short_band, siteid=siteid, runids=runids, phaseids=PSids, exclude_evids = [evid,])

                    loaded_dad_params = gp_params[siteid]["S" if is_s else "P"][chan][short_band]
                    if len(loaded_dad_params.keys()) == 3:
                        my_dad_params = loaded_dad_params
                    else:
                        my_dad_params = dad_params

                    try:
                        cm = CodaModel(fit_data=fit_data, band_dir = None, phaseids=PSids, chan=chan, target_str="decay", earthmodel=earthmodel, sigmodel = sigmodel, sites=sites, dad_params=my_dad_params["decay"], debug=False)
                        model_dict[siteid][band][chan][is_s]["decay"] = cm

                        cm = CodaModel(fit_data=fit_data, band_dir = None, phaseids=PSids, chan=chan, target_str="onset", earthmodel=earthmodel, sigmodel = sigmodel, sites=sites, dad_params=my_dad_params["onset"], debug=False)
                        model_dict[siteid][band][chan][is_s]["onset"] = cm

                        cm = CodaModel(fit_data=fit_data, band_dir = None, phaseids=PSids, chan=chan, target_str="amp_transfer", earthmodel=earthmodel, sigmodel = sigmodel, sites=sites, dad_params=my_dad_params["amp_transfer"], debug=False)
                        model_dict[siteid][band][chan][is_s]["amp_transfer"] = cm

                    except:
                        import traceback, pdb
                        traceback.print_exc()
                        raise

    return model_dict, cursor, sigmodel, earthmodel, sites

def main():

    parser = OptionParser()

    parser.add_option("-e", "--evid", dest="evid", default=None, type="int", help="event ID to locate")
    parser.add_option("-s", "--siteids", dest="siteids", default=None, type="str", help="comma-separated list of station siteid's with which to locate the event")
    parser.add_option("-r", "--runids", dest="runids", default=None, type="str", help="train models using fits from a specific runid (default is to use the most recent)")
    parser.add_option("-w", "--map_width", dest="map_width", default=2, type="float", help="width in degrees of the plotted heat map (2)")
    parser.add_option("--iid", dest="iid", default=False, action="store_true", help="use a uniform iid noise model (instead of AR)")
    parser.add_option("--method", dest="method", default="monte_carlo", help="method for signal likelihood computation (monte_carlo)")

    (options, args) = parser.parse_args()

    evid = options.evid
    siteids = [int(s) for s in options.siteids.split(',')]
    runids = [int(s) for s in options.runids.split(',')]


    map_width = options.map_width

    mt = CodaModel.MODEL_TYPE_GP_DAD #if model_type == 1 else CodaModel.MODEL_TYPE_GAUSSIAN

    # train / load coda models
    model_dict, cursor, sigmodel, earthmodel, sites = train_param_models(siteids, runids, evid)

    sta_string = ":".join([siteid_to_sta(sid,cursor) for sid in siteids])
    run_label = "%d_%d_%s_%s_%s" % (evid, map_width, sta_string, options.method, "iid" if options.iid else "arwiggle")
    out_fname = os.path.join("logs", "heatmap_%s.pdf" % run_label)
    pp = PdfPages(out_fname)
    print "saving plots to", out_fname


    load_wiggle_models(cursor, sigmodel, "parameters/signal_wiggles.txt")
    tm = TraceModel(cursor, sigmodel, model_dict, model_type=mt)

    plot_band = 'narrow_envelope_2.00_3.00'
    plot_chan = 'BHZ'


    ev = load_event(cursor, evid)
    print "loading signals..."
    signals = []
    for siteid in siteids:
        s, n, o, op, oa = load_signal_slice(cursor, evid, siteid, load_noise=True, learn_noise=True, earthmodel=earthmodel)
        signals.append(s[0])

        tr = s[0][plot_chan][plot_band]
        fig = plot.plot_trace(tr)
        tm.plot_predicted_signal(s[0], ev, pp, iid=True, chan=plot_chan, band=plot_band)
        
        pp.savefig()
        plt.close(fig)

#    print "computing sensitivity..."
#    tm.sensitivity(pp, signals, ev)

    print "computing heat map..."
    try:
        best_event = tm.event_heat_map(pp, signals, ev, map_width=map_width, n=19, iid=options.iid, marginalize_method=options.method, run_label=run_label)

        for s in signals:
            tm.plot_predicted_signal(s, best_event, pp, iid=True, chan=plot_chan, band=plot_band)

    finally:
        pp.close()

if __name__ == "__main__":
    main()
