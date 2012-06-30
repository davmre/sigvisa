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

import utils.nonparametric_regression as nr
from priors.coda_decay.coda_decay_common import *
from priors.coda_decay.train_coda_models import CodaModel

class TraceModel:

    def __init__(self, cursor, model_dict, model_type=CodaModel.MODEL_TYPE_GAUSSIAN):
        self.cursor=cursor
        self.model_dict = model_dict
        self.model_type = model_type

        for k in self.model_dict:
            self.sigmodel =  self.model_dict[k][0].sigmodel
            break

        self.sites = read_sites(cursor)

    def log_likelihood(self, segment, event, pp=None, marginalize_method = "monte_carlo"):
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

        for chan in chans:
            for band in bands:
                tr = segment[chan][band]

                start_time = tr.stats.starttime_unix
                end_time = start_time + tr.stats.npts / tr.stats.sampling_rate
                siteid = tr.stats.siteid
                noise_floor = tr.stats.noise_floor

                phaseids = [1,5]

                # compute p(trace | template)p(template|event)
                f = lambda params: -1 * c_cost(sigmodel, tr, phaseids, params, iid=False) + self.__param_log_likelihood(phaseids, event, siteid, params)

                if marginalize_method == "mode": #just use the mean parameters
                    params = self.__predict_params(phaseids, event, siteid)
                    ll = f(params)

                elif marginalize_method == "optimize": # delegate optimziation to scipy
                    params = self.__predict_params(phaseids, event, siteid)

                    pshape = params.shape
                    sf = lambda flat_params : -1 * f(np.reshape(flat_params, pshape))
                    x, nll, d = scipy.optimize.fmin(sf, params.flatten(), maxfun=30)
                    ll = -1 * nll
                    print "found best ll", ll
                    params = np.reshape(x, pshape)

                elif marginalize_method == "monte_carlo": # do a monte carlo integral over parameters.

                    sum_ll = np.float("-inf")
                    n = 50
                    for i in range(n):
                        params = self.__sample_params(phaseids, event, siteid)
                        ll = f(params)
                        sum_ll = np.logaddexp(sum_ll, ll)

                    ll = sum_ll - np.log(n)
                    print "got ll", ll
                    params = None

                total_ll += ll

        return total_ll

    def __generate_params(self, phaseids, ev, siteid, chan, band, tt_f, template_f):
        """ Either predict or sample template parameters (plus arrival
        time), depending on the tt_f and template_f arguments. """

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

            params[phaseidx, ARR_TIME_PARAM]  = tt_f(ev, siteid, phaseid)
            params[phaseidx, PEAK_OFFSET_PARAM]  = template_f(models["onset"], ev, self.model_type)
            params[phaseidx, CODA_HEIGHT_PARAM]  = ev[EV_MB_COL] + template_f(models["amp"], ev, self.model_type)
            params[phaseidx, CODA_DECAY_PARAM] = template_f(models["decay"], ev, self.model_type)

            try:
                # TODO: consider actually implementing peak height
                # models (currently this block will just cause a key
                # error and fill in the default values below)
                params[phaseidx, PEAK_HEIGHT_PARAM]  = ev[EV_MB_COL] + template_f(models["peak_amp"], ev, self.model_type)
                params[phaseidx, PEAK_DECAY_PARAM] = template_f(models["peak_decay"], ev, self.model_type)
            except KeyError:
                params[phaseidx, PEAK_HEIGHT_PARAM] = params[phaseidx, CODA_HEIGHT_PARAM]
                params[phaseidx, PEAK_DECAY_PARAM] = 1

        return params

    def predict_params(self, phaseids, ev, siteid, chan, band):

        tt_f = lambda ev, siteid, phaseid : self.sigmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid-1, phase-1) + ev[EV_TIME_COL]
        template_f = lambda cm, ev, model_type : cm.predict(ev, model_type)

        return self.__handle_params(phaseids, ev, siteid, chan, band, tt_f=tt_f, template_f=template_f)

    def __hack_sample_tt(self, ev, siteid, phaseid):

        meantt = self.sigmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid-1, phaseid-1)

        # peak of a laplace distribution is 1/2b, where b is the
        # scale param, so (HACK ALERT) we can recover b by
        # evaluating the density at the peak
        ttscale = 2.0 / np.exp(cm.sigmodel.arrtime_logprob(0, 0, 0, siteid-1, phaseid-1))

        # sample from a Laplace distribution:
        U = np.random.random() - .5
        tt = meantt - ttscale * np.sign(U) * np.log(1 - 2*np.abs(U))
        return tt

    def sample_params(self, phaseids, ev, siteid, chan, band):

        tt_f = lambda ev, siteid, phaseid : self.__hack_sample_tt(ev, siteid, phaseid)
        template_f = lambda cm, ev, model_type : cm.sample(ev, model_type)

        return self.__handle_params(phaseids, ev, siteid, chan, band, tt_f=tt_f, template_f=template_f)

    def __param_log_likelihood(self, phaseids, ev, siteid, chan, band, params):

        models_p = self.model_dict[siteid][band][chan][0]
        models_s = self.model_dict[siteid][band][chan][1]

        ll = 0

        for phaseidx,phaseid in enumerate(phaseids):
            if phaseid in P_PHASEIDS:
                models = models_p
            elif phaseid in S_PHASEIDS:
                models = models_s
            else:
                raise Exception("unknown phaseid %d" % phaseid)

            predtt = self.sigmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid-1, phase-1)
            tt = params[phaseidx, ARR_TIME_PARAM] - ev[EV_TIME_COL]
            ll =+ self.sigmodel.arrtime_logprob(tt, predtt, 0, siteid-1, phase-1)

            onset_time = params[phaseidx, PEAK_DELAY_PARAM]
            ll += models["onset"].log_likelihood(onset_time, ev, self.model_type)

            amp = params[phaseidx, CODA_HEIGHT_PARAM] - ev[EV_MB_COL]
            ll += models["amp"].log_likelihood(amp, ev, self.model_type)

            decay = params[phaseidx, CODA_DECAY_PARAM]
            ll += models["decay"].log_likelihood(decay,  ev, self.model_type)

            try:
                # TODO: consider actually implementing peak height
                # models (currently this block will just cause a key
                # error and fill in the default values below)

                peak_height = params[phaseidx, PEAK_HEIGHT_PARAM]
                ll += models["peak_amp"].log_likelihood(peak_height, ev, self.model_type)

                peak_decay = params[phaseidx, PEAK_DECAY_PARAM]
                ll += models["peak_decay"].log_likelihood(peak_decay, ev, self.model_type)
            except KeyError:
                pass

        return ll

    # get the likelihood of an event location, if we don't know the event time.
    def event_location_likelihood(self, ev, segments, pp, marginalize_method, true_ev_loc):

        (mb, evlon, evlat, evid, time, depth) = ev

        # use backprojection from the detection times to propose event times
        event_time_proposals = []
        for segment in segments:
            tr = segment['BHZ']['narrow_envelope_2.00_3.00']
            siteid = int(tr.stats.siteid)
            p_arrival = tr.stats.p_time
            s_arrival = tr.stats.s_time
            p_phaseid = int(tr.stats.p_phaseid)
            s_phaseid = int(tr.stats.s_phaseid)

            p_projection = p_arrival - self.sigmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid-1, p_phaseid-1)
            s_projection = s_arrival - self.sigmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid-1, s_phaseid-1)

            dist = utils.geog.dist_km((evlon, evlat), true_ev_loc)

            if dist < 300:
                print "siteid %d, lat %f lon %f, distance from true %f, backprojecting p_error %f s_error %f" % (siteid, evlon, evlat, dist, p_projection-ev[EV_TIME_COL], s_projection-ev[EV_TIME_COL])

            event_time_proposals.append(p_projection)
            event_time_proposals.append(s_projection)

        # find the event time that maximizes the likelihood
        maxll = np.float("-inf")
        maxt = 0

        f = lambda t: np.sum([self.log_likelihood(s, (mb, evlon, evlat, evid, t, depth), pp=pp, marginalize_method=marginalize_method)[0] for s in segments])
        for proposed_t in event_time_proposals:
            ll = f(proposed_t)
            if ll > maxll:
                maxll = ll
                maxt = proposed_t

        return maxll

    def event_heat_map(self, pp, segments, base_event, map_width=5, marginalize_method="monte_carlo", n=20, true_params = None):

        (mb, evlon, evlat, evid, time, depth) = base_event

        # THIS IS CHEATING -- it assumes we know the event time
        # timeless_f = lambda lon, lat: np.sum([self.log_likelihood(s, (mb, lon, lat, evid, time, depth), pp=pp, marginalize_method=marginalize_method)[0] for s in segments])

        # this is more legit -- propose the event time based on the arrival time and event location
        timed_f = lambda lon, lat: self.event_location_likelihood((mb, lon, lat, evid, time, depth), segments, pp=pp, marginalize_method=marginalize_method, true_ev_loc=(evlon, evlat))
        f = timed_f

        # plot the likelihood heat map
        bmap, max_lonlat = plot_heat(pp, f, center=(evlon, evlat), width = map_width, title=("%d" % (evid)), n=n)
        print "got maxlonlat", max_lonlat, "vs true", (evlon, evlat)

        # plot the true event, the likelihood peak, and the station locations
        from utils.draw_earth import draw_events
        draw_events(bmap, ((evlon, evlat),), marker="o", ms=5, mfc="none", mec="yellow", mew=2)
        draw_events(bmap, (max_lonlat,), marker="o", ms=5, mfc="none", mec="blue", mew=2)

        for s in segments:
            sid = s['BHZ']['narrow_envelope_2.00_3.00'].stats.siteid
            (slon, slat, selev, ar) = self.sites[sid - 1]

            draw_events(bmap, ((slon, slat),),  marker="x", ms=50, mfc="none", mec="red", mew=5)

        bestll = f(max_lonlat[0], max_lonlat[1])
        basell = f(evlon, evlat)
        plt.title("map width %d deg\n best lon %f lat %f ll %f\n true lon %f lat %f ll %f" % (map_width, max_lonlat[0], max_lonlat[1], bestll, evlon, evlat, basell))
        pp.savefig()


        # plot the signals predicted by the event location
        best_event = np.copy(np.array(base_event))
        best_event[1] = max_lonlat[0]
        best_event[2] = max_lonlat[1]
        for s in segments:

            band = 'narrow_logenvelope_2.00_3.00'
            chan = 'BHZ'
            tr = s[chan][band]

            start_time = tr.stats.starttime_unix
            end_time = start_time + tr.stats.npts / tr.stats.sampling_rate
            siteid = tr.stats.siteid
            noise_floor = tr.stats.noise_floor

            ll, params = self.log_likelihood(s, best_event, pp=pp, marginalize_method="mode")
            if params is not None and true_params is not None:

#                gen_seg = self.sigmodel.generate_segment(start_time, end_time, siteid, tr.stats.sampling_rate, [1,5], params)
                gen_tr = self.sigmodel.generate_trace(start_time, end_time, siteid, chan, band, tr.stats.sampling_rate, [1,5], params)
                fig = plot.plot_trace(gen_tr, title="siteid %d ll %f \n p_arr %f p_height %f decay %f\n true p_arr %f height %f decay %f" % (siteid, ll, params[0, ARR_TIME_PARAM], params[0, CODA_HEIGHT_PARAM], params[0, CODA_DECAY_PARAM], true_params[siteid][0, ARR_TIME_PARAM], true_params[siteid][0, CODA_HEIGHT_PARAM], true_params[siteid][0, CODA_DECAY_PARAM]))
                pp.savefig()
                plt.close(fig)

def train_param_models(siteids, runids, evid):

    cursor, sigmodel, earthmodel, sites, dbconn = init_sigmodel()

    lldda_sum_params = {"decay": [.01, .05, 1, 0.00001, 20, 0.000001, 1, .05, 300], "onset": [2, 5, 1, 0.00001, 20, 0.000001, 1, 5, 300], "amp": [.4, 0.00001, 1, 0.00001, 20, 0.00001, 1, .4, 800] }

    model_dict = NestedDict()

    for siteid in siteids:
        for band in bands:
            for chan in chans:
                for (is_s, PSids) in enumerate((P_PHASEIDS, S_PHASEIDS)):

                    short_band = band[16:]
                    fit_data = load_shep_data(cursor, chan=chan, short_band=short_band, siteid=siteid, runids=runids, phaseids=PSids, exclude_evids = [evid,])

                    cm = CodaModel(fit_data=fit_data, band_dir = None, phaseids=phaseids, chan=chan, target_str="decay", earthmodel=earthmodel, sigmodel = sigmodel, sites=sites, lldda_sum_params=lldda_sum_params, debug=False)
                    model_dict[siteid][band][chan][is_s]["decay"] = cm

                    cm = CodaModel(fit_data=fit_data, band_dir = None, phaseids=phaseids, chan=chan, target_str="onset", earthmodel=earthmodel, sigmodel = sigmodel, sites=sites, lldda_sum_params=lldda_sum_params, debug=False)
                    model_dict[siteid][band][chan][is_s]["onset"] = cm

                    cm = CodaModel(fit_data=fit_data, band_dir = None, phaseids=phaseids, chan=chan, target_str="amp_transfer", earthmodel=earthmodel, sigmodel = sigmodel, sites=sites, lldda_sum_params=lldda_sum_params, debug=False)
                    model_dict[siteid][band][chan][is_s]["amp_transfer"] = cm

    return model_dict, cursor, sigmodel, earthmodel, sites

def main():


    parser = OptionParser()

    parser.add_options("-e", "--evid", dest="evid", default=None, type="int", help="event ID to locate")
    parser.add_option("-s", "--siteids", dest="siteids", default=None, type="str", help="comma-separated list of station siteid's with which to locate the event")
    parser.add_option("-r", "--runids", dest="runids", default=None, type="int", help="train models using fits from a specific runid (default is to use the most recent)")
    parser.add_options("-w", "--map_width", dest="map_width", default=2, type="float", help="width in degrees of the plotted heat map (2)")

    (options, args) = parser.parse_args()

    evid = options.evid
    siteids = [int(s) for s in options.siteids.split(',')]
    runids = [int(s) for s in options.runids.split(',')]

    map_width = options.map_width

    out_fname = os.path.join("logs", "heatmap_%d_%d_%d_%d.pdf" % (evid, model_type, map_width))
    pp = PdfPages(out_fname)
    print "saving plots to", out_fname

    # train / load coda models
    model_dict, cursor, sigmodel, earthmodel, sites = train_param_models(siteids, runids, evid)
    mt = CodaModel.MODEL_TYPE_GP #if model_type == 1 else CodaModel.MODEL_TYPE_GAUSSIAN

    tm = TraceModel(cursor, model_dict, model_type=mt)

    ev = load_event(cursor, evid)

    signals = []
    for siteid in siteids:
        s, n, o, op = load_signal_slice(cursor, evid, 2, load_noise=True)
        signals.append(s)

        fig = plot.plot_trace(s2[0]['BHZ']['narrow_envelope_2.00_3.00'])
        pp.savefig()
        plt.close(fig)

    tm.event_heat_map(pp, signals, ev, map_width=map_width, n=19, true_params = true_params)
    pp.close()

if __name__ == "__main__":
    main()
