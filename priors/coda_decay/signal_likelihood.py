import os, errno, sys, time, traceback
import numpy as np, scipy
from guppy import hpy; hp = hpy()

from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from optparse import OptionParser

import plot
import learn, sigvisa_util
import priors.SignalPrior
from utils.waveform import *
import utils.geog
import obspy.signal.util


import utils.nonparametric_regression as nr
from priors.coda_decay.coda_decay_common import *
from priors.coda_decay.train_coda_models import CodaModel

ARR_TIME_PARAM, CODA_HEIGHT_PARAM, CODA_DECAY_PARAM, NUM_PARAMS = range(3+1)

class TraceModel:

    def __init__(self, cursor, model_dict, model_type=CodaModel.MODEL_TYPE_GAUSSIAN):
        self.cursor=cursor
        self.model_dict = model_dict
        self.model_type = model_type

        for k in self.model_dict:
            self.netmodel =  self.model_dict[k][0].netmodel
            break

        self.sites = read_sites(cursor)

    def __param_scales(self):
        scales = np.zeros((1, NUM_PARAMS))
        scales[0, ARR_TIME_PARAM] = 1
        scales[0, CODA_HEIGHT_PARAM] = 0.5
        scales[0, CODA_DECAY_PARAM] = 0.01
        return scales

    def log_likelihood(self, segment, event, pp=None, integrate_method = 0):

        # for now we focus on one band and one channel
        band = 'narrow_logenvelope_2.00_3.00'
        chan = 'BHZ'

        tr = segment[chan][band]

        start_time = tr.stats.starttime_unix
        end_time = start_time + tr.stats.npts / tr.stats.sampling_rate
        siteid = tr.stats.siteid
        noise_floor = tr.stats.noise_floor

        phaseids = [1,5]

        # initialize the typical scale at which each parameter varies
        param_scales = self.__param_scales()
        param_scales = np.tile(param_scales, (2, 1))


        trueshape = param_scales.shape
        f = lambda params: logenv_l1_cost(tr.data, self.generate_trace(start_time, end_time, siteid, noise_floor, phaseids, np.reshape(params, trueshape), srate=tr.stats.sampling_rate).data) - self.__param_log_likelihood(phaseids, event, siteid, np.reshape(params, trueshape))

        if integrate_method == 0: #just use the mean parameters
            params = self.__predict_params(phaseids, event, siteid)
            nll = f(params)

        elif integrate_method == 1: # optimize parameters iteratively
            base_params = self.__predict_params(phaseids, event, siteid)
            params = np.copy(base_params)

            best_nll = np.float("inf")

            for i in range(3):
                for phaseidx, phaseid in enumerate(phaseids):
                    for paramidx in range(NUM_PARAMS):
                        base_val = base_params[phaseidx, paramidx]
                        scale = param_scales[phaseidx, paramidx]

                        best_val = params[phaseidx, paramidx]
                        print "on (%d, %d, %d)" % (i, phaseidx, paramidx)
                        for val in np.linspace(base_val - scale*3, base_val + scale*3, 5):
                            params[phaseidx, paramidx] = val
                            nll = f(params)
                            if nll < best_nll:
                                print "updating best_ll to", best_nll
                                best_nll = nll
                                best_val = val
                        params[phaseidx, paramidx] = best_val

            nll = best_nll

        elif integrate_method == 2: # delegate optimziation to scipy
            params = self.__predict_params(phaseids, event, siteid)

            lbound = params - param_scales * 5
            ubound = params + param_scales * 5
            ubound[:, CODA_DECAY_PARAM] = 0
            lbound = lbound.flatten()
            ubound = ubound.flatten()
            bounds = zip(lbound, ubound)

            x, nll, d = scipy.optimize.fmin_l_bfgs_b(f, params.flatten(), approx_grad=1, bounds=bounds, maxfun=30)
            print "found best nll", nll
            params = np.reshape(x, trueshape)

        elif integrate_method == 3: # do a monte carlo integral over parameters.

            sum_ll = np.float("-inf")
            n = 50
            for i in range(n):
                params = self.__sample_params(phaseids, event, siteid)
                ll = -1 * f(params)
                sum_ll = np.logaddexp(sum_ll, ll)

            nll = np.log(n) - sum_ll
            print "got nll", nll
            params = None


        ll = -1 * nll
        return ll, params


    def generate_trace(self, start_time, end_time, siteid, noise_floor, phaseids, params, srate=40):

        npts = int((end_time-start_time)*srate)

        data = noise_floor * np.ones( (npts,) )

        for (i, phase) in enumerate(phaseids):
            arr_idx = int((params[i, ARR_TIME_PARAM]-start_time)*srate)
            peak_idx = int((params[i, ARR_TIME_PARAM]+3-start_time)*srate)
            coda_height = params[i, CODA_HEIGHT_PARAM]
            coda_decay = params[i, CODA_DECAY_PARAM]

            legit_height = np.log(np.exp(coda_height) + np.exp(noise_floor))
            onset_slope = (legit_height-noise_floor)/(peak_idx-arr_idx)

#            print "generating with", start_time, arr_times[i], peak_times[i], end_time
#            print "  at indices", arr_idx, peak_idx, npts

            for t in range(arr_idx, peak_idx):
                if t < 0 or t >= npts:
                    continue
                data[t] += max(0, (t-arr_idx)*onset_slope)
            for t in range(peak_idx, npts):
                if t < 0 or t >= npts:
                    continue
                data[t] += max(0, (legit_height-noise_floor) + (t-peak_idx)*coda_decay/srate)

        stats = {"npts": npts, "starttime_unix": start_time, "sampling_rate": srate, "noise_floor": noise_floor, "siteid": siteid}
        return Trace(data=data, header=stats)

    def __predict_params(self, phaseids, ev, siteid):
        cm_p = self.model_dict[siteid][0]
        cm_s = self.model_dict[siteid][1]

        params = np.zeros((len(phaseids), NUM_PARAMS))

        for phaseidx,phase in enumerate(phaseids):
            if phase == 0 or phase == 1 or phase == 2:
                cm = cm_p
            else:
                cm = cm_s

            tt = cm.netmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid-1, phase-1)
            pred_arr_time = ev[EV_TIME_COL] + tt

            params[phaseidx, ARR_TIME_PARAM]  = pred_arr_time
            params[phaseidx, CODA_HEIGHT_PARAM]  = ev[EV_MB_COL] + cm.predict_peak_amp( ev, self.model_type)
            params[phaseidx, CODA_DECAY_PARAM] = cm.predict_decay( ev, self.model_type )

        #TODO: predict peak height seperately from coda height
        return params

    def __sample_params(self, phaseids, ev, siteid):
        cm_p = self.model_dict[siteid][0]
        cm_s = self.model_dict[siteid][1]

        params = np.zeros((len(phaseids), NUM_PARAMS))

        for phaseidx,phase in enumerate(phaseids):
            if phase == 0 or phase == 1 or phase == 2:
                cm = cm_p
            else:
                cm = cm_s

            meantt = cm.netmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid-1, phase-1)
            # peak of a laplace distribution is 1/2b, where b is the
            # scale param, so (HACK ALERT) we can recover b by
            # evaluating the density at the peak
            ttscale = 2.0 / np.exp(cm.netmodel.arrtime_logprob(0, 0, 0, siteid-1, phase-1))

            # sample from a Laplace distribution:
            U = np.random.random() - .5
            tt = meantt - ttscale * np.sign(U) * np.log(1 - 2*np.abs(U))

            params[phaseidx, ARR_TIME_PARAM]  = ev[EV_TIME_COL] + tt
            params[phaseidx, CODA_HEIGHT_PARAM]  = ev[EV_MB_COL] + cm.sample_peak_amp( ev, self.model_type)
            params[phaseidx, CODA_DECAY_PARAM] = cm.sample_decay( ev, self.model_type )

        return params

    def __param_log_likelihood(self, phaseids, ev, siteid, params):
        cm_p = self.model_dict[siteid][0]
        cm_s = self.model_dict[siteid][1]

        ll = 0

        for phaseidx,phase in enumerate(phaseids):
            if phase == 0 or phase == 1 or phase == 2:
                cm = cm_p
            else:
                cm = cm_s

            predtt = cm.netmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid-1, phase-1)
            tt = params[phaseidx, ARR_TIME_PARAM] - ev[EV_TIME_COL]
            ll =+ cm.netmodel.arrtime_logprob(tt, predtt, 0, siteid-1, phase-1)

            amp = params[phaseidx, CODA_HEIGHT_PARAM] - ev[EV_MB_COL]
            ll += cm.log_likelihood_peak_amp(amp, ev, self.model_type)

            decay = params[phaseidx, CODA_DECAY_PARAM]
            ll += cm.log_likelihood_decay(decay,  ev, self.model_type)

        return ll

    def predict_trace(self, ev, noise_floor, siteid, p_phaseid = 1, s_phaseid = 5):

        siteid = int(row[SITEID_COL])
        evid = int(ev[EV_EVID_COL])

        sql_query="SELECT l.time, l.arid FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d order by l.time" % (evid, siteid)
        self.cursor.execute(sql_query)
        other_arrivals = np.array(cursor.fetchall())
        other_arrivals = other_arrivals[:, 0]

        starttime = np.min(other_arrivals)- 30
        endtime = np.max(other_arrivals) + 150
        srate = 10

        phaseids = [p_phaseid, s_phaseid]

        return self.generate_trace(starttime, endtime, siteid, noise_floor, phaseids, params, srate=srate)


    # get the likelihood of an event location, if we don't know the event time.
    def event_location_likelihood(self, ev, segments, pp, integrate_method, true_ev_loc):

        (mb, evlon, evlat, evid, time, depth) = ev

        # use backprojection from the detection times to propose event times
        event_time_proposals = []
        for segment in segments:
            tr = segment['BHZ']['narrow_logenvelope_2.00_3.00']
            siteid = int(tr.stats.siteid)
            p_arrival = tr.stats.p_time
            s_arrival = tr.stats.s_time
            p_phaseid = int(tr.stats.p_phaseid)
            s_phaseid = int(tr.stats.s_phaseid)

            p_projection = p_arrival - self.netmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid-1, p_phaseid-1)
            s_projection = s_arrival - self.netmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid-1, s_phaseid-1)

            dist = utils.geog.dist_km((evlon, evlat), true_ev_loc)

            if dist < 300:
                print "siteid %d, lat %f lon %f, distance from true %f, backprojecting p_error %f s_error %f" % (siteid, evlon, evlat, dist, p_projection-ev[EV_TIME_COL], s_projection-ev[EV_TIME_COL])

            event_time_proposals.append(p_projection)
            event_time_proposals.append(s_projection)

        # find the event time that maximizes the likelihood
        maxll = np.float("-inf")
        maxt = 0

        f = lambda t: np.sum([self.log_likelihood(s, (mb, evlon, evlat, evid, t, depth), pp=pp, integrate_method=integrate_method)[0] for s in segments])
        for proposed_t in event_time_proposals:
            ll = f(proposed_t)
            if ll > maxll:
                maxll = ll
                maxt = proposed_t

        return maxll

    def event_heat_map(self, pp, segments, base_event, map_width=5, integrate_method=0, n=20, true_params = None):
        sites = read_sites(self.cursor)

        (mb, evlon, evlat, evid, time, depth) = base_event

        timeless_f = lambda lon, lat: np.sum([self.log_likelihood(s, (mb, lon, lat, evid, time, depth), pp=pp, integrate_method=integrate_method)[0] for s in segments])
        timed_f = lambda lon, lat: self.event_location_likelihood((mb, lon, lat, evid, time, depth), segments, pp=pp, integrate_method=integrate_method, true_ev_loc=(evlon, evlat))
        f = timed_f


        bmap, max_lonlat = plot_heat(pp, f, center=(evlon, evlat), width = map_width, title=("%d" % (evid)), n=n)
        print "got maxlonlat", max_lonlat, "vs true", (evlon, evlat)

        from utils.draw_earth import draw_events
        draw_events(bmap, ((evlon, evlat),), marker="o", ms=5, mfc="none", mec="yellow", mew=2)
        draw_events(bmap, (max_lonlat,), marker="o", ms=5, mfc="none", mec="blue", mew=2)

        for s in segments:
            sid = s['BHZ']['narrow_logenvelope_2.00_3.00'].stats.siteid
            (slon, slat, selev, ar) = sites[sid - 1]

            if sid == 2:
                draw_events(bmap, ((slon, slat),),  marker="x", ms=50, mfc="none", mec="purple", mew=5)
            elif sid == 109:
                draw_events(bmap, ((slon, slat),),  marker="x", ms=50, mfc="none", mec="green", mew=5)
            else:
                draw_events(bmap, ((slon, slat),),  marker="x", ms=50, mfc="none", mec="black", mew=5)

        bestll = f(max_lonlat[0], max_lonlat[1])
        basell = f(evlon, evlat)
        plt.title("map width %d deg\n best lon %f lat %f ll %f\n true lon %f lat %f ll %f" % (map_width, max_lonlat[0], max_lonlat[1], bestll, evlon, evlat, basell))
        pp.savefig()

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

            ll, params = self.log_likelihood(s, best_event, pp=pp, integrate_method=integrate_method)
            if params is not None and true_params is not None:
                gen_tr = self.generate_trace(start_time, end_time, siteid, noise_floor, [1,5], params, srate=tr.stats.sampling_rate)
                fig = plot.plot_trace(gen_tr, title="siteid %d ll %f \n p_arr %f p_height %f decay %f\n true p_arr %f height %f decay %f" % (siteid, ll, params[0, ARR_TIME_PARAM], params[0, CODA_HEIGHT_PARAM], params[0, CODA_DECAY_PARAM], true_params[siteid][0, ARR_TIME_PARAM], true_params[siteid][0, CODA_HEIGHT_PARAM], true_params[siteid][0, CODA_DECAY_PARAM]))
                pp.savefig()
                plt.close(fig)

def main():

    cursor = db.connect().cursor()

    evid = int(sys.argv[1])
    integrate_method = int(sys.argv[2]) # 0=none, 1=sequential, 2=lbfgs, 3=sampling
    model_type = int(sys.argv[3]) # 0=1d gaussian, 1=GP
    map_width = int(sys.argv[4]) # in degrees

    base_coda_dir2 = get_base_dir(int(2), None, int(1332146399))
    base_coda_dir109 = get_base_dir(int(109), None, int(1332146405))

    fname = os.path.join(base_coda_dir2, 'all_data')
    all_data, bands = read_shape_data(fname)
    band_idx = 1
    band = bands[1]
    all_data = add_depth_time(cursor, all_data)
    band_data2 = extract_band(all_data, band_idx)
    clean_p_data2 = clean_points(band_data2, P=True, vert=True)
    clean_s_data2 = clean_points(band_data2, P=False, vert=True)

    include_rows = np.array([(int(row[EVID_COL]) != evid) for row in clean_p_data2])
    clean_p_data2 = clean_p_data2[include_rows, :]

    include_rows = np.array([(int(row[EVID_COL]) != evid) for row in clean_s_data2])
    clean_s_data2 = clean_s_data2[include_rows, :]

    fname = os.path.join(base_coda_dir109, 'all_data')
    all_data, bands = read_shape_data(fname)
    all_data = add_depth_time(cursor, all_data)
    band_data109 = extract_band(all_data, band_idx)
    clean_p_data109 = clean_points(band_data109, P=True, vert=True)
    clean_s_data109 = clean_points(band_data109, P=False, vert=True)

    include_rows = np.array([(int(row[EVID_COL]) != evid) for row in clean_p_data109])
    clean_p_data109 = clean_p_data109[include_rows, :]

    include_rows = np.array([(int(row[EVID_COL]) != evid) for row in clean_s_data109])
    clean_s_data109 = clean_s_data109[include_rows, :]
#    for (band_idx, band) in enumerate(bands):

    out_fname = os.path.join("logs", "heatmap_lesscheat_%d_%d_%d_%d.pdf" % (evid, integrate_method, model_type, map_width))
    pp_s = PdfPages(out_fname)
    print "saving plots to", out_fname


    band_dir2 = os.path.join(base_coda_dir2, band[19:])
    band_dir109 = os.path.join(base_coda_dir109, band[19:])


    cm_p2 = CodaModel(clean_p_data2, band_dir2, True, True, sigma_f = [0.01, 1, 1], w = [100, 100, 100], sigma_n = [0.01, 0.01, 0.01])
    cm_s2 = CodaModel(clean_s_data2, band_dir2, False, True, sigma_f = [0.01, 1, 1], w = [100, 100, 100], sigma_n = [0.01, 0.01, 0.01])

    cm_p109 = CodaModel(clean_p_data109, band_dir109, True, True, sigma_f = [0.01, 1, 1], w = [100, 100, 100], sigma_n = [0.01, 0.01, 0.01])
    cm_s109 = CodaModel(clean_s_data109, band_dir109, False, True, sigma_f = [0.01, 1, 1], w = [100, 100, 100], sigma_n = [0.01, 0.01, 0.01])


    true2 = np.zeros((2, NUM_PARAMS))
    true109 = np.zeros((2, NUM_PARAMS))

    sql_query="SELECT l.time, l.arid, pid.id FROM leb_arrival l , static_siteid sid, static_phaseid pid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d and leba.phase=pid.phase and (pid.id=%d or pid.id=%d) order by l.time" % (evid, 2, 1, 2)
    cursor.execute(sql_query)
    other_arrivals = np.array(cursor.fetchall())
    print other_arrivals
    true_p2 = other_arrivals[0, 0]
    sql_query="SELECT l.time, l.arid FROM leb_arrival l , static_siteid sid, static_phaseid pid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d and pid.phase=leba.phase and (pid.id=%d or pid.id=%d) order by l.time" % (evid, 2, 4, 5)
    cursor.execute(sql_query)
    other_arrivals = np.array(cursor.fetchall())
    true_s2 = other_arrivals[0, 0]
    sql_query="SELECT l.time, l.arid FROM leb_arrival l , static_siteid sid, static_phaseid pid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d and pid.phase=leba.phase and (pid.id=%d or pid.id=%d) order by l.time" % (evid, 109, 1, 2)
    cursor.execute(sql_query)
    other_arrivals = np.array(cursor.fetchall())
    true_p109 = other_arrivals[0, 0]
    sql_query="SELECT l.time, l.arid FROM leb_arrival l , static_siteid sid, static_phaseid pid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d and pid.phase=leba.phase and (pid.id=%d or pid.id=%d) order by l.time" % (evid, 109, 4, 5)
    cursor.execute(sql_query)
    other_arrivals = np.array(cursor.fetchall())
    true_s109 = other_arrivals[0, 0]

    gp = construct_output_generators(cursor,cm_p2.netmodel,True, True)
    gs = construct_output_generators(cursor,cm_s2.netmodel,False, True)
    for row in band_data2:
        if int(row[EVID_COL]) == evid:
            true2[0,ARR_TIME_PARAM] = true_p2
            true2[0,CODA_HEIGHT_PARAM] = gp[3](row) + row[MB_COL]
            true2[0,CODA_DECAY_PARAM] = gp[1](row)
            true2[1,ARR_TIME_PARAM] = true_s2
            true2[1,CODA_HEIGHT_PARAM] = gs[3](row) + row[MB_COL]
            true2[1,CODA_DECAY_PARAM] = gs[1](row)
            break
    for row in band_data109:
        if int(row[EVID_COL]) == evid:
            true109[0,ARR_TIME_PARAM] = true_p109
            true109[0,CODA_HEIGHT_PARAM] = gp[3](row) + row[MB_COL]
            true109[0,CODA_DECAY_PARAM] = gp[1](row)
            true109[1,ARR_TIME_PARAM] = true_s109
            true109[1,CODA_HEIGHT_PARAM] = gs[3](row) + row[MB_COL]
            true109[1,CODA_DECAY_PARAM] = gs[1](row)
            break

    print true2
    print true109
    true_params = {2 : true2, 109: true109}
    print true_params
    mt = CodaModel.MODEL_TYPE_GP if model_type == 1 else CodaModel.MODEL_TYPE_GAUSSIAN

    tm = TraceModel(cursor, {2: [cm_p2, cm_s2], 109: [cm_p109, cm_s109]}, model_type=mt)

    ev = load_event(cursor, evid)
    s2, n2, o, op = load_signal_slice(cursor, evid, 2, load_noise=True)
    s109, n109, o, op = load_signal_slice(cursor, evid, 109, load_noise=True)

    fig = plot.plot_trace(s2[0]['BHZ']['narrow_logenvelope_2.00_3.00'])
    pp_s.savefig()
    plt.close(fig)

    fig = plot.plot_trace(s109[0]['BHZ']['narrow_logenvelope_2.00_3.00'])
    pp_s.savefig()
    plt.close(fig)



    print "2 alone"
    tm.event_heat_map(pp_s, [s2[0],], ev, map_width=map_width, integrate_method=integrate_method, n=19, true_params = true_params)

    print "109 alone"
    tm.event_heat_map(pp_s, [s109[0],], ev, map_width=map_width, integrate_method=integrate_method, n=19, true_params = true_params)

    print "2 and 109"
    tm.event_heat_map(pp_s, [s2[0],s109[0]], ev, map_width=map_width, integrate_method=integrate_method, n=19, true_params = true_params)


#    print "109 alone"
#    tm.event_heat_map(pp_s, [s109[0],], ev, map_width=20, integrate=False)
#    print "2 and 109 together"
#    tm.event_heat_map(pp_s, [s2[0], s109[0]], ev, map_width=20, integrate=False)
#    tm.event_heat_map(pp_s, [s2[0],], ev, map_width=20, integrate=True)
#    tm.event_heat_map(pp_s, [s109[0],], ev, map_width=20, integrate=True)
#    tm.event_heat_map(pp_s, [s2[0], s109[0]], ev, map_width=20, integrate=True)


    pp_s.close()




if __name__ == "__main__":
    main()
