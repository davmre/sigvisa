import os, errno, sys, time, traceback
import numpy as np, scipy

from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plot
import learn, sigvisa_util
import signals.SignalPrior
from utils.waveform import *
import utils.geog
import obspy.signal.util


import utils.nonparametric_regression as nr
from priors.coda_decay.coda_decay_common import *
from priors.coda_decay.plot_coda_decays import *

def arrival_peak_offset(trace, window_start_offset, window_end_offset = None):
    srate = trace.stats.sampling_rate

    if window_end_offset is None:
        window_end_offset = window_start_offset + 15

    i = np.floor(window_start_offset*srate)
    j = np.floor(window_end_offset*srate)

    print window_start_offset, window_end_offset, i, j, srate, trace.data.shape

    pt = np.argmax(trace.data[i:j]) / srate
    return (pt +window_start_offset, trace.data[(pt+window_start_offset) * srate ])

def print_params(params):
    n = params.shape[0]
    for i in range(n):
        print "%d: st: %.1f pdelay: %.1f pheight: %.2f pdecay: %.4f cheight: %.2f cdecay: %.4f" % (i, params[i, 0], params[i, 1], params[i, 2], params[i, 3], params[i, 4], params[i, 5])

def learn_wiggle_params(sigmodel, env, smoothed, phaseids, params):
    srate = env.stats.sampling_rate
    st = env.stats.starttime_unix
    nf = env.stats.noise_floor


    tmpl = get_template(sigmodel, env, phaseids, params)
    diff = subtract_traces(env, tmpl)

    # we don't want wiggles from the first (say) 25 secs, since some of that is onset and some of that is peak
    # we also only trust our wiggles if the envelope is, say, 10 times greater than the noise floor (so a height of 2.5 in natural-log-space)
    models = []
    for (phase_idx, phase) in enumerate(phaseids):
        start_wiggle = params[phase_idx, ARR_TIME_PARAM] + 20
        start_idx = np.ceil((start_wiggle - st)*srate)
        for t in range(100):
            end_idx = start_idx + np.ceil(srate*t)
            if tmpl[end_idx] - nf < 3:
                break
        l = (end_idx - start_idx)/srate
        if l < 8:
            import pdb
            pdb.set_trace()
            print "couldn't learn wiggle params for phase %d!" % (phase,)
            models.append(None)
            continue
        wiggle_train = diff.data[start_idx:end_idx]

        f = open("wiggles%d" % (phase,), 'w')
        for d in wiggle_train:
            f.write(str(d) + "\n")
        f.close()

        ar_learner = ARLearner(wiggle_train, srate)
        arparams, std = ar_learner.yulewalker(17)
        em = ErrorModel(0, std)
        print "learned wiggle params %s std %f mean %f from %f seconds of data" % (arparams, std, ar_learner.c, (end_idx - start_idx)/srate)
        wiggle_model = ARModel(arparams, em, c = ar_learner.c)
        models.append(wiggle_model)
    return models

def c_cost(sigmodel, smoothed, phaseids, params, iid=False):

#    noise_floor = params[-1]
#    params = np.reshape(params[:-1], (len(phaseids), -1))

    params = np.reshape(params, (len(phaseids), -1))

    for i, pid in enumerate(phaseids):
        if np.isnan(params[i, PEAK_HEIGHT_PARAM]) or np.isnan(params[i, CODA_HEIGHT_PARAM]):
            return np.float('inf')
        if params[i, PEAK_HEIGHT_PARAM] < 0:
            return np.float('inf')
        if params[i, CODA_HEIGHT_PARAM] > 1.1 * params[i, PEAK_HEIGHT_PARAM]:
            return np.float('inf')
        if params[i, CODA_DECAY_PARAM] >= 0 or params[i, CODA_DECAY_PARAM] >= 0:
            return np.float('inf')
        if params[i, PEAK_DECAY_PARAM] < 0 or params[i, PEAK_DECAY_PARAM] < 0:
            return np.float('inf')

#    print "trying heights ppeak %f pcoda %f speak %f scoda %f" % (params[0, PEAK_HEIGHT_PARAM], params[0, CODA_HEIGHT_PARAM], params[1, PEAK_HEIGHT_PARAM], params[1, CODA_HEIGHT_PARAM] )

    # we assume the noise params are already set...

    print "params"
    print_params(params)

    if iid:
        env = get_template(sigmodel, smoothed, phaseids, params)
        c = logenv_l1_cost(smoothed.data, env.data)
    else:
        c = -1 *sigmodel.trace_likelihood(smoothed, phaseids, params);

    print "cost", c

    return c


#def coord_ascent(f, params, bounds):
#
#    # move through arrivals one by one
#    # for each one, optimize coda height and decay
#    # then optimize peak
#
#    n_arrivals = len(params)/NUM_PARAMS
#
#    best_params = np.copy(params)
#    best_cost = f(best_params)
#
#    for i in range(1, 5):
#
#        for arr_idx in range(n_arrivals):
#            i1 = arr_idx * NUM_PARAMS
#
#            for cdecay in [-.1, -.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, -0.005]:
#                for cheight in np.linspace[]

# params with peak but without arrtime
def remove_peak(pp):
    newp = np.zeros((pp.shape[0], NUM_PARAMS-3))
    newp[:, 0] = pp[:, PEAK_OFFSET_PARAM-1]
    newp[:, 1] = pp[:, CODA_HEIGHT_PARAM-1]
    newp[:, 2] = pp[:, CODA_DECAY_PARAM-1]
    return newp

def restore_peak(peakless_params):
    p = peakless_params
    newp = np.zeros((p.shape[0], NUM_PARAMS-1))
    newp[:, 0] = p[:, 0]
    newp[:, 1] = p[:, 1]
    newp[:, 2] = 1
    newp[:, 3] = p[:, 1]
    newp[:, 4] = p[:, 2]
    return newp

def fit_elephant_envelope(sigmodel, pp, arrivals, env, smoothed, defaults, fix_peak = True, evid=None):

    arr_bounds = [ (0, 15), (0, 15) , (0, 10), (0, 15), (-.2, 0) ]
    arr_bounds_fixed_peak = [ (0, 15), (0, 10), (-.15, 0) ]
    arrivals = [arr for arr in arrivals if arr is not None]

    start_params = np.zeros((len(arrivals), NUM_PARAMS))
    bounds = []
    bounds_fp = []
    phaseids = []
    arr_times = np.zeros((len(arrivals), 1))
    for i, arr in enumerate(arrivals):
        time = arr[AR_TIME_COL]
#        (peak_offset_time, peak_height) = arrival_peak_offset(smoothed, time - smoothed.stats.starttime_unix)

        a = defaults[i]
        noise_floor = smoothed.stats.noise_model.c

        fit_peak_height = logsub_noise(a[FIT_PEAK_HEIGHT], noise_floor)
        fit_coda_height = logsub_noise(a[FIT_HEIGHT] - a[FIT_B] *(a[FIT_CODA_START_OFFSET] - a[FIT_PEAK_OFFSET]), noise_floor)

        start_params[i, PEAK_OFFSET_PARAM] = ( a[FIT_PEAK_OFFSET] + smoothed.stats.starttime_unix) - time
        print "init peak offset to ", start_params[i, PEAK_OFFSET_PARAM], "for phase", i
        start_params[i, PEAK_HEIGHT_PARAM] = fit_peak_height if fit_peak_height > 0 else 1
        start_params[i, PEAK_DECAY_PARAM] = 5
        start_params[i, CODA_HEIGHT_PARAM] = fit_coda_height if fit_coda_height > 0 else 1
        start_params[i, CODA_DECAY_PARAM] = a[FIT_B] if a[FIT_B] < 0 else -0.03

        bounds = bounds + arr_bounds
        bounds_fp = bounds_fp + arr_bounds_fixed_peak
        phaseids.append(arr[AR_PHASEID_COL])
        arr_times[i] = time

    start_params = start_params[:, 1:]

    if fix_peak:
        print start_params
        start_params = remove_peak(start_params)
        bounds = bounds_fp
        print start_params, bounds
        assem_params = lambda params: np.hstack([arr_times, restore_peak(np.reshape(params, (len(arrivals), -1))) ])
    else:
        assem_params = lambda params: np.hstack([arr_times, np.reshape(params, (len(arrivals), -1))])

    start_params = start_params.flatten()

    print "start params", start_params

    c = sigvisa.canonical_channel_num(smoothed.stats.channel)
    b = sigvisa.canonical_band_num(smoothed.stats.band)
    arm = smoothed.stats.noise_model
    sarm = smoothed.stats.smooth_noise_model

    # learn from original data
    sigmodel.set_noise_process(smoothed.stats.siteid, b, c, arm.c, arm.em.std**2, np.array(arm.params))
    sigmodel.set_wiggle_process(smoothed.stats.siteid, b, 0, arm.em.std**2, np.array(arm.params))

    f = lambda params : c_cost(sigmodel, env, phaseids, assem_params(params))

    start_cost = f(start_params)
    print "start params cost (w/ orig data)", start_cost
    plot_channels_with_pred(sigmodel, pp, env, assem_params(start_params), phaseids, None, None, title = "start orig (cost %f, evid %s)" % (start_cost, evid))


    print "bounds", bounds

#    best_params, best_cost, d = scipy.optimize.fmin_tnc(f, start_params, approx_grad=1, bounds=bounds)
    best_params, best_cost, d = scipy.optimize.fmin_l_bfgs_b(f, start_params, approx_grad=1, bounds=bounds, epsilon = 1e-3, factr=1e12)
    plot_channels_with_pred(sigmodel, pp, env, assem_params(best_params), phaseids, None, None, title = "best orig (cost %f, evid %s)" % (best_cost, evid))

    # learn from smoothed data w/ orig model
    sigmodel.set_noise_process(smoothed.stats.siteid, b, c, arm.c, arm.em.std**2, np.array(arm.params))
    sigmodel.set_wiggle_process(smoothed.stats.siteid, b, 0, arm.em.std**2, np.array(arm.params))

    f = lambda params : c_cost(sigmodel, smoothed, phaseids, assem_params(params))

    best_params_smooth, best_cost_smooth, d = scipy.optimize.fmin_l_bfgs_b(f, start_params, approx_grad=1, bounds=bounds, epsilon = 1e-3, factr=1e12)
    plot_channels_with_pred(sigmodel, pp, smoothed, assem_params(best_params_smooth), phaseids, None, None, title = "best hybrid (smooth data, original model) (cost %f, evid %s)" % (best_cost_smooth, evid))

    # learn from smoothed data
    sigmodel.set_noise_process(smoothed.stats.siteid, b, c, sarm.c, sarm.em.std**2, np.array(sarm.params))
    sigmodel.set_wiggle_process(smoothed.stats.siteid, b, 0, sarm.em.std**2, np.array(sarm.params))

    f = lambda params : c_cost(sigmodel, smoothed, phaseids, assem_params(params))

    start_cost = f(start_params)
    print "start params cost (w/ smoothed data)", start_cost
    plot_channels_with_pred(sigmodel, pp, smoothed, assem_params(start_params), phaseids, None, None, title = "start smoothed (cost %f, evid %s)" % (start_cost, evid))
    best_params_smooth, best_cost_smooth, d = scipy.optimize.fmin_l_bfgs_b(f, start_params, approx_grad=1, bounds=bounds, epsilon = 1e-3, factr=1e12)
    plot_channels_with_pred(sigmodel, pp, smoothed, assem_params(best_params_smooth), phaseids, None, None, title = "best smoothed (cost %f, evid %s)" % (best_cost_smooth, evid))

    # learn from smoothed data w/ iid noise
    f = lambda params : c_cost(sigmodel, smoothed, phaseids, assem_params(params), iid=True)

    start_cost = f(start_params)
    print "start params cost (w/ smoothed data and iid noise)", start_cost
    plot_channels_with_pred(sigmodel, pp, smoothed, assem_params(start_params), phaseids, None, None, title = "start smoothed iid (cost %f, evid %s)" % (start_cost, evid))
    best_params_iid, best_cost_iid, d = scipy.optimize.fmin_l_bfgs_b(f, start_params, approx_grad=1, bounds=bounds, epsilon = 1e-3, factr=1e12)
    plot_channels_with_pred(sigmodel, pp, smoothed, assem_params(best_params_iid), phaseids, None, None, title = "best iid (cost %f, evid %s)" % (best_cost_iid, evid))

#    wrm = learn_wiggle_params(sigmodel, env, smoothed, phaseids, assem_params(true_params))
#    wrm = wrm[1]
#    sigmodel.set_wiggle_process(smoothed.stats.siteid, b, 0, wrm.em.std**2, np.array(wrm.params))
#    sys.exit(1)


    """true_params = (8, 2.8, -0.02, 14, 3.1, -0.06)
    true_cost = f(true_params)
    print "true params cost", true_cost
    plot_channels_with_pred(sigmodel, pp, smoothed, assem_params(true_params), phaseids, None, None, title = "true")

#    best_params = (9.3, 0, -.1, 14.5, 0, -.1)
    best_cost = f(best_params)
    print "best params cost", best_cost
    plot_channels_with_pred(sigmodel, pp, smoothed, assem_params(best_params), phaseids, None, None, title = "so-called best")
    """

    return best_params, phaseids, best_cost

#######################################################
# TODO: delete all of the stuff in the following region

def fit_specific(trace, coda_start_time, coda_len):
    srate = trace.stats['sampling_rate']
    true_env = trace.data[coda_start_time*srate : (coda_start_time + coda_len)*srate]
    cost = lambda(height, b): logenv_l1_cost(true_env, gen_logenvelope(len(true_env)/srate, srate, height, 0, b))
    start_height = true_env[0]
    bounds = ((start_height-2, start_height+1), (-.1, 0),)
    results = scipy.optimize.brute(cost, bounds, Ns=15, full_output=0)
    avg_cost = cost(results)/len(true_env)
    return results, avg_cost

def fit_logenvelope(trace, peak_offset_time, peak_height, max_coda_length, min_coda_length):

    best_cost = 100000
    best_tradeoff = 100000
    best_results = None
    best_start_time = peak_offset_time
    best_length = max_coda_length

    best_results, best_cost = fit_specific(trace, peak_offset_time, max_coda_length)

    tradeoff_score = lambda cost, l, min_l: cost / np.sqrt(l / min_l)

    if max_coda_length > min_coda_length:
        for coda_start_time in np.linspace(peak_offset_time, min(peak_offset_time+10, peak_offset_time + max_coda_length - min_coda_length), 4):
            real_max_len = max_coda_length - (coda_start_time - peak_offset_time)
            for coda_len in np.linspace(min_coda_length, real_max_len, np.ceil((real_max_len - min_coda_length)/5)):
                results, cost = fit_specific(trace, coda_start_time, coda_len)
                tradeoff = tradeoff_score(cost, coda_len, min_coda_length)
                if tradeoff < best_tradeoff:
                    best_tradeoff = tradeoff
                    best_cost = cost
                    best_results = results
                    best_start_time = coda_start_time
                    best_length = coda_len

#    print "returning", best_results, best_cost, best_start_time, best_length
    return best_results, best_cost, best_start_time, best_length

def fit_phase_coda(phase_arrival, smoothed, other_arrivals, other_arrival_phases, noise_floor):
    npts = smoothed.stats.npts
    srate = smoothed.stats.sampling_rate
    stime = smoothed.stats.starttime_unix

    P = True if int(phase_arrival[AR_PHASEID_COL]) in P_PHASEIDS else False

    phase_length = 200
    if other_arrivals.shape[0] > 0:
        for (a, pa) in zip(other_arrivals, other_arrival_phases):
            if a > phase_arrival[AR_TIME_COL] and pa != "LR":
                phase_length = np.min([a - phase_arrival[AR_TIME_COL], phase_length])

    # compute phase arrival times relative to the trace start time
    phase_start_time = phase_arrival[AR_TIME_COL] - stime
    phase_length = np.min([phase_length, npts/srate - phase_start_time])

    try:
        (peak_offset_time, peak_height) = arrival_peak_offset(smoothed, phase_start_time, phase_start_time + phase_length )
    except:
        print "segment does not contain the arrival we care about"
        return None

    if (npts/srate - peak_offset_time) < 10:
        print "skipping segment because peak is within 10s of end"
        return None

    try:
        max_coda_length = find_coda_max_length(smoothed, peak_offset_time, peak_offset_time - (peak_offset_time - phase_start_time)  + phase_length, noise_floor)
        max_coda_length = np.min([max_coda_length, phase_length - (peak_offset_time - phase_start_time)])
    except:
        print "error finding coda length"
        print traceback.format_exc()
        print phase_length, peak_offset_time, phase_start_time
        return None

    (fit_height, b), avg_cost, coda_start_time, coda_length = fit_logenvelope(smoothed, peak_offset_time, peak_height, max_coda_length, min_p_coda_length if P else min_s_coda_length)

    return (b, fit_height, phase_start_time, phase_length, peak_offset_time, peak_height, coda_start_time, coda_length, max_coda_length, avg_cost)

def find_coda_max_length(trace, peak_offset_time, phase_end_time, noise_floor):
    srate = trace.stats.sampling_rate

# choose coda end time as the first 5s window when we fit a positive slope

    num_windows = np.floor((phase_end_time - peak_offset_time - 10 )/5) -1

    for i in np.linspace((peak_offset_time + 10)*srate, (phase_end_time - 10)*srate, num_windows):
        height = trace.data[i]
        cost = lambda(b): logenv_l1_cost(trace.data[i:i+10*srate], gen_logenvelope(10, srate, height, 0, b))
        bounds = ((-.1, 0.1),)
        results = scipy.optimize.brute(cost, bounds, Ns=10, full_output=0)
        b = results[0]

        if np.mean(trace.data[i:i+10*srate]) < noise_floor(i/srate) + 0.5:
            return (i)/srate - peak_offset_time + 10


    return phase_end_time - peak_offset_time

# end old model region
####################################################

def get_first_p_s_arrivals(cursor, event, siteid):
    phase_condition = "(" + " or ".join(["leba.phase='%s'" % (pn) for pn in S_PHASES + P_PHASES]) + ")"
    sql_query="SELECT l.time, l.azimuth, l.snr, pid.id, sid.id FROM leb_arrival l , static_siteid sid, static_phaseid pid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and leba.arid=l.arid and lebo.orid=leba.orid and %s and sid.sta=l.sta and sid.statype='ss' and sid.id=%d and pid.phase=leba.phase" % (event[EV_EVID_COL], phase_condition, siteid)
    cursor.execute(sql_query)
    arrivals = np.array(cursor.fetchall())

    first_p_arrival = None
    for arrival in arrivals:
        if int(arrival[AR_PHASEID_COL]) in P_PHASEIDS:
            first_p_arrival = arrival
            break
    first_s_arrival = None
    for arrival in arrivals:
        if int(arrival[AR_PHASEID_COL]) in S_PHASEIDS:
            first_s_arrival = arrival
            break

    return (first_p_arrival, first_s_arrival)


def get_densest_azi(cursor, siteid):
    max_azi_count = -1
    max_azi = -1
    max_azi_condition = ""
    for azi in np.linspace(0, 330, 12):
        if azi == 330:
            azi_condition = "(l.azimuth between 0 and 30 or l.azimuth between 330 and 360)"
        else:
            azi_condition = "l.azimuth between %f and %f" % (azi, azi+60)
        sql_query="SELECT count(distinct(l.arid)) FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where l.time between 1238889600 and 1245456000 and lebo.mb>4 and leba.arid=l.arid and l.snr > 2 and lebo.orid=leba.orid and sid.sta=l.sta and sid.id=%d and %s" % (siteid, azi_condition)
        cursor.execute(sql_query)
        azi_count = cursor.fetchall()
        if azi_count > max_azi_count:
            max_azi_count = azi_count
            max_azi = azi
            max_azi_condition = azi_condition
    print "max azi is", max_azi, "with count", max_azi_count
    return max_azi




def main():
# boilerplate initialization of various things
    siteid = int(sys.argv[1])
    elephant_model = False
    if len(sys.argv) > 2:

        # "with five parameters I can fit an elephant"
        if sys.argv[2] == "elephant":
            elephant_model = True

    cursor = db.connect().cursor()
    sites = read_sites(cursor)
    st  = 1237680000
    et = st + 3600*24
    site_up = read_uptime(cursor, st, et)
    detections, arid2num = read_detections(cursor, st, et, arrival_table="leb_arrival", noarrays=True)
    phasenames, phasetimedef = read_phases(cursor)
    earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
    sigmodel = learn.load_sigvisa("parameters", st, et, "spectral_envelope", site_up, sites, phasenames, phasetimedef, load_signal_params = False)

    max_azi = get_densest_azi(cursor, siteid)


# want to select all events, with certain properties, which have a P or S phase detected at this station
    phase_condition = "(" + " or ".join(["leba.phase='%s'" % (pn) for pn in S_PHASES + P_PHASES]) + ")"
    sql_query="SELECT distinct lebo.mb, lebo.lon, lebo.lat, lebo.evid, lebo.time, lebo.depth FROM leb_arrival l , static_siteid sid, static_phaseid pid, leb_origin lebo, leb_assoc leba where l.time between 1238889600 and 1245456000 and lebo.mb>4 and leba.arid=l.arid and l.snr > 2 and lebo.orid=leba.orid and %s and sid.sta=l.sta and sid.statype='ss' and sid.id=%d and pid.phase=leba.phase" % (phase_condition, siteid)
#5308821
#5301405
# and lebo.evid=5301449

    print sql_query
    cursor.execute(sql_query)
    events = np.array(cursor.fetchall())

#    bands = ['narrow_logenvelope_4.00_6.00', 'narrow_logenvelope_2.00_3.00', 'narrow_logenvelope_1.00_1.50', 'narrow_logenvelope_0.70_1.00']
    bands = ['narrow_logenvelope_2.00_3.00']
    short_bands = [b[19:] for b in bands]

    runid = int(time.time())
    base_coda_dir = get_base_dir(siteid, None, runid)
    print "writing data to directory", base_coda_dir

    f = open(os.path.join(base_coda_dir, 'all_data'), 'w')
    for b in bands:
        f.write(b + " ")
    f.write("\n")

    learned_p = [[] for b in bands]
    learned_s = [[] for b in bands]
    learned_sp = [[] for b in bands]

    for event in events:

        distance = utils.geog.dist_km((event[EV_LON_COL], event[EV_LAT_COL]), (sites[siteid-1][0], sites[siteid-1][1]))
        azimuth = utils.geog.azimuth((event[EV_LON_COL], event[EV_LAT_COL]), (sites[siteid-1][0], sites[siteid-1][1]))

        (first_p_arrival, first_s_arrival) = get_first_p_s_arrivals(cursor, event, siteid)

        if first_p_arrival is not None and first_s_arrival is not None and first_p_arrival[AR_TIME_COL] > first_s_arrival[AR_TIME_COL]:
            print "skipping evid %d because S comes before P..." % (event[EV_EVID_COL])
            continue

        try:
            (arrival_segment, noise_segment, other_arrivals, other_arrival_phases) = load_signal_slice(cursor, event[EV_EVID_COL], siteid, load_noise = True)
        except:
            print traceback.format_exc()
            continue

        for (band_idx, band) in enumerate(bands):
            short_band = short_bands[band_idx]


            vert_noise_floor = arrival_segment[0]["BHZ"][band].stats.noise_floor
            horiz_noise_floor = arrival_segment[0]["horiz_avg"][band].stats.noise_floor

            try:
                vert, horiz = arrival_segment[0]["BHZ"][band], arrival_segment[0]["horiz_avg"][band]
                vert_smoothed, horiz_smoothed = smoothed_traces(arrival_segment, band)

            except:
                print traceback.format_exc()
                continue

            vnf = lambda t : vert_noise_floor
            hnf = lambda t : horiz_noise_floor
            fit_p_vert = None
            fit_p_horiz = None
            fit_s_vert = None
            fit_s_horiz = None
            accept_p_vert = False
            accept_p_horiz = False
            accept_s_vert = False
            accept_s_horiz = False
            if first_p_arrival is not None:
                fit_p_vert = fit_phase_coda(first_p_arrival, vert_smoothed, other_arrivals, other_arrival_phases, vnf)
                fit_p_horiz = fit_phase_coda(first_p_arrival, horiz_smoothed, other_arrivals, other_arrival_phases, hnf)
                accept_p_vert = accept_fit(fit_p_vert, min_coda_length=min_p_coda_length, max_avg_cost = avg_cost_bound)
                accept_p_horiz = accept_fit(fit_p_horiz, min_coda_length=min_p_coda_length, max_avg_cost = avg_cost_bound)


            if first_s_arrival is not None:

                # if we got a good fit to the P coda, use the continuing P coda as a secondary noise floor for the S coda
                if accept_p_vert:
                    vnf = lambda t : max(vert_noise_floor, fit_p_vert[FIT_HEIGHT] + fit_p_vert[FIT_B]*(t - fit_p_vert[FIT_CODA_START_OFFSET]))
                if accept_p_horiz:
                    hnf = lambda t : max(horiz_noise_floor, fit_p_horiz[FIT_HEIGHT] + fit_p_horiz[FIT_B]*(t - fit_p_horiz[FIT_CODA_START_OFFSET]))

                fit_s_vert = fit_phase_coda(first_s_arrival, vert_smoothed, other_arrivals, other_arrival_phases, vnf)
                fit_s_horiz = fit_phase_coda(first_s_arrival, horiz_smoothed, other_arrivals, other_arrival_phases, hnf)
                accept_s_vert = accept_fit(fit_s_vert, min_coda_length=min_s_coda_length, max_avg_cost = avg_cost_bound)
                accept_s_horiz = accept_fit(fit_s_horiz, min_coda_length=min_s_coda_length, max_avg_cost = avg_cost_bound)



            # DO THE FITTING
            if elephant_model:

                # plot!
                pdf_dir = get_dir(os.path.join(base_coda_dir, short_band))
                pp = PdfPages(os.path.join(pdf_dir, str(int(event[EV_EVID_COL])) + ".pdf"))
                # DO THE FITTING
                fit_vert_params, phaseids, vert_cost = fit_elephant_envelope(sigmodel, pp, [first_p_arrival, first_s_arrival], vert, vert_smoothed, [ x for x in [fit_p_vert, fit_s_vert] if x is not None], evid = str(int(event[EV_EVID_COL])))
#                fit_horiz_params, phaseids, horiz_cost = fit_elephant_envelope(sigmodel, [first_p_arrival, first_s_arrival], horiz, horiz_smoothed, [ x for x in [fit_p_horiz, fit_s_horiz] if x is not None])



#                gen_title = lambda event, fit: "%s evid %d siteid %d mb %f \n dist %f azi %f \n p: %s \n s: %s " % (band, event[EV_EVID_COL], siteid, event[EV_MB_COL], distance, azimuth, fit[0,:],fit[1,:] if fit.shape[0] > 1 else "")
                try:
#                    plot_channels_with_pred(sigmodel, pp, vert_smoothed, fit_vert_params, phaseids, horiz_smoothed, fit_horiz_params, title = gen_title(event, fit_vert_params))
#                 plot_channels_with_pred(sigmodel, pp, vert_smoothed, fit_vert_params, phaseids, horiz_smoothed, None, title = gen_title(event, fit_vert_params))
                    pass
                except:
                    print "error plotting:"
                    print traceback.format_exc()
                print "wrote plot", os.path.join(pdf_dir, str(int(event[EV_EVID_COL])) + ".pdf")

#                pp.close()
#                sys.exit(1)

            else:


        #        print first_p_arrival
        #        print first_s_arrival
        #        print "p vert" , fit_p_vert
        #        print "s horiz", fit_s_horiz

                if accept_p_vert:
                    learned_p[band_idx].append((distance, azimuth, fit_p_vert[FIT_B]))
                if accept_s_horiz:
                    learned_s[band_idx].append((distance, azimuth, fit_s_horiz[FIT_B]))
                if accept_p_vert and accept_s_horiz:
                    learned_sp[band_idx].append((fit_p_horiz[FIT_B], fit_s_horiz[FIT_B]))


                # plot!
                pdf_dir = get_dir(os.path.join(base_coda_dir, short_band))
                pp = PdfPages(os.path.join(pdf_dir, str(int(event[EV_EVID_COL])) + ".pdf"))
                gen_title = lambda event, p_fit, s_fit: "%s evid %d siteid %d mb %f \n dist %f azi %f \n p_b %f p_acost %f p_len %f \n s_b %f s_acost %f s_len %f " % (band, event[EV_EVID_COL], siteid, event[EV_MB_COL], distance, azimuth, safe_lookup(p_fit, FIT_B), safe_lookup(p_fit, FIT_AVG_COST), safe_lookup(p_fit, FIT_CODA_LENGTH), safe_lookup(s_fit, FIT_B), safe_lookup(s_fit, FIT_AVG_COST), safe_lookup(s_fit, FIT_CODA_LENGTH))
                try:
                    plot_channels(pp, vert_smoothed, vert_noise_floor, [fit_p_vert, fit_s_vert], ["g-" if accept_p_vert else "r-", "g-" if accept_s_vert else "r-"], horiz_smoothed, horiz_noise_floor, [fit_p_horiz, fit_s_horiz], ["g-" if accept_p_horiz else "r-", "g-" if accept_s_horiz else "r-"], all_det_times = other_arrivals, all_det_labels = other_arrival_phases, title = gen_title(event, fit_p_vert, fit_s_horiz))
                except:
                    print "error plotting:"
                    print traceback.format_exc()
                print "wrote plot", os.path.join(pdf_dir, str(int(event[EV_EVID_COL])) + ".pdf")


                # write a line to the output file
                f.write("%d %d %d %d %d %f %f %f %f %f " % (event[EV_EVID_COL], siteid, band_idx, first_p_arrival[AR_PHASEID_COL] if first_p_arrival is not None else -1, first_s_arrival[AR_PHASEID_COL] if first_s_arrival is not None else -1, distance, azimuth, event[EV_LON_COL], event[EV_LAT_COL], event[EV_MB_COL]))
                write_fit = lambda f, fit: map(lambda x : f.write("%f " % (x) ), fit)
                if first_p_arrival is not None:
                    write_fit(f, fit_p_vert)
                    write_fit(f, fit_p_horiz)
                else:
                    f.write("-1 " * FIT_NUM_COLS)
                    f.write("-1 " * FIT_NUM_COLS)
                if first_s_arrival is not None:
                    write_fit(f, fit_s_vert)
                    write_fit(f, fit_s_horiz)
                else:
                    f.write("-1 " * FIT_NUM_COLS)
                    f.write("-1 " * FIT_NUM_COLS)

                f.write("%f %f\n" % (vert_noise_floor, horiz_noise_floor))

            pp.close()

        del arrival_segment
        del noise_segment
        del other_arrivals
        del other_arrival_phases
    #    print hp.heap()

    f.close()

if __name__ == "__main__":
    main()






