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

from optparse import OptionParser

import utils.nonparametric_regression as nr
from priors.coda_decay.coda_decay_common import *
from priors.coda_decay.plot_coda_decays import *
from priors.coda_decay.train_wiggles import *
from priors.coda_decay.templates import *

def arrival_peak_offset(trace, window_start_offset, window_end_offset = None):
    srate = trace.stats.sampling_rate

    if window_end_offset is None:
        window_end_offset = window_start_offset + 15

    i = np.floor(window_start_offset*srate)
    j = np.floor(window_end_offset*srate)

    print window_start_offset, window_end_offset, i, j, srate, trace.data.shape

    pt = np.argmax(trace.data[i:j]) / srate
    return (pt +window_start_offset, trace.data[(pt+window_start_offset) * srate ])




def coord_descent(f, x0, converge=0.1, steps=None, maxiters=500):
    ncoords = len(x0)
    x = x0.copy()
    v = f(x)
    for i in range(maxiters):
        incr = 0
        for p in np.random.permutation(ncoords):

            # try taking steps in both directions
            step = steps[p]
            x[p] = x[p] + step
            v1 = f(x)
            x[p] = x[p] - 2*step
            v2 = f(x)
            if v <= v1 and v <= v2:
                x[p] = x[p] + step
                continue

            # continue stepping in the best direction, until there's
            # no more improvement.
            if v1 < v2:
                vold = v1
                x[p] = x[p] + 3 * step
                sign = 1
            else:
                vold = v2
                sign = -1
                x[p] = x[p] - step
            vnew = f(x)
            while vnew <= vold:
                x[p] = x[p] + sign*step
                vold = vnew
                vnew = f(x)

            x[p] = x[p] - sign*step
            incr = np.max([v - vold, incr])
            v = vold
        if incr < converge:
            break
        if i % 10 == 0:
            print "coord iter %d incr %f" % (i, incr)
    return x

def optimize(f, start_params, bounds, method, phaseids=None, maxfun=None):
    if phaseids is not None:
        return optimize_by_phase(f, start_params, bounds, phaseids, method=method,maxfun=maxfun)
    else:
        return minimize(f, start_params, bounds=bounds, method=method, steps=[.1, .1, .005] * (len(start_params)/3), maxfun=maxfun)

def minimize(f, x0, method="bfgs", bounds=None, steps=None, maxfun=None):
    if method=="bfgs":
        x1, best_cost, d = scipy.optimize.fmin_l_bfgs_b(f, x0, approx_grad=1, bounds=bounds, epsilon = 1e-1, factr=1e12, maxfun=maxfun)
    elif method=="tnc":
        x1, nfeval, rc = scipy.optimize.fmin_tnc(f, x0, approx_grad=1, bounds=bounds, maxfun=maxfun)
        x1 = np.array(x1)
    elif method=="simplex":
        x1 = scipy.optimize.fmin(f, x0, maxfun=maxfun, xtol=0.01, ftol=0.01)
    elif method=="anneal":
        x1, jmin, T, feval, iters, accept, retval = scipy.optimize.anneal(f, x0, maxeval=maxfun)
    elif method=="coord":
        x1 = coord_descent(f, x0, steps=steps)
    else:
        raise Exception("unknown optimization method %s" % (method))
    return x1, f(x1)


def optimize_by_phase(f, start_params, bounds, phaseids, method="bfgs", iters=3, maxfun=None):
    nphase_params = len(start_params) / len(phaseids)
    params = start_params.copy()
    for i in range(iters):
        for (pidx, phaseid) in enumerate(phaseids):
            sidx = pidx*nphase_params
            eidx = (pidx+1)*nphase_params
            phase_params = params[sidx:eidx]
            phase_bounds = bounds[sidx:eidx]
            apf = lambda pp : f(np.concatenate([params[:sidx], pp, params[eidx:]]))
            phase_params, c = minimize(apf, phase_params, method=method, bounds=phase_bounds, steps = [.1, .1, .005], maxfun=maxfun)
            print "params", phase_params, "cost", c
            params = np.concatenate([params[:sidx], phase_params, params[eidx:]])
    return params, c

def extract_wiggles(tr, tmpl, arrs, threshold=2.5):

    srate = tr.stats.sampling_rate
    st = tr.stats.starttime_unix
    nf = tmpl.stats.noise_floor

    wiggles = []
    for (phase_idx, phase) in enumerate(arrs["arrival_phases"]):
        start_wiggle = arrs["arrivals"][phase_idx]+1
        start_idx = np.ceil((start_wiggle - st)*srate)

        try:
            next_phase_idx = np.ceil((arrs["arrivals"][phase_idx+1] - st)*srate)
        except:
            next_phase_idx = np.float('inf')
        for t in range(150):
            end_idx = start_idx + np.ceil(srate*t)
            if (end_idx >= next_phase_idx) or (tmpl[end_idx]/nf < np.exp(threshold) ):
                break

        wiggle = tr[start_idx:end_idx] / tmpl[start_idx:end_idx]
        wiggles.append(wiggle)

    return wiggles


def learn_wiggle_params(sigmodel, env, smoothed, phaseids, params):


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

def find_starting_params(arr, smoothed):
    """ Uses various heuristics to come up with a good initialization
    for the fitting process. Also constructs a list of bounds
    appropriate for passing to a scipy optimization function."""

    smoothed = Trace(np.log(smoothed.data), smoothed.stats.copy())
    smoothed.stats.noise_floor = np.log(smoothed.stats.noise_floor)
    noise_floor = smoothed.stats.noise_floor

    arr_bounds = [ (0, 15), (0, 15) , (0, 10), (0, 15), (-.2, 0) ]
    arr_bounds_fixed_peak = [ (0, 15), (0, 10), (-.15, 0) ]

    nf = lambda t : noise_floor
    accept_p = False
    accept_s = False

    arrivals = []
    defaults = []
    if arr["first_p_arrival"] is not None:
        fit_p = fit_phase_coda(arr["first_p_arrival"], smoothed, arr["arrivals"], arr["arrival_phases"], nf)
        smoothed.stats.fit_p = fit_p
        accept_p = accept_fit(smoothed.stats.fit_p, min_coda_length=min_p_coda_length, max_avg_cost = avg_cost_bound)
        arrivals.append(arr["first_p_arrival"])
        defaults.append(smoothed.stats.fit_p)

    if arr["first_s_arrival"] is not None:
        # if we got a good fit to the P coda, use the continuing P coda as a secondary noise floor for the S coda
        if accept_p:
            nf = lambda t : max(noise_floor, fit_p[HEURISTIC_FIT_HEIGHT] + fit_p[HEURISTIC_FIT_B]*(t - fit_p[HEURISTIC_FIT_CODA_START_OFFSET]))

        fit_s = fit_phase_coda(arr["first_s_arrival"], smoothed, arr["arrivals"], arr["arrival_phases"], nf)
        smoothed.stats.fit_s = fit_s
        arrivals.append(arr["first_s_arrival"])
        defaults.append(smoothed.stats.fit_s)

    start_params = np.zeros((len(arrivals), NUM_PARAMS))
    bounds = []
    bounds_fp = []
    phaseids = []
    arr_times = np.zeros((len(arrivals), 1))
    for i, arr in enumerate(arrivals):
        time = arr[AR_TIME_COL]

        a = defaults[i]
        noise_floor = smoothed.stats.noise_floor

        fit_peak_height = logsub_noise(a[HEURISTIC_FIT_PEAK_HEIGHT], noise_floor)
        fit_coda_height = logsub_noise(a[HEURISTIC_FIT_HEIGHT] - a[HEURISTIC_FIT_B] *(a[HEURISTIC_FIT_CODA_START_OFFSET] - a[HEURISTIC_FIT_PEAK_OFFSET]), noise_floor)

        start_params[i, PEAK_OFFSET_PARAM] = ( a[HEURISTIC_FIT_PEAK_OFFSET] + smoothed.stats.starttime_unix) - time
        print "init peak offset to ", start_params[i, PEAK_OFFSET_PARAM], "for phase", i
        start_params[i, PEAK_HEIGHT_PARAM] = fit_peak_height if fit_peak_height > 0 else 1
        start_params[i, PEAK_DECAY_PARAM] = 5
        start_params[i, CODA_HEIGHT_PARAM] = fit_coda_height if fit_coda_height > 0 else 1
        start_params[i, CODA_DECAY_PARAM] = a[HEURISTIC_FIT_B] if a[HEURISTIC_FIT_B] < 0 else -0.03

        bounds = bounds + arr_bounds
        bounds_fp = bounds_fp + arr_bounds_fixed_peak
        phaseids.append(arr[AR_PHASEID_COL])
        arr_times[i] = time

    start_params = start_params[:, 1:]
    return start_params, phaseids, bounds, bounds_fp



def fit_template(sigmodel, pp, arrs, env, smoothed, fix_peak = True, evid=None, method="bfgs", by_phase=False, wiggles=None, cursor=None, init_runid=None):

    start_params, phaseids, bounds, bounds_fp = find_starting_params(arrs, smoothed)
    narrs = len(arrs["arrivals"])
    arr_times = np.reshape(np.array(arrs["arrivals"]), (narrs, -1))

    if fix_peak:
        start_params = remove_peak(start_params)
        bounds = bounds_fp
        assem_params = lambda params: np.hstack([arr_times, restore_peak(np.reshape(params, (narrs, -1)))])
    else:
        assem_params = lambda params: np.hstack([arr_times, np.reshape(params, (narrs, -1))])

    start_params = start_params.flatten()

    print "start params", start_params

    #gen_title = lambda event, fit: "%s evid %d siteid %d mb %f \n dist %f azi %f \n p: %s \n s: %s " % (band, event[EV_EVID_COL], siteid, event[EV_MB_COL], distance, azimuth, fit[0,:],fit[1,:] if fit.shape[0] > 1 else "")

    set_ar_processes(sigmodel, env, phaseids)

    f = lambda params : c_cost(sigmodel, smoothed, phaseids, assem_params(params), iid=True)
    start_cost = f(start_params)
    print "start params cost (w/ smoothed data and iid noise)", start_cost
    if pp is not None:
        plot_channels_with_pred(sigmodel, pp, smoothed, assem_params(start_params), phaseids, None, None, title = "start smoothed iid (cost %f, evid %s)" % (start_cost, evid))

    best_params = None
    if init_runid is not None:
        best_params, phaseids_loaded, fit_cost = load_template_params(cursor, int(evid), env.stats.channel, env.stats.short_band, init_runid, env.stats.siteid)
        best_params = best_params[:, 1:]
        if fix_peak:
            best_params = remove_peak(best_params)
        best_params = best_params.flatten()
        print "loaded"
        print_params(assem_params(best_params))

        if phaseids_loaded != phaseids:
            best_params = None
        elif pp is not None:
            plot_channels_with_pred(sigmodel, pp, smoothed, assem_params(best_params), phaseids, None, None, title = "loaded smoothed iid (cost %f, evid %s)" % (start_cost, evid))

    if wiggles is None or best_params is None:
        # learn from smoothed data w/ iid noise
        best_params, best_cost = optimize(f, start_params, bounds, method=method, phaseids= (phaseids if by_phase else None))
        if pp is not None:
            plot_channels_with_pred(sigmodel, pp, smoothed, assem_params(best_params), phaseids, None, None, title = "best iid (cost %f, evid %s)" % (best_cost, evid))

    if wiggles is not None:
        load_wiggle_models(cursor, sigmodel, wiggles)
        f = lambda params : c_cost(sigmodel, env, phaseids, assem_params(params))
        print "loaded cost is", f(best_params)
        best_params, best_cost = optimize(f, best_params, bounds, method=method, phaseids= (phaseids if by_phase else None))
        if pp is not None:
            plot_channels_with_pred(sigmodel, pp, env, assem_params(best_params), phaseids, None, None, title = "best (cost %f, evid %s)" % (best_cost, evid))

    return assem_params(best_params), phaseids, best_cost

#######################################################
# Begin old fitting routines

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

def fit_phase_coda(phase_arrival, smoothed, arrivals, arrival_phases, noise_floor):
    npts = smoothed.stats.npts
    srate = smoothed.stats.sampling_rate
    stime = smoothed.stats.starttime_unix

    P = True if int(phase_arrival[AR_PHASEID_COL]) in P_PHASEIDS else False

    phase_length = 200
    if len(arrivals) > 0:
        for (a, pa) in zip(arrivals, arrival_phases):
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

def get_first_p_s_arrivals(cursor, evid, siteid):
    phase_condition = "(" + " or ".join(["leba.phase='%s'" % (pn) for pn in S_PHASES + P_PHASES]) + ")"
    sql_query="SELECT l.time, l.azimuth, l.snr, pid.id, sid.id FROM leb_arrival l , static_siteid sid, static_phaseid pid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and leba.arid=l.arid and lebo.orid=leba.orid and %s and sid.sta=l.sta and sid.statype='ss' and sid.id=%d and pid.phase=leba.phase" % (evid, phase_condition, siteid)
    cursor.execute(sql_query)
    arrivals = np.array(cursor.fetchall())

    first_p_arrival = None
    p_phaseid = None
    for arrival in arrivals:
        if int(arrival[AR_PHASEID_COL]) in P_PHASEIDS:
            first_p_arrival = arrival
            p_phaseid = int(arrival[AR_PHASEID_COL])
            break
    first_s_arrival = None
    s_phaseid = None
    for arrival in arrivals:
        if int(arrival[AR_PHASEID_COL]) in S_PHASEIDS:
            first_s_arrival = arrival
            s_phaseid = int(arrival[AR_PHASEID_COL])
            break

    if first_p_arrival is not None and first_s_arrival is not None and first_p_arrival[AR_TIME_COL] > first_s_arrival[AR_TIME_COL]:
        print "warning: %d S arrival comes before P, ignoring..." % (evid)
        first_p_arrival=None
        first_s_arrival=None

    arrivals = [first_p_arrival, first_s_arrival]
    arrivals = [x[AR_TIME_COL] for x in arrivals if x is not None]
    phaseids = [p_phaseid, s_phaseid]
    phases= [phaseid_to_name(x) for x in phaseids if x is not None]
    return first_p_arrival, first_s_arrival, arrivals, phases

def load_segments(cursor, evid, siteid, ar_noise=True, chans=None, bands=None):
    print "bands", bands
    print "chans", chans
    (arrival_segment, noise_segment, all_arrivals, all_arrival_phases, all_arrival_arids) = load_signal_slice(cursor, evid, siteid, load_noise = True, learn_noise=ar_noise, chans=chans, bands=bands)
    arrival_segment = arrival_segment[0]

    # reject segments too short to do an accurate coda fit
    c = arrival_segment.keys()[0]
    b = arrival_segment[c].keys()[0]
    tr = arrival_segment[c][b]
    npts = tr.stats.npts
    srate = tr.stats.sampling_rate
    if npts < srate * MIN_SEGMENT_LENGTH:
        raise Exception("minimum segment length %.2fs, skipping segment withlength %.2f" % (MIN_SEGMENT_LENGTH,  npts/srate))

    # package together information about arriving phases into a single
    # dictionary
    arrs = {"all_arrivals": all_arrivals, "all_arrival_phases": all_arrival_phases, "all_arrival_arids": all_arrival_arids}
    arrs["first_p_arrival"], arrs["first_s_arrival"], arrs["arrivals"], arrs["arrival_phases"] = get_first_p_s_arrivals(cursor, evid, siteid)

    smoothed_segment = smooth_segment(arrival_segment, bands=bands, chans=chans)
    return arrival_segment, smoothed_segment, arrs


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


def demo_get_wiggles():

    cursor, sigmodel, earthmodel, sites, dbconn = sigvisa_util.init_sigmodel()
    tr, smoothed, tmpl, phases, wiggles, wiggles_smooth = get_wiggles(cursor, sigmodel, 5301405, 2)
    print tr
    print smoothed
    print tmpl
    print phases
    print wiggles
    print wiggles_smooth

def get_wiggles(cursor, sigmodel, evid, siteid, chan='BHZ', band='narrow_envelope_2.00_3.00', wiggle_threshold=2):
    """

    Arguments:
    cursor, sigmodel: objects initialized with sigvisa_util.init_sigmodel()
    evid, siteid: event ID (from leb_origin DB table) and site id (from static_siteid table)
    chan, band: channel and band strings
    wiggle_threshold: log-height above the noise level at which we cut off wiggle extraction (too close to the noise level and the fluctuations we see might be from noise rather than from wiggles). (TODO: determine this automatically using the learned noise variance)

    Returns:
    tr: A Trace object containing the log-envelope for the given band/channel, beginning 30 seconds before the first phase arrival associated with the given event, and continuing for 170 seconds after the final phase arrival.
    smoothed: The same as tr, but smoothed using a moving average (currently a Hamming window of length approximately 7.5 seconds)
    tmpl: A Trace object covering the same time period as tr and smoothed, but containing an empirically-fit log-envelope template.
    phases: a list of strings, giving the phase names for which wiggles were extracted
    wiggles: a list of wiggles (each in the form of an np.array object) extracted from the (unsmoothed) log-envelope.
    wiggles_smooth: a list of wiggles extracted from the smoothed log-envelope.
    """


    # load the relevant traces
    arrival_segment, smoothed_segment, arrs = load_segments(cursor, evid, siteid, ar_noise=False, chans=[chan,], bands=[band,])
    tr = arrival_segment[chan][band]
    smoothed = smoothed_segment[chan][band]

    # fit an envelope template
    start_params, phaseids, bounds, bounds_fp = find_starting_params(arrs, smoothed)
    start_params = remove_peak(start_params)
    start_params = start_params.flatten()
    bounds = bounds_fp

    c = sigvisa.canonical_channel_num(chan)
    b = sigvisa.canonical_band_num(band)
    sigmodel.set_noise_process(siteid, b, c, smoothed.stats.noise_floor, 1, np.array((.8,)))
    sigmodel.set_wiggle_process(siteid, b, 1, 1, np.array((.8,)))

    narrs = len(arrs["arrivals"])
    arr_times = np.reshape(np.array(arrs["arrivals"]), (narrs, -1))
    assem_params = lambda params: np.hstack([arr_times, restore_peak(np.reshape(params, (narrs, -1)))])
    f = lambda params : c_cost(sigmodel, smoothed, phaseids, assem_params(params), iid=True)
    best_params, best_cost = optimize(f, start_params, bounds, phaseids, method="simplex", by_phase=False)

    print "start params"
    print_params(assem_params(start_params))
    print "found params"
    print_params(assem_params(best_params))

    tmpl = get_template(sigmodel, tr, phaseids, assem_params(best_params))
    tmpls = get_template(sigmodel, tr, phaseids, assem_params(start_params))
    diff = subtract_traces(tr, tmpl)
    diff_smooth = subtract_traces(smoothed, tmpl)

    # p/s wiggles
    wiggles = extract_wiggles(tr, tmpl, arrs, threshold=1)
    wiggles_smooth = extract_wiggles(smoothed, tmpl, arrs, threshold=1)

    return tr, smoothed, tmpl, arrs["arrival_phases"], wiggles, wiggles_smooth

def main():
# boilerplate initialization of various things


    parser = OptionParser()

    parser.add_option("-s", "--siteid", dest="siteid", default=None, type="int", help="siteid of station for which to fit templates")
    parser.add_option("-m", "--method", dest="method", default="simplex", type="str", help="fitting method (iid)")
    parser.add_option("-r", "--runid", dest="runid", default=None, type="int", help="runid")
    parser.add_option("-e", "--evid", dest="evid", default=None, type="int", help="event ID")
    parser.add_option("-w", "--wiggles", dest="wiggles", default=None, type="str", help="filename of wiggle-model params to load (default is to ignore wiggle model and do iid fits)")
    parser.add_option("--init_runid", dest="init_runid", default=None, type="int", help="initialize template fitting with results from this runid")
    parser.add_option("-p", "--plot", dest="plot", default=False, action="store_true", help="save plots")

    (options, args) = parser.parse_args()

    siteid = options.siteid
    method = options.method
    runid = options.runid
    evid = options.evid

    iid=True
    by_phase=False
    snr_threshold=2

    evid_condition = "and lebo.mb>5 and d.label='training' and l.time between d.start_time and d.end_time and l.snr > 5" if evid is None else "and evid=%d" % (evid)

    cursor, sigmodel, earthmodel, sites, dbconn = sigvisa_util.init_sigmodel()
    load_wiggle_models(cursor, sigmodel, "parameters/signal_wiggles.txt")
    
# want to select all events, with certain properties, which have a P or S phase detected at this station
    phase_condition = "(" + " or ".join(["leba.phase='%s'" % (pn) for pn in S_PHASES + P_PHASES]) + ")"
    sql_query="SELECT distinct lebo.lon, lebo.lat, lebo.depth, lebo.time, lebo.mb, lebo.orid, lebo.evid FROM leb_arrival l , static_siteid sid, static_phaseid pid, leb_origin lebo, leb_assoc leba, dataset d where leba.arid=l.arid and lebo.orid=leba.orid and %s and sid.sta=l.sta and sid.statype='ss' and sid.id=%d %s and pid.phase=leba.phase" % (phase_condition, siteid, evid_condition)
#5308821
#5301405
# and lebo.evid=5301449



    print sql_query
    cursor.execute(sql_query)
    events = np.array(cursor.fetchall())

#    bands = ['narrow_envelope_4.00_6.00', 'narrow_envelope_2.00_3.00', 'narrow_envelope_1.00_1.50', 'narrow_envelope_0.70_1.00']
    short_bands = [b[16:] for b in bands]

    if runid is None:
        cursor.execute("select max(runid) from sigvisa_coda_fits")
        runid, = cursor.fetchone()
        if runid is None:
            runid=0
        else:
            runid = int(runid)+1

    base_coda_dir = get_base_dir(siteid, runid)

    for event in events:
        evid = int(event[EV_EVID_COL])

        if len(events)>1:
            cmd_str = "python2.6 -m priors.coda_decay.fit_shape_params -r %d -e %d -m %s -s %d" % (runid, evid, method, siteid)
            print "running", cmd_str
            os.system(cmd_str)
            continue

        distance = utils.geog.dist_km((event[EV_LON_COL], event[EV_LAT_COL]), (sites[siteid-1][0], sites[siteid-1][1]))
        azimuth = utils.geog.azimuth((sites[siteid-1][0], sites[siteid-1][1]), (event[EV_LON_COL], event[EV_LAT_COL]))

        try:
            arrival_segment, smoothed_segment, arrs = load_segments(cursor, event[EV_EVID_COL], siteid, bands=bands, chans=chans)

            for (band_idx, band) in enumerate(bands):
                short_band = short_bands[band_idx]
                pdf_dir = get_dir(os.path.join(base_coda_dir, short_band))

                for chan in chans:

                    if options.plot:
                        fname = os.path.join(pdf_dir, "%d_%s.pdf" % (evid, chan))
                        print "writing to %s..." % (fname,)
                        pp = PdfPages(fname)
                    else:
                        pp = None
                    tr = arrival_segment[chan][band]
                    smoothed = smoothed_segment[chan][band]

                    st = tr.stats.starttime_unix
                    srate = tr.stats.sampling_rate
                    time_len = len(tr.data)/srate
                    et = st + time_len

                    # DO THE FITTING
                    if method == "load":
                        fit_params, phaseids, fit_cost = load_template_params(cursor, evid, chan, short_band, options.init_runid, siteid)
                        if fit_params is None:
                            print "no params in database for evid %d siteid %d runid %d chan %s band %s, skipping" % (evid, siteid, runid, chan, short_band)
                            continue
                        set_noise_process(sigmodel, tr)
                        fit_cost = fit_cost * time_len
                    else:
                        fit_params, phaseids, fit_cost = fit_template(sigmodel, pp, arrs, tr, smoothed, evid = str(evid), method=method, wiggles=options.wiggles, by_phase=by_phase, cursor=cursor, init_runid=options.init_runid)
                        if pp is not None:
                            print "wrote plot", os.path.join(pdf_dir, "%d_%s.pdf" % (evid, chan))

                    tmpl = get_template(sigmodel, tr, phaseids, fit_params)
                    wiggles = extract_wiggles(tr, tmpl, arrs, threshold=snr_threshold)
                    wiggles2 = extract_wiggles(arrival_segment[chan]['broadband'], tmpl, arrs, threshold=snr_threshold)
                    for (pidx, phaseid) in enumerate(phaseids):
                        if wiggles[pidx] is None or len(wiggles[pidx]) == 0:
                            continue
                        else:
                            dirname = os.path.join("wiggles", str(int(runid)), str(int(siteid)), str(int(phaseid)), short_band)
                            dirname2 = os.path.join("wiggles", str(int(runid)), str(int(siteid)), str(int(phaseid)))
                            fname = os.path.join(dirname, "%d_%s.dat" % (evid, chan))
                            fname2 = os.path.join(dirname2, "%d_%s_raw.dat" % (evid, chan))
                            get_dir(dirname)
                            get_dir(dirname2)
                            print "saving phase %d len %d" % (phaseid, len(wiggles[pidx]))
                            np.savetxt(fname, np.array(wiggles[pidx]))
                            sql_query = "INSERT INTO sigvisa_wiggle_wfdisc (runid, arid, siteid, phaseid, band, chan, evid, fname, snr) VALUES (%d, %d, %d, %d, '%s', '%s', %d, '%s', %f)" % (runid, arrs["all_arrival_arids"][pidx], siteid, phaseid, short_band, chan, evid, fname, snr_threshold)
                            cursor.execute(sql_query)

                            np.savetxt(fname2, np.array(wiggles2[pidx]))
                            sql_query = "INSERT INTO sigvisa_wiggle_wfdisc (runid, arid, siteid, phaseid, band, chan, evid, fname, snr) VALUES (%d, %d, %d, %d, '%s', '%s', %d, '%s', %f)" % (runid, arrs["all_arrival_arids"][pidx], siteid, phaseid, "broadband", chan, evid, fname, snr_threshold)
                            cursor.execute(sql_query)

                    s = [method,]
                    if by_phase:
                        s.append('byphase')
                    method_str = '_'.join(s)
                    if method != "load":
                        for (i, arid) in enumerate(arrs["all_arrival_arids"]):
                            sql_query = "INSERT INTO sigvisa_coda_fits (runid, arid, chan, band, peak_delay, peak_height, peak_decay, coda_height, coda_decay, optim_method, iid, stime, etime, acost, dist, azi) VALUES (%d, %d, '%s', '%s', %f, NULL, NULL, %f, %f, '%s', %d, %f, %f, %f, %f, %f)" % (runid, arid, chan, short_band, fit_params[i, PEAK_OFFSET_PARAM], fit_params[i, CODA_HEIGHT_PARAM], fit_params[i, CODA_DECAY_PARAM], method_str, 1 if iid else 0, st, et, fit_cost/time_len, distance, azimuth)
                            print sql_query
                            cursor.execute(sql_query)
                    dbconn.commit()
                    if pp is not None:
                        pp.close()

        except KeyboardInterrupt:
            dbconn.commit()
            if pp is not None:
                pp.close()
            raise
        except:
            dbconn.commit()
            if pp is not None:
                pp.close()
            print traceback.format_exc()
            continue

    dbconn.close()

if __name__ == "__main__":
    main()






