#######################################
# Begin old hack heuristic coda fitting
# Here be dragons
#######################################

import numpy as np
import scipy
import sys, os, traceback
from sigvisa import *
from database.dataset import *
from noise.noise_model import *

from source.event import Event

ARR_TIME_PARAM, PEAK_OFFSET_PARAM, CODA_HEIGHT_PARAM, CODA_DECAY_PARAM, NUM_PARAMS = range(4+1)

(HEURISTIC_FIT_B, HEURISTIC_FIT_HEIGHT, HEURISTIC_FIT_PHASE_START_TIME, HEURISTIC_FIT_PHASE_LENGTH, HEURISTIC_FIT_PEAK_OFFSET, HEURISTIC_FIT_PEAK_HEIGHT, HEURISTIC_FIT_CODA_START_OFFSET, HEURISTIC_FIT_CODA_LENGTH, HEURISTIC_FIT_MAX_CODA_LENGTH, HEURISTIC_FIT_AVG_COST, HEURISTIC_FIT_NUM_COLS) = range(10+1)


avg_cost_bound = .28
min_p_coda_length = 30
min_s_coda_length = 45

def logsub_noise(log_height, log_noise):
    return np.log ( np.exp(log_height) - np.exp(log_noise) )

def accept_fit(fit, min_coda_length=40, max_avg_cost=avg_cost_bound):
# print fit[HEURISTIC_FIT_B], fit[HEURISTIC_FIT_CODA_LENGTH], fit[HEURISTIC_FIT_AVG_COST]
    return fit[HEURISTIC_FIT_B] > -0.15 and fit[HEURISTIC_FIT_B] <= 0 and fit[HEURISTIC_FIT_CODA_LENGTH] >= (min_coda_length-0.1) and fit[HEURISTIC_FIT_AVG_COST] <= max_avg_cost


def find_starting_params(smoothed, detected_phases_only=True):
    """ Uses various heuristics to come up with a good initialization
    for the fitting process. Also constructs a list of bounds
    appropriate for passing to a scipy optimization function."""

    s = Sigvisa()

    avg_cost_bound = 0.2

    noise_floor = smoothed.stats.noise_floor
    nf = lambda t : noise_floor
    accept_p = False
    accept_s = False

    ev = Event(evid=smoothed.stats['evid'])
    sta = smoothed.stats['sta']
    siteid = smoothed.stats['siteid']

    heuristic_fits = []

    arrivals = smoothed.stats['event_arrivals']
    arrival_times = arrivals[:, DET_TIME_COL]
    arrival_phases = [s.phasenames[pid_m1] for pid_m1 in arrivals[:, DET_PHASE_COL]]

    accepted_previous_fit = False
    for arrival,phase in zip(arrivals, arrival_phases):
        if accepted_previous_fit:
            nf = lambda t : max(noise_floor, fit[HEURISTIC_FIT_HEIGHT] + fit[HEURISTIC_FIT_B]*(t - fit[HEURISTIC_FIT_CODA_START_OFFSET]))

        fit = fit_phase_coda(arrival, smoothed, arrivals, arrival_phases, nf)

        min_coda_length = 30 if phase in s.P_phases else 45
        accepted_previous_fit = accept_fit(fit, min_coda_length=min_coda_length, max_avg_cost = avg_cost_bound)
        heuristic_fits.append(fit)

    # initialize default params for all arriving phases (including those not actually detected)

    if detected_phases_only:
        all_phases = arrival_phases
    else:
        all_phases = s.arriving_phases(ev, sta)

    start_params = np.zeros((len(all_phases), NUM_PARAMS))
    for (i, phase) in enumerate(all_phases):
        start_params[i, ARR_TIME_PARAM] = ev.time + s.sigmodel.mean_travel_time(ev.lon, ev.lat, ev.depth, siteid-1, s.phaseids[phase]-1)
        start_params[i, PEAK_OFFSET_PARAM] = 3
        start_params[i, CODA_HEIGHT_PARAM] = 1e-10
        start_params[i, CODA_DECAY_PARAM] = -0.03

    # copy over heuristic coda fits for detected phases
    for (time, phase, fit) in zip(arrival_times, arrival_phases, heuristic_fits):
        i = all_phases.index(phase)
        start_params[i, ARR_TIME_PARAM] = time
        fit_peak_height = logsub_noise(fit[HEURISTIC_FIT_PEAK_HEIGHT], noise_floor)
        fit_coda_height = logsub_noise(fit[HEURISTIC_FIT_HEIGHT] - fit[HEURISTIC_FIT_B] *(fit[HEURISTIC_FIT_CODA_START_OFFSET] - fit[HEURISTIC_FIT_PEAK_OFFSET]), noise_floor)

        start_params[i, PEAK_OFFSET_PARAM] = fit[HEURISTIC_FIT_PEAK_OFFSET] - fit[HEURISTIC_FIT_PHASE_START_TIME]
        start_params[i, CODA_HEIGHT_PARAM] = fit_coda_height if fit_coda_height > 0 else 1
        start_params[i, CODA_DECAY_PARAM] = fit[HEURISTIC_FIT_B] if fit[HEURISTIC_FIT_B] < 0 else -0.03

    return (all_phases, start_params)

def fit_phase_coda(phase_arrival, smoothed, arrivals, arrival_phases, noise_floor):
    """
    phase_arrival: Arrival array for a single phase (loaded from e.g. the leb_arrival db table)
    smoothed: smoothed waveform trace
    arrivals: matrix of all arrivals for this event at this station
    arrival_phases: names of these arrivals
    noise_floor: real number giving the background noise level
    """

    npts = smoothed.stats.npts
    srate = smoothed.stats.sampling_rate
    stime = smoothed.stats.starttime_unix

    s = Sigvisa()
    phase = s.phasenames[phase_arrival[DET_PHASE_COL]]
    P = True if phase in s.P_phases else False

    phase_length = 200
    if len(arrivals) > 0:
        for (a, pa) in zip(arrivals[:, DET_TIME_COL], arrival_phases):
            if a > phase_arrival[DET_TIME_COL] and pa != "LR":
                phase_length = np.min([a - phase_arrival[DET_TIME_COL], phase_length])

    # compute phase arrival times relative to the trace start time
    phase_start_time = phase_arrival[DET_TIME_COL] - stime
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
    except IndexError:
        print "error finding coda length"
        print traceback.format_exc()
        print phase_length, peak_offset_time, phase_start_time
        return None

    (fit_height, b), avg_cost, coda_start_time, coda_length = fit_logenvelope(smoothed, peak_offset_time, peak_height, max_coda_length, min_p_coda_length if P else min_s_coda_length)

    return (b, fit_height, phase_start_time, phase_length, peak_offset_time, peak_height, coda_start_time, coda_length, max_coda_length, avg_cost)

def logenv_l1_cost(true_env, logenv):
    n = len(true_env)
    n2 = len(logenv)
    if n != n2:
        if np.abs(n-n2) > 5:
            print "warning: comparing unequal-length traces (%d vs %d)" % (n, n2)
        n = np.min([n, n2])
    c = np.sum (np.abs(true_env[:n] - logenv[:n]))
    return c

def gen_logenvelope(length, sampling_rate, peak_height, gamma, b):
# print length, sampling_rate, peak_height, gamma, b
    t = np.linspace(1/sampling_rate, length, length*sampling_rate)
    f = (gamma*-1)*np.log(t) + (b * t)

    offset = peak_height - f[0]
    f = f + offset

    return f

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

def arrival_peak_offset(trace, window_start_offset, window_end_offset = None):
    srate = trace.stats.sampling_rate

    if window_end_offset is None:
        window_end_offset = window_start_offset + 15

    i = np.floor(window_start_offset*srate)
    j = np.floor(window_end_offset*srate)

#    print window_start_offset, window_end_offset, i, j, srate, trace.data.shape

    pt = np.argmax(trace.data[i:j]) / srate
    return (pt +window_start_offset, trace.data[(pt+window_start_offset) * srate ])


# end old model region
####################################################
