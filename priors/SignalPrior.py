import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import time

from database.dataset import *

from obspy.core import Trace, Stream, UTCDateTime
import learn as l, sigvisa_util
import sigvisa

LEARN_NOISE = True
LEARN_SHAPE = True
LEARN_PERTURB = False

def learn(cursor, earthmodel, sigmodel, events, leb_evlist, detections, arid2num, param_dirname, start_time, end_time, fake_det, sites):
    """
    Find maximum-likelhood parameter estimates. 
    """

    sigmodel.set_fake_detections(fake_det)

    try:
        params = read_params(os.path.join(param_dirname, 
                                          "EnvelopeSignalModel.txt"))
        sigmodel.set_all_signal_params(siteid, params)
    except:
        params = default_params(sites)

    # iterate over stations to build a model at each station
    for (id_minus1, site) in enumerate(sites):
        if site[SITE_IS_ARRAY] == 1:
            continue

        siteid = id_minus1 + 1

        print "at siteid %d:" % (siteid)

        # load signals for this station
        energies, traces = sigvisa_util.load_and_process_traces(cursor, start_time, end_time, stalist=[siteid, ])

        if len(energies) == 0:
            print "no waveform data is available in this time period for siteid %d, skipping..." % (siteid)
            continue

        if LEARN_NOISE:
            print " learning noise params..."
            ttimes = ttimes_from_assoc(leb_evlist, events, detections, arid2num)
            sta_noise_means, sta_noise_vars = learn_noise_params(energies, events, 
                                                                 ttimes)
            params = build_noise_params(params, sta_noise_means, sta_noise_vars)
            sigmodel.set_signal_params(siteid, params[siteid])

        sigmodel.set_signals(energies)

        if LEARN_SHAPE:
            print " learning envelope params..."
            params = learn_envelope_params(sigmodel, siteid, params)
            sigmodel.set_signal_params(siteid, params[siteid])            

        if LEARN_PERTURB:
            print ' learning AR params...'
            params = learn_ar_params(sigmodel, siteid, params)
            sigmodel.set_signal_params(siteid, params[siteid])

        save_params(os.path.join(param_dirname, "EnvelopeSignalModel.txt"), params)

    return params

def param_likelihood(sigmodel, siteid, params):
    sigmodel.set_signal_params(siteid, params)
    print "trying params", params
    r = sigmodel.detection_likelihood(0)
    print "got ll", r
    return r

def learn_envelope_params(sigmodel, siteid, params):
    ll = lambda (p_decay, p_onset, s_decay, s_onset): -1 * param_likelihood(sigmodel, siteid, {'env_p_decay': p_decay, 'env_p_onset': p_onset, 'env_s_decay': s_decay, 'env_s_onset': s_onset, 'env_height': 1})

    print "starting the optimizer"
    #xopt, fopt, iters, evals, flags  = scipy.optimize.fmin(ll, np.array((1.5, .08, .4)), full_output=True)

    l1 = ll((0.01, 0.3, 0.01, 0.3))
    lp = ll((0.5, 0.01, 0.01, 0.3))
    ls = ll((0.01, 0.3, 0.5, 0.01))

    print [l1,lp,ls]

    if lp==ls and lp==l1:
        xopt = np.array([0.04, 0.8, 0.04, 0.8])
    # if there are no S detections, learn only P params
    elif l1 == ls:
        pll = lambda(p_decay, p_onset) : ll((p_decay, p_onset, 0.04, 0.8))
        ranges = ((0.01, 0.5), (0.1, 5))
        xopt  = scipy.optimize.brute(pll, ranges, Ns=8, full_output=0)
        xopt = np.concatenate((xopt, [0.04, 0.8]))
    # and vice versa
    elif l1 == lp:
        sll = lambda(s_decay, s_onset) : ll((0.04, 0.8, s_decay, s_onset))
        ranges = ((0.01, 0.5), (0.1, 5))
        xopt  = scipy.optimize.brute(sll, ranges, Ns=8, full_output=0)
        xopt = np.concatenate(([0.04, 0.8], xopt))
    else:        
        ranges = ((0.01, 0.5), (0.1, 5), (0.01, 0.5), (0.1, 5))
        xopt  = scipy.optimize.brute(ll, ranges, Ns=8, full_output=0)

    if (xopt[0:2] == (0.01, 0.1)).all():
        print "p learning failed, using default params..."
        xopt[0:2] = (0.04, 0.8)
    if (xopt[2:4] == (0.01, 0.1)).all():
        print "s learning failed, using default params..."
        xopt[2:4] = (0.04, 0.8)

    if siteid not in params:
        params[siteid] = dict()

    params[siteid]['env_height'] = 1
    params[siteid]['env_p_onset'] = xopt[1]
    params[siteid]['env_p_decay'] = xopt[0]
    params[siteid]['env_s_onset'] = xopt[3]
    params[siteid]['env_s_decay'] = xopt[2]


    print "learned params: ", xopt
    print "give likelihood ", ll(xopt)
    return params
    
def learn_ar_params(sigmodel, siteid, params):
    ll = lambda (coeff1, sigma2): -1 * param_likelihood(sigmodel, siteid, {'ar_noise_sigma2': sigma2, 'ar_coeffs': (coeff1,)})
    #ll = lambda (coeff1, sigma2): -1 * 12

    print "starting the optimizer"
    #xopt, fopt, iters, evals, flags  = scipy.optimize.fmin(ll, np.array((1.5, .08, .4)), full_output=True)

    ranges = ((0.1, .99), (0.01, 0.15))

    ll((0.1, 0.01))
    ll((0.2, 0.01))
    ll((0.3, 0.01))
    ll((0.1, 0.02))

    xopt  = scipy.optimize.brute(ll, ranges, Ns=10, full_output=0)
    
    params[siteid]['ar_noise_sigma2'] = xopt[1]
    params[siteid]['ar_coeffs'] = (xopt[0],)

    print "learned params: ", xopt
    print "give likelihood ", ll(xopt)
    return params
    
def expectation_over_noise(fn, traces, events, ttimes):
    MAX_EVENT_LENGTH = 50 # seconds
    
    results = dict()
    normalizer = dict()

    # compute the expectation of some function over all the noisey parts of the signal
    for channel_bundle in traces:
        for chan_trace in channel_bundle:
            
            siteid = chan_trace.stats["siteid"]
            chan = chan_trace.stats["channel"]
            samprate = __samprate(chan_trace.stats) # hz
            max_event_samples = MAX_EVENT_LENGTH * samprate

            if siteid not in ttimes.keys():
                ttimes[siteid] = __nanlist(len(events))

            prev_arrival_end = 0
            if siteid not in normalizer.keys():
                results[siteid] = dict()
                normalizer[siteid] = dict()
            if chan not in normalizer[siteid].keys():
                results[siteid][chan] = 0
                normalizer[siteid][chan] = 0

            compute_rel_atime = lambda (event, ttime): event[3] - chan_trace.stats['starttime'].getTimeStamp() + ttime
            rel_atimes = map(compute_rel_atime, zip(events, ttimes[siteid]))
            sorted_by_atime = sorted(zip(rel_atimes, events, ttimes[siteid]), key = lambda triple: triple[0])

            for (rel_atime, event, ttime) in sorted_by_atime:
                if np.isnan(ttime):
                    continue

            # everything within max_event_samples of the current
            # arrival IN BOTH DIRECTIONS will be off limits, just
            # for safety
                arrival_i = int( (rel_atime - max_event_samples) * samprate )
                if (arrival_i - prev_arrival_end) > 0:
                    results[siteid][chan] = results[siteid][chan] + np.sum(fn(chan_trace.data[prev_arrival_end:arrival_i], siteid, chan))
                    normalizer[siteid][chan] = normalizer[siteid][chan] + (arrival_i-prev_arrival_end)
                    prev_arrival_end = arrival_i + max_event_samples*2

            if prev_arrival_end == 0:
            # if no arrivals recorded during this trace, we assume the whole thing is noise
                results[siteid][chan] = results[siteid][chan] + np.sum(fn(chan_trace.data[prev_arrival_end:], siteid, chan))
                normalizer[siteid][chan] = normalizer[siteid][chan] + len(chan_trace.data)

    for siteid in results.keys():
        for chan in results[siteid].keys():
            results[siteid][chan] = results[siteid][chan] / normalizer[siteid][chan]
    return results


def build_noise_params(params, sta_noise_means, sta_noise_vars):
    for siteid in sta_noise_means.keys():
        if siteid not in params:
            params[siteid] = dict()
        for chan in sta_noise_means[siteid].keys():
            params[siteid]['chan_mean_'+chan] = sta_noise_means[siteid][chan]
            params[siteid]['chan_var_'+chan] = sta_noise_vars[siteid][chan]
    return params


def learn_noise_params(traces, events, ttimes):
    expectation_f = lambda x, siteid, chan: x
    sta_noise_means = expectation_over_noise(expectation_f, traces, events, ttimes)
        
    variance_f = lambda x, siteid, chan: (x - sta_noise_means[siteid][chan])**2
    sta_noise_vars = expectation_over_noise(variance_f, traces, events, ttimes)

    return sta_noise_means, sta_noise_vars


def plot_envelope(trace, means, variances):
    sta = trace.stats["station"]
    chan = trace.stats["channel"]
    npts = trace.stats["npts"]
    samprate = trace.stats["sampling_rate"]
    siteid = trace.stats["siteid"]

    start_time = trace.stats['starttime'].getTimeStamp()
    timerange = np.arange(start_time, start_time+npts/samprate, 1.0/samprate)
    plt.figure()
    plt.suptitle("%s -- %s" % (sta, chan))
    plt.xlabel("Time (s)")
    
    axes = plt.subplot(3, 1, 1)
    plt.plot(timerange, trace.data)

    plt.subplot(3, 1, 2, sharex=axes, sharey=axes)
    plt.plot(timerange, means)
    plt.ylabel("Envelope Mean")

    plt.subplot(3, 1, 3, sharex=axes)
    plt.plot(timerange, variances)
    plt.ylabel("Envelope Variance")


def ttimes_from_assoc(evlist, events, detections, arid2num):

    ttimes = dict()

    # generate reverse index, to save doing lots of searches
    arid2siteid = dict()
    for det in detections:
        arid2siteid[int(det[1])] = int(det[0])

    # we have a list of detections for each event. for each event,
    # then, we go through, calculate the station corresponding to
    # each detection, and then calculate the travel time that this
    # implies from that event to that station. we then update the
    # corresponding element of the travel time matrix that we're
    # calculating.


    for (evnum, event) in enumerate(evlist):
        for (phaseid, detnum) in event:

            # for now we're only going to concern ourselves with P arrivals
                #if phaseid != 0:
            #    continue
            
            arid = __reverse_conv(detnum, arid2num)
            siteid = arid2siteid[arid]

            event_time = events[evnum][3]
            arrival_time = detections[detnum][2]
            travel_time = arrival_time - event_time
            
            if siteid not in ttimes.keys():
                ttimes[siteid] = __nanlist(len(evlist))
            elif not np.isnan(ttimes[siteid][evnum]):
                travel_time = np.amin((travel_time, ttimes[siteid][evnum]))
            #print "set ttimes[", siteid, "][", evnum, "] = ", travel_time
            ttimes[siteid][evnum] = travel_time

    return ttimes

def default_params(sites):
    params = dict()
    for (id_minus1, site) in enumerate(sites):
        params[id_minus1 + 1] = dict()
    params = default_env_params(params)
    params = default_AR_params(params)
    return params

def default_env_params(params):
    for siteid in params.keys():
        params[siteid]['env_height'] = 1
        params[siteid]['env_p_onset'] = 0.8
        params[siteid]['env_p_decay'] = 0.04
        params[siteid]['env_s_onset'] = 0.8
        params[siteid]['env_s_decay'] = 0.04
    return params

def default_AR_params(params):
    for siteid in params.keys():
        params[siteid]['ar_noise_sigma2'] = 0.05
        params[siteid]['ar_coeffs'] = [0.8,]
    return params

def save_params(filename, params):
    with open(filename, 'w') as f:
        f.write(repr(params))
        f.write('\n')

def read_params(filename):
    with open(filename, 'r') as f:
        return eval(f.read())
            

def __reverse_conv(num, xxx2num):
    for (xxx, n) in xxx2num.items():
        if n == num:
            return xxx
    return None

def __samprate(stats):
    return stats["sampling_rate"]
#    return 1 / (stats["window_size"]*stats["overlap"])

def __nanlist(n):
    l = np.empty((n, 1))
    l[:] = np.NAN
    return l
