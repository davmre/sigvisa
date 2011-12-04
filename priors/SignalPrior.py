import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import time

from obspy.core import Trace, Stream, UTCDateTime
import sigvisa



def learn(filename, sigmodel, earthmodel, traces, events, leb_evlist, detections, arid2num):
    """
    Find maximum-likelhood parameter estimates 
    """

    #ttimes = ttimes_from_assoc(leb_evlist, events, detections, arid2num)

    f = open(filename, 'w')
    f.write("5000 500 500\n")
    f.write("2 .2 .7 10\n")

    print "learning envelope params"
    xopt = learn_envelope_params(sigmodel)
    #sta_noise_means, sta_noise_vars = learn_noise_params(f, sigmodel, traces, events, ttimes)

    f.close()

    #return sta_noise_means, sta_noise_vars


def learn_envelope_params(sigmodel):
    ll = lambda (decay, onset): -1 * sigmodel.detection_likelihood(1, decay, onset)

    print "starting the optimizer"
    #xopt, fopt, iters, evals, flags  = scipy.optimize.fmin(ll, np.array((1.5, .08, .4)), full_output=True)

    ranges = ((0.01, 0.5), (0.05, 5))
    xopt  = scipy.optimize.brute(ll, ranges, Ns=15, full_output=0)

    print "learned params: ", xopt
    print "give likelihood ", ll(xopt)
    return xopt
    

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

def learn_noise_params(filehandle, sigmodel, traces, events, ttimes):
    expectation_f = lambda x, siteid, chan: x
    sta_noise_means = expectation_over_noise(expectation_f, traces, events, ttimes)
        
    variance_f = lambda x, siteid, chan: (x - sta_noise_means[siteid][chan])**2
    sta_noise_vars = expectation_over_noise(variance_f, traces, events, ttimes)

    filehandle.write(str(len(sta_noise_means)) + "\n")
    for siteid in sta_noise_means.keys():
        filehandle.write(str(siteid) + " " + str(len(sta_noise_means[siteid].keys())))
        for chan in sta_noise_means[siteid].keys():
            chan_num = sigmodel.canonical_channel_num(chan)
            filehandle.write(" " + str(chan_num) + " " + str(sta_noise_means[siteid][chan]) + " " + str(sta_noise_vars[siteid][chan]))
        filehandle.write("\n")
    return sta_noise_means, sta_noise_vars


def plot_envelope(trace, means, variances):
    sta = trace.stats["station"]
    chan = trace.stats["channel"]
    npts = trace.stats["npts_processed"]
    samprate = 1 / (trace.stats["window_size"]*trace.stats["overlap"])
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
            print "set ttimes[", siteid, "][", evnum, "] = ", travel_time
            ttimes[siteid][evnum] = travel_time

    return ttimes

def __reverse_conv(num, xxx2num):
    for (xxx, n) in xxx2num.items():
        if n == num:
            return xxx
    return None

def __samprate(stats):
    return 1 / (stats["window_size"]*stats["overlap"])

def __nanlist(n):
    l = np.empty((n, 1))
    l[:] = np.NAN
    return l
