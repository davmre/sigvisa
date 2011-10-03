
import time
import sys, MySQLdb,struct
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from optparse import OptionParser
from obspy.core import Trace, Stream, UTCDateTime

from database import db, dataset
import utils.waveform
import netvisa, learn

MAX_TRAVEL_TIME = 2000 # seconds


class SigModel:
    def __init__(self, start_time, end_time, earthmodel, netmodel, siteid):
        self.start_time = start_time
        self.end_time = end_time
        self.earthmodel = earthmodel
        self.netmodel = netmodel
        self.siteid = siteid

        # parameters

        self.sta_noise_means = dict()
        self.sta_noise_vars = dict()
        for s in siteid.values():
            self.sta_noise_means[s] = 10
            self.sta_noise_vars[s] = 10

        self.envelope_peak_coeff = 5000
        self.envelope_decay_coeff = 10

    def learn_from_scratch(self, cursor, energies, start_time, end_time, detections, arid2num, event_set="leb", runid=None):
        earliest_event_time = start_time - MAX_TRAVEL_TIME
        events, orid2num = dataset.read_events(cursor, earliest_event_time, end_time, event_set, runid)
        print "loaded ", len(events), " events."

    # read associations, as training data for learning
        evlist = dataset.read_assoc(cursor, earliest_event_time, end_time, orid2num, arid2num, event_set, runid)
        print "loaded associations for ", len(events), " events."
        assoc_ttimes = self.ttimes_from_assoc(evlist, events, detections, arid2num)
        self.learn(energies, events, assoc_ttimes)
    
        return assoc_ttimes

    def learn(self, traces, events, ttimes):
        """
        Find maximum-likelhood parameter estimates 
        """

        # need to learn noise mean and var at each station. and envelop height, and decay.
        # we'll know where the envelopes start, but we don't know where they end unless we know the envelope params. without that, we don't know what's noise and what's not. but we can guess - take the mode (or equivalent) under the assumption that most of the time we're seeing noise. the noise variance is uninteresting, since I don't think it changes the model that much.
        # alternately we could take some "known-clean" times which are far away from any event arrival, and just find the mean and variance in those regions. 
        # as far as the envelope parameters, for any given signal we know where all of the envelopes start. if we know the noise mean, then we can essentially zero things out, and then it should be easy to tell where they end. 

        # the likelihood decomposes over traces, so at each trace we can find values of all four parameters which maximize it. That's good for the noise params, since those are trace-specific anyway, but it's not necessarily good for the other params - we really want to optimize those over all traces.

        # I forsee a two-step process then. First, we estimate noise levels in some hacky way. Then, search for param values to optimize the likelihood. In principle we could do this with numpy's optimize, if computing the likelihood doesn't take too long. how long does it take? about .6 seconds with my current settings (just three stations of traces), which is much too long. maybe when things are in C... anyway, learning is not the problem; it only has to run once.

# 

        self.__learn_noise_params(traces, events, ttimes)
        #self.__learn_envelope_params(traces, events, ttimes)

    def __learn_envelope_params(self, traces, events, ttimes):
        ll = lambda (peak, decay): -1 * self.log_likelihood_complete(traces, events, ttimes, peak, decay)
        xopt, fopt, iters, evals, flags  = scipy.optimize.fmin(ll, np.array((5000, 1)), full_output=True)
        print "learned params: ", xopt
        print "give likelihood ", fopt
        self.envelope_peak_coeff = xopt[0]
        self.envelope_decay_coeff = xopt[1]

    def __expectation_over_noise(self, f, traces, events, ttimes):
        MAX_EVENT_LENGTH = 50 # seconds

        results = dict()
        normalizer = dict()

        # compute the expectation of some function over all the noisey parts of the signal
        for trace in traces:
            siteid = self.siteid[trace.stats["station"]]
            samprate = self.__samprate(trace.stats) # hz
            max_event_samples = MAX_EVENT_LENGTH * samprate

            if siteid not in ttimes.keys():
                ttimes[siteid] = self.__nanlist(len(events))

            prev_arrival_end = 0
            if siteid not in normalizer.keys():
                results[siteid] = 0
                normalizer[siteid] = 0

            compute_rel_atime = lambda (event, ttime): event[3] - trace.stats['starttime'].getTimeStamp() + ttime
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
                    
                    results[siteid] = results[siteid] + np.sum(trace.data[prev_arrival_end:arrival_i])
                    normalizer[siteid] = normalizer[siteid] + (arrival_i-prev_arrival_end)
                    prev_arrival_end = arrival_i + max_event_samples*2


            if prev_arrival_end == 0:
            # if no arrivals recorded during this trace, we assume the whole thing is noise
                results[siteid] = results[siteid] + np.sum(f(trace.data[prev_arrival_end:], siteid))
                normalizer[siteid] = normalizer[siteid] + len(trace.data)

        for siteid in results.keys():
            results[siteid] = results[siteid] / normalizer[siteid]
        return results

    def __learn_noise_params(self, traces, events, ttimes):
        expectation_f = lambda x, siteid: x
        self.sta_noise_means = self.__expectation_over_noise(expectation_f, traces, events, ttimes)
        
        variance_f = lambda x, siteid: (x - self.sta_noise_means[siteid])**2
        self.sta_noise_vars = self.__expectation_over_noise(variance_f, traces, events, ttimes)


    def envelope(self, event, siteid, hz, peak_coeff=None, decay_coeff=None):
        """
        Return the mean and variance of the effect of an event at a station.
        """

        if peak_coeff is None:
            peak_coeff = self.envelope_peak_coeff
        if decay_coeff is None:
            decay_coeff = self.envelope_decay_coeff

        # model is a triangle with constant slope
        
        # initial height depends on event magnitude (mb) and distance (compute from lat/long)
        distance_deg= self.earthmodel.Delta(event[0], event[1], siteid)
        distance_mi=12000 * (distance_deg/180)

        mb = event[4]

        # TODO: figure out what this model should actually be
        envelopem = []
        envelopev = []
        newmean = peak_coeff / distance_mi * np.exp(mb)
        while newmean > 0:
            envelopem.append(newmean)
            envelopev.append(newmean/2)
            newmean = newmean - decay_coeff/hz
        return np.array(envelopem), np.array(envelopev)

    def all_envelopes(self, stats, events, ttimes, peak_coeff=None, decay_coeff=None):
        """
        Combine information from all events at a single station, and return the resulting mean and variance.
        """

        if peak_coeff is None:
            peak_coeff = self.envelope_peak_coeff
        if decay_coeff is None:
            decay_coeff = self.envelope_decay_coeff

        siteid = self.siteid[stats["station"]]
        npts = stats["npts_processed"]
        samprate = self.__samprate(stats)

        means = np.ones((npts, 1))*self.sta_noise_means[siteid]
        variances = np.ones((npts, 1))*self.sta_noise_vars[siteid]

        for (event, ttime) in zip(events, ttimes):
            if len(event) < 4:
                continue
            if np.isnan(ttime):
                continue

            rel_time = event[3] - stats['starttime'].getTimeStamp()
            (emeans, evars) = self.envelope(event, siteid, samprate, peak_coeff, decay_coeff)

#           print rel_time, event[3], stats['starttime'].getTimeStamp(), ttime
#           print "event at t=", rel_time+ttime, ", mb = ", event[4]
            base_t = int( (rel_time + ttime) * samprate  )
            for i in range(len(emeans)):
               # print "  adding emeans[", i, "] = ", emeans[i], " to means[", base_t+i, "]."
                try:
                    #print len(means), len(variances), base_t, i, base_t+1
                    means[base_t +i] = means[base_t+i] + emeans[i]
                    variances[base_t +i] = variances[base_t+i] + evars[i]
                except IndexError:
                    #sys.stderr.write("Predicted arrival at time " + str(event[3] + ttime) + " - " + str(event[3] + ttime + i*samprate) + " falls outside trace boundaries " + str(stats['starttime'].getTimeStamp()) + " - " + str(stats['starttime'].getTimeStamp() + len(means)/samprate) + ".\n")
                    break
        return (means, variances)

    def plot_envelope(self, trace, means, variances):
        sta = trace.stats["station"]
        chan = trace.stats["channel"]
        npts = trace.stats["npts_processed"]
        samprate = 1 / (trace.stats["window_size"]*trace.stats["overlap"])
        siteid = int(self.siteid[trace.stats["station"]])

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


    def waveform_model_trace(self, events, trace, ttimes, peak_coeff=None, decay_coeff=None):

        if peak_coeff is None:
            peak_coeff = self.envelope_peak_coeff
        if decay_coeff is None:
            decay_coeff = self.envelope_decay_coeff

        (means, variances) = self.all_envelopes(trace.stats, events, ttimes, peak_coeff, decay_coeff)
        #self.plot_envelope(trace, events, means, variances)

        ll=0
        for i in range(len(means)):
            ll = ll - .5 * np.log(2*np.pi * variances[i]) - .5 * (trace.data[i] - means[i])**2 / variances[i]
            
        return ll

    def ttime_point_matrix(self, siteids, events, phase):
        ttimes = dict()

        for siteid in siteids:
            #siteid = int(self.siteid[trace.stats["station"]])
            #if siteid in ttimes.keys():
            #    continue
            siteid = int(siteid)

            ttime_list = np.zeros((len(events),1))
            for (i, event) in enumerate(events):
                lat = event[0]
                lon = event[1]
                depth = event[2]
           
                ttime_list[i] = self.ttime_model_point(lat, lon, depth, siteid, phase)
            ttimes[siteid] = ttime_list
        return ttimes


    def ttimes_from_assoc(self, evlist, events, detections, arid2num):

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
            
                arid = self.__reverse_conv(detnum, arid2num)
                siteid = arid2siteid[arid]

                event_time = events[evnum][3]
                arrival_time = detections[detnum][2]
                travel_time = arrival_time - event_time

                if siteid not in ttimes.keys():
                    ttimes[siteid] = self.__nanlist(len(evlist))
                elif not np.isnan(ttimes[siteid][evnum]):
                    travel_time = np.amin((travel_time, ttimes[siteid][evnum]))
                #print "set ttimes[", siteid, "][", evnum, "] = ", travel_time
                ttimes[siteid][evnum] = travel_time

        return ttimes

    def ttime_model_point(self, lat, lon, depth, siteid, phase):
        return self.earthmodel.ArrivalTime(lat, lon, depth, 0, phase, siteid)

    def ttime_model_logprob(self, ttime, lat, lon, depth, siteid, phase):
        """
        Compute p(ttime to siteid | event at lat, lon, depth, etime). 
        """

        pred_ttime = self.earthmodel.ArrivalTime(lat, lon, depth, 0, phase, siteid)
        return self.netmodel.arrtime_logprob(ttime, pred_ttime, 0, siteid, phase)

    def log_likelihood_complete(self, traces, events, ttimes, peak_coeff=None, decay_coeff=None):
        """
        Compute p(signals| events, travel_times) for a given set of travel times, expressed as a dictionary mapping siteids to n-element arrays (n=#of events). 
        """

        if peak_coeff is None:
            peak_coeff = self.envelope_peak_coeff
        if decay_coeff is None:
            decay_coeff = self.envelope_decay_coeff

        start = time.time()
        
        print "likelihood called w/ params peak = ", peak_coeff, ", decay_coeff = ", decay_coeff


        ll = 0
        for (i, trace) in enumerate(traces):
            trace_start = trace.stats['starttime'].getTimeStamp()
            siteid = int(self.siteid[trace.stats["station"]])

            if siteid not in ttimes.keys():
                ttimes[siteid] = self.__nanlist(len(events))


            wavell = self.waveform_model_trace(events, trace, np.array(ttimes[siteid]), peak_coeff, decay_coeff)
            ttll = 0
            for (j, event) in enumerate(events):

                # if we don't have a travel time from an event to this
                # station, then there still *is* a travel time, we
                # just don't observe it. we should really sum across
                # all travel times, but since this missing travel time
                # issue really only comes up in parameter learning,
                # and our calculations here have no impact on
                # parameter choices, I'm just going to ignore the
                # issue for now.
                if np.isnan(ttimes[siteid][j]):
                    continue

                etime = event[3]
                lat = event[0]
                lon = event[1]
                depth = event[2]
                
                ttll = ttll + self.ttime_model_logprob(ttimes[siteid][j], lat, lon, depth, siteid, 0)
            ll = ll + wavell + ttll

        end = time.time()
        print "computed likelihood ", ll, " in time ", (end-start), " seconds."
        return ll

    def infer(self, runid, traces, numsamples, birthsteps, window, step, threads, propose_events, verbose, write_cb):

        return py_infer_sig(runid, traces, numsamples, birthsteps, window, step, threads, propose_events, verbose, write_cb)

    #TODO: move this and following to a util package
    def __log_addprob__(self, lp1, lp2):
        a = np.max((lp1, lp2))
        b = np.min((lp1, lp2))
        if a-b > 50:
            return a
        else:
            return a + np.log(1 + np.exp(a-b))

    def __decode_counter__(self, c, base):
        if base <= 1:
            return [0]
        counts = []
        while c > 0:
            counts.append(np.mod(c, base))
            c = int(np.floor(c/base))
        return counts
        
    def __copy_trace__(self, trace, newdata):
        new_header = trace.stats.copy()
        new_header["npts_processed"] = len(newdata)
        processed_trace = Trace(newdata, header=new_header)
        return processed_trace    

    def __samprate(self, stats):
        return 1 / (stats["window_size"]*stats["overlap"])

    def __reverse_conv(self, num, xxx2num):
        for (xxx, n) in xxx2num.viewitems():
            if n == num:
                return xxx
        return None

    def __nanlist(self, n):
        l = np.empty((n, 1))
        l[:] = np.NAN
        return l

    

def window_energies(trace, window_size=1, overlap=0.5):
  """
  Returns a vector giving the signal energy in each of many
  windows. 

  The length of each window in seconds is given by
  window_size. 

  The overlap argument specifies (1 - the fraction of a window length
  to wait before starting the next window); for example, overlap=0
  causes successive windows to be disjoint, while overlap=0.99 causes
  successive windows to share 99% of their data.
  """
  samprate = trace.stats["sampling_rate"]
  data = trace.data

  nsamples_window = window_size * samprate
  offset = nsamples_window * (1-overlap)
  nwindows = int((len(data) - nsamples_window) / offset)
  #print 'computing windows on data on length ', len(data)
  windows = np.zeros((nwindows, 1))

  for i in range(nwindows):
    wstart = offset*i
    wstop = wstart+window_size
    
    #print "window ", i, " runs from ", wstart, '-', wstop
    window = data[wstart:wstop]
    windows[i] = np.linalg.norm(window, 2)
  
  return windows

def load_traces(cursor, stations, start_time, end_time):
    traces = []
    for sta in stations:
        cursor.execute("select chan from idcx_wfdisc where sta='%s'" % sta)
        chan, = cursor.fetchone()

        # select all waveforms which overlap with the designated period
        cursor.execute("select time,endtime from idcx_wfdisc where sta = '%s' and chan ='%s' "
                       "and endtime > %f and time < %f" %
                       (sta, chan, start_time, end_time))
        waveforms = cursor.fetchall()
        for (stime, etime) in waveforms:
            
            st = np.max((stime, start_time))
            et = np.min((etime, end_time))

            print "fetching waveform {sta: ", sta, ", chan: ", chan, ", start_time: ", st, ", end_time: ", et, "}", 
            try:
                trace = utils.waveform.fetch_waveform(sta, chan, int(np.ceil(st)), int(np.floor(et)))
                traces.append(trace)
                print " ... successfully loaded."
            except (utils.waveform.MissingWaveform, IOError):
                print " ... not found, skipping."
                continue
    print "fetched ", len(traces), " waveforms."
    return traces
    
def process_traces(traces, f, opts):
    # process each trace to yield a representation suitable for inference.
    # currently, that means computing energies within overlapping windows
    processed_traces = []
    for trace in traces:
        processed_data = f(trace)
        new_header = trace.stats.copy()
        new_header.update(opts)
        new_header["npts_processed"] = len(processed_data)
        processed_trace = Trace(processed_data, header=new_header)
        processed_traces.append(processed_trace)    
    return processed_traces

def main():
    parser = OptionParser() 
    # make options for which set of events to load, and what time period
    parser.add_option("--window", dest="window_size", default=1,
                    type="int",
                    help = "length of signal energy windows, in seconds (1)")
    parser.add_option("--overlap", dest="overlap", default=0.5,
                    type="float",
                    help = "fraction of a window length between the start of successive windows (0.5)")
    parser.add_option("--start", dest="start_time", type="float", default=1237680000,
                    help = "start time")
    parser.add_option("--end", dest="end_time", type="float", default=1237683600,
                    help = "end time")
    parser.add_option("--events", dest="event_set", default="leb",
                    help = "set of events for which to compute likelihood: visa, leb, or sel3 (leb)")
    parser.add_option("--runid", dest="runid", default=None, type="int",
                    help = "ID of netvisa run from which to use events (0)")
    parser.add_option("-g", "--gui", dest="gui", default=None,
                    help = "show trace and envelope plots (False)")
    (options, args) = parser.parse_args()

    # read traces for each station
    cursor = db.connect().cursor()
    cursor.execute("select sta, id from static_siteid where statype='ss'")
    stations = np.array(cursor.fetchall())
    siteids = dict(stations)
    stations = stations[:,0]

    # convert all values in the siteids dictionary from strings to ints
    # TODO: figure out the proper pythonic way to do this
    siteids_ints = dict()
    for sta in siteids.keys():
        siteids_ints[sta] = int(siteids[sta])
    siteids = siteids_ints

    traces = load_traces(cursor, stations, options.start_time, options.end_time)
    f = lambda trace: window_energies(trace, window_size=options.window_size, overlap=options.overlap)
    opts = dict(window_size=options.window_size, overlap=options.overlap)
    energies = process_traces(traces, f, opts)

    # load earth and net models
    # read the detections and uptime
    detections, arid2num = dataset.read_detections(cursor, options.start_time, options.end_time, "idcx_arrival", 1)
    site_up = dataset.read_uptime(cursor, options.start_time, options.end_time,
                          "idcx_arrival")

    # read the rest of the static data
    sites = dataset.read_sites(cursor)
    phasenames, phasetimedef = dataset.read_phases(cursor)
    assert(len(phasenames) == len(phasetimedef))
    sm = learn.load_sigvisa("parameters",
                                options.start_time, options.end_time,
                                detections, site_up, sites, siteids, phasenames,
                                phasetimedef)
    

    # read appropriate event set (e.g. netvisa)
    earliest_event_time = options.start_time - MAX_TRAVEL_TIME
    events, orid2num = dataset.read_events(cursor, earliest_event_time, options.end_time, options.event_set, options.runid)
    print "loaded ", len(events), " events."
    
    # read associations, as training data for learning
    evlist = dataset.read_assoc(cursor, earliest_event_time, options.end_time, orid2num, arid2num, options.event_set, options.runid)
    print "loaded associations for ", len(events), " events."

    # calculate likelihood
    peak_ttimes = sm.ttime_point_matrix(siteids.values(), events, 0)
    assoc_ttimes = sm.ttimes_from_assoc(evlist, events, detections, arid2num)
    ttimes = peak_ttimes
    sm.learn(energies, events, assoc_ttimes)


    ll = sm.log_likelihood_complete(energies, events, ttimes)
    print "log-likelihood is ", ll

    #print "learning params..."


    if options.gui is not None:
        for trace in energies:   
            siteid = siteids[trace.stats["station"]]       
            if options.gui == "all" or trace.stats["station"] in options.gui:
                (means, variances) = sm.all_envelopes(trace.stats, events, ttimes[siteid])
                sm.plot_envelope(trace, means, variances)
        plt.show()




if __name__ == "__main__":
    main()
