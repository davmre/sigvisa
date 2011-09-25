

import sys, MySQLdb,struct
import matplotlib.pyplot as plt
import numpy as np

from optparse import OptionParser
from obspy.core import Trace, Stream, UTCDateTime

from database import db, dataset
import utils.waveform
import netvisa, learn

class SigModel:
    def __init__(self, start_time, end_time, earthmodel, netmodel, siteid):
        self.start_time = start_time
        self.end_time = end_time
        self.earthmodel = earthmodel
        self.netmodel = netmodel
        self.siteid = siteid

        self.noise_mean=660
        self.noise_variance=2500


    def learn(self, events, ttimes):
        """
        Find maximum-likelhood parameter estimates 
        """
        
    def envelope(self, event, siteid, hz):
        """
        Return the mean and variance of the effect of an event at a station.
        """

        # model is a triangle with constant slope
        
        # initial height depends on event magnitude (mb) and distance (compute from lat/long)
        dist= self.earthmodel.Delta(event[0], event[1], siteid)
        mb = event[4]

        # TODO: figure out what this model should actually be
        envelopem = []
        envelopev = []
        newmean = (200-dist)/180 * mb * 20
        while newmean > 0:
            envelopem.append(newmean)
            envelopev.append(newmean/2)
            newmean = newmean - 1/(2*hz)
        return np.array(envelopem), np.array(envelopev)

    def all_envelopes(self, stats, events, ttimes):
        """
        Combine information from all events at a single station, and return the resulting mean and variance.
        """

        siteid = int(self.siteid[stats["station"]])
        npts = stats["npts_processed"]
        samprate = 1 / (stats["window_size"]*stats["overlap"])

        means = np.ones((npts, 1))*self.noise_mean
        variances = np.ones((npts, 1))*self.noise_variance

        for (event, ttime) in zip(events, ttimes):
            if len(event) < 4:
                continue

            rel_time = event[3] - stats['starttime'].getTimeStamp()
            (emeans, evars) = self.envelope(event, siteid, samprate)

#           print rel_time, event[3], stats['starttime'].getTimeStamp(), ttime
            #print "event at t=", rel_time+ttime, ", mb = ", event[4]
            base_t = int( (rel_time + ttime) * samprate  )
            for i in range(len(emeans)):
               # print "  adding emeans[", i, "] = ", emeans[i], " to means[", base_t+i, "]."
                try:
                    #print len(means), len(variances), base_t, i, base_t+1
                    means[base_t +i] = means[base_t+i] + emeans[i]
                    variances[base_t +i] = variances[base_t+i] + evars[i]
                except IndexError:
                    sys.stderr.write("Predicted arrival at time " + str(event[3] + ttime) + " - " + str(event[3] + ttime + i*samprate) + " falls outside trace boundaries " + str(stats['starttime'].getTimeStamp()) + " - " + str(stats['starttime'].getTimeStamp() + len(means)/samprate) + ".\n")
                    break
        return (means, variances)

    def plot_envelope(self, trace, events, means=None, variances=None):
        sta = trace.stats["station"]
        chan = trace.stats["channel"]
        npts = trace.stats["npts_processed"]
        samprate = 1 / (trace.stats["window_size"]*trace.stats["overlap"])
        siteid = int(self.siteid[trace.stats["station"]])

        if means is None or variances is None:
            (means, variances) = self.all_envelopes(trace.stats, events)

        timerange = np.arange(0, npts/samprate, 1.0/samprate)
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


    def waveform_model_trace(self, events, trace, ttimes):
        (means, variances) = self.all_envelopes(trace.stats, events, ttimes)
        #self.plot_envelope(trace, events, means, variances)

        ll=0
        for i in range(len(means)):
            ll = ll - .5 * np.log(2*np.pi * variances[i]) - .5 * (trace.data[i] - means[i])**2 / variances[i]
            
        return ll

    def waveform_model(self, traces, events, ttimes):
        """
        Compute p(signals | events, travel times). Combined with the travel time model, we can then integrate over travel times to get p(signals | events), i.e. the overall likelihood.

        The current implementation is a toy model, in which Gaussian
        white noise is added to a deterministic envelope.
        """

        ll = 0
        for (i, trace) in enumerate(traces):
            ll = ll + self.waveform_model_trace(events, trace, ttimes[i])
            

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

    def ttime_model_point(self, lat, lon, depth, siteid, phase):
        return self.earthmodel.ArrivalTime(lat, lon, depth, 0, phase, siteid)

    def ttime_model_logprob(self, ttime, lat, lon, depth, siteid, phase):
        """
        Compute p(ttime to siteid | event at lat, lon, depth, etime). 
        """

        pred_ttime = self.earthmodel.ArrivalTime(lat, lon, depth, 0, phase, siteid)
        return self.netmodel.arrtime_logprob(ttime, pred_ttime, 0, siteid, phase)

    def log_likelihood_complete(self, traces, events, ttimes):
        """
        Compute p(signals| events, travel_times) for a given set of travel times, expressed as a dictionary mapping siteids to n-element arrays (n=#of events). 
        """
        
        ll = 0
        for (i, trace) in enumerate(traces):
            trace_start = trace.stats['starttime'].getTimeStamp()
            siteid = int(self.siteid[trace.stats["station"]])

            wavell = self.waveform_model_trace(events, trace, np.array(ttimes[siteid]))
            ttll = 0
            for (j, event) in enumerate(events):
                etime = event[3]
                lat = event[0]
                lon = event[1]
                depth = event[2]
                
                ttll = ttll + self.ttime_model_logprob(ttimes[siteid][j], lat, lon, depth, siteid, 0)
            ll = ll + wavell + ttll
        return ll

    def log_likelihood(self, traces, events):
        """ 
        Compute p(signals | events) by integrating over travel times:

        p(signals | events) = int_ttimes p(signals | ttimes, events) p(ttimes | events)
                            = int_ttimes prod_signal prod_event p(signal | ttime, event)p(ttime | event)
                            = prod_signal prod_event int_ttimes p(signal | ttime, event)p(ttime | event)
        """

        TT_SEARCH_STEP = 1 # seconds
        
        # TODO: make this depend on some probability threshold, rather than absolute time
        TT_SEARCH_WIDTH = float(1.0) # seconds
        MAX_TT = 2000 # seconds
        MAX_ENVELOPE_LEN = 100 # seconds
        SUBTRACE_LEN = 900.0 # seconds
        

        ll = 0
        for trace in traces:
            trace_start = trace.stats['starttime'].getTimeStamp()
            siteid = int(self.siteid[trace.stats["station"]])

            nsubtraces = np.ceil(len(trace.data)/(SUBTRACE_LEN / (trace.stats["window_size"] * trace.stats["overlap"])))
            subtraces = np.array_split(trace.data, nsubtraces)

            trace_ll = 0

            # compute loglikelihood for each subtrace
            for (i, subtrace_data) in enumerate(subtraces):
                subtrace_start = trace_start + i * SUBTRACE_LEN
                subtrace_end = subtrace_start + len(subtrace_data) * (trace.stats["window_size"] * trace.stats["overlap"])
                
                subtrace_ll = float("-inf")

                subtrace = self.__copy_trace__(trace, subtrace_data)
                subtrace.stats['starttime'] = UTCDateTime(subtrace_start)

                relevant_events = []
                arrtimess = []
                for event in events:
                    etime = event[3]
                    lat = event[0]
                    lon = event[1]
                    depth = event[2]
                    point_arrtime = etime + self.ttime_model_point(lat, lon, depth, siteid, 0)
                    min_arrtime = point_arrtime - TT_SEARCH_WIDTH/2
                    max_arrtime = point_arrtime + TT_SEARCH_WIDTH/2
#                    print  min_arrtime, max_arrtime, point_arrtime, TT_SEARCH_WIDTH, TT_SEARCH_WIDTH/2

                    # don't consider events that can't possibly affect the current subtrace
                    if max_arrtime + MAX_ENVELOPE_LEN < subtrace_start or min_arrtime > subtrace_end:
                        continue

                    arrtimes = np.arange(min_arrtime, max_arrtime, TT_SEARCH_STEP)
                    relevant_events.append(event)
                    #print "building artimess: appending ", arrtimes
                    arrtimess.append(arrtimes)
                

                total_combos = int((max_arrtime - min_arrtime) ** len(relevant_events))
                print "looping over ", total_combos, " total combos on subtrace ", i, "."
                for i in range(total_combos):
                    indices = self.__decode_counter__(i, max_arrtime - min_arrtime)
                    ttimes = []
                    ttll = 0

                    if len(relevant_events) > 0:
                        for (i, idx) in enumerate(indices):
#                            print i,idx, len(ttimes), relevant_events
                            ttimes.append(arrtimess[i][idx]-relevant_events[i][3])
                            ttll = ttll + self.ttime_model_logprob(ttimes[-1], lat, lon, depth, siteid, 0)
                    
                    wavell = self.waveform_model_trace(relevant_events, subtrace, ttimes)
#                    print "testing atimes ", ttimes, " with lp ", ttll, " and wavell ", wavell

                    # equiv to p() = p() + p(subtrace | ttimes, events)p(ttimes|events)
                    # which will eventually sum over all choices of travel times.
                    subtrace_ll = self.__log_addprob__(subtrace_ll, ttll + wavell)
#                    print "added subtrace_ll now ", subtrace_ll
                trace_ll = trace_ll + subtrace_ll
                print "added trace_ll now ", trace_ll
            ll = ll + trace_ll
            print "added ll now ", ll
        return ll

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
        print "fetching waveform {sta: ", sta, ", chan: ", chan, ", start_time: ", start_time, ", end_time: ", end_time, "}", 
        try:
            trace = utils.waveform.fetch_waveform(sta, chan, start_time, end_time)
            print " ... successfully loaded."
        except (utils.waveform.MissingWaveform, IOError):
            print " ... not found, skipping."
            continue
        traces.append(trace)
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
    parser.add_option("--start", dest="start_time", type="float",
                    help = "start time")
    parser.add_option("--end", dest="end_time", type="float",
                    help = "end time")
    parser.add_option("--events", dest="event_set", default="leb",
                    help = "set of events for which to compute likelihood: visa, leb, or sel3 (leb)")
    parser.add_option("--runid", dest="runid", default=None, type="int",
                    help = "ID of netvisa run from which to use events (0)")
    parser.add_option("-g", "--gui", dest="gui", default=False,
                    action = "store_true",
                    help = "show trace and envelope plots (False)")
    (options, args) = parser.parse_args()

    # read traces for each station
    cursor = db.connect().cursor()
    cursor.execute("select sta, id from static_siteid where statype='ss'")
    stations = np.array(cursor.fetchall())
    siteids = dict(stations)
    stations = stations[:,0]

    traces = load_traces(cursor, stations, options.start_time, options.end_time)
    f = lambda trace: window_energies(trace, window_size=options.window_size, overlap=options.overlap)
    opts = dict(window_size=options.window_size, overlap=options.overlap)
    energies = process_traces(traces, f, opts)

    # load earth and net models
    # read the detections and uptime
    detections = dataset.read_detections(cursor, options.start_time, options.end_time,
                                 "idcx_arrival", 1)[0]
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
    events, stuff = dataset.read_events(cursor, options.start_time, options.end_time, options.event_set, options.runid)
    

    # calculate likelihood
    peak_ttimes = sm.ttime_point_matrix(siteids.values(), events, 0)
    ll = sm.log_likelihood_complete(energies, events, peak_ttimes)

    if options.gui:
        plt.show()

    print "log-likelihood is ", ll
 
if __name__ == "__main__":
    main()
