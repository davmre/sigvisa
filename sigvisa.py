

import sys, MySQLdb,struct
import matplotlib.pyplot as plt
import numpy as np

from optparse import OptionParser
from obspy.core import Trace

from database import db, dataset
import utils.waveform
import netvisa, learn

class SigModel:
    def __init__(self, start_time, end_time, traces, earthmodel, netmodel, siteid):
        self.start_time = start_time
        self.end_time = end_time
        self.traces = traces
        self.earthmodel = earthmodel
        self.netmodel = netmodel
        self.siteid = siteid

        self.noise_mean=660
        self.noise_variance=2500

        
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

    def all_envelopes(self, stats, events):
        """
        Combine information from all events at a single station, and return the resulting mean and variance.
        """

        siteid = int(self.siteid[stats["station"]])
        npts = stats["npts_processed"]
        samprate = 1 / (stats["window_size"]*stats["overlap"])

        means = np.ones((npts, 1))*self.noise_mean
        variances = np.ones((npts, 1))*self.noise_variance

        for event in events:
            if len(event) < 4:
                continue

            rel_time = event[3] - stats['starttime'].getTimeStamp()
            ttime = self.netmodel.mean_travel_time(event[0], event[1], event[2], siteid, 0)
            (emeans, evars) = self.envelope(event, siteid, samprate)

            print "event at t=", rel_time+ttime, ", mb = ", event[4]
            base_t = int( (rel_time + ttime) * samprate  )
            for i in range(len(emeans)):
               # print "  adding emeans[", i, "] = ", emeans[i], " to means[", base_t+i, "]."
                try:
                    means[base_t +i] = means[base_t+i] + emeans[i]
                    variances[base_t +i] = variances[base_t+i] + evars[i]
                except IndexError:
                    sys.stderr.write("Predicted arrival at time " + str(event[3] + ttime) + " falls outside trace boundary.\n")
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


    def waveform_model(self, events, ttimes):
        """
        Compute p(signals | events, travel times). Combined with the travel time model, we can then integrate over travel times to get p(signals | events), i.e. the overall likelihood.
        """

    def log_likelihood(self, events):

        ll = 0
        for trace in self.traces:

            (means, variances) = self.all_envelopes(trace.stats, events)

            for i in range(len(means)):
                ll = ll - .5 * np.log(2*np.pi * variances[i]) - .5 * (trace.data[i] - means[i])**2 / variances[i]

            self.plot_envelope(trace, events, means, variances)
        return ll


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

    traces = []
    for sta in stations:
        cursor.execute("select chan from idcx_wfdisc where sta='%s'" % sta)
        chan, = cursor.fetchone()
        print "fetching waveform {sta: ", sta, ", chan: ", chan, ", start_time: ", options.start_time, ", end_time: ", options.end_time, "}", 
        try:
            trace = utils.waveform.fetch_waveform(sta, chan, options.start_time, options.end_time)
            print " ... successfully loaded."
        except (utils.waveform.MissingWaveform, IOError):
            print " ... not found, skipping."
            continue
        traces.append(trace)
    print "fetched ", len(traces), " waveforms."
    

    # process each trace to yield a representation suitable for inference.
    # currently, that means computing energies within overlapping windows
    print "processing..."    
    processed_traces = []
    for trace in traces:
        processed_data = window_energies(trace, window_size=options.window_size, overlap=options.overlap)
        new_header = trace.stats.copy()
        new_header["window_size"] = options.window_size
        new_header["overlap"] = options.overlap
        new_header["npts_processed"] = len(processed_data)
        processed_trace = Trace(processed_data, header=new_header)
        processed_traces.append(processed_trace)    
    print "done."

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
    earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
    netmodel = learn.load_netvisa("parameters",
                                options.start_time, options.end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

    sm = SigModel(options.start_time, options.end_time, processed_traces, earthmodel, netmodel, siteids)
    

    # read appropriate event set (e.g. netvisa)
    events, stuff = dataset.read_events(cursor, options.start_time, options.end_time, options.event_set, options.runid)
    

    # calculate likelihood
    ll = sm.log_likelihood(events)

    if options.gui:
        plt.show()

    print "log-likelihood is ", ll
 
if __name__ == "__main__":
    main()
