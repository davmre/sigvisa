
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


def arrtime_point_matrix(numsites, events, numphases, earthmodel):
    arrtimes = np.zeros((numsites, len(events), numphases))
        
    for siteid in range(numsites):
        for (eventid, event) in enumerate(events):
            lat = event[0]
            lon = event[1]
            depth = event[2]
            evtime = event[3]

            for phaseid in range(numphases):
                arrtimes[siteid,eventid,phaseid] = earthmodel.ArrivalTime(lat, lon, depth, evtime, phaseid, siteid)
    return arrtimes
    
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
        cursor.execute("select chan,time,endtime from idcx_wfdisc where sta='%s'" 
                       "and endtime > %f and time < %f order by time,endtime" % 
                       (sta, start_time, end_time))
        wfdiscs = cursor.fetchall()

        last_stime = -1
        last_etime = -1
        segment_chans = []
        for (chan, stime, etime) in waveforms:
            if stime != last_stime or etime != last_etime:
                if len(segment_chans) > 0:
                    traces.append(segment_chans)
                last_stime = stime
                last_etime = etime
                segment_chans = []

            st = np.max((stime, start_time))
            et = np.min((etime, end_time))

            print "fetching waveform {sta: ", sta, ", chan: ", chan, ", start_time: ", st, ", end_time: ", et, "}", 
            try:
                trace = utils.waveform.fetch_waveform(sta, chan, int(np.ceil(st)), int(np.floor(et)))
                trace.data = obspy.signal.filter.bandpass(trace.data,1,4,trace.stats['sampling_rate'])
                segment_chans.append(trace)
                print " ... successfully loaded."
            except (utils.waveform.MissingWaveform, IOError):
                print " ... not found, skipping."
                continue

    print "fetched ", len(traces), " segments."
    return traces

def max_over_channels(channel_bundle):
    max_data = []
    for channel in channel_bundle:
        if len(max_data) == 0:
            max_data = np.copy(channel.data)
            continue
        for i in range(len(channel.data)):
            max_data[i] = np.max((max_data[i], channel.data[i]))
    return max_data

def estimate_azi_amp(channel_bundle, det_time, det_len):

    for chan in channel_bundle:
        det_idx = round((det_time - chan.stats['starttime_unix']) * chan.stats['sampling_rate'])
        det_end_idx = det_idx + round(det_len * chan.stats['sampling_rate'])

        if chan.stats['channel'] == "BHE":
            bhe_det_signal = chan.data[det_idx:det_end_idx]
            bhe_det_energy = np.linalg.norm(bhe_det_signal, 2)
        else if chan.stats['channel'] == "BHN":
            bhn_det_signal = chan.data[det_idx:det_end_idx]
            bhn_det_energy = np.linalg.norm(bhn_det_signal, 2)
        else if chan.stats['channel'] == "BHZ":
            bhz_det_signal = chan.data[det_idx:det_end_idx]
            bhz_det_energy = np.linalg.norm(bhz_det_signal, 2)

        # TODO: learn these functions properly

        # angle of 0 means due north
        tan_azi = bhe_det_energy/bhn_det_energy
        tan_decl = bhz_det_energy/bhe_det_energy
        energy = np.linalg.norm((bhz_det_energy, bhn_det_energy, bhe_det_energy), 2)
        
        return np.degrees(np.arctan(tan_azi)), energy, np.degrees(np.arctan(tan_decl))

def det2fake(detections):
    fake = []
    for det in detections:
        fake.append((len(fake), det[0], det[2], det[10], det[4], det[6]))

def fake_detections(traces, sta_high_thresholds, sta_low_thresholds):
    # loop over traces
    # for each trace, compute sta/lta and apply a station-specific threshold
    # whenever this threshold is crossed, register a detection with appropriate amplitude, azimuth (from relative channels), and time

    detections = []
    for channel_bundle in traces:
        max_data = max_over_channels(channel_bundle)
        chan = channel_bundle[0].stats["channel"]
        siteid = channel_bundle[0].stats["siteid"]
        samprate = channel_bundle[0].stats["sampling_rate"]
        cft_data = obspy.signal.recStalta(filtered_data, int(1.5 * samprate),
                                          int(60 * samprate))
        triggers = triggerOnset(cft_data, sta_high_thresholds[siteid], sta_low_thresholds[siteid], 50 * samprate)
    
        for (trigger_start, trigger_end) in triggers:
            det_id = len(detections)
            det_time = trigger_start
            det_len = trigger_end - trigger_start
            det_azi, det_amp, det_slo = estimate_azi_amp_slow(channel_bundle, det_time, det_len)
            detections.append((det_id, siteid, det_time, det_amp, det_azi, det_slo))

def process_traces(traces, f, opts):
    # process each trace to yield a representation suitable for inference.
    # currently, that means computing energies within overlapping windows
    processed_traces = []
    for sta_traces in traces:
        processed_traces_sta = []
        for trace in sta_traces:
            processed_data = f(trace)
            new_header = trace.stats.copy()
            new_header.update(opts)
            new_header["npts_processed"] = len(processed_data)
            processed_trace = Trace(processed_data, header=new_header)
            processed_traces_sta.append(processed_trace)    
        processed_traces.append(processed_traces_sta)
    return processed_traces

def load_and_process_traces(cursor, start_time, end_time, window_size=1, overlap=0.5):
    cursor.execute("select sta, id from static_siteid where statype='ss'")
    stations = np.array(cursor.fetchall())
    stations = stations[:,0]

    traces = load_traces(cursor, stations, start_time, end_time)
    f = lambda trace: window_energies(trace, window_size=window_size, overlap=overlap)
    opts = dict(window_size=window_size, overlap=overlap)
    energies = process_traces(traces, f, opts)
    return energies, traces

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
    energies, traces = load_and_process_traces(cursor, options.start_time, options.end_time, options.window_size, options.overlap)
    print "loaded energies"

    # load earth and net models
    # read the detections and uptime
    detections, arid2num = dataset.read_detections(cursor, options.start_time, options.end_time, "idcx_arrival", 1)
    site_up = dataset.read_uptime(cursor, options.start_time, options.end_time,
                          "idcx_arrival")

    # read the rest of the static data
    sites = dataset.read_sites(cursor)
    phasenames, phasetimedef = dataset.read_phases(cursor)
    assert(len(phasenames) == len(phasetimedef))
    print "loaded other info from DB, now loading earth model..."

    earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)

    # sta_noise_means, sta_noise_vars = learn.learn_sigvisa("parameters", earthmodel)

    print "now loading sigvisa..."
    sm = learn.load_sigvisa("parameters",
                                options.start_time, options.end_time,
                                site_up, sites, phasenames, phasetimedef)
    
    print "sigvisa loaded, setting signals"
    sm.set_signals(energies)

    sta_high_thresholds = dict()
    sta_low_thresholds = dict()
    for i in range(200):
        sta_high_thresholds[i] = 1.5
        sta_low_thresholds[i] = 0.6
    print "signals set, computing fake detections"
    fake_det = fake_detections(traces, sta_high_thresholds, sta_low_thresholds)

    # read appropriate event set (e.g. netvisa)
    MAX_TRAVEL_TIME = 2000
    earliest_event_time = options.start_time - MAX_TRAVEL_TIME
    print "really reading events"
    events, orid2num = dataset.read_events(cursor, earliest_event_time, options.end_time, options.event_set, options.runid)
    print "loaded ", len(events), " events."
    
    # read associations, as training data for learning
    evlist = dataset.read_assoc(cursor, earliest_event_time, options.end_time, orid2num, arid2num, options.event_set, options.runid)
    print "loaded associations for ", len(events), " events."

    # calculate likelihood
    arrtimes = arrtime_point_matrix(len(sites), events, 1, earthmodel)
#   assoc_ttimes = sm.ttimes_from_assoc(evlist, events, detections, arid2num)

    #sm.learn(energies, events, assoc_ttimes)
    #print "scoring world!"
    #print arrtimes[0, 0, 0]
    #ll = sm.score_world(events, arrtimes)
    #print "log-likelihood is ", ll

    #print "learning params..."

    sm.set_fake_detections(det2fake(detections))

    
    

    if options.gui is not None:
        for trace in energies:   
            siteid = siteids[trace.stats["station"]]       
            if options.gui == "all" or trace.stats["station"] in options.gui:
                (means, variances) = sm.all_envelopes(trace.stats, events, ttimes[siteid])
                sm.plot_envelope(trace, means, variances)
        plt.show()




if __name__ == "__main__":
    main()
