
import time
import sys, MySQLdb,struct
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from optparse import OptionParser
from obspy.core import Trace, Stream, UTCDateTime
import obspy.signal.filter
from obspy.signal.trigger import triggerOnset

from database import db, dataset
import utils.waveform
import netvisa, learn
import sys

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

def load_traces(cursor, stations, start_time, end_time, small):
    traces = []

    for (idx, sta) in enumerate(stations):
        if (small and idx > 2):
            break

        sql = "select chan,time,endtime from idcx_wfdisc where sta='%s' and endtime > %f and time < %f order by time,endtime" % (sta, start_time, end_time)
        cursor.execute(sql)
        wfdiscs = cursor.fetchall()

        last_stime = -1
        last_etime = -1
        segment_chans = []
        for (chan, stime, etime) in wfdiscs:
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



def enframe(x, win, inc):
    """
    Splits the vector up into (overlapping) frames beginning at increments
    of inc. Each frame is multiplied by the window win().
    The length of the frames is given by the length of the window win().
    The centre of frame I is x((I-1)*inc+(length(win)+1)/2) for I=1,2,...
    
    :param x: signal to split in frames
    :param win: window multiplied to each frame, length determines frame length
    :param inc: increment to shift frames, in samples
    :return f: output matrix, each frame occupies one row
    :return length, no_win: length of each frame in samples, number of frames
    """
    nx = len(x)
    nwin = len(win)
    if (nwin == 1):
        length = win
    else:
	        #length = nextpow2(nwin)
        length = nwin
    nf = int(np.fix((nx - length + inc) // inc))
	    #f = np.zeros((nf, length))
    indf = inc * np.arange(nf)
    inds = np.arange(length) + 1
    f = x[(np.transpose(np.vstack([indf] * length)) + \
               np.vstack([inds] * nf)) - 1]
    if (nwin > 1):
        w = np.transpose(win)
        f = f * np.vstack([w] * nf)
    no_win, _ = f.shape
    return f, length, no_win
    

def estimate_azi_amp_slo(channel_bundle_det_window):

    FRAME_LEN = 20
    FRAME_INC = 5

    bhe_num = 0
    bhn_num = 1
    bhz_num = 2

    # first dim is channel, second is frame, third is idx
    frames = np.array(map(lambda chan : enframe(chan, obspy.signal.invsim.cosTaper(FRAME_LEN, .1), FRAME_INC)[0], channel_bundle_det_window))

    maxamp = -1
    maxamp_azi = None
    maxamp_slo = None
    nframes = frames.shape[1]
    for frameno in range(nframes):
        A = np.matrix(frames[:,frameno,:]).squeeze()

        M = (A-np.mean(A,axis=1))
        covM = np.cov(M)
        l,u = np.linalg.eigh(covM)

        # sort so that largest eigenvalue is first
        reorder = np.argsort(l)[::-1]
        l = np.take(l, reorder)
        u = np.take(u, reorder, axis=1)

        amp = np.linalg.norm(l, 2)
        rect = 1 - (l[2] + l[1])/ (2 * l[0])

        inang1 = np.rad2deg(np.arccos(np.abs(u[bhz_num,0])))
        inang3 = np.rad2deg(np.arccos(np.abs(u[bhz_num,2])))

        seazp = np.rad2deg(np.arctan2(u[bhn_num,0], u[bhe_num,0])) + 180

        if amp > maxamp:
            maxamp = amp
            maxamp_azi = seazp
            maxamp_slo = inang1

    return maxamp_azi, maxamp, maxamp_slo

def det2fake(detections):
    fake = []
    for det in detections:
        fake.append((len(fake), det[0], det[2], det[10], det[4], det[6]))

def order_channels(channel_bundle):
    nchans = len(channel_bundle)
    bhe_num = None
    bhz_num = None
    bhn_num = None
    for chan_num in range(nchans):
        chan = channel_bundle[chan_num]
        if chan.stats['channel'] == "BHE":
            bhe_num = chan_num
        elif chan.stats['channel'] == "BHN":
            bhn_num = chan_num
        elif chan.stats['channel'] == "BHZ":
            bhz_num = chan_num
    return channel_bundle[bhe_num], channel_bundle[bhn_num], channel_bundle[bhz_num]

def fake_detections(traces, sta_high_thresholds, sta_low_thresholds):
    # loop over traces
    # for each trace, compute sta/lta and apply a station-specific threshold
    # whenever this threshold is crossed, register a detection with appropriate amplitude, azimuth (from relative channels), and time

    detections = []
    for channel_bundle in traces:
        channel_bundle = order_channels(channel_bundle)
        siteid = channel_bundle[0].stats["siteid"]
        samprate = channel_bundle[0].stats["sampling_rate"]
        start_time = channel_bundle[0].stats["starttime_unix"]
            
        print "computing sta/lta and triggers at ", siteid
        max_data = max_over_channels(channel_bundle)
        cft_data = obspy.signal.recStalta(max_data, int(1.5 * samprate),
                                          int(60 * samprate))
        triggers = triggerOnset(cft_data, sta_high_thresholds[siteid], sta_low_thresholds[siteid], 50 * samprate)

        print " estimating detection info ..., "
        i=0
        for (trigger_start, trigger_end) in triggers:
            det_id = len(detections)
            det_time = trigger_start / samprate + start_time
            
            channel_bundle_window = map(lambda c : c[trigger_start:trigger_end], channel_bundle)

            det_azi, det_amp, det_slo = estimate_azi_amp_slo(channel_bundle_window)
            if det_azi is not None:
                det = (det_id, siteid, det_time, det_amp, det_azi, det_slo)
                print "adding det", det
                detections.append(det)
                i = i+1
    return detections

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

def load_and_process_traces(cursor, start_time, end_time, window_size=1, overlap=0.5, small=False):
    cursor.execute("select sta, id from static_siteid where statype='ss'")
    stations = np.array(cursor.fetchall())
    stations = stations[:,0]

    traces = load_traces(cursor, stations, start_time, end_time, small)
    f = lambda trace: window_energies(trace, window_size=window_size, overlap=overlap)
    opts = dict(window_size=window_size, overlap=overlap)
    energies = process_traces(traces, f, opts)
    return energies, traces

def print_trace(trace):
    prevx = None
    for (i, x) in enumerate(trace.data):
        if x != prevx:
            print i, x
            prevx = x

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
    parser.add_option("--small", dest="small", default=False, action = "store_true",
                    help = "read only a small subset of signals, for debugging (False)")
    parser.add_option("-g", "--gui", dest="gui", default=None,
                    help = "show trace and envelope plots (False)")
    (options, args) = parser.parse_args()

    # read traces for each station
    cursor = db.connect().cursor()
    energies, traces = load_and_process_traces(cursor, options.start_time, options.end_time, options.window_size, options.overlap, options.small)
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

    print "set energies"

    recovered = sm.get_signals()
    print "recovered this:"
    #utils.waveform.plot_segment(energies[0])
    utils.waveform.plot_segment(recovered[0])

    plt.show()

    return
    
    sta_high_thresholds = dict()
    sta_low_thresholds = dict()
    for i in range(200):
        sta_high_thresholds[i] = 1.5
        sta_low_thresholds[i] = 0.6
    print "signals set, computing fake detections"
    fake_det = fake_detections(traces, sta_high_thresholds, sta_low_thresholds)

    return

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

    #sm.set_fake_detections(det2fake(detections))

    
    

    if options.gui is not None:
        for trace in energies:   
            siteid = siteids[trace.stats["station"]]       
            if options.gui == "all" or trace.stats["station"] in options.gui:
                (means, variances) = sm.all_envelopes(trace.stats, events, ttimes[siteid])
                sm.plot_envelope(trace, means, variances)
        plt.show()




if __name__ == "__main__":
    main()
