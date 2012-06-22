import time
import sys, struct
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.optimize

from optparse import OptionParser
from obspy.core import Trace, Stream, UTCDateTime
import obspy.signal.filter
from obspy.signal.trigger import triggerOnset

from database import db, dataset
import utils.waveform
import plot
import learn
import sys, os

# Frequency bands in Hz. From Mayeda et. al., "Stable and Transportable Regional Magnitudes Based on Coda-Derived Moment-Rate Spectra". (2003)
FREQ_BANDS = ((0.02, 0.03), (0.03, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0), (0.7, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 8.0))

# also need to update sigvisa.h whenever this changes
#FREQ_BANDS = ((0.5, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 8.0))


class MissingChannel(Exception):
  pass

def real_to_fake_det(det):
  return (det[dataset.DET_ARID_COL], det[dataset.DET_SITE_COL], det[dataset.DET_TIME_COL], det[dataset.DET_AMP_COL], det[dataset.DET_AZI_COL], det[dataset.DET_SLO_COL], det[dataset.DET_PHASE_COL])


def init_sigmodel(st=1237680000, hours=24):
  et = st + 3600*hours
  dbconn = db.connect()
  cursor = dbconn.cursor()
  sites = dataset.read_sites(cursor)
  site_up = dataset.read_uptime(cursor, st, et)
  phasenames, phasetimedef = dataset.read_phases(cursor)
  earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
  sigmodel = learn.load_sigvisa("parameters", st, et, "spectral_envelope", site_up, sites, phasenames, phasetimedef, load_signal_params = False)
  return cursor, sigmodel, earthmodel, sites, dbconn

def log_trace(trc, filename, format):

  real_fn = 'logs/%s.pdf' % (filename)

#  while (os.path.exists(real_fn)):
#    real_fn = real_fn + "_"

  print "logging to file", real_fn

  pp = PdfPages(real_fn)

  siteid = trc.stats["siteid"]
  start_time = trc.stats["starttime_unix"]
  if trc.stats["window_size"] is not None:
    srate = 1/ ( trc.stats.window_size * (1- trc.stats.overlap) )
    npts = trc.stats.npts_processed
  else:
    srate = trc.stats.sampling_rate
    npts = trc.stats.npts
  end_time = start_time + npts/srate

  text = "%s: siteid %d" % (filename, siteid)
  print text
  utils.waveform.plot_trace(trc, title = text, format=format)
  pp.savefig()
  pp.close()

  return True

def process_trace(trace, f, opts):
  new_header = trace.stats.copy()

  processed_data = f(trace)

  new_header["npts_processed"] = len(processed_data)
  new_header.update(opts)

  processed_trace = Trace(processed_data, header=new_header)
  return processed_trace

# return the fraction of the timespan of the first segment, which is overlapped by the second
def seconds_overlap(seg1, seg2):

    s1 = seg1[0][1]
    e1 = seg1[0][2]
    s2 = seg2[0][1]
    e2 = seg2[0][2]

    assert s1 <= s2

    if s2 >= e1:
        overlap=0
    else:
        overlap=min(e1,e2)-s2

    return overlap,s1,e1,s2,e2

def getchans(segment):
    return map(lambda x : x[0], segment)

def buildsegment(chans, start_time, end_time):
    segment = [(chan, start_time, end_time) for chan in chans]
    return segment

def insert_sorted(segments, segment, start=0):
    for i in range(start, len(segments)):
        if segment[0][1] < segments[i][0][1]:
            segments.insert(i, segment)
            return
    segments.append(segment)

def aggregate_segments(segments):
    aggregate_threshold = 1
    cutoff_threshold = 10

    min_len = 1

    changed = True

    while changed:
        changed = False

        n = len(segments)
        overlaps = np.zeros((n,n))

        # merge segments in order of how much they overlap
        for i in range(n-1):
          for j in range(i+1, n):
            overlap,s1,e1,s2,e2 = seconds_overlap(segments[i], segments[j])
            overlaps[i,j] = overlap
        amax = np.argmax(overlaps)
        i = amax/n
        j = amax % n
        overlap,s1,e1,s2,e2 = seconds_overlap(segments[i], segments[j])

        chans1 = getchans(segments[i])
        chans2 = getchans(segments[j])


        if overlap > aggregate_threshold and len(set(chans1).intersection(set(chans2))) == 0:

            agg_start = max(s1,s2)
            agg_end = min(e1,e2)

            del segments[j]
            del segments[i]

            agg = buildsegment(chans1 + chans2, agg_start, agg_end)


            if np.abs(agg_start - s1) > cutoff_threshold:
              pre_agg = buildsegment(chans1, s1, agg_start)
              insert_sorted(segments, pre_agg, i)

            if agg_end-agg_start > min_len:
              insert_sorted(segments, agg, i)

            post_agg = None
            if np.abs(e1 - agg_end) > cutoff_threshold:
              post_agg = buildsegment(chans1, agg_end, e1)
            elif np.abs(e2 - agg_end) > cutoff_threshold:
              post_agg = buildsegment(chans2, agg_end, e2)
              if post_agg is not None:
                insert_sorted(segments, post_agg, i)

            changed = True

    return segments

def print_segments(segments):
    for seg in segments:
        print seg


def load_event_traces(cursor, evids, evtype="leb", stations=None, process=None, downsample_factor = 1):
    traces = []
    traces_processed = []

    if stations is not None:
      stalist_str = str(stations).replace('[', '(').replace(']', ')')
      sta_cmd = "site.id in " + stalist_str
    else:
      sta_cmd = "site.statype='ss'"

    # get all arrivals for the specified events, at the given stations
    # (or at all non-array stations if not specified).
    format_strings = ','.join(['%s'] * len(evids))
    evid_list_str = str(evids).replace('[', '(').replace(']', ')')
    sql_query = "select site.id-1, iarr.arid, iarr.time, iarr.deltim, iarr.azimuth, iarr.delaz, iarr.slow, iarr.delslo, iarr.snr, ph.id-1, iarr.amp, iarr.per, site.sta from %s_origin ior, %s_assoc iass, %s_arrival iarr, static_siteid site, static_phaseid ph where ior.evid in %s and iass.orid=ior.orid and iarr.arid=iass.arid and iarr.delaz > 0 and iarr.delslo > 0 and iarr.snr > 0 and iarr.sta=site.sta and iarr.iphase=ph.phase and ascii(iarr.iphase) = ascii(ph.phase) and %s order by iarr.time, iarr.arid" %  (evtype, evtype, evtype, evid_list_str, sta_cmd)
    print "executing query ", sql_query
    cursor.execute(sql_query)
    arrivals = cursor.fetchall()

    for (idx, arr) in enumerate(arrivals):
      sta = arr[12]
      stm = float(arr[2]) - 4
      etm = float(arr[2]) +30

      segment_chans = dict()

      for chan in ("BHZ", "BHN", "BHE", "BH1", "BH2"):
        print "fetching waveform {sta: ", sta, ", chan: ", chan, ", start_time: ", stm, ", end_time: ", etm, "}",
        try:
          trace = utils.waveform.fetch_waveform(sta, chan, stm, etm)
          if chan == "BH1":
            trace.stats['channel'] = "BHE"
          if chan == "BH2":
            trace.stats['channel'] = "BHN"

          if trace.stats['channel'] not in segment_chans:
            segment_chans[trace.stats['channel']] = dict()

          if process is not None:
            trace_processed = process(trace)
            trace_processed.decimate(downsample_factor)
            segment_chans[trace.stats['channel']]["broadband_envelope"] =  trace_processed
          trace.decimate(downsample_factor)
          segment_chans[trace.stats['channel']]["broadband"] = trace
          print " ... successfully loaded."
        except (utils.waveform.MissingWaveform, IOError):
          print " ... not found, skipping."
          continue

      if len(segment_chans) > 0:
        traces.append(segment_chans)

    return traces


def load_traces(cursor, stations, start_time, end_time, process=None, downsample_factor = 1):
    traces = []

    for (idx, sta) in enumerate(stations):
        sql = "select distinct chan,time,endtime from idcx_wfdisc where sta='%s' and endtime > %f and time < %f order by time,endtime" % (sta, start_time, end_time)
        cursor.execute(sql)
        wfdiscs = cursor.fetchall()
        wfdiscs = filter(lambda x: x[0] in ["BHE", "BHN", "BHZ", "BH1", "BH2"], wfdiscs)

        if len(wfdiscs) == 0:
            continue

        segments = map(lambda x: [x], wfdiscs)
        segments = aggregate_segments(segments)
#        print "aggregated to "
#        print_segments(segments)

#        print "trying ", segments

        for segment in segments:
            segment_chans = dict()
            for (chan, st, et) in segment:

              stm = max(st, float(start_time))

              etm = min(et, float(end_time))

              # TODO: figure out why this happens
              if etm <= stm:
                continue

              print "fetching waveform {sta: ", sta, ", chan: ", chan, ", start_time: ", stm, ", end_time: ", etm, "}",
              try:
                trace = utils.waveform.fetch_waveform(sta, chan, stm, etm)


                if chan == "BH1":
                  trace.stats['channel'] = "BHE"
                if chan == "BH2":
                  trace.stats['channel'] = "BHN"

                if trace.stats['channel'] not in segment_chans:
                  segment_chans[trace.stats['channel']] = dict()

                if process is not None:
                  trace_processed = process(trace)
                  trace_processed.decimate(downsample_factor)
                  segment_chans[trace.stats['channel']]["broadband_envelope"] =  trace_processed
                print trace
                trace.decimate(downsample_factor)
                segment_chans[trace.stats['channel']]["broadband"] = trace
                print " ... successfully loaded."
              except (utils.waveform.MissingWaveform, IOError):
                print " ... not found, skipping."
                continue

            traces.append(segment_chans)

   # print "fetched ", len(traces), " segments."
    return traces

def max_over_channels(channel_bundle):
    max_data = []

    maxlen = -1
    maxidx = None
    for (idx, channel) in enumerate(channel_bundle):
        if len(channel) > maxlen:
            maxlen = len(channel)
            maxidx = idx
    max_data = np.copy(channel_bundle[maxidx])

    for channel in channel_bundle:
        chanlen = len(channel.data)
        for i in range(chanlen):
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

    bhe_num = None
    bhn_num = None
    bhz_num = None
    for (idx, chan) in enumerate(channel_bundle_det_window):
        if chan.stats["channel"] in ["BHE", "BH1"]:
            bhe_num = idx
        if chan.stats["channel"] in ["BHN", "BH2"]:
            bhn_num = idx
        if chan.stats["channel"] in ["BHZ"]:
            bhz_num = idx
    if bhe_num is None or bhn_num is None or bhz_num is None:
        raise MissingChannel("can't do polarization analysis without all three channels present")

    # first dim is channel, second is frame, third is idx
    frames = np.array(map(lambda chan : enframe(chan, obspy.signal.invsim.cosTaper(FRAME_LEN, .1), FRAME_INC)[0], channel_bundle_det_window))
    #frames = np.array(map(lambda chan : enframe(chan, np.ones((FRAME_LEN, 1)), FRAME_INC)[0], channel_bundle_det_window))


    maxrect = -1
    maxrect_amp = -1
    maxrect_azi = None
    maxrect_iangle = None
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

        pc = u[:,0]

        v =  np.dot(A.T, pc)
        #print "v is ", v
        if np.median(np.array(v)) < 0:
            pc = pc*-1

        #print "pc is ", pc, " means are ", np.mean(A,axis=1), " amp is ", amp

# flip according to the projection onto the first principle component?
        y = pc[bhn_num]
        x = pc[bhe_num]
        seazp = np.rad2deg(np.arctan2(y,x))
        if seazp < 0:
            seazp = seazp + 360
        if seazp > 360:
            seazp = seazp - 360

        if rect > maxrect:
            maxrect = rect
            maxrect_amp = amp
            maxrect_azi = seazp
            maxrect_iangle = inang1

    return maxrect_azi, maxrect_amp, maxrect_iangle

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
        else:
            print "strange channel", chan.stats['channel']
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

            try:
                det_azi, det_amp, det_slo = estimate_azi_amp_slo(channel_bundle_window)
            except:
                continue
            if det_azi is not None:
                det = (det_id, siteid, det_time, det_amp, det_azi, det_slo, 0)
                print "adding det", det
                detections.append(det)
                i = i+1

    return detections


def compute_narrowband_envelopes(segments):

  for segment_chans in segments:

    for (chan, chan_bands) in segment_chans.items():

      broadband_signal = chan_bands["broadband"]

      # compute log envelope for each frequency band in this component
      for band in FREQ_BANDS:
        band_data = obspy.signal.filter.bandpass(broadband_signal.data, band[0], band[1], broadband_signal.stats['sampling_rate'], corners = 4, zerophase=True)
        band_env = obspy.signal.filter.envelope(band_data)
        band_name = "narrow_envelope_%1.2f_%1.2f" % (band[0], band[1])
        band_trace = Trace(band_env, dict(broadband_signal.stats.items() + [("band", band_name)]))
        chan_bands[band_name] = band_trace

    # average the two horizonal components, if they're both present
    horiz_avg = None
    chan1 = None
    chan2 = None
    if "BHE" in segment_chans:
      horiz_avg = segment_chans["BHE"]
      chan1 = segment_chans["BHE"]
    if "BH1" in segment_chans:
      horiz_avg = segment_chans["BH1"]
      chan1 = segment_chans["BH1"]
    if "BHN" in segment_chans:
      horiz_avg = segment_chans["BHN"]
      chan2 = segment_chans["BHN"]
    if "BH2" in segment_chans:
      horiz_avg = segment_chans["BH2"]
      chan2 = segment_chans["BH2"]

    if chan1 is not None and chan2 is not None:
      horiz_avg = dict()
      for band in chan1.keys():
        if not band.startswith("narrow_envelope"):
          continue

        horiz_avg_data = np.exp(( np.log(chan1[band].data) + np.log(chan2[band].data) ) /2)
        horiz_avg_trace = Trace(horiz_avg_data, header = chan1[band].stats.copy())
        horiz_avg_trace.stats["channel"] = "horiz_avg"
        horiz_avg[band] = horiz_avg_trace
    if horiz_avg is not None:
      segment_chans["horiz_avg"] = horiz_avg

# given a list of segments (e.g. as loaded by
# load_and_process_traces), extract a signle segment containing a
# single time window at a single statement, possibly by concatenating
# several of the original segments
def extract_timeslice_at_station(traces, start_time, end_time, siteid):

  new_segment = dict()
  last_segment_end = float("NaN")

  for segment in traces:

    c = segment.keys()[0]
    b = segment[c].keys()[0]
    trc = segment[c][b]
    if trc.stats.siteid != siteid:
      continue

    npts = trc.stats.npts
    srate = trc.stats.sampling_rate
    st = trc.stats.starttime_unix
    et = st + npts / srate

    if et < start_time or st > end_time:
      continue

    # to avoid discontinuities, we require each new segment starts within two seconds of the end of the previous.
    if not np.isnan(last_segment_end) and np.abs( st - last_segment_end ) > 2:
      continue
    last_segment_end = min(et, end_time)

    if len(new_segment.keys()) == 0:
      # initialize the extracted segment
      for chan in segment.keys():
        new_segment[chan] = dict()
        for band in segment[chan].keys():
          trc = segment[chan][band]
          new_stats = trc.stats.copy()
          start_point = int((max(start_time,st) - st) * srate)
          end_point = int((min(et, end_time)-st) * srate)
          new_trc = Trace(trc.data[start_point:end_point], new_stats)
#          print "initializing with data from %f to %f" % (max(start_time, st), min(end_time, et))
          new_trc.stats.starttime_unix = start_time
          new_trc.stats.npts = len(new_trc.data)
          new_segment[chan][band] = new_trc
    else:
        # extend the extracted segment
      for chan in new_segment.keys():
        for band in new_segment[chan].keys():

          trc = segment[chan][band]
          new_trc = new_segment[chan][band]
          start_point = 0
          end_point = int((min(et, end_time)-st) * srate)
#          print "continuing with data from %f to %f" % (st, min(end_time, et))
          new_trc.data = np.concatenate((new_trc.data, trc.data[start_point:end_point]))
          new_trc.stats.npts = len(new_trc.data)


  return [new_segment,]

def load_and_process_traces(cursor, start_time, end_time, stalist=None, downsample_factor=1):

    if stalist is None:
        cursor.execute("select sta, id from static_siteid where statype='ss'")
        stations = np.array(cursor.fetchall())
        stations = stations[:,0]
    else:
        stations = []
        for siteid in stalist:
            cursor.execute("select sta, id from static_siteid where id=%d" % (siteid))
            a = cursor.fetchone()
            if a is not None:
                stations.append(np.array(a)[0])
        print "loading traces from stations", stations
    opts = dict()
    f = lambda trace: obspy.signal.filter.envelope(trace.data)
    pr = lambda trace: process_trace(trace, f=f, opts=opts)
    traces = load_traces(cursor, stations, start_time, end_time, process=pr, downsample_factor = downsample_factor)

    return traces

def load_and_process_event_traces(cursor, evids, evtype="leb", window_size=1, overlap=0.5, stations=None, downsample_factor=8):
    f = lambda trace: obspy.signal.filter.envelope(trace.data)
    pr = lambda trace: process_trace(trace, f=f, opts=dict())
    traces = load_event_traces(cursor, evids, evtype, stations=stations, process=pr, downsample_factor=downsample_factor)

    return traces

def trim_first_n_seconds(segments, n):
  for segment in segments:
    for trc in segment.values():
      srate = trc.stats.sampling_rate
      npts = trc.stats.npts

      to_trim = n*srate

      if to_trim < npts:
        trc.data = trc.data[to_trim:]
        trc.stats.npts = len(trc.data)
        trc.stats.starttime_unix = trc.stats.starttime_unix + n
  return segments


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
    parser.add_option("--start", dest="start_time", type="float", default=1238889600,
                    help = "start time")
    parser.add_option("--end", dest="end_time", type="float", default=1238893000,
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
    traces = load_and_process_traces(cursor, options.start_time, options.end_time, stalist=(2,))
    print "loaded energies"

    for segment in traces:
      for (chan, chan_dict) in segment.items():
        print chan
        print chan_dict

    compute_narrowband_envelopes(traces)

    for segment in traces:
      for (chan, chan_dict) in segment.items():
        print chan
        print chan_dict


    print "loaded", traces[0]["BHZ"]["broadband"].data, "len", len(traces[0]["BHZ"]["broadband"].data)

    pp = PdfPages('logs/test_bands.pdf')
    plot.plot_bands(traces[0]["BHZ"])
    pp.savefig()
    pp.close()

    sys.exit(0)


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


if __name__ == "__main__":
    main()
