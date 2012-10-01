import time
import sys, struct
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import numpy.ma as ma
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

from signals.coda_decay_common import *

CHAN_SYNONYMS = (("BHE", "BH1"), ("BHN", "BH2"))

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

    if len(segments) == 0:
      return segments

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


def has_trace(cursor, sta=None, start_time=None, end_time=None, evid=None, earthmodel=None, siteid=None):

    if start_time is None or end_time is None:
      arr_times, phaseids = predict_event_arrivals(cursor, earthmodel, evid, siteid, [1,2,4,5])
      arr_times = [a for a in arr_times if a > 0]
      start_time = np.min(arr_times)-5
      end_time = np.max(arr_times) + 300

    if sta is None:
        sta = siteid_to_sta(siteid, cursor)

    sql = "select distinct chan,time,endtime from idcx_wfdisc where sta='%s' and endtime > %f and time < %f order by time,endtime" % (sta, start_time, end_time)
    cursor.execute(sql)
    wfdiscs = cursor.fetchall()
    wfdiscs = filter(lambda x: x[0] in ["BHE", "BHN", "BHZ", "BH1", "BH2"], wfdiscs)

    s1 = map(lambda x: [x], wfdiscs)
    s2 = reduce(lambda a,b : a+b, aggregate_segments(s1), [])
    s3 = [s for s in s2 if max(s[1], float(start_time)) < min(s[2], float(end_time))]

    return len(s3) > 0


def load_segments(cursor, stations, chans, start_time, end_time):
    traces = []

    for (idx, sta) in enumerate(stations):
      for (chanidx, chan) in enumerate(chans):

        trace = utils.waveform.fetch_waveform(sta, chan, start_time, end_time)

        if trace.stats['channel'] not in segment_chans:
          segment_chans[trace.stats['channel']] = dict()

                print trace
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


def compute_narrowband_envelopes(segments):

  for segment_chans in segments:

    for (chan, chan_bands) in segment_chans.items():

      broadband_signal = chan_bands["broadband"]

      # compute log envelope for each frequency band in this component
      for band in FREQ_BANDS:
        band_data = obspy.signal.filter.bandpass(broadband_signal.data, band[0], band[1], broadband_signal.stats['sampling_rate'], corners = 4, zerophase=True)
        band_env = obspy.signal.filter.envelope(band_data)
        band_name = "narrow_envelope_%1.2f_%1.2f" % (band[0], band[1])
        short_band = "%1.2f_%1.2f" % (band[0], band[1])
        band_trace = Trace(band_env, dict(broadband_signal.stats.items() + [("band", band_name)] + [("short_band", short_band)]))
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
              sta = np.array(a)[0]
              if sta=="MKAR":
                sta = 'MK31'
              stations.append(sta)
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



def fetch_waveform(station, chan, stime, etime):
  """
  Returns a single obspy trace corresponding to the waveform for the given
  channel at the station in the given interval.
  """
  cursor = database.db.connect().cursor()

  # scan the waveforms for the given interval
  samprate = None

  # the global_data array is initialized below once we know the
  # samprate.za
  global_data = None
  global_stime = stime
  global_etime = etime

  # get the equivalence class of names for this channel
  chan_equiv = (chan,)
  for chansyn in CHAN_SYNONYMS:
    if chan in chansyn:
      chan_equiv = chansyn
      break
  chan = chansyn[0]

  while True:

    sql = "select * from idcx_wfdisc where sta = '%s' and %s and time <= %f and %f < endtime" % (station, sql_values("chan", chan_list), stime, stime)

    cursor.execute(sql)
    waveform_values = cursor.fetchone()

    if waveform_values is None:
      raise MissingWaveform("Can't find data for sta %s chan %s time %d"
                            % (station, chan, stime))

    waveform = dict(zip([x[0].lower() for x in cursor.description], waveform_values))
    cursor.execute("select id from static_siteid where sta = '%s'" % (station))
    try:
      if station=="MK31":
        siteid=66
      else:
        siteid = cursor.fetchone()[0]
    except:
      raise MissingWaveform("couldn't get siteid for station %s" % (station))

    # check the samprate is consistent for all waveforms in this interval
    assert(samprate is None or samprate == waveform['samprate'])
    if samprate is None:
      samprate = waveform['samprate']

      # initialize the data array full of nans.
      # these will be replace by signals that we load.
      global_data = np.empty((int((global_etime-global_stime)*samprate),))
      global_data.fill(np.nan)

    # at which offset should we start collecting samples
    first_off = int((stime-waveform['time'])*samprate)
    # how many samples are needed remaining
    desired_samples = int(round((etime - stime) * samprate))
    # how many samples are actually available
    available_samples = waveform['nsamp'] - first_off
    # grab the available and needed samples
    try:
      wave = _read_waveform_from_file(waveform, first_off,
                                       min(desired_samples, available_samples))
    except IOError:
      raise MissingWaveform("Can't find data for sta %s chan %s time %d"
                            % (station, chan, stime))


    # copy the data we loaded into the global array
    t_start = max(0, int((waveform['time'] - global_stime) *samprate))
    t_end = t_start + len(wave)
    global_data[t_start:t_end] = wave

    # do we have all the data that we need
    if desired_samples <= available_samples:
      break

    # otherwise move the start time forward for the next file
    stime = waveform['endtime']
    # and adust the end time to ensure that the correct number of samples
    # will be selected in the next file
    etime = stime + (desired_samples - available_samples) / float(samprate)

  stats = {'network': station, 'station': station, 'location': '',
           'channel': chan, 'npts': len(global_data), 'sampling_rate': samprate,
           'mseed' : {'dataquality' : 'D'}}
  stats['starttime_unix'] = stime
  stats['siteid'] = int(siteid)
  stats['chanid'] = int(waveform['chanid'])
  stats['starttime'] = UTCDateTime(stime)

  masked_data = ma.masked_invalid(global_data)

  return Trace(data=np.array(masked_data), header=stats)

  #return samprate, np.array(data)

def _read_waveform_from_file(waveform, skip_samples, read_samples):
  """
  waveform -- row queried from wfdisc table
  """
  # open the waveform file
  #filename = os.path.join(*(waveform['dir'].split("/")
  #                          + [waveform['dfile']]))
  filename = waveform['dir'] + waveform['dfile']
  try:
    datafile = open(filename, "rb")
  except IOError, e:
    print "cannot open file ", filename
    # the file could be compressed try .gz extension
    datafile = gzip.open(filename+".gz")

  assert(waveform['datatype'] in ["s3", "s4"])
  bytes_per_sample = int(waveform['datatype'][-1])

  # seek to the desired offset
  datafile.seek(waveform['foff'] + skip_samples * bytes_per_sample)
  # and read the number of bytes required
  assert(read_samples <= waveform['nsamp'])
  bytes = datafile.read(read_samples * bytes_per_sample)
  datafile.close()

  # now convert the bytes into an array of integers

  data = np.ndarray((read_samples,), int)

  if waveform['datatype'] == "s4":
    data = struct.unpack(">%di" % read_samples, bytes)

  else:
    # s3
    for dest in xrange(read_samples):
      src = dest * 3

      # if the first byte's MSB is set then add an FF to the number
      first = struct.unpack("B", bytes[src])[0]
      if first >= 128:
        data[dest] = struct.unpack(">i", "\xff" + bytes[src:src+3])[0]
      else:
        data[dest] = struct.unpack(">i", "\x00" + bytes[src:src+3])[0]

  # convert the raw values into nm (nanometers)
  calib = float(waveform['calib'])
  return [float(x) * calib for x in data]
