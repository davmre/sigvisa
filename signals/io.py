import time
import sys, struct, sys, os, gzip, functools32
import numpy as np
import numpy.ma as ma
import scipy.optimize

from optparse import OptionParser
from obspy.core import Trace, Stream, UTCDateTime
import obspy.signal.filter
from obspy.signal.trigger import triggerOnset

from database.dataset import *
import plot

from signals.common import *
from signals.mask_util import *
# from signals.coda_decay_common import *

from sigvisa import Sigvisa

class MissingWaveform(Exception):
  pass

def load_event_station_chan(evid, sta, chan, evtype="leb", cursor=None):
  if cursor is None:
    cursor = Sigvisa().dbconn.cursor()

  arrivals = read_event_detections(cursor, evid, (sta,), evtype=evtype)
  arrival_times = arrivals[:, DET_TIME_COL]

  wave = fetch_waveform(sta, chan, np.min(arrival_times)-10, np.max(arrival_times)+200)
  wave.segment_stats['evid'] = evid
  wave.segment_stats['event_arrivals'] = arrivals

  return wave


def load_event_station(evid, sta, evtype="leb", cursor=None):
  if cursor is None:
    cursor = Sigvisa().dbconn.cursor()

  arrivals = read_event_detections(cursor, evid, (sta,), evtype=evtype)
  arrival_times = arrivals[:, DET_TIME_COL]
  seg = load_segments(cursor, (sta,), np.min(arrival_times)-10, np.max(arrival_times)+200)[0]
  seg.stats['evid'] = evid
  seg.stats['event_arrivals'] = arrivals

  return seg

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


def load_segments(cursor, stations, start_time, end_time, chans=None):
  """
  Return a list of waveform segments corresponding to the given channels
  at the given stations over the given time period.
  """

  segments = []

  if chans is None:
    chans = Sigvisa().chans

  # standardize channel names to avoid duplicates
  chans = [Sigvisa().canonical_channel_name[c] for c in chans]

  for (idx, sta) in enumerate(stations):

    waves = []

    for (chanidx, chan) in enumerate(chans):
      print "loading sta %s chan %s time [%.1f, %.1f]" % (sta, chan, start_time, end_time),
      sys.stdout.flush()
      try:
        wave = fetch_waveform(sta, chan, start_time, end_time)
        print " ... successfully loaded."
      except (MissingWaveform, IOError) as e:
        print " ... not found, skipping. (%s)" % e
        continue
      waves.append(wave)

    if len(waves) > 0:
      segment = Segment(waves)
      segments.append(segment)

  if len(segments) == 0:
    raise MissingWaveform("couldn't load any waveforms for this segment; check that the data files are in the correct location.")

  return segments


@functools32.lru_cache(maxsize=1024)
def fetch_waveform(station, chan, stime, etime, pad_seconds=20):
  """
  Returns a single Waveform for the given channel at the station in
  the given interval. If there are periods for which data are not
  available, they are marked as missing data.

  Loads data for pad_seconds before and after the true start time, and
  masks the additional data. This is used to absorb filtering
  artifacts.
  """
  s = Sigvisa()
  cursor = s.dbconn.cursor()

  # scan the waveforms for the given interval
  samprate = None

  orig_stime = stime
  orig_etime = etime
  stime = stime-pad_seconds
  etime = etime+pad_seconds

  # the global_data array is initialized below once we know the
  # samprate
  global_data = None
  global_stime = stime
  global_etime = etime

  chan = s.canonical_channel_name[chan]
  chan_list = s.equivalent_channels(chan)

  if station=="MKAR":
    selection="MK31"
  else:
    selection=station

  # explicitly do BETWEEN queries (with generous bounds) rather than just checking time < etime and endtime > stime, because the latter creates a monstrous slow database join
  MAX_SIGNAL_LEN=3600*8
  sql = "select * from idcx_wfdisc where sta = '%s' and %s and time between %f and %f and endtime between %f and %f" % (selection, sql_multi_str("chan", chan_list), stime-MAX_SIGNAL_LEN, etime, stime, etime+MAX_SIGNAL_LEN)
  cursor.execute(sql)
  waveforms = cursor.fetchall()
  if not waveforms:
    raise MissingWaveform("Can't find data for sta %s chan %s time %d"
                          % (station, chan, stime))
  table_description = cursor.description

  for waveform_values in waveforms:

    waveform = dict(zip([x[0].lower() for x in table_description], waveform_values))
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

    # at which offset into this waveform should we start collecting samples
    first_offset_time = max(stime-waveform['time'], 0)
    first_offset = int(np.floor(first_offset_time*samprate))
    # how many samples are needed remaining
    load_start_time = waveform['time'] + first_offset_time
    desired_samples = int(np.floor((global_etime - load_start_time) * samprate))
    # how many samples are actually available
    available_samples = waveform['nsamp'] - first_offset
    # grab the available and needed samples
    try:
      wave = _read_waveform_from_file(waveform, first_offset,
                                       min(desired_samples, available_samples))
    except IOError:
      raise MissingWaveform("Can't find data for sta %s chan %s time %d"
                            % (station, chan, stime))

    # copy the data we loaded into the global array
    t_start = max(0, int((waveform['time'] - global_stime) *samprate))
    t_end = t_start + len(wave)
    global_data[t_start:t_end] = wave

#    print "   loaded data from %d to %d (%.1f to %.1f)" % (t_start, t_end, t_start/samprate, t_end/samprate)

    # do we have all the data that we need
    if desired_samples <= available_samples:
      break

    # otherwise move the start time forward for the next file
    stime = waveform['endtime']
    # and adust the end time to ensure that the correct number of samples
    # will be selected in the next file

  masked_data = mirror_missing(ma.masked_invalid(global_data))

  if pad_seconds > 0:
    pad_samples = pad_seconds * samprate
    masked_data[0:pad_samples] = ma.masked
    masked_data[-pad_samples:] = ma.masked

  return Waveform(data=masked_data, sta=station, stime=global_stime, srate=samprate, chan=chan)

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
    print "cannot open file ", filename,
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
