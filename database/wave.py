import sys, MySQLdb,struct
import numpy as np
import gzip

import database.db

class MissingWaveform(Exception):
  pass

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

def fetch_waveform(station, chan, stime, etime):
  """
  Returns a single obspy trace corresponding to the waveform for the given
  channel at the station in the given interval.
  """
  cursor = database.db.connect().cursor(MySQLdb.cursors.DictCursor)
  
  # scan the waveforms for the given interval
  samprate = None
  data = []
  
  while True:
    cursor.execute("select * from idcx_wfdisc where sta = '%s' and chan ='%s' "
                   "and time <= %f and %f < endtime" %
                   (station, chan, stime, stime))
    
    waveform = cursor.fetchone()
    if waveform is None:
      raise MissingWaveform("Can't find data for sta %s chan %s time %d"
                            % (station, chan, stime))
    
    # check the samprate is consistent for all waveforms in this interval
    assert(samprate is None or samprate == waveform['samprate'])
    if samprate is None:
      samprate = waveform['samprate']
    
    # at which offset should we start collecting samples
    first_off = int((stime-waveform['time'])*samprate)
    # how many samples are needed remaining
    desired_samples = int(round((etime - stime) * samprate))
    # how many samples are actually available
    available_samples = waveform['nsamp'] - first_off
    # grab the available and needed samples
    segment = _read_waveform_from_file(waveform, first_off,
                                       min(desired_samples, available_samples))
    data.extend(segment)
    
    # do we have all the data that we need
    if desired_samples <= available_samples:
      break
    
    # otherwise move the start time forward for the next file
    stime = waveform['endtime']
    # and adust the end time to ensure that the correct number of samples
    # will be selected in the next file
    etime = stime + (desired_samples - available_samples) / float(samprate)

  return data, samprate


# http://en.wikipedia.org/wiki/High-pass_filter
def highpass_filter(data, samprate, cutoff_freq):
  """
  Returns the data with the lower frequencies below cutoff_freq removed
  """
  rc = 1.0 / (2 * np.pi * cutoff_freq)
  alpha = rc / (rc + 1.0/samprate)

  output = [0 for _ in data]
  output[0] = data[0]
  for i in xrange(1, len(data)):
    output[i] = alpha * output[i-1] + alpha * (data[i] - data[i-1])
  return output

def lowpass_filter(data, samprate, cutoff_freq):
  """
  Returns the data with the higher frequencies above cutoff_freq removed
  """
  rc = 1.0 / (2 * np.pi * cutoff_freq)
  alpha = (1.0/samprate) / (rc + (1.0/samprate))

  output = [0 for _ in data]
  output[0] = data[0]
  for i in xrange(1, len(data)):
    output[i] = (1-alpha) * output[i-1] + alpha * data[i]
  return output

