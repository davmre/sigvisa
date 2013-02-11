# Copyright (c) 2012, Bayesian Logic, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Bayesian Logic, Inc. nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# Bayesian Logic, Inc. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#

import sys, struct, time
import numpy as np
import gzip


import sigvisa.database.db

class MissingWaveform(Exception):
  pass

class MissingSite(Exception):
  pass

def read_site(cursor, sta, epoch):
  """
  reads the static site table for this station around the epoch time
  raises MissingSite
  """
  epochtime = time.gmtime(epoch)
  yrday = epochtime.tm_year * 1000 + epochtime.tm_yday
  rowcnt = cursor.execute("select lon, lat, elev, statype, refsta, "
                          "dnorth, deast, staname from static_site where "
                          "((%d <= offdate and %d >= ondate) "
                          " or (offdate=-1 and %d >= ondate))"
                          "and sta='%s' order by ondate"
                          % (yrday, yrday, yrday, sta))
  if not rowcnt:
    raise MissingSite()

  lon, lat, elev, statype, refsta, dnorth, deast, staname = cursor.fetchone()

  return {"lon":lon, "lat":lat, "elev":elev, "statype": statype,
          "refsta":refsta, "dnorth":dnorth, "deast":deast, "staname":staname}

def read_site_ss(cursor, refsta, epoch):
  """
  reads the all the static site records for single station sites
  which have this refsta around the epoch

  raises MissingSite
  """
  epochtime = time.gmtime(epoch)
  yrday = epochtime.tm_year * 1000 + epochtime.tm_yday
  rowcnt = cursor.execute("select lon, lat, elev, sta, "
                          "dnorth, deast, staname from static_site where "
                          "((%d <= offdate and %d >= ondate) "
                          " or (offdate=-1 and %d >= ondate))"
                          "and refsta='%s' and statype='ss' order by ondate"
                          % (yrday, yrday, yrday, refsta))
  if not rowcnt:
    raise MissingSite()

  seen_sta = set()
  retval = []
  for lon, lat, elev, sta, dnorth, deast, staname in cursor.fetchall():
    if sta not in seen_sta:
      seen_sta.add(sta)
      retval.append({"lon":lon, "lat":lat, "elev":elev,
                     "sta":sta, "dnorth":dnorth, "deast":deast,
                     "staname":staname})
  return retval

def read_sitechan(cursor, sta, epoch):
  """
  returns all the static sitechan table for this station around the epoch time
  """
  epochtime = time.gmtime(epoch)
  yrday = epochtime.tm_year * 1000 + epochtime.tm_yday
  rowcnt = cursor.execute("select chan, ctype, edepth, hang, vang, descrip "
                          "from static_sitechan where sta='%s' and "
                          "((%d <= offdate and %d >= ondate) "
                          " or (offdate=-1 and %d >= ondate))"
                          " order by ondate"
                          % (sta, yrday, yrday, yrday))
  seen_chan = set()
  retval = []
  for chan, ctype, edepth, hang, vang, descrip in cursor.fetchall():
    if chan not in seen_chan:
      seen_chan.add(chan)
      retval.append({"chan":chan, "ctype":ctype, "edepth":edepth, "hang":hang,
                     "vang":vang, "descrip":descrip})

  return retval

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
  cursor = database.db.connect().cursor()

  # scan the waveforms for the given interval
  samprate = None
  data = []

  while True:
    TIME_COL, CHANID_COL, ENDTIME_COL, NSAMP_COL, SAMPRATE_COL = range(5)
    sql = "select * from idcx_wfdisc where sta = '%s' and chan ='%s' and time <= %f and %f < endtime" % (station, chan, stime, stime)
    cursor.execute(sql)
    waveform_values = cursor.fetchone()
    if waveform_values is None:
      raise MissingWaveform("Can't find data for sta %s chan %s time %d"
                            % (station, chan, stime))

    waveform = dict(zip([x[0].lower() for x in cursor.description], waveform_values))
    print cursor.description
    print waveform

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
