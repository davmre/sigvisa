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
# draw the waveforms for an LEB event at some of the stations

import sys, struct
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import gzip

from obspy.core import Trace, Stream, UTCDateTime
from obspy.signal import filter as sig_filter
from obspy.signal.array_analysis import sonic

from database.dataset import *
import database.db
import learn
from utils.geog import dist_deg, azimuth

from priors.coda_decay.coda_decay_common import *
from plot import *

from optparse import OptionParser


class MissingWaveform(Exception):
  pass

def plot_ss_waveforms(siteid, start_time, end_time, detections, earthmodel,
                      event):
  cursor = database.db.connect().cursor()
  cursor.execute("select sta from static_siteid where id=%d" % (siteid+1,))
  sta, = cursor.fetchone()
  print "Station:", sta

  cursor.execute("select distinct chan from idcx_wfdisc where sta='%s'" % sta)
  chans = [x for x, in cursor.fetchall() if x in priors.coda_decay.coda_decay_common.chans]
  print "Channels:", chans

  # get the times of all the arrivals at this site within the window
  all_det_times = []
  for det in detections:
    if int(det[DET_SITE_COL]) == siteid:
      arrtime = det[DET_TIME_COL]
      if arrtime > start_time and arrtime < end_time:
        all_det_times.append(arrtime - start_time)

  # get the times of all the phases at this site from the event in the window
  all_phase_times, all_phase_names = [], []
  for phaseid in xrange(earthmodel.NumTimeDefPhases()):
    arrtime = earthmodel.ArrivalTime(event[ EV_LON_COL],
                                     event[ EV_LAT_COL],
                                     event[ EV_DEPTH_COL],
                                     event[ EV_TIME_COL],
                                     phaseid,
                                     siteid)
    if arrtime > start_time and arrtime < end_time:
      all_phase_times.append(arrtime - start_time)
      all_phase_names.append(earthmodel.PhaseName(phaseid))

  plt.figure()
  plt.xlabel("Time (s)")
  for chidx, chan in enumerate(chans):
    if chidx == 0:
      axes = plt.subplot(len(chans), 1, 1)
    else:
      plt.subplot(len(chans), 1, chidx+1, sharex=axes)

    plt.ylabel(sta+" - "+chan)
    try:
      trc = fetch_waveform(sta, chan, start_time, end_time)
    except:
      plt.close()
      raise MissingWaveform("Can't find signal for sta %s chan %s time %f - %f"
                            % (sta, chan, start_time, end_time))
    trc.filter('bandpass', freqmin=1.0, freqmax=2.0, corners=2,
               zerophase=True)
    env = sig_filter.envelope(trc.data)


    timevals = np.arange(0, trc.stats.npts/trc.stats.sampling_rate,
                         1.0 /trc.stats.sampling_rate)
    plt.plot(timevals, trc.data, 'k:')
    plt.plot(timevals, env, 'r')

    maxtrc, mintrc = float(max(trc.data)), float(min(trc.data))
    plt.bar(left=all_det_times, height=[maxtrc-mintrc for _ in all_det_times],
            width=.25, bottom=mintrc, color="red", linewidth=0, alpha=.5)

    plt.bar(left=all_phase_times,
            height=[maxtrc-mintrc for _ in all_phase_times],
            width=.25, bottom=mintrc, color="blue", linewidth=0, alpha=.5)

    for arrtime, phname in zip(all_phase_times, all_phase_names):
      plt.text(arrtime, 0, phname)

    #st.plot(color='k')


def fetch_array_elements(siteid):
  cursor = database.db.connect().cursor()
  # extract the station name and the reference station
  cursor.execute("select s1.sta, s2.refsta, s2.staname from static_siteid s1, "
                 "static_site s2 where s1.id=%d and "
                 "s2.sta=s1.sta and s2.offdate=-1" % (siteid+1,))
  sta, refsta, staname = cursor.fetchone()
  # deduce the channel with the most number of array elements
  #  sql_query = "select c.chan, count(*) from static_sitechan c, static_site s where c.offdate=-1 and s.offdate=-1 and c.sta=s.sta and s.statype='ss' and s.refsta='%s' and c.vang=0 and c.hang=-1 group by 1 order by 2 desc limit 1" % (refsta,)
  sql_query = "select * from (select c.chan, count(*) cnt from static_sitechan c, static_site s where c.offdate=-1 and s.offdate=-1 and c.sta=s.sta and s.statype='ss' and s.refsta='%s' and c.vang=0 and c.hang=-1 group by c.chan order by cnt desc) where rownum <= 1" % (refsta,)
  print sql_query
  cursor.execute(sql_query)
  chan,chan_cnt = cursor.fetchone()
  # now, return the name and locations of all these array elements
  cursor.execute("select s.sta, s.lon, s.lat, s.elev-c.edepth from "
                 "static_site s, static_sitechan c "
                 "where s.refsta='%s' and s.offdate=-1 and s.statype='ss' "
                 "and c.sta=s.sta and c.offdate=-1 and c.vang=0 and c.hang=-1 "
                 "and c.chan='%s'" % (refsta, chan))
  arrsta = np.array(cursor.fetchall())
  print "Station: %s, Refsta: %s, Chan: %s, %d array elements" \
        % (sta, refsta, chan, len(arrsta))

  return sta, chan, arrsta

def plot_fk_arr(siteid, start_time, end_time):
  sta, chan, arrsta = fetch_array_elements(siteid)

  # query waveforms from all the array elements
  trcs = []
  for arridx in range(len(arrsta)):
    try:
      trc = fetch_waveform(arrsta[arridx, 0], chan, start_time, end_time)
    except MissingWaveform:
      print "Warning: station %s, chan %s missing" % (arrsta[arridx, 0], chan)
      continue

    trc.stats.network, trc.stats.station = sta, arrsta[arridx, 0]
    trc.stats.coordinates = {"longitude": float(arrsta[arridx, 1]),
                             "latitude": float(arrsta[arridx, 2]),
                             "elevation": float(arrsta[arridx, 3])}
    trcs.append(trc)

  if len(trcs) == 0:
    raise MissingWaveform("Can't find data for array sta %s"
                            % (sta))
  st = Stream(trcs)

  # perform F-K analysis to compute slowness and backazimuth
  kwargs = dict(
    # slowness grid: X min, X max, Y min, Y max, Slow Step
    sll_x=-3.0, slm_x=3.0, sll_y=-3.0, slm_y=3.0, sl_s=0.1, # .03 was too slow!
    # sliding window propertieds
    win_len=5, win_frac=.1,
    # frequency properties
    frqlow=1.0, frqhigh=2.0, prewhiten=1,
    # restrict output
    semb_thres=-1e9, vel_thres=-1e9, verbose=False, timestamp='mlabhour',
    stime=trcs[0].stats.starttime, etime=trcs[0].stats.endtime
    )
  out = sonic(st, **kwargs)

  labels = 'rel.power abs.power baz slow'.split()

  fig = plt.figure()
  plt.suptitle(sta)
  for i, lab in enumerate(labels):
    ax = fig.add_subplot(4,1,i+1)
    ax.scatter(out[:,0], out[:,i+1], c=out[:,1], alpha=0.6, edgecolors='none')
    ax.set_ylabel(lab)
    ax.xaxis_date()

  fig.autofmt_xdate()
  fig.subplots_adjust(top=0.95, right=0.95, bottom=0.2, hspace=0)


  return


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


def dict_cursor(cursor):
  description = [x[0] for x in cursor.description]
  for row in cursor:
    yield dict(zip(description, row))


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

    sql = "select * from idcx_wfdisc where sta = '%s' and chan ='%s' and time <= %f and %f < endtime" % (station, chan, stime, stime)
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

    # at which offset should we start collecting samples
    first_off = int((stime-waveform['time'])*samprate)
    # how many samples are needed remaining
    desired_samples = int(round((etime - stime) * samprate))
    # how many samples are actually available
    available_samples = waveform['nsamp'] - first_off
    # grab the available and needed samples
    try:
      segment = _read_waveform_from_file(waveform, first_off,
                                         min(desired_samples, available_samples))
    except IOError:
      raise MissingWaveform("Can't find data for sta %s chan %s time %d"
                            % (station, chan, stime))

    data.extend(segment)

    # do we have all the data that we need
    if desired_samples <= available_samples:
      break

    # otherwise move the start time forward for the next file
    stime = waveform['endtime']
    # and adust the end time to ensure that the correct number of samples
    # will be selected in the next file
    etime = stime + (desired_samples - available_samples) / float(samprate)

  stats = {'network': station, 'station': station, 'location': '',
           'channel': chan, 'npts': len(data), 'sampling_rate': samprate,
           'mseed' : {'dataquality' : 'D'}}
  stats['starttime_unix'] = stime
  stats['siteid'] = int(siteid)
  stats['chanid'] = int(waveform['chanid'])
  stats['starttime'] = UTCDateTime(stime)
  return Trace(data=np.array(data), header=stats)

  #return samprate, np.array(data)




def plot_focused_arr(earthmodel, event, siteid, window):
  sta, chan, arrsta = fetch_array_elements(siteid)

  # for each array element fetch the waveform in a window around the
  # arrival time at that station
  trcs = []
  for arridx in range(len(arrsta)):
    arrtime = earthmodel.ArrivalTimeCoord(event[ EV_LON_COL],
                                          event[ EV_LAT_COL],
                                          event[ EV_DEPTH_COL],
                                          event[ EV_TIME_COL],
                                          0, # P-phase,
                                          float(arrsta[arridx, 1]), # site lon
                                          float(arrsta[arridx, 2]), # site lat
                                          float(arrsta[arridx, 3])) # site elev
    try:
      trc = fetch_waveform(arrsta[arridx, 0], chan, arrtime-window,
                           arrtime + window)
    except MissingWaveform:
      print "Warning: station %s, chan %s missing" % (arrsta[arridx, 0], chan)
      continue

    trcs.append(trc)

  # now sum up all the waveforms
  sumdata = sum([x.data for x in trcs])
  env = sig_filter.envelope(sumdata)

  plt.figure()
  plt.xlabel("Time (s)")
  plt.ylabel(sta + " - " + chan)
  timevals = np.arange(0, trcs[0].stats.npts/trcs[0].stats.sampling_rate,
                       1.0 /trcs[0].stats.sampling_rate)
  plt.plot(timevals, sumdata, 'k')
  plt.plot(timevals, env, 'k:')


def main(param_dirname):


  parser = OptionParser()

  parser.add_option("-s", "--siteids", dest="siteids", default="", type="str")
  parser.add_option("-e", "--evid", dest="evid", default=None, type="int")
  parser.add_option("--orid", dest="orid", default=None, type="int")
  parser.add_option("-o", "--outfile", dest="outfile", default=None, type="str")

  (options, args) = parser.parse_args()
  cursor, sigmodel, earthmodel, sites, dbconn = sigvisa_util.init_sigmodel()

  if options.orid is not None:
    cursor.execute("select evid from leb_origin where orid=%d" % options.orid)
    evid = cursor.fetchone()[0]
    print "using evid %d for orid %d" % (evid, options.orid)
  else:
      evid = options.evid
  siteids = [int(x) for x in options.siteids.split(',')]

  if options.outfile is None:
    outfile = "logs/%d_detections.pdf" % (evid)
  else:
    outfile=options.outfile
  pp = PdfPages(outfile)
  print "saving to %s..." % (outfile)



  cursor.execute("select sta from static_siteid site order by id")
  sitenames = np.array(cursor.fetchall())[:,0]


  event = load_event(cursor, evid)
  print "Ev Mag %.1f Ev Time %.1f" % (event[ EV_MB_COL],
                                      event[ EV_TIME_COL])

  try:
    for siteid in siteids:
      (arrival_segment, noise_segment, all_arrivals, all_arrival_phases, all_arrival_arids) = load_signal_slice(cursor, evid, siteid, bands=["narrow_envelope_2.00_3.00"], earthmodel=earthmodel)

      if len(arrival_segment[0]) == 0:
        print "no signal at siteid %d"
        continue

      try:
        plot_segment(arrival_segment[0], title = sitenames[siteid-1], all_det_times = all_arrivals, all_det_labels=all_arrival_phases, band="narrow_envelope_2.00_3.00", chans=["BHZ", "BHE", "BHN"], logscale=False)
        pp.savefig()
        plot_segment(arrival_segment[0], title = sitenames[siteid-1] + " (log)", all_det_times = all_arrivals, all_det_labels=all_arrival_phases, band="narrow_envelope_2.00_3.00", chans=["BHZ", "BHE", "BHN"], logscale=True)
        pp.savefig()
      except KeyboardInterrupt:
        raise
      except:
        import traceback, pdb
        traceback.print_exc()
        continue
  finally:
    pp.close()

#  plt.show()

if __name__ == "__main__":
  try:
    main("parameters")
  except SystemExit:
    raise
  except:
    import pdb, traceback, sys
    traceback.print_exc(file=sys.stdout)
    pdb.post_mortem(sys.exc_traceback)
    raise


