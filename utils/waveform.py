# draw the waveforms for an LEB event at some of the stations

import sys, MySQLdb, struct
import matplotlib.pyplot as plt
import gzip

from obspy.core import Trace, Stream, UTCDateTime
from obspy.signal import filter as sig_filter
from obspy.signal.array_analysis import sonic

from database.dataset import *
import database.db
import learn, netvisa
from utils.geog import dist_deg, azimuth

class MissingWaveform(Exception):
  pass

def plot_ss_waveforms(siteid, start_time, end_time, detections, earthmodel,
                      event):
  cursor = database.db.connect().cursor()
  cursor.execute("select sta from static_siteid where id=%d" % (siteid+1,))
  sta, = cursor.fetchone()
  print "Station:", sta

  cursor.execute("select distinct chan from idcx_wfdisc where sta='%s'" % sta)
  chans = [x for x, in cursor.fetchall()]
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
    trc = fetch_waveform(sta, chan, start_time, end_time)
    trc.filter('bandpass', freqmin=1.0, freqmax=2.0, corners=2,
               zerophase=True)
    env = sig_filter.envelope(trc.data)

    
    timevals = np.arange(0, trc.stats.npts/trc.stats.sampling_rate,
                         1.0 /trc.stats.sampling_rate)
    plt.plot(timevals, trc.data, 'k')
    plt.plot(timevals, env, 'k:')

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
  cursor.execute("select c.chan, count(*) from static_sitechan c, "
                 "static_site s where c.offdate=-1 and s.offdate=-1 and "
                 "c.sta=s.sta and s.statype='ss' and s.refsta='%s' and "
                 "c.vang=0 and c.hang=-1 group by 1 order by 2 desc limit 1" %
                 (refsta,))
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

  stats = {'network': station, 'station': station, 'location': '',
           'channel': chan, 'npts': len(data), 'sampling_rate': samprate,
           'mseed' : {'dataquality' : 'D'}}
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
  if len(sys.argv) not in [3,4] or (len(sys.argv)==3 and sys.argv[1] != "leb")\
         or (len(sys.argv)==4 and sys.argv[1] != "visa"):
    print "Usage: python waveform.py leb leb-orid | visa run-id orid"
    sys.exit(1)

  evtype = sys.argv[1]
  if evtype == "leb":
    leb_orid = int(sys.argv[2])
  else:
    visa_runid = int(sys.argv[2])
    visa_orid= int(sys.argv[3])
  
  # load the event
  cursor = database.db.connect().cursor()
  if evtype == "leb":
    cursor.execute("select time from leb_origin where orid=%d" % (leb_orid,))
  else:
    cursor.execute("select time from visa_origin where runid=%d and orid=%d" %
                   (visa_runid, visa_orid))
  fetch = cursor.fetchone()
  if fetch is None:
    print "Event not found"
    sys.exit(2)
  evtime, = fetch
  
  # start a dataset from 5 minutes before the event to 1 hour after
  start_time, end_time = evtime - 5 * 60, evtime + 60*60
  
  detections, arid2num = read_detections(cursor, start_time, end_time)
  if evtype == "leb":
    leb_events, leb_orid2num = read_events(cursor, start_time, end_time, "leb")
    leb_evlist = read_assoc(cursor, start_time, end_time, leb_orid2num,
                            arid2num, "leb")
    evnum = leb_orid2num[leb_orid]
    event = leb_events[evnum]
    event_detlist = leb_evlist[evnum]
  else:
    visa_events, visa_orid2num = read_events(cursor, start_time, end_time,
                                             "visa", runid = visa_runid)
    visa_evlist = read_assoc(cursor, start_time, end_time, visa_orid2num,
                             arid2num, "visa", runid = visa_runid)
    evnum = visa_orid2num[visa_orid]
    event = visa_events[evnum]
    event_detlist = visa_evlist[evnum]
    
  sites = read_sites(cursor)
  site_up = read_uptime(cursor, start_time, end_time)
  phasenames, phasetimedef = read_phases(cursor)
  cursor.execute("select sta from static_siteid site order by id")
  sitenames = np.array(cursor.fetchall())[:,0]
  
  # load the earthmodel and the NET-VISA model
  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  print "Ev Mag %.1f Ev Time %.1f" % (event[ EV_MB_COL],
                                      event[ EV_TIME_COL])
  
  # find the travel-time of the event to all the stations that are up
  all_site_arrtimes = []
  for siteid, site in enumerate(sites):
    arrtime = earthmodel.ArrivalTime(event[ EV_LON_COL],
                                     event[ EV_LAT_COL],
                                     event[ EV_DEPTH_COL],
                                     event[ EV_TIME_COL],
                                     0, # P-phase
                                     siteid)

    # is the site up at the time the P wave is expected to arrive?
    if arrtime > 0 \
           and site_up[siteid, int((arrtime-start_time) // UPTIME_QUANT)]:
      # find this site in the detections of the event
      isdet = False
      for phaseid, detid in event_detlist:
        if phaseid == 0 and int(detections[detid, DET_SITE_COL]) == siteid:
          isdet = True
          break
        
      all_site_arrtimes.append((arrtime, siteid, isdet))

  all_site_arrtimes.sort()

  print "Detected At:"
  last_det_idx = -1
  for idx, (arrtime, siteid, isdet) in enumerate(all_site_arrtimes):
    
    if sites[siteid, SITE_IS_ARRAY]:
      site_type = "ARR"
    else:
      site_type = "3-C"
    
    if isdet:
      if idx > (last_det_idx+1):
        print "..... missed by %d sites" % (idx - last_det_idx - 1)
      print "ttime %d, %s (%s, siteid %d) seaz %d esaz %d" \
            % (arrtime - event[ EV_TIME_COL],
               sitenames[siteid], site_type, siteid,
               azimuth(sites[siteid, [SITE_LON_COL, SITE_LAT_COL]],
                       event[ [EV_LON_COL, EV_LAT_COL]]),
               azimuth(event[ [EV_LON_COL, EV_LAT_COL]],
                       sites[siteid, [SITE_LON_COL, SITE_LAT_COL]]))
      last_det_idx = idx

  if idx > last_det_idx:
    print "..... missed by %d sites" % (idx - last_det_idx)

  print "Total sites which were up:", len(all_site_arrtimes)
  # for the first site which detected the event draw the waveform

  # plot the waveforms at all the sites where it was detected
  for arrtime, siteid, isdet in all_site_arrtimes:
    if isdet:
      if not sites[siteid, SITE_IS_ARRAY]:
        plot_ss_waveforms(siteid, arrtime-20, arrtime+20, detections,
                          earthmodel, event)
      else:
        plot_fk_arr(siteid, arrtime-20, arrtime+20)
        #plot_focused_arr(earthmodel, event, siteid, 20)
        
  plt.show()
  
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

  
