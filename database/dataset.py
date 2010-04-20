# python imports
import numpy as np
from time import strftime, gmtime
from math import ceil
import os

# local imports
import db
from az_slow_corr import load_az_slow_corr

# events
EV_LON_COL, EV_LAT_COL, EV_DEPTH_COL, EV_TIME_COL, EV_MB_COL, EV_ORID_COL,\
            EV_NUM_COLS = range(6+1)

# detections
DET_SITE_COL, DET_ARID_COL, DET_TIME_COL, DET_DELTIM_COL, DET_AZI_COL,\
              DET_DELAZ_COL, DET_SLO_COL, DET_DELSLO_COL, DET_SNR_COL,\
              DET_PHASE_COL, DET_AMP_COL, DET_PER_COL,\
              DET_NUM_COLS = range(12+1)

# sites
SITE_LON_COL, SITE_LAT_COL, SITE_ELEV_COL, SITE_IS_ARRAY, \
              SITE_NUM_COLS = range(4+1)


MIN_MAGNITUDE = 2.0

UPTIME_QUANT = 3600                     # 1 hour

MAX_TRAVEL_TIME = 2000.0

def read_timerange(cursor, label, hours, skip):
  # determine the start and end time for the specified label
  cursor.execute("select start_time, end_time from dataset where label=%s",
                 (label,))
  row = cursor.fetchone()
  if row is None:
    raise ValueError("Unknown label %s" % label)
  start_time, end_time = row
  
  # compute the subset of time within the dataset to actually read
  if skip >= start_time and skip <= end_time:
    stime = skip
  else:
    stime = start_time + skip * 60 * 60
    
  if hours is None:
    etime = end_time
  elif hours >= start_time and hours <= end_time:
    etime = hours
  else:
    etime = stime + hours * 60 * 60
  
  if stime > end_time or stime < start_time or etime < start_time\
         or etime > end_time:
    raise ValueError("invalid skip %d and hours %d parameter"
                     % (skip, hours))

  return stime, etime
  
def read_events(cursor, start_time, end_time, evtype, runid=None):
  if runid is None:
    cursor.execute("select lon, lat, depth, time, mb, orid from %s_origin "
                   "where time between %d and %d order by time"
                   % (evtype, start_time, end_time))
  else:
    cursor.execute("select lon, lat, depth, time, mb, orid from %s_origin "
                   "where runid=%s and time between %d and %d order by time"
                   % (evtype, runid, start_time, end_time))
    
  events = np.array(cursor.fetchall())

  # change -999 mb to MIN MAG
  events[:, EV_MB_COL][events[:, EV_MB_COL] == -999] = MIN_MAGNITUDE

  orid2num = {}
  
  for ev in events:
    orid2num[ev[EV_ORID_COL]] = len(orid2num)
  
  return events, orid2num

def read_detections(cursor, start_time, end_time):
  cursor.execute("select site.id-1, iarr.arid, iarr.time, iarr.deltim, "
                 "iarr.azimuth, iarr.delaz, iarr.slow, iarr.delslo, iarr.snr, "
                 "ph.id-1, iarr.amp, iarr.per from idcx_arrival_net iarr, "
                 "static_siteid site, static_phaseid ph where "
                 "iarr.sta=site.sta and iarr.iphase=ph.phase and "
                 "iarr.time between %s and %s order by iarr.time, iarr.arid",
                 (start_time, end_time))
  
  detections = np.array(cursor.fetchall())

  cursor.execute("select sta from static_siteid site order by id")
  sitenames = np.array(cursor.fetchall())[:,0]
  corr_dict = load_az_slow_corr(os.path.join('parameters', 'sasc'))
  #print len(corr_dict), "SASC corrections loaded"
  
  arid2num = {}
  
  for det in detections:
    arid2num[det[DET_ARID_COL]] = len(arid2num)

    # apply SASC correction
    (det[DET_AZI_COL], det[DET_SLO_COL], det[DET_DELAZ_COL],
     det[DET_DELSLO_COL]) = corr_dict[sitenames[det[DET_SITE_COL]]].correct(
      det[DET_AZI_COL], det[DET_SLO_COL], det[DET_DELAZ_COL],
      det[DET_DELSLO_COL])

  return detections, arid2num

def read_assoc(cursor, start_time, end_time, orid2num, arid2num, evtype):
  cursor.execute("select lass.orid, lass.arid, ph.id-1 from %s_assoc lass, "
                 "%s_origin lori, static_phaseid ph where "
                 "ph.timedef='d' and "
                 "lass.orid=lori.orid and lass.phase=ph.phase and lori.time "
                 "between %s and %s" % (evtype, evtype, start_time, end_time))
  
  evlist = [[] for _ in range(len(orid2num))]
  for orid, arid, phaseid in cursor:
    evnum = orid2num[orid]
    if arid in arid2num:
      detnum = arid2num[arid]
      evlist[evnum].append((int(phaseid), int(detnum)))
        
  return evlist

def read_uptime(cursor, start_time, end_time):
  cursor.execute("select count(*) from static_siteid")
  numsites, = cursor.fetchone()
  
  uptime = np.zeros((numsites,
                     int(ceil((end_time - start_time) / UPTIME_QUANT))),
                    bool)
  
  cursor.execute("select site.id-1, truncate((arr.time-%s)/3600, 0), count(*) "
                 "from idcx_arrival arr, static_siteid site "
                 "where arr.sta = site.sta and "
                 "arr.time between %s and %s group by 1, 2",
                 (start_time, start_time, end_time))
  
  for (siteidx, timeidx, cnt) in cursor.fetchall():
    uptime[siteidx, timeidx] = True
  
  return uptime

def read_sites(cursor):
  cursor.execute("select lon, lat, elev, statype='ar' from static_siteid "
                 "order by id")
  return np.array(cursor.fetchall())

def read_phases(cursor):
  cursor.execute("select phase from static_phaseid "
                 "order by id")
  phasenames = np.array(cursor.fetchall())[:,0]

  cursor.execute("select timedef='d' from static_phaseid "
                 "order by id")
  phasetimedef = np.array(cursor.fetchall())[:,0].astype(bool)

  return phasenames, phasetimedef

def read_data(label="training", hours=None, skip=0, verbose=1):
  """
  Reads
  - LEB events/assoc,
  - SEL3 events/assoc,
  - IDCX arrivals with delaz>0 and delsnr>0
  - Site information
  - Phase information


  The data is divided into the following labels:
  
  - validation
  - test
  - training
  
  Setting 'hours' to None will return all the data from the specified
  dataset. 'skip' controls the number of initial hours to skip, for example
  for testing the second 2 hour window set hours=2 skip=2

  Returns the following tuple
         start_time, end_time, detections, leb_events, leb_evlist, sel3_events,
         sel3_evlist, site_up, sites, phasenames, phasetimedef

  where:

  events:
    o. lon, lat, depth
    o. time
    o. mb
    o. orid

  evlist maps evnum -> list of detection numbers

  detections:
    sitenum
    arid
    time
    deltim
    azimuth
    delaz
    slow
    delslo
    snr
    phasenum
    amp
    period
    
  sites:
    o. lon, lat, elev, is_array?
  """
  cursor = db.connect().cursor()
  start_time, end_time = read_timerange(cursor, label, hours, skip)

  if verbose:
    print "Dataset: %.1f hrs from %d" % ((end_time-start_time)/3600,
                                         start_time),
    print "i.e. %s to %s" % (strftime("%x %X", gmtime(start_time)),
                             strftime("%x %X", gmtime(end_time)))

  if verbose:
    print "Reading detections...",
    
  det, arid2num = read_detections(cursor, start_time, end_time)

  if verbose:
    print "done (%d detections)" % len(det)
    print "Reading LEB events and associations...",
  
  leb_events, leb_orid2num = read_events(cursor, start_time, end_time, "leb")
  leb_evlist = read_assoc(cursor, start_time, end_time, leb_orid2num,
                          arid2num, "leb")

  if verbose:
    print "done (%d events %d with >=3 det %d with >=3 P det)" \
          % (len(leb_events), sum(len(e)>=3 for e in leb_evlist),
             sum(sum(p==0 for p,dnum in alist)>=3 for alist in leb_evlist))
    print "Reading SEL3 events and associations...",
    
  sel3_events, sel3_orid2num = read_events(cursor, start_time, end_time,"sel3")
  sel3_evlist = read_assoc(cursor, start_time, end_time, sel3_orid2num,
                           arid2num, "sel3")

  if verbose:
    print "done (%d events %d with >=3 det %d with >=3 P det)" \
          % (len(sel3_events), sum(len(e)>=3 for e in sel3_evlist),
             sum(sum(p==0 for p,dnum in alist)>=3 for alist in sel3_evlist))
    print "Reading Uptime, site and phase info...",
    
  site_up = read_uptime(cursor, start_time, end_time)
  
  sites = read_sites(cursor)
  
  phasenames, phasetimedef = read_phases(cursor)

  assert(len(phasenames) == len(phasetimedef))
  
  if verbose:
    print "done (%d sites %d array, %d phases %d timedef)" \
          % (len(sites), sites[:,SITE_IS_ARRAY].sum(), len(phasenames),
             sum(phasetimedef))
    
  return start_time, end_time, det, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef
