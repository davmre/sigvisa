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
import numpy as np
from time import strftime, gmtime
from math import ceil
import os

# local imports
import db
from az_slow_corr import load_az_slow_corr

# events

EV_LON_COL, EV_LAT_COL, EV_DEPTH_COL, EV_TIME_COL, EV_MB_COL, EV_ORID_COL, EV_EVID_COL, \
    EV_NUM_COLS = range(7+1)

# detections
DET_SITE_COL, DET_ARID_COL, DET_TIME_COL, DET_DELTIM_COL, DET_AZI_COL,\
              DET_DELAZ_COL, DET_SLO_COL, DET_DELSLO_COL, DET_SNR_COL,\
              DET_PHASE_COL, DET_AMP_COL, DET_PER_COL,\
              DET_NUM_COLS = range(12+1)

# sites
SITE_LON_COL, SITE_LAT_COL, SITE_ELEV_COL, SITE_IS_ARRAY, \
              SITE_NUM_COLS = range(4+1)


MIN_MAGNITUDE = 2.0

MAX_MAGNITUDE = 8.0

MIN_LOGAMP = -25.0
MAX_LOGAMP = +15.0
STEP_LOGAMP = 0.1

UPTIME_QUANT = 3600                     # 1 hour

MAX_TRAVEL_TIME = 2000.0

MAX_DEPTH = 700.0

AVG_EARTH_RADIUS_KM = 6371.0            # when modeled as a sphere

def sql_multi(key, values):
    return "(" + " or ".join(["%s=%s" % (key, val) for val in values])  + ")"

def sql_multi_str(key, values):
    return "(" + " or ".join(["%s='%s'" % (key, val) for val in values])  + ")"

def read_timerange(cursor, label, hours, skip):
  # determine the start and end time for the specified label
  cursor.execute("select start_time, end_time from dataset where "
                 "label='%s'" % label)
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

def read_event(cursor, evid, evtype="leb", runid=None):
  if runid is None:
    cursor.execute("select lon, lat, depth, time, mb, orid, evid from %s_origin "
                   "where evid=%d" % (evtype, int(evid)))
  else:
    cursor.execute("select lon, lat, depth, time, mb, orid, evid from %s_origin "
                   "where evid=%d" % (evtype, int(evid)))
  event = np.array(cursor.fetchone())
  return event

def read_events(cursor, start_time, end_time, evtype, runid=None):
  if runid is None:
    cursor.execute("select lon, lat, depth, time, mb, orid, evid from %s_origin "
                   "where time between %d and %d order by time"
                   % (evtype, start_time, end_time))
  else:
    cursor.execute("select lon, lat, depth, time, mb, orid, evid from %s_origin "
                   "where runid=%s and time between %d and %d order by time"
                   % (evtype, runid, start_time, end_time))

  events = np.array(cursor.fetchall())

  # change -999 mb to MIN MAG
  if len(events):
    events[:, EV_MB_COL][events[:, EV_MB_COL] == -999] = MIN_MAGNITUDE

  orid2num = {}

  for ev in events:
    orid2num[ev[EV_ORID_COL]] = len(orid2num)

  return events, orid2num

def read_isc_events(cursor, start_time, end_time, author):
  if author is None:
    # no point getting IDC events from ISC bulletin, we already have these
    cursor.execute("select lon, lat, depth, time, mb, eventid from "
                   "isc_events where time between %d and %d and author != 'IDC'"
                   " order by time"
                   % (start_time, end_time))
  else:
    cursor.execute("select lon, lat, depth, time, mb, eventid from "
                   "isc_events where author='%s' and time between %d "
                   "and %d order by time"% (author, start_time, end_time))

  events = np.array(cursor.fetchall())

  # change -999 mb to MIN MAG and -999 depth to 0
  if len(events):
    events[:, EV_MB_COL][events[:, EV_MB_COL] == -999] = MIN_MAGNITUDE
    events[:, EV_DEPTH_COL][events[:, EV_DEPTH_COL] == -999] = 0

  return events


def read_detections(cursor, start_time, end_time,arrival_table="idcx_arrival", noarrays=False):

  print "reading detections... "

  sql_query = "select site.id-1, iarr.arid, iarr.time, iarr.deltim, iarr.azimuth, iarr.delaz, iarr.slow, iarr.delslo, iarr.snr, ph.id-1, iarr.amp, iarr.per from %s iarr, static_siteid site, static_phaseid ph where %s iarr.delaz > 0 and iarr.delslo > 0 and iarr.snr > 0 and iarr.sta=site.sta and iarr.iphase=ph.phase and ascii(iarr.iphase) = ascii(ph.phase) and iarr.time between %d and %d order by iarr.time, iarr.arid" %  (arrival_table, ("site.statype='ss' and " if noarrays else ""), start_time, end_time)

  cursor.execute(sql_query)

  detections = np.array(cursor.fetchall())

  sql_query = "select sta from static_siteid site order by id"
  cursor.execute(sql_query)
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


def read_event(cursor, evid, evtype="leb"):
    sql_query = "SELECT lon, lat, depth, time, mb, orid, evid from %s_origin where evid=%d" % (evtype, evid)
    cursor.execute(sql_query)
    return np.array(cursor.fetchone())

def read_event_detections(cursor, evid, stations=None, evtype="leb"):

  if stations is not None:
    s = str(stations)
    stalist_str = s.replace('[', '(').replace(']', ')').replace(',)', ')')
    sta_cmd = "site.sta in " + stalist_str
  else:
    sta_cmd = "site.statype='ss'"

  sql_query = "select site.id-1, iarr.arid, iarr.time, iarr.deltim, iarr.azimuth, iarr.delaz, iarr.slow, iarr.delslo, iarr.snr, ph.id-1, iarr.amp, iarr.per from %s_origin ior, %s_assoc iass, %s_arrival iarr, static_siteid site, static_phaseid ph where ior.evid=%d and iass.orid=ior.orid and iarr.arid=iass.arid and iarr.delaz > 0 and iarr.delslo > 0 and iarr.snr > 0 and iarr.sta=site.sta and iarr.iphase=ph.phase and ascii(iarr.iphase) = ascii(ph.phase) and %s order by iarr.time, iarr.arid" %  (evtype, evtype, evtype, int(evid), sta_cmd)

  cursor.execute(sql_query)
  detections = np.array(cursor.fetchall())

  return np.array(detections)

def read_station_detections(cursor, sta, start_time, end_time,arrival_table="idcx_arrival"):

  print "reading detections... "

  sql_query = "select site.id-1, iarr.arid, iarr.time, iarr.deltim, iarr.azimuth, iarr.delaz, iarr.slow, iarr.delslo, iarr.snr, ph.id-1, iarr.amp, iarr.per from %s iarr, static_siteid site, static_phaseid ph where site.sta='%f' iarr.delaz > 0 and iarr.delslo > 0 and iarr.snr > 0 and iarr.sta=site.sta and iarr.iphase=ph.phase and ascii(iarr.iphase) = ascii(ph.phase) and iarr.time between %d and %d order by iarr.time, iarr.arid" %  (arrival_table, sta, start_time, end_time)

  cursor.execute(sql_query)

  detections = np.array(cursor.fetchall())

  sql_query = "select sta from static_siteid site order by id"
  cursor.execute(sql_query)
  sitenames = np.array(cursor.fetchall())[:,0]
  corr_dict = load_az_slow_corr(os.path.join('parameters', 'sasc'))
  #print len(corr_dict), "SASC corrections loaded"

  for det in detections:

    # apply SASC correction
    (det[DET_AZI_COL], det[DET_SLO_COL], det[DET_DELAZ_COL],
     det[DET_DELSLO_COL]) = corr_dict[sitenames[det[DET_SITE_COL]]].correct(
      det[DET_AZI_COL], det[DET_SLO_COL], det[DET_DELAZ_COL],
      det[DET_DELSLO_COL])

  return detections


def read_assoc(cursor, start_time, end_time, orid2num, arid2num, evtype,
               runid=None):
  if evtype == "visa":
    cursor.execute("select vass.orid, vass.arid, ph.id-1 from visa_assoc vass,"
                   "visa_origin vori, static_phaseid ph where "
                   "ph.timedef='d' and "
                   "vass.orid=vori.orid and vass.phase=ph.phase and vori.time "
                   "between %f and %f and vass.runid=vori.runid "
                   "and vass.runid=%d" % (start_time, end_time, runid))
  else:
    cursor.execute("select lass.orid, lass.arid, ph.id-1 from %s_assoc lass, "
                   "%s_origin lori, static_phaseid ph where "
                   "ph.timedef='d' and "
                   " ascii(lass.phase) = ascii(ph.phase) and "
                   "lass.orid=lori.orid and lass.phase=ph.phase and lori.time "
                   "between %f and %f"
                   % (evtype, evtype, start_time, end_time))

  evlist = [[] for _ in range(len(orid2num))]
  for orid, arid, phaseid in cursor:
    if orid not in orid2num:
      continue
    evnum = orid2num[orid]
    if arid in arid2num:
      detnum = arid2num[arid]
      evlist[evnum].append((int(phaseid), int(detnum)))

  return evlist

def read_uptime(cursor, start_time, end_time, arrival_table="idcx_arrival"):
  cursor.execute("select count(*) from static_siteid")
  numsites, = cursor.fetchone()

  uptime = np.zeros((numsites,
          int(ceil((MAX_TRAVEL_TIME + end_time - start_time) / UPTIME_QUANT))),
                    bool)

  cursor.execute("select snum, hnum, count(*) from "
                 "(select site.id-1 snum,trunc((arr.time-%d)/%d, 0) hnum "
                 "from %s arr, static_siteid site "
                 "where arr.sta = site.sta and "
                 "arr.time between %d and %d) sitearr group by snum, hnum" %
                 (start_time, UPTIME_QUANT, arrival_table, start_time,
                  end_time + MAX_TRAVEL_TIME))

  for (siteidx, timeidx, cnt) in cursor.fetchall():
    uptime[siteidx, timeidx] = True

  return uptime

def read_sites_by_name(cursor):
  sites = read_sites(cursor)
  cursor.execute("select sta from static_siteid order by id")
  names = [r[0] for r in cursor.fetchall()]


  # returns stations, name_to_siteid_minus1, siteid_minus1_to_name
  return dict(zip(names, sites)), dict(zip(names, range(len(names)))), names


def read_sites(cursor):
  cursor.execute("select lon, lat, elev, "
                 "(case statype when 'ar' then 1 else 0 end) "
                 "from static_siteid "
                 "order by id")
  return np.array(cursor.fetchall())

def read_phases(cursor):
  cursor.execute("select phase from static_phaseid "
                 "order by id")
  phasenames = np.array(cursor.fetchall())[:,0]

  cursor.execute("select (case timedef when 'd' then 1 else 0 end) "
                 "from static_phaseid "
                 "order by id")
  phasetimedef = np.array(cursor.fetchall())[:,0].astype(bool)

  return phasenames, phasetimedef

def read_data(label="training", hours=None, skip=0, verbose=1,
              visa_leb_runid=None, read_leb_detections=False, noarrays=False):
  """
  Reads
  - LEB events/assoc, with IDCX arrivals
  - SEL3 events/assoc,
  - IDCX arrivals with delaz>0 and delsnr>0 (upto 2000 secs beyond end_time)
  - Site information
  - Phase information
  - LEB assoc with LEB arrivals
  - LEB arrivals


  The data is divided into the following labels:

  - validation
  - test
  - training

  Setting 'hours' to None will return all the data from the specified
  dataset. 'skip' controls the number of initial hours to skip, for
  example for testing the second 2 hour window set hours=2
  skip=2. 'noarrays' prohibits using data from array stations.

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

  visa_leb_runid is a special parameter which says that a certain visa run
  is to be treated as leb
  """
  cursor = db.connect().cursor()
  start_time, end_time = read_timerange(cursor, label, hours, skip)

  if verbose:
    print "Dataset: %.1f hrs from %d to %d " % ((end_time-start_time)/3600.,
                                                start_time, end_time),
    print "i.e. %s to %s" % (strftime("%x %X", gmtime(start_time)),
                             strftime("%x %X", gmtime(end_time)))

  if verbose:
    print "Reading SEL3 detections...",

  det, arid2num = read_detections(cursor, start_time, end_time, noarrays=noarrays)

  if verbose:
    print "done (%d detections)" % len(det)

  if read_leb_detections:
    leb_det, leb_arid2num = read_detections(cursor, start_time, end_time,
                                            "leb_arrival", noarrays)
    if verbose:
      print "done (%d detections)" % len(leb_det)

  if verbose:
    print "Reading LEB events and associations...",

  if visa_leb_runid is None:
    leb_events, leb_orid2num = read_events(cursor, start_time, end_time, "leb")
    leb_evlist = read_assoc(cursor, start_time, end_time, leb_orid2num,
                            arid2num, "leb")
    if read_leb_detections:
      leb_leb_evlist = read_assoc(cursor, start_time, end_time, leb_orid2num,
                                  leb_arid2num, "leb")
  else:
    leb_events, leb_orid2num = read_events(cursor, start_time, end_time,"visa",
                                           visa_leb_runid)
    leb_evlist = read_assoc(cursor, start_time, end_time, leb_orid2num,
                            arid2num, "visa", visa_leb_runid)

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

  if read_leb_detections:
    return start_time, end_time, det, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef,\
         leb_det, leb_leb_evlist
  else:
    return start_time, end_time, det, leb_events, leb_evlist, sel3_events, \
           sel3_evlist, site_up, sites, phasenames, phasetimedef, arid2num

def compute_arid2num(detections):
  return dict((det[DET_ARID_COL], detnum) for detnum, det
              in enumerate(detections))

def compute_orid2num(events):
  return dict((event[EV_ORID_COL], evnum) for evnum, event
              in enumerate(events))

def read_sitenames():
  cursor = db.connect().cursor()
  cursor.execute("select sta from static_siteid site order by id")
  sitenames = np.array(cursor.fetchall())[:,0]
  return sitenames
