#checks whether invert detection works correctly

import os, sys, time
import numpy as np
from optparse import OptionParser

from database.dataset import *
from database.db import connect
import netvisa, learn
from results.compare import *
from utils.geog import dist_km, dist_deg

def fixup_event_lon(event):
  if event[EV_LON_COL] < - 180:
    event[EV_LON_COL] += 360
  elif event[EV_LON_COL] >= 180:
    event[EV_LON_COL] -= 360

def fixup_event_lat(event):
  if event[EV_LAT_COL] < -90:
    event[EV_LAT_COL] = -180 - event[EV_LAT_COL]
  elif event[EV_LAT_COL] > 90:
    event[EV_LAT_COL] = 180 - event[EV_LAT_COL]

def round_to(value, bucket):
  return value - (value % bucket) + round((value % bucket) / bucket) * bucket

def score_event_posdet(netmodel, event, detlist):
  posdetlist = []
  for phaseid, detid in detlist:
    if netmodel.score_event_det(event, phaseid, detid) > 0:
      posdetlist.append((phaseid, detid))
  return netmodel.score_event(event, posdetlist)

def score_event_best_mag_depth(netmodel, event, detlist):
  score = -np.inf
  for mag in [3,4]:
    event[EV_MB_COL] = mag
    for depth in np.linspace(0,700,6):
      event[EV_DEPTH_COL] = depth
      score = max(score, score_event_posdet(netmodel, event, detlist))
  return score

def print_event(netmodel, earthmodel, detections, event, event_detlist):
  print ("Event: lon %4.2f lat %4.2f depth %3.1f mb %1.1f time %.1f orid %d"
         % (event[ EV_LON_COL], event[ EV_LAT_COL],
            event[ EV_DEPTH_COL], event[ EV_MB_COL],
            event[ EV_TIME_COL], event[ EV_ORID_COL]))
  print "Detections:",
  detlist = [x for x in event_detlist]
  detlist.sort()
  for phaseid, detid in detlist:
    print "(%s, %d)" % (earthmodel.PhaseName(phaseid), detid),
  print
  for phaseid, detid in detlist:
    if not earthmodel.IsTimeDefPhase(int(detections[detid, DET_PHASE_COL])):
      phaseid = 0                       # P phase
    else:
      phaseid = int(detections[detid, DET_PHASE_COL])

    tres = detections[detid, DET_TIME_COL] \
           - earthmodel.ArrivalTime(event[EV_LON_COL],
                                    event[EV_LAT_COL],
                                    event[EV_DEPTH_COL],
                                    event[EV_TIME_COL],
                                    phaseid,
                                    int(detections[detid, DET_SITE_COL]))
    
    azres = detections[detid, DET_AZI_COL]\
            - earthmodel.ArrivalAzimuth(event[EV_LON_COL],
                                        event[EV_LAT_COL],
                                        int(detections[detid, DET_SITE_COL]))
    sres = detections[detid, DET_SLO_COL] \
           - earthmodel.ArrivalSlowness(event[EV_LON_COL],
                                        event[EV_LAT_COL],
                                        event[EV_DEPTH_COL],
                                        phaseid,
                                        int(detections[detid, DET_SITE_COL]))

    print "%d: tres %.1f azres %.1f sres %.1f" % (detid, tres, azres, sres)
    
    inv = netmodel.invert_det(detid, 0)
    if inv is None:
      print None
    else:
      print "lon %.2f lat %.2f depth %.1f time %.1f" % inv

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-s", "--seed", dest="seed", default=123456789,
                    type="int",
                    help = "random number generator seed (123456789)")
  parser.add_option("-r", "--hours", dest="hours", default=None,
                    type="float",
                    help = "inference on HOURS worth of data (all)")
  parser.add_option("-k", "--skip", dest="skip", default=0,
                    type="float",
                    help = "skip the first HOURS of data (0)")
  
  parser.add_option("-i", "--runid", dest="runid", default=None,
                    type="int",
                    help = "the run-identifier to treat as LEB (None)")

  parser.add_option("-d", "--degree_step", dest="degree_step", default=1.0,
                    type="float",
                    help = "degree step (1.0)", metavar="degrees")

  parser.add_option("-t", "--time_step", dest="time_step", default=5.0,
                    type="float",
                    help = "time step (5.0)", metavar="seconds")

  (options, args) = parser.parse_args()

  netvisa.srand(options.seed)
  
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data("validation", hours=options.hours, skip=options.skip)

  # treat a VISA run as LEB
  if options.runid is not None:
    print "Reading VISA runid %d..." % options.runid,
    cursor = connect().cursor()
    detections, arid2num = read_detections(cursor, start_time, end_time)
    visa_events, visa_orid2num = read_events(cursor, start_time, end_time,
                                             "visa", options.runid)
    visa_evlist = read_assoc(cursor, start_time, end_time, visa_orid2num,
                             arid2num, "visa", runid=options.runid)
    leb_events, leb_evlist = visa_events, visa_evlist
    print "done"
  
  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  t1 = time.time()
  tot_leb, pos_leb, pos_rnd_leb, tot_hits, tot_3hits, tot_2hits = 0, 0, 0, 0,0,0
  tot_inv_hits = 0
  for leb_evnum, leb_event in enumerate(leb_events):
    leb_detlist = leb_evlist[leb_evnum]
    tot_leb += 1
    
    if score_event_best_mag_depth(netmodel, leb_event, leb_detlist) > 0:
      pos_leb += 1

    # move the event to the nearest space-time bucket
    event = leb_event.copy()
    event[EV_LON_COL] = round_to(event[EV_LON_COL], options.degree_step)
    fixup_event_lon(event)
    event[EV_LAT_COL] = round_to(event[EV_LAT_COL], options.degree_step)
    fixup_event_lat(event)
    event[EV_TIME_COL] = round_to(event[EV_TIME_COL], options.time_step)
    
    if score_event_best_mag_depth(netmodel, event, leb_detlist) > 0:
      pos_rnd_leb += 1

    # now see how many of the detections hit this bucket
    ev_hits = 0
    for phaseid, detid in leb_detlist:
      trvtime = earthmodel.ArrivalTime(event[EV_LON_COL], event[EV_LAT_COL],
                                       0, 0, 0,
                                       int(detections[detid, DET_SITE_COL]))
      if trvtime is not None and trvtime > 0:
        bucket_time = round_to(detections[detid, DET_TIME_COL] - trvtime,
                               options.time_step)
        if bucket_time == event[EV_TIME_COL]:
          tot_hits += 1
          ev_hits += 1

    if ev_hits >= 3:
      tot_3hits += 1
    if ev_hits >= 2:
      tot_2hits += 1

    # now count how many of the inverted detections hit this bucket
    ev_hits = 0
    for phaseid, detid in leb_detlist:
      inverted_event = netmodel.invert_det(detid, 0)
      if inverted_event is None:
        continue
      inv_lon, inv_lat, inv_depth, inv_time = inverted_event
      
      inv_event = event.copy()
      inv_event[EV_LON_COL] = round_to(inv_lon, options.degree_step)
      fixup_event_lon(inv_event)
      inv_event[EV_LAT_COL] = round_to(inv_lat, options.degree_step)
      fixup_event_lat(inv_event)
      inv_event[EV_TIME_COL] = round_to(inv_time, options.time_step)

      if (inv_event[EV_LON_COL] == event[EV_LON_COL] 
          and inv_event[EV_LAT_COL] == event[EV_LAT_COL]
          and inv_event[EV_TIME_COL] == event[EV_TIME_COL]):
        tot_inv_hits += 1
      
          
  t2 = time.time()
  print "%.1f secs elapsed" % (t2 - t1)
  print "%.1f %% LEB events had +ve score"\
        % (pos_leb * 100. / tot_leb)
  print "%.1f %% LEB events with rounded location had +ve score"\
        % (pos_rnd_leb * 100. / tot_leb)
  print "Avg. of %.1f detections hit event bucket" % \
        (float(tot_hits) / tot_leb)
  print "%.1f %% LEB events had 3 hits or more"\
        % (tot_3hits * 100. / tot_leb)
  print "%.1f %% LEB events had 2 hits or more"\
        % (tot_2hits * 100. / tot_leb)
  print "Avg. of %.1f inverted detections hit event bucket" % \
        (float(tot_inv_hits) / tot_leb)
  
if __name__ == "__main__":
  main("parameters")
