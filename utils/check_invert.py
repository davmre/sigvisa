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

  (options, args) = parser.parse_args()

  netvisa.srand(options.seed)
  
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data("validation", hours=options.hours, skip=options.skip)

  # treat a VISA run as LEB
  if options.runid is not None:
    cursor = connect().cursor()
    detections, arid2num = read_detections(cursor, start_time, end_time)
    visa_events, visa_orid2num = read_events(cursor, start_time, end_time,
                                             "visa", options.runid)
    visa_evlist = read_assoc(cursor, start_time, end_time, visa_orid2num,
                             arid2num, "visa", runid=options.runid)
    leb_events, leb_evlist = visa_events, visa_evlist
  
  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  t1 = time.time()
  tot_leb, pos_leb, nearby_inv, pos_inv = 0, 0, 0, 0
  for leb_evnum, leb_event in enumerate(leb_events):
    leb_detlist = leb_evlist[leb_evnum]
    tot_leb += 1
    if netmodel.score_event(leb_event, leb_detlist) > 0:
      pos_leb += 1

    has_nearby_inv = False
    has_pos_inv = False
    
    for phaseid, detid in leb_detlist:
      inv_ev = netmodel.invert_det(detid, 0)
      if (inv_ev is None or (inv_ev[EV_TIME_COL] - leb_event[EV_TIME_COL] > 100)
          or (dist_deg((inv_ev[EV_LON_COL], inv_ev[EV_LAT_COL]),
                      (leb_event[EV_LON_COL], leb_event[EV_LAT_COL])) > 10)):
        continue
      else:
        has_nearby_inv = True
        
        tmp_ev = leb_event.copy()
        tmp_ev[[EV_LON_COL, EV_LAT_COL, EV_DEPTH_COL, EV_TIME_COL]] = inv_ev

        for mag in [2.,3.,4.,5.]:
          tmp_ev[EV_MB_COL] = mag
          
          for loni in range(-10,11):
            tmp_ev[EV_LON_COL] = inv_ev[EV_LON_COL] + loni * 1.0
            fixup_event_lon(tmp_ev)
            
            for lati in range(-10,11):
              tmp_ev[EV_LAT_COL] = inv_ev[EV_LAT_COL] + lati * 1.0
              fixup_event_lat(tmp_ev)

              trvtime = earthmodel.ArrivalTime(
                tmp_ev[EV_LON_COL], tmp_ev[EV_LAT_COL], tmp_ev[EV_DEPTH_COL],
                0, 0, int(detections[detid, DET_SITE_COL]))

              if trvtime is None or trvtime < 0:
                continue

              tmp_ev[EV_TIME_COL] = detections[detid, DET_TIME_COL] - trvtime
        
              if tmp_ev[EV_TIME_COL] < start_time \
                     or tmp_ev[EV_TIME_COL] > end_time:
                continue
            
              inv_score = netmodel.score_event(tmp_ev, leb_detlist)

              if inv_score > 0:
                has_pos_inv = True

    if has_nearby_inv:
      nearby_inv += 1
    
    if has_pos_inv:
      pos_inv += 1

  t2 = time.time()
  print "%.1f secs elapsed" % (t2 - t1)
  print "%.1f %% LEB events had +ve score"\
        % (pos_leb * 100. / tot_leb)
  print "%.1f %% LEB events had a nearby invert"\
        % (nearby_inv * 100. / tot_leb)
  print "%.1f %% LEB events had +ve score from some nearby invert"\
        % (pos_inv * 100. / tot_leb)
  
  
if __name__ == "__main__":
  main("parameters")
