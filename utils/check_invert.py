#checks whether invert detection works correctly

import os, sys
import numpy as np
from optparse import OptionParser

from database.dataset import *
from database.db import connect
import netvisa, learn
from results.compare import *
from utils.geog import dist_km, dist_deg

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
  parser.add_option("-n", "--numsamples", dest="numsamples", default=1,
                    type="int",
                    help = "number of perturbations to inverted event (1)")
  parser.add_option("-r", "--hours", dest="hours", default=None,
                    type="float",
                    help = "inference on HOURS worth of data (all)")
  parser.add_option("-k", "--skip", dest="skip", default=0,
                    type="float",
                    help = "skip the first HOURS of data (0)")
  parser.add_option("-s", "--seed", dest="seed", default=123456789,
                    type="int",
                    help = "random number generator seed (123456789)")
  parser.add_option("-v", "--verbose", dest="verbose", default=False,
                    action = "store_true",
                    help = "verbose output (False)")

  (options, args) = parser.parse_args()

  netvisa.srand(options.seed)
  
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data("validation", hours=options.hours, skip=options.skip)

  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  inv_events = [netmodel.invert_det(detnum, 0)
                for detnum in range(len(detections))]

  low_detnum = 0
  num_matched_leb = 0
  err_matched_leb = 0.
  num_pos_leb = 0
  for leb_evnum, leb_event in enumerate(leb_events):
    leb_detlist = leb_evlist[leb_evnum]

    num_inv_mat = 0
    err_inv_mat = 0.
    num_inv_pos = 0
    
    for detnum in xrange(low_detnum, len(detections)):
      if leb_event[EV_TIME_COL] >= detections[detnum, DET_TIME_COL]:
        low_detnum = detnum + 1
        continue
      
      if (detections[detnum, DET_TIME_COL] - leb_event[EV_TIME_COL]
          > MAX_TRAVEL_TIME):
        break

      inv_ev = inv_events[detnum]

      # try to find a matching invert event
      if (inv_ev is None or inv_ev[EV_TIME_COL] - leb_event[EV_TIME_COL] > 50
          or dist_deg((inv_ev[EV_LON_COL], inv_ev[EV_LAT_COL]),
                      (leb_event[EV_LON_COL], leb_event[EV_LAT_COL])) > 5):
        continue

      # compute the distance
      dist = dist_km((inv_ev[EV_LON_COL], inv_ev[EV_LAT_COL]),
                     (leb_event[EV_LON_COL], leb_event[EV_LAT_COL]))
      
      err_inv_mat += dist
      num_inv_mat += 1

      # compute the score
      event = leb_event.copy()
      event[EV_MB_COL] = MIN_MAGNITUDE

      score = 0
      repeat = 0
      for repeat in range(options.numsamples+1):
        # try not perturbing once
        if repeat == 0:
          perturb = 0
        # other times perturb the invert
        else:
          perturb = 1
          
        inv_ev = netmodel.invert_det(detnum, perturb)
        
        if inv_ev is None:
          continue
        
        event[[EV_LON_COL, EV_LAT_COL, EV_DEPTH_COL, EV_TIME_COL]] = inv_ev
        
        score = netmodel.score_event(event, leb_detlist)
        
        if score > 0:
          break
        
      if score > 0:
        num_inv_pos += 1
        if options.verbose:
          print "LEB ORID %d detnum %d score %.1f  on %d th try" \
                % (leb_event[EV_ORID_COL], detnum, score, repeat)

    if num_inv_mat > 0:
      err_matched_leb += (err_inv_mat / num_inv_mat)
      num_matched_leb += 1

    if num_inv_pos > 0:
      num_pos_leb += 1

  print "%.1f %% LEB events matched with avg. avg. invert error %.1f km"\
        % (100. * num_matched_leb / len(leb_events),
           err_matched_leb / num_matched_leb)

  print "%.1f %% LEB events had inverts with +ve score on leb detections"\
        % (100. * float(num_pos_leb) / len(leb_events))
       
  
if __name__ == "__main__":
  main("parameters")
