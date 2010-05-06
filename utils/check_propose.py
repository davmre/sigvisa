#checks whether events generated from the proposer are good enough

import os, sys, time
import numpy as np
from optparse import OptionParser
import random

from database.dataset import *
import netvisa, learn
from utils.geog import dist_deg
from infer import print_events

def print_event(netmodel, earthmodel, detections, event, event_detlist):
  print ("%d: lon %4.2f lat %4.2f depth %3.1f mb %1.1f time %.1f"
         % (event[ EV_ORID_COL], event[ EV_LON_COL], event[ EV_LAT_COL],
            event[ EV_DEPTH_COL], event[ EV_MB_COL],
            event[ EV_TIME_COL]))
  print "Detections:",
  detlist = [x for x in event_detlist]
  detlist.sort()
  for phaseid, detid in detlist:
    print "(%s, %d, %s)" % (earthmodel.PhaseName(phaseid), detid,
                  earthmodel.PhaseName(int(detections[detid, DET_PHASE_COL]))),
  print "\nEv Score:", netmodel.score_event(event, event_detlist)
  
def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-d", "--degree_step", dest="degree_step", default=2.0,
                    type="float",
                    help = "degree step (2)")
  parser.add_option("-t", "--time_step", dest="time_step", default=5.0,
                    type="float",
                    help = "time step (5)")
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
  random.seed(options.seed)
  
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data("validation", hours=options.hours, skip=options.skip)

  if (end_time - MAX_TRAVEL_TIME - options.degree_step) <= start_time:
    print "Error: too short an interval"
    sys.exit(1)
    
  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  t1 = time.time()
  prop_events, prop_evlist = netmodel.propose(start_time,
                                              end_time - MAX_TRAVEL_TIME,
                                              0, len(detections),
                                              options.degree_step,
                                              options.time_step)
  t2 = time.time()

  print "%.1f seconds to propose %d event" % (t2-t1, len(prop_events))

  print_events(netmodel, earthmodel, prop_events, prop_evlist, "PROP")
  
  for leb_evnum, leb_event in enumerate(leb_events):
    
    # we can only predict events for which we have all the detections
    if leb_event[EV_TIME_COL] > (end_time - MAX_TRAVEL_TIME):
      continue
    
    print_event(netmodel, earthmodel, detections, leb_event,
                leb_evlist[leb_evnum])
    
    for prop_evnum, prop_event in enumerate(prop_events):
      
      if (dist_deg(leb_event[[EV_LON_COL, EV_LAT_COL]],
                   prop_event[[EV_LON_COL, EV_LAT_COL]]) < 5
          and abs(leb_event[EV_TIME_COL] - prop_event[EV_TIME_COL]) < 50):
        print "--> MATCH"
        print_event(netmodel, earthmodel, detections, prop_event,
                    prop_evlist[prop_evnum])
        break
      
    else:
      print "--> MISS"
  
if __name__ == "__main__":
  main("parameters")