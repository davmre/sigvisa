import os, sys
import numpy as np
from optparse import OptionParser

from database.dataset import *
import netvisa, learn
from results.compare import *

def print_events(earthmodel, detections, leb_events, leb_evlist):
  for evnum in range(len(leb_events)):
    print "Event: lon %.1f lat %.1f depth %.1f mb %.1f time %.1f orid %d" %\
          (leb_events[evnum, EV_LON_COL], leb_events[evnum, EV_LAT_COL],
           leb_events[evnum, EV_DEPTH_COL], leb_events[evnum, EV_MB_COL],
           leb_events[evnum, EV_TIME_COL], leb_events[evnum, EV_ORID_COL])
    print "Detections:",
    for phaseid, detid in leb_evlist[evnum]:
      print "(%s, %d)" % (earthmodel.PhaseName(phaseid),
                          detections[detid, DET_SITE_COL]),
    print

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-n", "--numsamples", dest="numsamples", default=10,
                    type="int",
                    help = "number of samples per second of input data (10)")
  parser.add_option("-r", "--hours", dest="hours", default=None,
                    type="float",
                    help = "inference on HOURS worth of data (False)")
  parser.add_option("-x", "--text", dest="gui", default=True,
                    action = "store_false",
                    help = "text only output (False)")
  (options, args) = parser.parse_args()
  
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data("validation", hours=options.hours)

  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  print "LEB:"
  print_events(earthmodel, detections, leb_events, leb_evlist)
  netmodel.score_world(leb_events, leb_evlist, 1)

  print "SEL3:"
  print_events(earthmodel, detections, sel3_events, sel3_evlist)
  netmodel.score_world(sel3_events, sel3_evlist, 1)

  netmodel.infer(options.numsamples)

if __name__ == "__main__":
  main("parameters")
