import os, sys
import numpy as np
from optparse import OptionParser

from database.dataset import *
import netvisa, learn
from results.compare import *
from utils.geog import dist_km

def print_events(netmodel, earthmodel, leb_events, leb_evlist):
  print "=" * 60
  score = 0
  for evnum in range(len(leb_events)):
    score += print_event(netmodel, earthmodel, leb_events[evnum],
                         leb_evlist[evnum])
    print "-" * 60
  print "Total: %.1f" % score
  print "=" * 60
    
def print_event(netmodel, earthmodel, event, event_detlist):
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
  score = netmodel.score_event(event, event_detlist)
  print "Ev Score: %.1f" % score
  return score

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-n", "--numsamples", dest="numsamples", default=1,
                    type="int",
                    help = "number of samples per second of input data (1)")
  parser.add_option("-r", "--hours", dest="hours", default=None,
                    type="float",
                    help = "inference on HOURS worth of data (all)")
  parser.add_option("-k", "--skip", dest="skip", default=0,
                    type="float",
                    help = "skip the first HOURS of data (0)")
  parser.add_option("-s", "--seed", dest="seed", default=123456789,
                    type="int",
                    help = "random number generator seed (123456789)")
  parser.add_option("-x", "--text", dest="gui", default=True,
                    action = "store_false",
                    help = "text only output (False)")
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

  #print "===="
  #print "LEB:"
  #print "===="
  #print_events(netmodel, earthmodel, leb_events, leb_evlist)
  #netmodel.score_world(leb_events, leb_evlist, 1)

  #print "===="
  #print "SEL3:"
  #print "===="
  #print_events(netmodel, earthmodel, sel3_events, sel3_evlist)
  #netmodel.score_world(sel3_events, sel3_evlist, 1)

  print "===="
  print "NET"
  print "===="  
  events, ev_detlist = netmodel.infer(options.numsamples)
  #print_events(netmodel, earthmodel, events, ev_detlist)

  print "===="
  print "EVAL"
  print "===="  
  true_idx, false_idx, mat_idx = find_true_false_guess(leb_events, events)
  unmat_idx = [x for x in range(len(leb_events))]
  print "Matched:"
  for lebevnum, evnum in mat_idx:
    print "/--"
    print_event(netmodel, earthmodel, leb_events[lebevnum],
                leb_evlist[lebevnum])
    print "~~ %.1f km ~~>" % dist_km(leb_events[lebevnum,
                                                [EV_LON_COL, EV_LAT_COL]],
                                     events[evnum, [EV_LON_COL, EV_LAT_COL]])
    print_event(netmodel, earthmodel, events[evnum], ev_detlist[evnum])
    print "\--"
    unmat_idx.remove(lebevnum)
  if len(unmat_idx):
    print "Unmatched:"
  for lebevnum in unmat_idx:
    print "--"
    print_event(netmodel, earthmodel, leb_events[lebevnum],
                leb_evlist[lebevnum])
    print "--"
  if len(false_idx):
    print "Spurious:"
  for evnum in false_idx:
    print "~~"
    print_event(netmodel, earthmodel, events[evnum], ev_detlist[evnum])
    print "~~"

if __name__ == "__main__":
  main("parameters")
