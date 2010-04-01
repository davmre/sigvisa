import os, sys
import numpy as np

from database.dataset import *
import netvisa, learn
from results.compare import *

def main(param_dirname):
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data("validation")

  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)
  
  print "LEB:"
  netmodel.score_world(leb_events, leb_evlist, 1)

  for i in range(len(leb_events)):
    if np.isinf(netmodel.score_event(leb_events[i], leb_evlist[i])):
      print "-Infinite score"
      print leb_events[i]
      print leb_evlist[i]
      sys.exit(1)
      
  print "Avg. LEB Event Score:", sum(netmodel.score_event(leb_events[i],
                                                      leb_evlist[i])
                                 for i in range(len(leb_events)))\
                                 / len(leb_events)
  print "SEL3:"
  netmodel.score_world(sel3_events, sel3_evlist, 1)

  true_sel3_idx, false_sel3_idx, mat_idx = find_true_false_guess(leb_events,
                                                                 sel3_events)

  lebi, sel3i = mat_idx[0]
  print "Sample LEB, SEL3 matched event:"
  print leb_events[lebi], leb_evlist[lebi]
  print sel3_events[sel3i], sel3_evlist[sel3i]

  print "TRUE SEL3"
  netmodel.score_world(sel3_events[true_sel3_idx,:],
                       [sel3_evlist[i] for i in true_sel3_idx], 1)

  print "Avg. True SEL3 Score:", sum(netmodel.score_event(sel3_events[i],
                                                          sel3_evlist[i])
                                     for i in true_sel3_idx)\
                                     / len(true_sel3_idx)

  print "FALSE SEL3"
  netmodel.score_world(sel3_events[false_sel3_idx,:],
                       [sel3_evlist[i] for i in false_sel3_idx], 1)

  print "Avg. False SEL3 Score:", sum(netmodel.score_event(sel3_events[i],
                                                           sel3_evlist[i])
                                      for i in false_sel3_idx)\
                                      / len(false_sel3_idx)

if __name__ == "__main__":
  main("parameters")
