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

  print "SEL3:"
  netmodel.score_world(sel3_events, sel3_evlist, 1)

  # separate the SEL3 events into true and false events
  true_sel3_idx, false_sel3_idx, mat_idx = find_true_false_guess(leb_events,
                                                                 sel3_events)

  #lebi, sel3i = mat_idx[0]
  #print "Sample LEB, SEL3 matched event:"
  #print leb_events[lebi], leb_evlist[lebi]
  #print sel3_events[sel3i], sel3_evlist[sel3i]

  print "TRUE SEL3"
  netmodel.score_world(sel3_events[true_sel3_idx,:],
                       [sel3_evlist[i] for i in true_sel3_idx], 1)
  true_sel3 = [netmodel.score_event(sel3_events[i], sel3_evlist[i])
               for i in true_sel3_idx]

  idx = true_sel3_idx[0]

  #print "Testing on one event:"
  #netmodel.score_world(sel3_events[[idx],:], [sel3_evlist[idx]], 1)
  #print "Score_Event:", netmodel.score_event(sel3_events[idx],
  #                                           sel3_evlist[idx])

  print "FALSE SEL3"
  netmodel.score_world(sel3_events[false_sel3_idx,:],
                       [sel3_evlist[i] for i in false_sel3_idx], 1)

  false_sel3 = [netmodel.score_event(sel3_events[i], sel3_evlist[i])
                for i in false_sel3_idx]
  
  import matplotlib.pyplot as plt

  bins = range(-100,200,5)
  plt.figure()
  plt.title("SEL3 event scores")
  plt.hist(true_sel3, bins, facecolor = "blue", label = "true events",
           alpha = .5)
  plt.hist(false_sel3, bins, facecolor = "red", label = "false events",
           alpha = .5)
  plt.legend(loc="upper right")
  plt.xlim(-100,200)
  plt.ylim(0,100)
  plt.xlabel("score")
  plt.ylabel("number of events")

  plt.show()

if __name__ == "__main__":
  main("parameters")
