import os, sys
import numpy as np
from optparse import OptionParser

from database.dataset import *
import netvisa, learn
from results.compare import *

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-x", "--text", dest="gui", default=True,
                    action = "store_false",
                    help = "text only output (False)")
  parser.add_option("-w", "--writefile", dest="writefile", default=None,
                    type="str",
                    help = "file to write the sel3 scores output to")
  (options, args) = parser.parse_args()
  
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

  #idx = true_sel3_idx[0]
  #print "Testing on one event:"
  #netmodel.score_world(sel3_events[[idx],:], [sel3_evlist[idx]], 1)
  #print "Score_Event:", netmodel.score_event(sel3_events[idx],
  #                                           sel3_evlist[idx])

  print "FALSE SEL3"
  netmodel.score_world(sel3_events[false_sel3_idx,:],
                       [sel3_evlist[i] for i in false_sel3_idx], 1)

  if options.writefile is not None:
    fp = open(options.writefile, "w")
    for evnum, event in enumerate(sel3_events):
      print >>fp, int(event[EV_ORID_COL]),
      if evnum in true_sel3_idx:
        print >>fp, 1,
      else:
        print >>fp, 0,
      print >>fp, netmodel.score_event(event, sel3_evlist[evnum])
    fp.close()
    
  if options.gui:
    leb_scores = [netmodel.score_event(leb_events[i], leb_evlist[i])
                  for i in range(len(leb_events))]

    true_sel3 = [netmodel.score_event(sel3_events[i], sel3_evlist[i])
                 for i in true_sel3_idx]
    
    false_sel3 = [netmodel.score_event(sel3_events[i], sel3_evlist[i])
                  for i in false_sel3_idx]


    # compute the ROC curve
    x_pts, y_pts = [], []
    for sep in range(-100, 200, 2):
      y = float(len(filter(lambda x: x>sep, true_sel3))) / len(true_sel3)
      x = float(len(filter(lambda x: x>sep, false_sel3))) / len(false_sel3)
      
      x_pts.append(x)
      y_pts.append(y)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.title("ROC curve for true SEL3 events")
    plt.plot(x_pts, y_pts)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("false events")
    plt.ylabel("true events")
    plt.grid(True)

    bins = range(-100,200,5)
    
    plt.figure()
    plt.title("SEL3 event scores")
    plt.hist(true_sel3, bins, facecolor = "blue", label = "true events",
             alpha = .5)
    plt.hist(false_sel3, bins, facecolor = "red", label = "false events",
             alpha = .5)
    plt.legend(loc="upper right")
    plt.xlim(-100,200)
    plt.ylim(0,150)
    plt.xlabel("score")
    plt.ylabel("number of events")
    
    plt.figure()
    plt.title("LEB event scores")
    plt.hist(leb_scores, bins, facecolor = "blue", label = "true events",
             alpha = .5)
    #plt.legend(loc="upper right")
    plt.xlim(-100,200)
    plt.ylim(0,150)
    plt.xlabel("score")
    plt.ylabel("number of events")
    
    plt.show()

if __name__ == "__main__":
  main("parameters")
