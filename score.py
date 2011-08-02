import os, sys, time
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt

from database.dataset import *
import netvisa, learn
from results.compare import *
from utils.kstest import kstest

def prune_detections(netmodel, events, event_detlists):
  for evnum, event in enumerate(events):
    dellist = []
    for evdetnum, (phaseid, detid) in enumerate(event_detlists[evnum]):
      detsc = netmodel.score_event_det(event, phaseid, detid)
      if detsc is None or np.isinf(detsc):
        dellist.append(evdetnum)
    while len(dellist):
      evdetnum = dellist.pop()
      event_detlists[evnum].pop(evdetnum)

def extract_false_detections(numdets, leb_evlist):
  """
  Returns a list of false detnums
  """
  falsedets = set(range(numdets))
  for evlist in leb_evlist:
    for phaseid, detid in evlist:
      falsedets.remove(detid)
  falsedets = list(falsedets)
  falsedets.sort()
  return falsedets

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-x", "--text", dest="gui", default=True,
                    action = "store_false",
                    help = "text only output (False)")
  
  parser.add_option("-p", "--prob", dest="prob", default=False,
                    action = "store_true",
                    help = "write probabilities instead of densities (False)")
  
  parser.add_option("-w", "--writefile", dest="writefile", default=None,
                    type="str",
                    help = "file to write the sel3 scores output to")
  
  (options, args) = parser.parse_args()
  
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data("validation")
  
  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  prune_detections(netmodel, leb_events, leb_evlist)
  prune_detections(netmodel, sel3_events, sel3_evlist)
  
  print "LEB:"
  netmodel.score_world(leb_events, leb_evlist, 1)

  falsedets = extract_false_detections(len(detections), leb_evlist)

  print "FALSE:"
  netmodel.logprob_false(falsedets, 1)
  
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
      
      if options.prob:
        t1 = time.time()
        prob = netmodel.prob_event(event, sel3_evlist[evnum])
        print >>fp, prob
        t2 = time.time()
        print "%.1f sec" % (t2 - t1),
        print int(event[EV_ORID_COL]),
        if evnum in true_sel3_idx:
          print 1,
        else:
          print 0,
        print prob, netmodel.score_event(event, sel3_evlist[evnum])
        
      else:
        print >>fp, netmodel.score_event(event, sel3_evlist[evnum])
      
    fp.close()
    
  leb_scores = [netmodel.score_event(leb_events[i], leb_evlist[i])
                for i in range(len(leb_events))]

  true_sel3 = [netmodel.score_event(sel3_events[i], sel3_evlist[i])
               for i in true_sel3_idx]
    
  false_sel3 = [netmodel.score_event(sel3_events[i], sel3_evlist[i])
                for i in false_sel3_idx]


  print "%.1f %% False-SEL3 events had +ve score" \
        % (100.0 * sum(s > 0 for s in false_sel3) / len(false_sel3))
    
  print "%.1f %% True-SEL3 events had +ve score" \
        % (100.0 * sum(s > 0 for s in true_sel3) / len(true_sel3))
  
  print "%.1f %% LEB events had +ve score" \
        % (100.0 * sum(s > 0 for s in leb_scores) / len(leb_scores))
  
  if options.gui:
    # compute the ROC curve
    x_pts, y_pts = [], []
    for sep in range(-100, 200, 2):
      y = float(len(filter(lambda x: x>sep, true_sel3))) / len(true_sel3)
      x = float(len(filter(lambda x: x>sep, false_sel3))) / len(false_sel3)
      
      x_pts.append(x)
      y_pts.append(y)

    plt.figure()
    plt.title("ROC curve for true SEL3 events")
    plt.plot(x_pts, y_pts)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
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
  try:
    main("parameters")
  except SystemExit:
    raise
  except:
    import pdb, traceback, sys
    traceback.print_exc(file=sys.stdout)
    pdb.post_mortem(sys.exc_traceback)
    raise

