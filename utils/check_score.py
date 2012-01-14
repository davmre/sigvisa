import os, sys, time
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt

from database.dataset import *
import netvisa, learn
from results.compare import *
from score import *
from priors.SecDetPrior import compute_secondary_dets

def main(param_dirname):
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

  leb_seclist = compute_secondary_dets(earthmodel, detections, leb_events,
                                       leb_evlist)
  sel3_seclist = compute_secondary_dets(earthmodel, detections, sel3_events,
                                        sel3_evlist)
  
  true_sel3_idx, false_sel3_idx, mat_idx = find_true_false_guess(leb_events,
                                                                 sel3_events)

  leb_scores = [netmodel.score_event(event, leb_seclist[evnum]) for
                evnum, event in enumerate(leb_events)]
  
  pos_leb = sum(1 for s in leb_scores if s>0)
  
  print "%.1f LEB events have +ve score" % (100. * pos_leb / len(leb_events))
  print "     Average %.1f" % np.average(leb_scores)

  pos_true_sel3 = sum(1 for evnum in true_sel3_idx
                      if netmodel.score_event(sel3_events[evnum],
                                              sel3_seclist[evnum]) > 0)
  
  print "%.1f True SEL3 events have +ve score" % \
        (100. * float(pos_true_sel3) / len(true_sel3_idx))
  
  pos_false_sel3 = sum(1 for evnum in false_sel3_idx
                       if netmodel.score_event(sel3_events[evnum],
                                               sel3_seclist[evnum]) > 0)
  
  print "%.1f False SEL3 events have +ve score" % \
        (100. * float(pos_false_sel3) / len(false_sel3_idx))


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

