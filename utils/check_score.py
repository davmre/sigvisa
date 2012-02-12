import os, sys, time
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt

from database.dataset import *
import database.db
from analyze import read_sel3_svm_scores
import netvisa, learn
from results.compare import *
from score import *
from priors.SecDetPrior import compute_secondary_dets

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-i", "--runid", dest="runid", default=None,
                    type="int",
                    help = "the run-identifier to score (None)")
  parser.add_option("-l", "--label", dest="label", default="validation",
                    help = "the data label (validation)")
  parser.add_option("-g", "--gui", dest="gui", default=False,
                    action="store_true", help="display graphs (False)")
  
  (options, args) = parser.parse_args()
  
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data(options.label)
  
  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)
  netmodel.disable_sec_arr()

  if options.runid:
    conn = database.db.connect()
    cursor = conn.cursor()
    print "Reading VISA Events...",
    visa_events, visa_orid2num = read_events(cursor, start_time, end_time,
                                             "visa", options.runid)
    print "Done (%d events)" % len(visa_events)
    print "Reading VISA Associations...",
    visa_assoc = read_assoc(cursor, start_time, end_time, visa_orid2num,
                            compute_arid2num(detections), "visa", options.runid)
    print "Done (%d associations)" % sum(len(evlist) for evlist in visa_assoc)
    
  
  prune_detections(netmodel, leb_events, leb_evlist)
  prune_detections(netmodel, sel3_events, sel3_evlist)

  leb_scores = [netmodel.score_event(event, leb_evlist[evnum]) for
                evnum, event in enumerate(leb_events)]
  
  pos_leb = sum(1 for s in leb_scores if s>0)
  
  print "%.1f %% LEB events have +ve score" % (100. * pos_leb / len(leb_events))
  print "     Average Score %.1f" % np.average(leb_scores)

  true_sel3_idx, false_sel3_idx, mat_idx = find_true_false_guess(leb_events,
                                                                 sel3_events)

  sel3_scores = [netmodel.score_event(event, sel3_evlist[evnum]) for
                 evnum, event in enumerate(sel3_events)]
  
  pos_true_sel3 = sum(1 for evnum in true_sel3_idx if sel3_scores[evnum] > 0)
  
  print "%.1f %% True SEL3 events have +ve score" % \
        (100. * float(pos_true_sel3) / len(true_sel3_idx))
  
  pos_false_sel3 = sum(1 for evnum in false_sel3_idx if sel3_scores[evnum] > 0)
  
  print "%.1f %% False SEL3 events have +ve score" % \
        (100. * float(pos_false_sel3) / len(false_sel3_idx))

  if options.gui:
    plt.figure()
    
  if options.runid:
    true_visa_idx, false_visa_idx, _ = find_true_false_guess(leb_events,
                                                             visa_events)

    visa_scores = [netmodel.score_event(event, visa_assoc[evnum]) for
                   evnum, event in enumerate(visa_events)]
    
    pos_true_visa = sum(1 for evnum in true_visa_idx if visa_scores[evnum] > 0)
    
    print "%.1f %% True VISA events have +ve score" % \
          (100. * float(pos_true_visa) / len(true_visa_idx))
    print "     Average %.1f" % (sum(visa_scores[evnum]
                                     for evnum in true_visa_idx)
                                 / len(true_visa_idx))
    
    pos_false_visa = sum(1 for evnum in false_visa_idx if visa_scores[evnum] >0)
    
    print "%.1f %% False VISA events have +ve score" % \
          (100. * float(pos_false_visa) / len(false_visa_idx))
    print "     Average %.1f" % (sum(visa_scores[evnum]
                                     for evnum in false_visa_idx)
                                 / len(false_visa_idx))
    
    if options.gui:
      draw_pr_curve("NET-VISA run %d" % options.runid, true_visa_idx,
                    visa_scores, len(leb_events))
      
  if options.gui:
    # first try the SVM scores, otherwise use NET-VISA scores for comparison
    try:
      sel3_svm_scores = read_sel3_svm_scores()
      sel3_svm_scores = [sel3_svm_scores[int(event[EV_ORID_COL])]
                         for evnum, event in enumerate(sel3_events)]
      draw_pr_curve("SEL3 (SVM scores)", true_sel3_idx, sel3_svm_scores,
                    len(leb_events))
    except:
      print "No SVM scores found"
      draw_pr_curve("SEL3 (NET-VISA scores)", true_sel3_idx, sel3_scores,
                    len(leb_events))
    
    plt.xlim(.39, 1)
    plt.ylim(.39, 1)
    plt.xlabel("precision")
    plt.ylabel("recall")
    plt.legend(loc = "upper right")
    plt.grid()
    
    plt.show()

def draw_pr_curve(label, true_evnums, evscores, total_true):
  true_evnums = set(true_evnums)
  scores = []
  for evnum, evscore in enumerate(evscores):
    if evnum in true_evnums:
      scores.append((evscore, 1))
    else:
      scores.append((evscore, 0))

  scores.sort()

  prec_pts, rec_pts, f1_pts = [], [], []
  # at each score we will compute the precision and recall if we throw away
  # all events with score less than that score
  numtrue = len(true_evnums)
  for idx, (score, istrue) in enumerate(scores):
    prec = float(numtrue) / (len(scores) - idx)
    recall = float(numtrue) / total_true
    if not prec or not recall:
      f1 = 0.
    else:
      f1 = 2 * prec * recall / (prec + recall)
      
    prec_pts.append(prec)
    rec_pts.append(recall)
    f1_pts.append(f1)
    
    if istrue:
      numtrue -= 1

  plt.plot(prec_pts, rec_pts, label=label, linewidth=3)

  idx = np.argmax(f1_pts)

  print "%s: prec=%.1f recall=%.1f MAX f1=%.1f, prec=%.1f, recall=%.1f" \
        % (label, 100. * float(len(true_evnums)) / len(scores),
           100. * float(len(true_evnums)) / total_true, 100. * f1_pts[idx],
           100. * prec_pts[idx], 100. * rec_pts[idx])
  
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

