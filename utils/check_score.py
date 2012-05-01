# Copyright (c) 2012, Bayesian Logic, Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Bayesian Logic, Inc. nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# Bayesian Logic, Inc. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# 
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
  parser.add_option("-r", "--hours", dest="hours", default=None,
                    type="float",
                    help = "inference on HOURS worth of data (all)")
  parser.add_option("-k", "--skip", dest="skip", default=0,
                    type="float",
                    help = "skip the first HOURS of data (0)")
  parser.add_option("-g", "--gui", dest="gui", default=False,
                    action="store_true", help="display graphs (False)")
  parser.add_option("--datafile", dest="datafile", default=None,
                    help = "tar file with data (None)", metavar="FILE")
  parser.add_option("--sel3", dest="sel3", default=False,
                    action = "store_true",
                    help = "show SEL3 scores (False)")
  
  (options, args) = parser.parse_args()

  if options.datafile is not None:
    start_time, end_time, detections, leb_events, leb_evlist,\
      sel3_events, sel3_evlist, site_up, sites, phasenames, \
      phasetimedef, sitenames \
      = learn.read_datafile_and_sitephase(options.datafile, param_dirname,
                                       hours = options.hours, skip=options.skip)
  else:
    start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
      sel3_evlist, site_up, sites, phasenames, phasetimedef \
      = read_data(options.label, hours = options.hours, skip=options.skip)
  
  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)
  netmodel.disable_sec_arr()
  netmodel.set_temperature(.0001)
  
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

  for evnum, event in enumerate(leb_events):
    evsc = netmodel.score_event(event, leb_evlist[evnum])
    if evsc is None or np.isinf(evsc):
      import pdb
      pdb.set_trace()
    
  leb_scores = dict((evnum, netmodel.score_event(event, leb_evlist[evnum])) for
                    evnum, event in enumerate(leb_events))
  
  pos_leb = sum(1 for s in leb_scores.values() if s>0)
  
  print "%.1f %% LEB events have +ve score" % (100. * pos_leb / len(leb_events))
  print "     Average Score %.1f Median %.1f" \
        % (np.average(leb_scores.values()), np.median(leb_scores.values()))
  
  # compute mis-detection statistics on events with negative scores
  tot_ev_misdet_lp, num_neg_ev = 0., 0
  for evnum in range(len(leb_events)):
    if leb_scores[evnum] < 0:
      det_ph_site = set((ph, int(detections[detnum, DET_SITE_COL]))\
                        for ph, detnum in leb_evlist[evnum])
      ev_misdet_lp = 0.
      for p in range(earthmodel.NumTimeDefPhases()):
        for s in range(earthmodel.NumSites()):
          if (p, s) not in det_ph_site:
            misdet_lp = netmodel.logprob_event_misdet(leb_events[evnum], p, s)
            if misdet_lp is not None:
              ev_misdet_lp += misdet_lp
      num_neg_ev += 1
      tot_ev_misdet_lp += ev_misdet_lp
  print "Of the %d negative LEB events average mis-detection logprob %.1f"\
        % (num_neg_ev, tot_ev_misdet_lp / num_neg_ev)

  print "Highest mb -ve LEB event:"
  print max((leb_events[evnum, EV_MB_COL], leb_events[evnum, EV_ORID_COL])
            for evnum, evscore in leb_scores.iteritems() if evscore < 0)
  
  if options.gui:
    plt.figure()
    plt.title("LEB scores")
    plt.hist(leb_scores.values())

    plt.figure()
    plt.title("mb of -ve LEB events")
    plt.hist([leb_events[evnum, EV_MB_COL]
              for evnum, evscore in leb_scores.iteritems() if evscore < 0])
    
  if not options.sel3 and options.gui:
    plt.show()

  if options.sel3:
    true_sel3_idx, false_sel3_idx, mat_idx = find_true_false_guess(leb_events,
                                                                   sel3_events)
  
    sel3_scores = [netmodel.score_event(event, sel3_evlist[evnum]) for
                   evnum, event in enumerate(sel3_events)]
    
    pos_true_sel3 = sum(1 for evnum in true_sel3_idx if sel3_scores[evnum] > 0)
    
    print "%.1f %% True SEL3 events have +ve score" % \
          (100. * float(pos_true_sel3) / len(true_sel3_idx))
    
    pos_false_sel3 = sum(1 for evnum in false_sel3_idx
                         if sel3_scores[evnum] > 0)
    
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

