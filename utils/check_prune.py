# prunes events which are within a space-time ball of better events
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

from database.dataset import *
from analyze import suppress_duplicates, compute_roc_curve
import database.db
import learn
import results.compare
from utils.geog import degdiff, dist_deg

def steal_dets(netmodel, visa_events, visa_evscores, visa_assoc):
  low_evnum = 0
  for evnum in range(len(visa_events)):
    for evnum2 in xrange(low_evnum, len(visa_events)):
      # we need the two events within 100 seconds of each other
      if abs(visa_events[evnum, EV_TIME_COL] -visa_events[evnum2, EV_TIME_COL])\
         > 100:
        if evnum2 < evnum:
          low_evnum = evnum2 + 1
          continue
        else:
          break
      # if the second one has a higher score than the first one then it can
      # try to steal from the first one
      if visa_evscores[int(visa_events[evnum2, EV_ORID_COL])]\
         > visa_evscores[int(visa_events[evnum, EV_ORID_COL])]:
        event2 = visa_events[evnum2]
        visa_assoc[evnum] = [(phaseid, detnum)
                             for phaseid, detnum in visa_assoc[evnum]
                             if netmodel.score_event_det(event2, phaseid,
                                                         detnum) < 0]
        
      
def drop_detections(visa_events, visa_evscores, visa_assoc, detections):
  # first find the evnum, phase for each detection
  det_evscore = {}
  for evnum, evlist in enumerate(visa_assoc):
    for phase, detnum in evlist:
      det_evscore[detnum] = visa_evscores[int(visa_events[evnum, EV_ORID_COL])]

  # for each detection check if there is an earlier phase arrival which could
  # be responsible for it
  sec_detnums = set()
  low_detnum2 = 0
  for detnum, det in enumerate(detections):
    sitenum = int(det[DET_SITE_COL])
    for detnum2 in xrange(low_detnum2, detnum):
      # we are not interested in very old prior detections
      if detections[detnum2, DET_TIME_COL] < (det[DET_TIME_COL] - 60):
        low_detnum2 = detnum2+1
        continue
      # we only want prior detections at the same station
      if int(detections[detnum2, DET_SITE_COL]) == sitenum:
        if detnum2 in det_evscore \
               and (detnum not in det_evscore
                    or det_evscore[detnum] < det_evscore[detnum2]):
#               and abs(degdiff(detections[detnum2, DET_AZI_COL],
#                               detections[detnum, DET_AZI_COL])) < 60:
          sec_detnums.add(detnum)
          # also allow other detections to hang off this one
          det_evscore[detnum] = det_evscore[detnum2]
          break

  # now remove the secondary detections from the assoc
  for evnum in range(len(visa_assoc)):
    visa_assoc[evnum] = [(phase, detnum) for (phase, detnum)
                         in visa_assoc[evnum] if detnum not in sec_detnums]

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-i", "--runid", dest="runid", default=None,
                    type="int",
                    help = "the run-identifier to prune (last runid)")
  parser.add_option("-l", "--label", dest="label", default="validation",
                    help = "the data label (validation)")
  
  (options, args) = parser.parse_args()

  conn = database.db.connect()
  cursor = conn.cursor()

  if options.runid is None:
    cursor.execute("select max(runid) from visa_run")
    options.runid, = cursor.fetchone()
  
  print "RUNID %d:" % options.runid,

  cursor.execute("select run_start, run_end, data_start, data_end, descrip, "
                 "numsamples, window, step from visa_run where runid=%d" %
                 options.runid)
  
  run_start, run_end, data_start, data_end, descrip, numsamples, window, step\
             = cursor.fetchone()
  
  if data_end is None:
    print "NO RESULTS"
    return


  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data(options.label, skip=data_start, hours=data_end)
  
  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)
  netmodel.disable_sec_arr()

  print "Reading VISA Events...",
  visa_events, visa_orid2num = read_events(cursor, data_start, data_end,
                                           "visa", options.runid)
  print "Done (%d events)" % len(visa_events)
  print "Reading VISA Associations...",
  visa_assoc = read_assoc(cursor, data_start, data_end, visa_orid2num,
                          compute_arid2num(detections), "visa", options.runid)
  print "Done (%d associations)" % sum(len(evlist) for evlist in visa_assoc)

  print "Reading VISA scores...",
  cursor.execute("select orid, score from visa_origin where runid=%d" %
                 (options.runid,))
  visa_evscores = dict(cursor.fetchall())
  print "Done (Average score %.1f)" % (sum(visa_evscores.itervalues())
                                       /len(visa_evscores))

  print "Suppressing VISA duplicate events...",
  unpruned_visa_events, unpruned_visa_orid2num = \
                        suppress_duplicates(visa_events, visa_evscores, False)
  print "done"
  
  print "Out of %d VISA events, %d left after suppressing duplicates" \
        % (len(visa_events), len(unpruned_visa_events))
  
  report_pruning_errors(leb_events, visa_events, unpruned_visa_events)

  print "Dropping secondary detections...",
  #drop_detections(visa_events, visa_evscores, visa_assoc, detections)
  steal_dets(netmodel, visa_events, visa_evscores, visa_assoc)

  # recompute VISA event scores after dropping secondaries
  sec_visa_evscores = dict ((int(visa_events[evnum, EV_ORID_COL]),
                             netmodel.score_event(visa_events[evnum],
                                                  visa_assoc[evnum]))
                            for evnum in range(len(visa_events)))
  
  positive_orids = set(int(visa_events[evnum, EV_ORID_COL])
                       for evnum in range(len(visa_events))
                       if netmodel.score_event(visa_events[evnum],
                                               visa_assoc[evnum]) > 0)
  
  pos_evnums = [evnum for evnum in range(len(visa_events))
                if int(visa_events[evnum, EV_ORID_COL]) in positive_orids]
  posvisa_events = visa_events[pos_evnums]
  print "done"
  
  print "Out of %d VISA events, %d left after dropping secondaries"\
        % (len(visa_events), len(posvisa_events))

  report_pruning_errors(leb_events, visa_events, posvisa_events)
  
  plt.figure()
  plt.title("Runid %d" % options.runid)
  print "Matching secondary-det VISA with LEB...",
  x_pts, y_pts = compute_roc_curve(leb_events, posvisa_events,
                                   sec_visa_evscores)
  plt.plot(x_pts, y_pts, label="secondary dets", color="blue",
           linestyle="-", linewidth=3)
  print "done"

  print "Matching pruned VISA with LEB...",
  x_pts, y_pts = compute_roc_curve(leb_events, unpruned_visa_events,
                                   visa_evscores)
  plt.plot(x_pts, y_pts, label="pruned", color="blue",
           linestyle="--", linewidth=3)
  print "done"
  
  plt.xlim(.39, 1)
  plt.ylim(.39, 1)
  plt.xlabel("precision")
  plt.ylabel("recall")
  plt.legend(loc = "upper right")
  plt.grid(True)
  plt.show()

def match_events(ev1, ev2):
  return ((ev1[EV_TIME_COL] - ev2[EV_TIME_COL]) < 50) \
         and dist_deg(ev1[[EV_LON_COL, EV_LAT_COL]],
                      ev2[[EV_LON_COL, EV_LAT_COL]]) < 5

def report_pruning_errors(true_events, orig_guess, final_guess):
  # for each true event try to match it with something in the original guess
  # and the final guess. report any event which is no longer matched
  orignum_low = 0
  finalnum_low = 0
  for true in true_events:
    final_matches = set()
    for evnum in range(finalnum_low, len(final_guess)):
      final = final_guess[evnum]
      if final[EV_TIME_COL] < (true[EV_TIME_COL] - 50):
        finalnum_low += 1
        continue
      elif final[EV_TIME_COL] > (true[EV_TIME_COL] + 50):
        break
      if match_events(true, final):
        final_matches.add(final[EV_ORID_COL])

    # no point analyzing this if we didn't break anything
    if len(final_matches):
      continue
    
    orig_matches = set()
    for evnum in range(orignum_low, len(orig_guess)):
      orig = orig_guess[evnum]
      if orig[EV_TIME_COL] < (true[EV_TIME_COL] - 50):
        orignum_low += 1
        continue
      elif orig[EV_TIME_COL] > (true[EV_TIME_COL] + 50):
        break
      if match_events(true, orig):
        orig_matches.add(orig[EV_ORID_COL])

    # again no point analyzing this if we didn't crack this event in the first
    # place
    if not len(orig_matches):
      continue

    print "Lost True Event %d Originally Matched" % true[EV_ORID_COL],
    for orid in orig_matches:
      print int(orid),
    print

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

  
