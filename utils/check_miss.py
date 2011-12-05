# check the events missed by a run
import sys, os
import matplotlib.pyplot as plt
import numpy as np

from database.dataset import *
import netvisa, learn
from results.compare import *
from utils.geog import dist_km, degdiff
import database.db

def main(param_dirname):
  if len(sys.argv) != 2:
    print "Error: check_miss <run-id>"
    sys.exit(1)
  
  runid = int(sys.argv[1])
  
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
              sel3_evlist, site_up, sites, phasenames, phasetimedef \
              = read_data("validation", hours=None, skip=0)
  
  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  cursor = database.db.connect().cursor()
  visa_events, visa_orid2num = read_events(cursor, start_time, end_time,
                                           "visa", runid)
  visa_evlist = read_assoc(cursor, start_time, end_time, visa_orid2num,
                           compute_arid2num(detections), "visa", runid)

  print "Computing unassociated detections...",
  unassoc_detnums = set(range(len(detections)))
  for evlist in visa_evlist:
    for phaseid, detnum in evlist:
      unassoc_detnums.remove(detnum)
  print "done"
  
  print "Computing matching...",
  missed_leb_idcs = find_unmatched(leb_events, visa_events)
  print "done"
  missed_leb_idcs_set = set(missed_leb_idcs)

  missed_det_cnt, missed_unassoc_cnt = 0, 0
  for evnum in missed_leb_idcs:
    for phaseid, detnum in leb_evlist[evnum]:
      if detnum in unassoc_detnums:
        missed_unassoc_cnt += 1
      missed_det_cnt += 1
      
  print "%d LEB events missed out of %d" % (len(missed_leb_idcs),
                                            len(leb_events))
  print "%d detections of missed events unassoc out of %d" \
        % (missed_unassoc_cnt, missed_det_cnt)
  
  print "Max missed mb", max(leb_events[evnum, EV_MB_COL]
                             for evnum in missed_leb_idcs)
  print "Avg. Missed mb", sum(leb_events[evnum, EV_MB_COL]
                              for evnum in missed_leb_idcs)/len(missed_leb_idcs)
  print "Avg. Overall mb", sum(leb_events[evnum, EV_MB_COL] for evnum in
                               range(len(leb_events)))/len(leb_events)

  print "Inverting detections...",
  # invert all the detections
  inv_events = []
  for detnum in range(len(detections)):
    inv_ev = netmodel.invert_det(detnum, 0)
    if inv_ev is not None:
      inv_events.append(inv_ev + (detnum,))
  inv_events.sort(key = lambda e: e[EV_TIME_COL])
  inv_events = np.array(inv_events)
  print "done"
  
  # find the nearest inverted event within 100 seconds of the LEB event time
  # also, find the weight distance = distance deg + time difference / 10
  missed_mindist, found_mindist = [], []
  missed_minwtdist, found_minwtdist = [], []
  hopeless_tot, hopeless_found = 0, 0

  detcnt_minwtdist = np.ndarray((0,2), float)
  unassoc_mindist, unassoc_minwtdist = [], []
  unassoc_detcnt_minwtdist = np.ndarray((0,2), float)
  
  low_invnum = 0
  for evnum in range(len(leb_events)):
    distlist, wtdistlist, unassoc_distlist, unassoc_wtdistlist = [], [], [], []
    for invnum in range(low_invnum, len(inv_events)):
      # no point at looking at a detection which is 100 seconds before the
      # current one
      if inv_events[invnum, EV_TIME_COL] < leb_events[evnum, EV_TIME_COL] - 100:
        low_invnum = invnum + 1
        continue
      
      # we can stop if we have gone 100 seconds past the event
      if inv_events[invnum, EV_TIME_COL] > leb_events[evnum, EV_TIME_COL] + 100:
        break
      
      dist = dist_deg(leb_events[evnum, [EV_LON_COL, EV_LAT_COL]],
                      inv_events[invnum, [EV_LON_COL, EV_LAT_COL]])
      wtdist = dist + abs(inv_events[invnum, EV_TIME_COL]
                          - leb_events[evnum, EV_TIME_COL]) / 10.
      distlist.append(dist)
      wtdistlist.append(wtdist)

      # the MB column stores the detnum
      if inv_events[invnum, EV_MB_COL] in unassoc_detnums:
        unassoc_distlist.append(dist)
        unassoc_wtdistlist.append(wtdist)
      
    if len(distlist):
      mindist = min(distlist)
      minwtdist = min(wtdistlist)
      
      if evnum in missed_leb_idcs_set:
        missed_mindist.append(mindist)
        missed_minwtdist.append(minwtdist)

        detcnt_minwtdist = np.vstack((detcnt_minwtdist,
                                      (len(leb_evlist[evnum]), minwtdist)))
        
        unassoc_detcnt_minwtdist = np.vstack((unassoc_detcnt_minwtdist,
                 (sum(1 for phaseid, detnum in leb_evlist[evnum]
                      if detnum in unassoc_detnums), minwtdist)))
        
      else:
        found_mindist.append(mindist)
        found_minwtdist.append(minwtdist)
    
    else:
      hopeless_tot += 1
      if evnum not in missed_leb_idcs_set:
        hopeless_found += 1

    if len(unassoc_distlist) and evnum in missed_leb_idcs_set:
      unassoc_mindist.append(min(unassoc_distlist))
      unassoc_minwtdist.append(min(unassoc_wtdistlist))
  
  print "%d LEB events hopeless (%d hopeless found!)" % (hopeless_tot,
                                                         hopeless_found)

  # plot a histogram showing distance from nearest inverted location
  maxdist = max(max(found_mindist), max(missed_mindist))
  bins = np.arange(0, maxdist+1)
  plt.figure()
  plt.title("Distance to nearest inverted detection")
  plt.hist(found_mindist, bins, facecolor="blue", edgecolor="none", alpha=0.5,
           label="Located Events")
  plt.hist(missed_mindist, bins, facecolor="red", edgecolor="none", alpha=0.5,
           label="Missed Events")
  plt.xlabel("Distance")
  plt.ylabel("Count")
  plt.legend(loc="upper right")
  plt.savefig(os.path.join("output", "inv-dist.png"))
  
  # plot a histogram showing weighted distance from nearest inverted location
  maxwtdist = max(max(found_minwtdist), max(missed_minwtdist))
  bins = np.arange(0, maxwtdist+1)
  plt.figure()
  plt.title("Weighted distance to nearest inverted detection")
  plt.hist(found_minwtdist, bins, facecolor="blue", edgecolor="none", alpha=0.5,
           label="Located Events")
  plt.hist(missed_minwtdist, bins, facecolor="red", edgecolor="none", alpha=0.5,
           label="Missed Events")
  plt.xlabel("Weighted Distance")
  plt.ylabel("Count")
  plt.legend(loc="upper right")
  plt.savefig(os.path.join("output", "inv-wtdist.png"))

  plt.figure()
  plt.title("Scatter plot of missed event detection count vs. min weighted"
            " distance")
  plt.scatter(detcnt_minwtdist[:,1], detcnt_minwtdist[:,0])
  plt.xlabel("Weighted Distance")
  plt.ylabel("Detection Count")
  plt.savefig(os.path.join("output", "inv-det-wtdist.png"))

  plt.figure()
  plt.title("Missed event detection count vs. min weighted"
            " distance")
  plt.hexbin(detcnt_minwtdist[:,1], detcnt_minwtdist[:,0], gridsize=10)
  cb = plt.colorbar()
  cb.set_label('counts')  
  plt.xlabel("Weighted Distance")
  plt.ylabel("Detection Count")
  plt.savefig(os.path.join("output", "inv-det-wtdist-hexbin.png"))

  plt.figure()
  plt.title("Missed event unassoc detection count vs. "
            "min weighted distance")
  plt.hexbin(unassoc_detcnt_minwtdist[:,1], unassoc_detcnt_minwtdist[:,0],
             gridsize=6)
  cb = plt.colorbar()
  cb.set_label('counts')
  plt.xlabel("Weighted Distance")
  plt.ylabel("Unassoc Detection Count")
  plt.savefig(os.path.join("output", "inv-unassoc-det-wtdist-hexbin.png"))
  
  
  plt.figure()
  plt.title("Distance of missed event to nearest inverted unassoc detection")
  plt.hist(unassoc_mindist, bins, facecolor="red", edgecolor="none", alpha=0.5)
  plt.xlabel("Distance")
  plt.ylabel("Count")
  plt.savefig(os.path.join("output", "inv-dist-missed-unassoc.png"))
  
  plt.figure()
  plt.title("Weighted Distance of missed event to nearest inverted unassoc"
            " detection")
  plt.hist(unassoc_minwtdist, bins, facecolor="red", edgecolor="none",
           alpha=0.5)
  plt.xlabel("Weighted Distance")
  plt.ylabel("Count")
  plt.savefig(os.path.join("output", "inv-wtdist-missed-unassoc.png"))
  
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
