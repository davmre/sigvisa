# interpret the results of a run
import os, sys
import numpy as np
from optparse import OptionParser

from database.dataset import *
from results.compare import *
from utils.geog import degdiff, dist_deg, dist_km
import database.db

# ignore warnings from matplotlib in python 2.4
import warnings
warnings.simplefilter("ignore",DeprecationWarning)
import matplotlib.pyplot as plt
# for type 1 fonts
#plt.rcParams['text.usetex'] = True
# for type 1 fonts
#plt.rcParams['ps.useafm'] = True
#plt.rcParams['pdf.use14corefonts'] = True
from utils.draw_earth import draw_events, draw_earth

AZGAP_RANGES = [(0, 90), (90, 180), (180, 270), (270, 360)]
DETCNT_RANGES = [(-1, 0), (0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,6), (6, 100)]
TPHASE_RANGES = [(-1, 0), (0, 100)]
HPHASE_RANGES = [(-1, 0), (0, 100)]
MAG_RANGES = [(0,2), (2,3), (3,4), (4,9)]

JAPAN_LON_1, JAPAN_LON_2 = 130, 145
JAPAN_LAT_1, JAPAN_LAT_2 = 30, 45

US_LON_1, US_LON_2 = -125, -70
US_LAT_1, US_LAT_2 = 25, 50

def read_sel3_svm_scores():
  evscores = {}
  for line in open(os.path.join("output",
                               "preds_linsvm_nimLabels_1_valNimarLabels.txt")):
    evid, istrue, score = line.rstrip().split()
    evscores[int(evid)] = float(score)
  return evscores

def compute_roc_curve(gold_events, guess_events, guess_ev_scores, freq=30):

  if len(guess_events) < 100:
    freq = 1
  else:
    freq = min(len(guess_events)/10, freq)
  
  true_idx, false_idx, mat = find_true_false_guess(gold_events, guess_events)
  true_set = set(true_idx)
  istrue_and_scores = [(guess_ev_scores[guess_events[i, EV_ORID_COL]],
                        int(i in true_idx))
                       for i in range(len(guess_events))]

  istrue_and_scores.sort(reverse = True)

  # compute the ROC curve
  x_pts, y_pts = [], []
  num_true = 0
  
  for cnt, (score, istrue) in enumerate(istrue_and_scores):
    num_true += istrue
    
    if cnt % freq == 0 or cnt == len(istrue_and_scores)-1:
      y_pts.append(float(num_true) / len(gold_events))
      x_pts.append(float(num_true) / (cnt+1.0))
  
  return x_pts, y_pts
  
def filter_by_lonlat(events, lon_1, lon_2, lat_1, lat_2):
  return events[(events[:,EV_LON_COL] > lon_1)
                & (events[:,EV_LON_COL] < lon_2)
                & (events[:,EV_LAT_COL] > lat_1)
                & (events[:,EV_LAT_COL] < lat_2), :]
  
def split_by_attr(attr_ranges, attr_vals):
  ev_buckets = [[] for _ in attr_ranges]
  for evnum, val in enumerate(attr_vals):
    for bucnum, (vallo, valhi) in enumerate(attr_ranges):
      if val > vallo and val <= valhi:
        ev_buckets[bucnum].append(evnum)
  return ev_buckets

def analyze_by_attr(attr_name, attr_ranges, leb_attrvals, leb_events,
                    sel3_events, visa_events, verbose):
  # for each SEL3 and VISA event find the attribute of the corresponding
  # LEB event
  sel3_attrvals = [leb_attrvals[x] for x in find_nearest(leb_events,
                                                         sel3_events)]
  visa_attrvals = [leb_attrvals[x] for x in find_nearest(leb_events,
                                                         visa_events)]

  # put the LEB, SEL3, and VISA events into buckets based on the LEB attribute
  leb_buckets = split_by_attr(attr_ranges, leb_attrvals)
  sel3_buckets = split_by_attr(attr_ranges, sel3_attrvals)
  visa_buckets = split_by_attr(attr_ranges, visa_attrvals)


  print "-" * 74
  print " %10s | #ev |          SEL3             |          VISA"\
        % attr_name
  print "    < _ <=  |     |  F1     P     R   err  sd |   F1    P     R "\
        "  err  sd"
  print "-" * 74
  
  for i, (vallo, valhi) in enumerate(attr_ranges):
    sel3_f, sel3_p, sel3_r, sel3_err = f1_and_error(
      leb_events[leb_buckets[i],:], sel3_events[sel3_buckets[i], :])

    visa_f, visa_p, visa_r, visa_err = f1_and_error(
      leb_events[leb_buckets[i],:], visa_events[visa_buckets[i], :])
    
    print (" %3d -- %3d | %3d | %5.1f %5.1f %5.1f %3.0f %3.0f " \
          "| %5.1f %5.1f %5.1f %3.0f %3.0f")\
          % (vallo, valhi, len(leb_buckets[i]), sel3_f, sel3_p, sel3_r,
             sel3_err[0], sel3_err[1],
             visa_f, visa_p, visa_r, visa_err[0], visa_err[1])
    
    if verbose:
      unmat_idx = find_unmatched(
        leb_events[leb_buckets[i], :], visa_events[visa_buckets[i], :])

      if len(unmat_idx):
        print "Unmatched:",
        for idx in unmat_idx:
          print int(leb_events[leb_buckets[i][idx], EV_ORID_COL]),
        print
      
    
def azimuth_gap(esazlist):
  if len(esazlist) < 2:
    return 360.
  
  azlist = [x for x in esazlist]        # copy the list
  azlist.sort()

  return max((360 + degdiff(azlist[i], azlist[(i+1) % len(azlist)])) % 360 \
             for i in range(len(azlist)))

# find the nearest leb event for each event
def find_nearest(leb_events, events):
  nearest = []
  for ev in events:
    minevnum, mindist = None, None
    for lebevnum, lebev in enumerate(leb_events):
      dist = dist_deg((ev[EV_LON_COL], ev[EV_LAT_COL]),
                      (lebev[EV_LON_COL], lebev[EV_LAT_COL]))\
                      + abs(ev[EV_TIME_COL] - lebev[EV_TIME_COL])/10.0
      if mindist is None or dist < mindist:
        minevnum, mindist = lebevnum, dist
    nearest.append(minevnum)
  return nearest

def gui(options, leb_events, sel3_events, events):
  #
  # draw and the leb, sel3 and predicted events
  #
  
  bmap = draw_earth("LEB(yellow), SEL3(red) and NET-VISA(blue)")
  draw_events(bmap, sel3_events[:,[EV_LON_COL, EV_LAT_COL]],
              marker="o", ms=8, mfc="none", mec="red", mew=1)
  draw_events(bmap, events[:,[EV_LON_COL, EV_LAT_COL]],
              marker="s", ms=8, mfc="none", mec="blue", mew=1)
  draw_events(bmap, leb_events[:,[EV_LON_COL, EV_LAT_COL]],
              marker="*", ms=8, mfc="yellow")
  if options.write:
    plt.savefig("output/run_%d_events.png" % (options.runid))

  if options.events:
    bmap = draw_earth("LEB(yellow) and SEL3(red)")
    draw_events(bmap, sel3_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="o", ms=8, mfc="none", mec="red", mew=1)
    draw_events(bmap, leb_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="*", ms=8, mfc="yellow")
    if options.write:
      plt.savefig("output/run_%d_leb_sel3.png" % (options.runid))
  
    bmap = draw_earth("LEB(yellow) and NET-VISA(blue)")
    draw_events(bmap, events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="s", ms=8, mfc="none", mec="blue", mew=1)
    draw_events(bmap, leb_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="*", ms=8, mfc="yellow")
    if options.write:
      plt.savefig("output/run_%d_leb_visa.png" % (options.runid))
  
    bmap = draw_earth("SEL3(red) and NET-VISA(blue)")
    draw_events(bmap, sel3_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="o", ms=8, mfc="none", mec="red", mew=1)
    draw_events(bmap, events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="s", ms=8, mfc="none", mec="blue", mew=1)
    if options.write:
      plt.savefig("output/run_%d_sel3_visa.png" % (options.runid))
  
  
    bmap = draw_earth("Missed LEB events")
    missed_leb_idx = find_unmatched(leb_events, events)
    draw_events(bmap, leb_events[missed_leb_idx][:,[EV_LON_COL,EV_LAT_COL]],
                marker="*", ms=8, mfc="yellow")
    if options.write:
      plt.savefig("output/run_%d_missed.png" % (options.runid))


  #
  # draw an ROC curve
  #
  cursor = database.db.connect().cursor()
  cursor.execute("select orid, score from visa_origin where runid=%s",
                 (options.runid,))
  evscores = dict(cursor.fetchall())

  plt.figure()
  plt.title("Precision-Recall curve with LEB as ground truth")
  
  sel3_f, sel3_p, sel3_r, sel3_err = f1_and_error(leb_events, sel3_events)
  
  plt.plot([(sel3_p/100.0)], [(sel3_r/100.0)], label="SEL3",
           marker='o', ms=10, mec="red",
           linestyle="none", mfc="none", linewidth=3)

  if options.svm:
    x_pts, y_pts = compute_roc_curve(leb_events, sel3_events,
                                     read_sel3_svm_scores())
    
    plt.plot(x_pts, y_pts, label="SEL3 extrapolation", color="red",
             linestyle=":", linewidth=3)
    
  x_pts, y_pts = compute_roc_curve(leb_events, events, evscores)
    
  plt.plot(x_pts, y_pts, label=options.run_name, color="blue",
           linestyle="-", linewidth=3)

  if options.runid2 is not None:
    events2 = read_events(cursor, options.data_start, options.data_end,
                          "visa", options.runid2)[0]
    
    cursor.execute("select orid, score from visa_origin where runid=%s",
                   (options.runid2,))
    evscores2 = dict(cursor.fetchall())

    if options.suppress:
      events2 = suppress_duplicates(events2, evscores2)[0]
    
    x_pts, y_pts = compute_roc_curve(leb_events, events2, evscores2)
    
    plt.plot(x_pts, y_pts, label=options.run2_name, color="green",
             linestyle="--", linewidth=3)

  if options.runid3 is not None:
    events3 = read_events(cursor, options.data_start, options.data_end,
                          "visa", options.runid3)[0]
    
    cursor.execute("select orid, score from visa_origin where runid=%s",
                   (options.runid3,))
    evscores3 = dict(cursor.fetchall())

    if options.suppress:
      events3 = suppress_duplicates(events3, evscores3)[0]
    
    x_pts, y_pts = compute_roc_curve(leb_events, events3, evscores3)
    
    plt.plot(x_pts, y_pts, label=options.run3_name, color="green",
             linestyle=":", linewidth=3)
    
  plt.xlim(.39, 1)
  plt.ylim(.39, 1)
  plt.xlabel("precision")
  plt.ylabel("recall")
  plt.legend(loc = "upper right")
  plt.grid(True)
  if options.write:
    plt.savefig("output/run_%d_roc.png" % (options.runid))
  
  plt.show()

def suppress_duplicates(events, evscores):
  # we'll figure out which events to keep, initially we decide to keep
  # everything
  keep_event = np.ones(len(events), bool)

  for evnum1 in range(len(events)):
    # if we have already discarded this event (due a colliding earlier event)
    # then move on
    if not keep_event[evnum1]:
      continue

    # otherwise try to find all colliding future events
    for evnum2 in range(evnum1+1, len(events)):
      # since events are sorted by time if the following condition fails
      # then there is no future colliding event
      if abs(events[evnum1, EV_TIME_COL]
             - events[evnum2, EV_TIME_COL]) > 50:
        break

      # the two events collide
      if dist_deg(events[evnum1, [EV_LON_COL, EV_LAT_COL]],
                  events[evnum2, [EV_LON_COL, EV_LAT_COL]]) < 5:
        
        # keep the better of the two events
        if (evscores[int(events[evnum1, EV_ORID_COL])] >
            evscores[int(events[evnum2, EV_ORID_COL])]):
          keep_event[evnum2] = False
        else:
          keep_event[evnum1] = False
          break
  
  events = events[keep_event]
  
  # recompute orid2num
  orid2num = {}
  
  for ev in events:
    orid2num[ev[EV_ORID_COL]] = len(orid2num)
  
  return events, orid2num

def assert_time_sorted(events):
  for evnum, event in enumerate(events):
    if evnum < len(events)-1 \
       and events[evnum+1, EV_TIME_COL] < event[EV_TIME_COL]:
      assert(False)

def kleinermackey_match(gold_events, guess_events):
  # check events are sorted
  assert_time_sorted(gold_events)
  assert_time_sorted(guess_events)

  err_dist = np.array([np.inf for _ in range(len(gold_events))])
  is_true = np.array([0 for _ in range(len(guess_events))])
  
  gold_evnum_start = 0
  for guess_evnum in range(len(guess_events)):
    for gold_evnum in range(gold_evnum_start, len(gold_events)):
      if gold_events[gold_evnum, EV_TIME_COL]\
         < guess_events[guess_evnum, EV_TIME_COL]-40:
        gold_evnum_start = max(gold_evnum_start, gold_evnum+1)
        continue
      elif gold_events[gold_evnum, EV_TIME_COL]\
         > guess_events[guess_evnum, EV_TIME_COL]+40:
        break
      dist = dist_km(gold_events[gold_evnum, [EV_LON_COL, EV_LAT_COL]],
                     guess_events[guess_evnum, [EV_LON_COL, EV_LAT_COL]])
      if dist < 250:
        is_true[guess_evnum] = 1
        err_dist[gold_evnum] = min(dist, err_dist[gold_evnum])
  errs = err_dist[err_dist < np.inf]
  return (100. * is_true.sum() / len(guess_events),
          100. * len(errs) / len(gold_events),
          np.average(errs), np.std(errs))

def kleinermackey_display(leb_events, sel3_events, visa_events):
  sel3_p, sel3_r, sel3_err, sel3_std \
          = kleinermackey_match(leb_events, sel3_events)
  visa_p, visa_r, visa_err, visa_std \
          = kleinermackey_match(leb_events, visa_events)
  
  print "Kleiner & Mackey criteria (any event in 250km, 40s ball):"
  print "SEL3: Precision %.1f, Recall %.1f, Avg Err %.1f, Std Err %.1f" %\
        (sel3_p, sel3_r, sel3_err, sel3_std)
  print "VISA: Precision %.1f, Recall %.1f, Avg Err %.1f, Std Err %.1f" %\
        (visa_p, visa_r, visa_err, visa_std)
  
def main():
  parser = OptionParser()
  parser.add_option("-i", "--runid", dest="runid", default=None,
                    type="int",
                    help = "the run-identifier to analyze (last runid)")

  parser.add_option("--run_name", dest="run_name", default="NET-VISA",
                    help = "the name of the run (NET-VISA)")

  parser.add_option("--runid2", dest="runid2", default=None,
                    type="int",
                    help = "the second run-identifier to analyze")
  
  parser.add_option("--run2_name", dest="run2_name", default="NET-VISA2",
                    help = "the name of run2 (NET-VISA2)")

  parser.add_option("--runid3", dest="runid3", default=None,
                    type="int",
                    help = "the third run-identifier to analyze")
  
  parser.add_option("--run3_name", dest="run3_name", default="NET-VISA3",
                    help = "the name of run3 (NET-VISA3)")


  parser.add_option("-m", "--maxtime", dest="maxtime", default=None,
                    type="float",
                    help = "Maximum time to analyze for")
  
  parser.add_option("-v", "--verbose", dest="verbose", default=False,
                    action = "store_true",
                    help = "verbose output (False)")

  parser.add_option("-1", "--type1", dest="type1", default=False,
                    action = "store_true",
                    help = "Type 1 fonts (False)")

  parser.add_option("-b", "--mag", dest="mag", default=False,
                    action = "store_true",
                    help = "analyze by magnitude (False)")
  
  parser.add_option("-d", "--detcnt", dest="detcnt", default=False,
                    action = "store_true",
                    help = "analyze by number of timedef detections (False)")

  parser.add_option("--missdet", dest="missdet", default=False,
                    action = "store_true",
                    help = "analyze by number of missed detections (False)")
  
  parser.add_option("-t", "--tphase", dest="tphase", default=False,
                    action = "store_true",
                    help = "analyze by number of T phases (False)")

  parser.add_option("-y", "--hphase", dest="hphase", default=False,
                    action = "store_true",
                    help = "analyze by number of H phases (False)")
  
  parser.add_option("-a", "--az", dest="azgap", default=False,
                    action = "store_true",
                    help = "analyze by azimuth gap (False)")

  parser.add_option("-g", "--gui", dest="gui", default=False,
                    action = "store_true",
                    help = "graphically display run (False)")

  parser.add_option("-s", "--suppress", dest="suppress", default=False,
                    action = "store_true",
                    help = "suppress duplicates (False)")

  parser.add_option("-n", "--NEIC", dest="neic", default=False,
                    action = "store_true",
                    help = "compare with NEIC events (False)")

  parser.add_option("-j", "--JMA", dest="jma", default=False,
                    action = "store_true",
                    help = "compare with JMA events (False)")

  parser.add_option("-z", "--svm", dest="svm", default=False,
                    action = "store_true",
                    help = "use svm scores to improve SEL3 (False)")

  parser.add_option("-e", "--error", dest="error", default=False,
                    action = "store_true",
                    help = "compute the error of VISA and SEL3 on "
                    "LEB events predicted by both")

  parser.add_option("-w", "--write", dest="write", default=False,
                    action = "store_true",
                    help = "write the results to output/ sub-directory")

  parser.add_option("--events", dest="events", default=False,
                    action = "store_true",
                    help="draw predicted events (False)")

  parser.add_option("-p", "--phase", dest="phase", default=False,
                    action = "store_true",
                    help = "analyze predictions by labels (False)")

  parser.add_option("-k", "--kleinermackey", dest="kleinermackey",
                    default=False, action = "store_true",
                    help = "Use Kleiner&Mackey 250km, 40s criteria, "
                    "with no matching")
  
  (options, args) = parser.parse_args()

  cursor = database.db.connect().cursor()

  if options.runid is None:
    cursor.execute("select max(runid) from visa_run")
    options.runid, = cursor.fetchone()
  
  print "RUNID %d:" % options.runid,

  cursor.execute("select run_start, run_end, data_start, data_end, descrip, "
                 "numsamples, window, step from visa_run where runid=%s",
                 options.runid)
  run_start, run_end, data_start, data_end, descrip, numsamples, window, step\
             = cursor.fetchone()

  if data_end is None:
    print "NO RESULTS"
    return

  if options.maxtime is not None:
    data_end = options.maxtime

  options.data_start, options.data_end = data_start, data_end
  
  print "%.1f - %.1f (%.1f hrs), runtime %s" \
        % (data_start, data_end, (data_end-data_start) / 3600.,
           str(run_end - run_start))
  
  print "D='%s' N=%d W=%s S=%s" % (descrip, numsamples, str(window), str(step))

  detections, arid2num = read_detections(cursor, data_start, data_end)
  
  leb_events, leb_orid2num = read_events(cursor, data_start, data_end, "leb")

  sel3_events, sel3_orid2num = read_events(cursor, data_start, data_end,
                                           "sel3")
  
  visa_events, visa_orid2num = read_events(cursor, data_start, data_end,
                                           "visa", options.runid)

  if options.neic:
    neic_events = read_isc_events(cursor, data_start, data_end, "NEIC")
  else:
    neic_events = None

  if options.jma:
    jma_events = read_isc_events(cursor, data_start, data_end, "JMA")
  else:
    jma_events = None
    
  if options.suppress:
    cursor.execute("select orid, score from visa_origin where runid=%s",
                   (options.runid,))
    visa_scores = dict(cursor.fetchall())    
    visa_events, visa_orid2num = suppress_duplicates(visa_events, visa_scores)

  leb_evlist = read_assoc(cursor, data_start, data_end, leb_orid2num,
                          arid2num, "leb")
  visa_evlist = read_assoc(cursor, data_start, data_end, visa_orid2num,
                           arid2num, "visa", runid=options.runid)  

  # use Type 1 fonts by invoking latex
  if options.type1:
    plt.rcParams['text.usetex'] = True
    
  if options.kleinermackey:
    kleinermackey_display(leb_events, sel3_events, visa_events)
    
  if options.phase:
    leb_visa = find_matching(leb_events, visa_events)
    buckets = [[] for b in MAG_RANGES]
    all = []
    for (leb_evnum, visa_evnum) in leb_visa:
      for bnum, (mag_low, mag_high) in enumerate(MAG_RANGES):
        if leb_events[leb_evnum, EV_MB_COL] > mag_low \
           and leb_events[leb_evnum, EV_MB_COL] <= mag_high:
          diff = len(visa_evlist[visa_evnum]) - len(leb_evlist[leb_evnum])
          buckets[bnum].append(diff)
          all.append(diff)
          break
    for bnum, (mag_low, mag_high) in enumerate(MAG_RANGES):
      print " %d -- %d | %3d | %3d %3d" %\
            (mag_low, mag_high, len(buckets[bnum]),
             np.average(buckets[bnum]), np.std(buckets[bnum]))
    print " all  | %3d | %3d %3d" % (len(all), np.average(all),
                                     np.std(all))
    
  if options.error:
    leb_sel3 = find_matching(leb_events, sel3_events)
    leb_visa = find_matching(leb_events, visa_events)
    # leb events common to both
    common_leb = list(set([x for (x,y) in leb_sel3])
                      & set([x for (x,y) in leb_visa]))
    common_leb_sel3 = dict((leb_evnum, sel3_evnum) for (leb_evnum, sel3_evnum)
                           in leb_sel3 if leb_evnum in common_leb)
    common_leb_visa = dict((leb_evnum, visa_evnum) for (leb_evnum, visa_evnum)
                           in leb_visa if leb_evnum in common_leb)
    
    buckets = [([], []) for b in MAG_RANGES]
    all = ([], [])
    for evnum in common_leb:
      for bnum, (mag_low, mag_high) in enumerate(MAG_RANGES):
        if leb_events[evnum, EV_MB_COL] > mag_low \
           and leb_events[evnum, EV_MB_COL] <= mag_high:
          leb_ev = leb_events[evnum]
          sel3_ev = sel3_events[common_leb_sel3[evnum]]
          visa_ev = visa_events[common_leb_visa[evnum]]
          sel3_dist = dist_km((leb_ev[EV_LON_COL], leb_ev[EV_LAT_COL]),
                              (sel3_ev[EV_LON_COL], sel3_ev[EV_LAT_COL]))
          visa_dist = dist_km((leb_ev[EV_LON_COL], leb_ev[EV_LAT_COL]),
                              (visa_ev[EV_LON_COL], visa_ev[EV_LAT_COL]))
          buckets[bnum][0].append(sel3_dist)
          buckets[bnum][1].append(visa_dist)
          all[0].append(sel3_dist)
          all[1].append(visa_dist)
          break
      else:
        raise ValueError("Event mag %f not found in any mag range"
                         % leb_events[evnum, EV_MB_COL])
      
    print "%d leb events detected by both sel3 and visa" % len(common_leb)
    print "   mb   | #ev |  SEL3   |  VISA"
    print "        |     | err  sd | err  sd"
    print "---------------------------------"
    for bnum, (mag_low, mag_high) in enumerate(MAG_RANGES):
      print " %d -- %d | %3d | %3d %3d | %3d %3d" \
            % (mag_low, mag_high, len(buckets[bnum][0]),
               np.average(buckets[bnum][0]), np.std(buckets[bnum][0]),
               np.average(buckets[bnum][1]), np.std(buckets[bnum][1]))
    print "   all  | %3d | %3d %3d | %3d %3d" \
          % (len(all[0]),
             np.average(all[0]), np.std(all[0]),
             np.average(all[1]), np.std(all[1]))


  if options.mag:
    analyze_by_attr("mb", MAG_RANGES, [ev[EV_MB_COL] for ev in leb_events],
                    leb_events, sel3_events, visa_events, options.verbose)

  if options.detcnt:
    detcnts = []
    for leb_event in leb_events:
      cursor.execute("select count(*) from leb_assoc join idcx_arrival_net "
                     "using (arid) where orid=%s and timedef='d'",
                     (int(leb_event[EV_ORID_COL]),))
      detcnts.append(cursor.fetchone()[0])
      
    analyze_by_attr("# Det", DETCNT_RANGES, detcnts,
                    leb_events, sel3_events, visa_events, options.verbose)

  if options.missdet:
    detcnts = []
    for leb_event in leb_events:
      cursor.execute("select count(*) from leb_assoc  "
                     "where orid=%s and timedef='d' and arid not in "
                     "(select arid from idcx_arrival_net)",
                     (int(leb_event[EV_ORID_COL]),))
      detcnts.append(cursor.fetchone()[0])
      
    analyze_by_attr("# Mis", DETCNT_RANGES, detcnts,
                    leb_events, sel3_events, visa_events, options.verbose)

  if options.tphase:
    tcnts = []
    for leb_event in leb_events:
      cursor.execute("select count(*) from leb_assoc "
                     "where orid=%s and phase='T'",
                     (int(leb_event[EV_ORID_COL]),))
      tcnts.append(cursor.fetchone()[0])
    
    analyze_by_attr("T Phases", TPHASE_RANGES, tcnts,
                    leb_events, sel3_events, visa_events, options.verbose)

  if options.hphase:
    hcnts = []
    for leb_event in leb_events:
      cursor.execute("select count(*) from leb_assoc "
                     "where orid=%s and phase='H'",
                     (int(leb_event[EV_ORID_COL]),))
      hcnts.append(cursor.fetchone()[0])
    
    analyze_by_attr("H Phases", HPHASE_RANGES, hcnts,
                    leb_events, sel3_events, visa_events, options.verbose)

  if options.azgap:
    # compute the azimuth gaps for each leb event
    leb_azgaps = [None for _ in leb_events]
    for (leb_orid, leb_evnum) in leb_orid2num.iteritems():
      cursor.execute("select esaz from leb_assoc join leb_arrival "
                     "using(arid,sta) where orid=%s and "
                     "timedef='d' and time between %s and %s",
                     (leb_orid, data_start, data_end))
      esazlist = [x for (x,) in cursor.fetchall()]
      leb_azgaps[leb_evnum] = azimuth_gap(esazlist)

    analyze_by_attr("AZIM. GAP", AZGAP_RANGES, leb_azgaps,
                    leb_events, sel3_events, visa_events, options.verbose)

  print "=" * 74
  print "            |     |  F1     P     R   err  sd |   F1    P     R "

  # finally, compute the overall scores
  sel3_f, sel3_p, sel3_r, sel3_err = f1_and_error(leb_events, sel3_events)
  
  visa_f, visa_p, visa_r, visa_err = f1_and_error(leb_events, visa_events)
  
  print "=" * 74
  print ("     --     | %3d | %5.1f %5.1f %5.1f %3.0f %3.0f " \
         "| %5.1f %5.1f %5.1f %3.0f %3.0f")\
         % (len(leb_events), sel3_f, sel3_p, sel3_r,
            sel3_err[0], sel3_err[1],
            visa_f, visa_p, visa_r, visa_err[0], visa_err[1])
  print "=" * 74


  if options.neic:
    neic_us_events = filter_by_lonlat(neic_events, US_LON_1, US_LON_2,
                                      US_LAT_1, US_LAT_2)
    leb_us_events = filter_by_lonlat(leb_events, US_LON_1, US_LON_2,
                                      US_LAT_1, US_LAT_2)
    sel3_us_events = filter_by_lonlat(sel3_events, US_LON_1, US_LON_2,
                                      US_LAT_1, US_LAT_2)
    visa_us_events = filter_by_lonlat(visa_events, US_LON_1, US_LON_2,
                                      US_LAT_1, US_LAT_2)
    
    leb_f, leb_p, leb_r, leb_err = f1_and_error(neic_us_events, leb_us_events)
    
    visa_f, visa_p, visa_r, visa_err = f1_and_error(neic_us_events,
                                                    visa_us_events)

    leb_recalled = find_matching(neic_us_events, leb_us_events)
    sel3_recalled = find_matching(neic_us_events, sel3_us_events)
    visa_recalled = find_matching(neic_us_events, visa_us_events)
    
    print "LEB recall", [(int(neic_us_events[i, EV_ORID_COL]),
                          int(leb_us_events[j, EV_ORID_COL]))
                         for (i,j) in leb_recalled]
    print "SEL3 recall",[(int(neic_us_events[i, EV_ORID_COL]),
                          int(sel3_us_events[j, EV_ORID_COL]))
                         for (i,j) in sel3_recalled]
    print "VISA recall",[(int(neic_us_events[i, EV_ORID_COL]),
                          int(visa_us_events[j, EV_ORID_COL]))
                         for (i,j) in visa_recalled]
    
    print "NEIC (US):         |          LEB              |          VISA"
    print "=" * 74
    print ("     --     | %3d | %5.1f %5.1f %5.1f %3.0f %3.0f " \
           "| %5.1f %5.1f %5.1f %3.0f %3.0f")\
           % (len(neic_us_events), leb_f, leb_p, leb_r,
              leb_err[0], leb_err[1],
              visa_f, visa_p, visa_r, visa_err[0], visa_err[1])
    print "=" * 74

    if options.gui is not None:
      bmap = draw_earth("NEIC(orange), LEB(yellow), and NET-VISA(blue)",
                        projection="mill",
                        resolution="l",
                        llcrnrlon = US_LON_1, urcrnrlon = US_LON_2,
                        llcrnrlat = US_LAT_1, urcrnrlat = US_LAT_2)
      draw_events(bmap, leb_us_events[:,[EV_LON_COL, EV_LAT_COL]],
                  marker="o", ms=10, mfc="none", mec="yellow", mew=1)
      draw_events(bmap, visa_us_events[:,[EV_LON_COL, EV_LAT_COL]],
                  marker="s", ms=10, mfc="none", mec="blue", mew=1)
      draw_events(bmap, neic_us_events[:,[EV_LON_COL, EV_LAT_COL]],
                  marker="*", ms=10, mfc="orange")

      cursor = database.db.connect().cursor()
      cursor.execute("select orid, score from visa_origin where runid=%s",
                     (options.runid,))
      evscores = dict(cursor.fetchall())
    
      plt.figure()
      plt.title("ROC curve with NEIC as ground truth over the Continental US")
    
      sel3_f, sel3_p, sel3_r, sel3_err = f1_and_error(neic_us_events,
                                                      sel3_us_events)
      
      plt.plot([(sel3_p/100.0)], [(sel3_r/100.0)], label="SEL3",
               marker='o', ms=10, mec="red",
               linestyle="none", mfc="none")

      plt.plot([(leb_p/100.0)], [(leb_r/100.0)], label="LEB",
               marker='o', ms=10, mec="yellow",
               linestyle="none", mfc="none")

      x_pts, y_pts = compute_roc_curve(neic_us_events, visa_us_events,
                                       evscores, freq=1)
      
      plt.plot(x_pts, y_pts, label="NetVISA", color="blue")
      
      plt.xlim(0, 1)
      plt.ylim(0, 1)
      plt.xlabel("precision")
      plt.ylabel("recall")
      plt.legend(loc = "upper right")
      plt.grid(True)

      

  if options.jma:
    leb_f, leb_p, leb_r, leb_err = f1_and_error(jma_events, leb_events)
    
    visa_f, visa_p, visa_r, visa_err = f1_and_error(jma_events, visa_events)
    
    print "JMA:              |          LEB              |          VISA"
    print "=" * 74
    print ("     --     | %3d | %5.1f %5.1f %5.1f %3.0f %3.0f " \
           "| %5.1f %5.1f %5.1f %3.0f %3.0f")\
           % (len(jma_events), leb_f, leb_p, leb_r,
              leb_err[0], leb_err[1],
              visa_f, visa_p, visa_r, visa_err[0], visa_err[1])
    print "=" * 74

    if jma_events is not None:
      bmap = draw_earth("JMA (orange), LEB (yellow), and NET-VISA (blue)",
                        projection="mill",
                        resolution="l",
                        llcrnrlon = JAPAN_LON_1, urcrnrlon = JAPAN_LON_2,
                        llcrnrlat = JAPAN_LAT_1, urcrnrlat = JAPAN_LAT_2)
      draw_events(bmap, leb_events[:,[EV_LON_COL, EV_LAT_COL]],
                  marker="o", ms=10, mfc="none", mec="yellow", mew=1)
      draw_events(bmap, events[:,[EV_LON_COL, EV_LAT_COL]],
                  marker="s", ms=10, mfc="none", mec="blue", mew=1)
      draw_events(bmap, jma_events[:,[EV_LON_COL, EV_LAT_COL]],
                  marker="*", ms=10, mfc="orange")
    
  if options.gui:
    gui(options, leb_events, sel3_events, visa_events)

if __name__ == "__main__":
  main()

