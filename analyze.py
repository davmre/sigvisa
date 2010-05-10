# interpret the results of a run
import os, sys
import numpy as np
from optparse import OptionParser

from database.dataset import *
from results.compare import *
from utils.geog import degdiff, dist_deg
import database.db

AZGAP_RANGES = [(0, 90), (90, 180), (180, 270), (270, 360)]
DETCNT_RANGES = [(2, 3), (3,4), (4,5), (5,6), (6, 100)]
TPHASE_RANGES = [(-1, 0), (0, 100)]
MAG_RANGES = [(0,2), (2,3), (3,4), (4,9)]

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

def main():
  parser = OptionParser()
  parser.add_option("-i", "--runid", dest="runid", default=None,
                    type="int",
                    help = "the run-identifier to analyze (last runid)")

  parser.add_option("-m", "--maxtime", dest="maxtime", default=None,
                    type="float",
                    help = "Maximum time to analyze for")
  
  parser.add_option("-v", "--verbose", dest="verbose", default=False,
                    action = "store_true",
                    help = "verbose output (False)")

  parser.add_option("-g", "--mag", dest="mag", default=False,
                    action = "store_true",
                    help = "analyze by magnitude (False)")
  
  parser.add_option("-d", "--detcnt", dest="detcnt", default=False,
                    action = "store_true",
                    help = "analyze by number of timedef detections (False)")
  
  parser.add_option("-t", "--tphase", dest="tphase", default=False,
                    action = "store_true",
                    help = "analyze by number of T phases (False)")
  
  parser.add_option("-a", "--az", dest="azgap", default=False,
                    action = "store_true",
                    help = "analyze by azimuth gap (False)")

  (options, args) = parser.parse_args()

  cursor = database.db.connect().cursor()

  if options.runid is None:
    cursor.execute("select max(runid) from visa_run")
    runid, = cursor.fetchone()
  
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
  
  print "%.1f - %.1f (%.1f hrs), runtime %s" \
        % (data_start, data_end, (data_end-data_start) / 3600.,
           str(run_end - run_start))
  
  print "D='%s' N=%d W=%s S=%s" % (descrip, numsamples, str(window), str(step))
  
  leb_events, leb_orid2num = read_events(cursor, data_start, data_end, "leb")

  sel3_events, sel3_orid2num = read_events(cursor, data_start, data_end,
                                           "sel3")
  
  visa_events, visa_orid2num = read_events(cursor, data_start, data_end,
                                           "visa", options.runid)

  if options.mag:
    analyze_by_attr("mb", MAG_RANGES, [ev[EV_MB_COL] for ev in leb_events],
                    leb_events, sel3_events, visa_events, options.verbose)

  if options.detcnt:
    detcnts = []
    for leb_event in leb_events:
      cursor.execute("select count(*) from leb_assoc "
                     "where orid=%s and timedef='d'",
                     (int(leb_event[EV_ORID_COL]),))
      detcnts.append(cursor.fetchone()[0])
      
    analyze_by_attr("# Det", DETCNT_RANGES, detcnts,
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
    
  
if __name__ == "__main__":
  main()

