# interpret the results of a run
import os, sys
import numpy as np
from optparse import OptionParser

from database.dataset import *
from results.compare import f1_and_error
from utils.geog import degdiff, dist_deg
import database.db

AZGAP_RANGES = [(0, 90), (90, 180), (180, 270), (270, 360)]

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

def split_by_azgap(azgaps):
  ev_buckets = [[] for _ in AZGAP_RANGES]
  for evnum, azgap in enumerate(azgaps):
    for bucnum, (azlo, azhi) in enumerate(AZGAP_RANGES):
      if azgap > azlo and azgap <= azhi:
        ev_buckets[bucnum].append(evnum)
  return ev_buckets
                                                   
def main():
  parser = OptionParser()
  parser.add_option("-i", "--runid", dest="runid", default=None,
                    type="int",
                    help = "the run-identifier to analyze (last runid)")

  (options, args) = parser.parse_args()

  analyze_runid(options.runid)

def analyze_runid(runid):
  cursor = database.db.connect().cursor()

  if runid is None:
    cursor.execute("select max(runid) from visa_run")
    runid, = cursor.fetchone()
  
  print "Analyzing RUNID %d:" % runid,

  cursor.execute("select data_start, data_end from visa_run where runid=%s",
                 runid)
  data_start, data_end = cursor.fetchone()

  print "%.1f -- %.1f  (%.1f hrs)," % (data_start, data_end,
                                        (data_end-data_start) / 3600.),

  leb_events, leb_orid2num = read_events(cursor, data_start, data_end, "leb")

  sel3_events, sel3_orid2num = read_events(cursor, data_start, data_end,
                                           "sel3")
  
  visa_events, visa_orid2num = read_events(cursor, data_start, data_end,
                                           "visa", runid)

  print "%d leb events" % len(leb_events)

  # compute the azimuth gaps for each leb event
  leb_azgaps = [None for _ in leb_events]
  for (leb_orid, leb_evnum) in leb_orid2num.iteritems():
    cursor.execute("select esaz from leb_assoc join leb_arrival "
                   "using(arid,sta) where orid=%s and "
                   "timedef='d' and time between %s and %s",
                   (leb_orid, data_start, data_end))
    esazlist = [x for (x,) in cursor.fetchall()]
    leb_azgaps[leb_evnum] = azimuth_gap(esazlist)
  
  sel3_azgaps = [leb_azgaps[x] for x in find_nearest(leb_events, sel3_events)]
  visa_azgaps = [leb_azgaps[x] for x in find_nearest(leb_events, visa_events)]

  # divide the events by azimuth gap
  leb_buckets = split_by_azgap(leb_azgaps)
  sel3_buckets = split_by_azgap(sel3_azgaps)
  visa_buckets = split_by_azgap(visa_azgaps)

  print "-" * 74
  print "  AZIM. GAP | #ev |          SEL3             |          VISA"
  print "            |     |  F1     P     R   err  sd |   F1    P     R "\
        "  err  sd"
  print "-" * 74
  for i, (azlo, azhi) in enumerate(AZGAP_RANGES):
    sel3_f, sel3_p, sel3_r, sel3_err = f1_and_error(
      leb_events[leb_buckets[i],:], sel3_events[sel3_buckets[i], :])

    visa_f, visa_p, visa_r, visa_err = f1_and_error(
      leb_events[leb_buckets[i],:], visa_events[visa_buckets[i], :])
    
    print (" %3d -- %3d | %3d | %5.1f %5.1f %5.1f %3.0f %3.0f " \
          "| %5.1f %5.1f %5.1f %3.0f %3.0f")\
          % (azlo, azhi, len(leb_buckets[i]), sel3_f, sel3_p, sel3_r,
             sel3_err[0], sel3_err[1],
             visa_f, visa_p, visa_r, visa_err[0], visa_err[1])
    
    
  
if __name__ == "__main__":
  main()

