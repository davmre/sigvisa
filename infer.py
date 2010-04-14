import os, sys
import numpy as np
from optparse import OptionParser

from database.dataset import *
import database.db
import netvisa, learn
from results.compare import *
from utils.geog import dist_km, degdiff

def analyze_leb(netmodel, earthmodel, leb_events, leb_evlist, detections):
  inv_evs = []
  inv_detnums = []
  inv_arids = []
  for detnum in range(len(detections)):
    ev = netmodel.invert_det(detnum)
    if ev is not None:
      inv_evs.append(ev)
      inv_detnums.append(detnum)
      inv_arids.append(detections[detnum, DET_ARID_COL])

  for evnum in range(len(leb_events)):
    leb_event = leb_events[evnum]
    leb_detlist = leb_evlist[evnum]
    print "%d: %.1f E %.1f N %.0f km %.0f s %.1f mb" % (
      leb_event[EV_ORID_COL], leb_event[EV_LON_COL], leb_event[EV_LAT_COL],
      leb_event[EV_DEPTH_COL], leb_event[EV_TIME_COL], leb_event[EV_MB_COL]),
    
    print "INV:",
    for invnum, (evlon, evlat, evdepth, evtime) in enumerate(inv_evs):
      detnum = inv_detnums[invnum]
      if abs(leb_event[EV_TIME_COL] - evtime) < 50 \
         and dist_deg((evlon, evlat),
                      (leb_event[EV_LON_COL], leb_event[EV_LAT_COL])) < 5:
        print "(%d, %s)" % (detnum,
                            earthmodel.PhaseName(int(detections[detnum,
                                                         DET_PHASE_COL]))),
    print
  
def print_events(netmodel, earthmodel, leb_events, leb_evlist):
  print "=" * 60
  score = 0
  for evnum in range(len(leb_events)):
    score += print_event(netmodel, earthmodel, leb_events[evnum],
                         leb_evlist[evnum])
    print "-" * 60
  print "Total: %.1f" % score
  print "=" * 60
  
def print_event(netmodel, earthmodel, event, event_detlist):
  print ("Event: lon %4.2f lat %4.2f depth %3.1f mb %1.1f time %.1f orid %d"
         % (event[ EV_LON_COL], event[ EV_LAT_COL],
            event[ EV_DEPTH_COL], event[ EV_MB_COL],
            event[ EV_TIME_COL], event[ EV_ORID_COL]))
  print "Detections:",
  detlist = [x for x in event_detlist]
  detlist.sort()
  for phaseid, detid in detlist:
    print "(%s, %d)" % (earthmodel.PhaseName(phaseid), detid),
  print
  score = netmodel.score_event(event, event_detlist)
  print "Ev Score: %.1f" % score
  return score

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-n", "--numsamples", dest="numsamples", default=1,
                    type="int",
                    help = "number of samples per second of input data (1)")
  parser.add_option("-r", "--hours", dest="hours", default=None,
                    type="float",
                    help = "inference on HOURS worth of data (all)")
  parser.add_option("-k", "--skip", dest="skip", default=0,
                    type="float",
                    help = "skip the first HOURS of data (0)")
  parser.add_option("-s", "--seed", dest="seed", default=123456789,
                    type="int",
                    help = "random number generator seed (123456789)")
  parser.add_option("-x", "--text", dest="gui", default=True,
                    action = "store_false",
                    help = "text only output (False)")
  parser.add_option("-v", "--verbose", dest="verbose", default=False,
                    action = "store_true",
                    help = "verbose output (False)")
  parser.add_option("-d", "--descrip", dest="descrip", default="",
                    help = "description of the run ('')")
  (options, args) = parser.parse_args()

  netvisa.srand(options.seed)
  
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data("validation", hours=options.hours, skip=options.skip)

  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  if options.verbose:
    print "===="
    print "LEB:"
    print "===="
    analyze_leb(netmodel, earthmodel, leb_events, leb_evlist, detections)
  
  #print_events(netmodel, earthmodel, leb_events, leb_evlist)
  #netmodel.score_world(leb_events, leb_evlist, 1)

  #print "===="
  #print "SEL3:"
  #print "===="
  #print_events(netmodel, earthmodel, sel3_events, sel3_evlist)
  #netmodel.score_world(sel3_events, sel3_evlist, 1)

  # create a runid
  cursor = database.db.connect().cursor()
  cursor.execute ("insert into visa_run(run_start, numsamples, seed, "
                  "data_start, data_end, descrip) values "
                  "(now(), %s, %s, %s, %s, %s)",
                  (options.numsamples, options.seed, start_time, end_time,
                   options.descrip))
  cursor.execute("select max(runid) from visa_run")
  runid, = cursor.fetchone()

  print "===="
  print "NET runid %d" % runid
  print "===="  
  events, ev_detlist = netmodel.infer(options.numsamples)
  #print_events(netmodel, earthmodel, events, ev_detlist)

  # store the events and associations
  cursor = database.db.connect().cursor()
  world_score = 0.0
  for evnum in range(len(events)):
    event = events[evnum]
    detlist = ev_detlist[evnum]
    evscore = netmodel.score_event(event, detlist)
    world_score += evscore
    
    cursor.execute("insert into visa_origin (runid, orid, lon, lat, depth, "
                   "time, mb, score) values (%s, %s, %s, %s, %s, %s, %s, %s)",
                   (runid, event[EV_ORID_COL], event[EV_LON_COL],
                    event[EV_LAT_COL], event[EV_DEPTH_COL], event[EV_TIME_COL],
                    event[EV_MB_COL], evscore))
    
    for phaseid, detnum in detlist:
      arrtime = earthmodel.ArrivalTime(event[EV_LON_COL], event[EV_LAT_COL],
                                       event[EV_DEPTH_COL], event[EV_TIME_COL],
                                       phaseid,
                                       int(detections[detnum, DET_SITE_COL]))
      if arrtime < 0:
        print "Warning: visa orid %d impossible at site %d with phase %d"\
              % (event[EV_ORID_COL], int(detections[detnum, DET_SITE_COL]),
                 phaseid)
        continue

      timeres = detections[detnum, DET_TIME_COL] - arrtime
      
      seaz = earthmodel.ArrivalAzimuth(event[EV_LON_COL], event[EV_LAT_COL],
                                       int(detections[detnum, DET_SITE_COL]))
      azres = degdiff(seaz, detections[detnum, DET_AZI_COL])

      arrslo = earthmodel.ArrivalSlowness(event[EV_LON_COL], event[EV_LAT_COL],
                                          event[EV_DEPTH_COL], phaseid,
                                       int(detections[detnum, DET_SITE_COL]))
      
      slores = detections[detnum, DET_SLO_COL] - arrslo
      
      cursor.execute("insert into visa_assoc(runid, orid, phase, arid, score, "
                     "timeres, azres, slores) "
                     "values (%s, %s, %s, %s, %s, %s, %s, %s)",
                     (runid, event[EV_ORID_COL],
                      earthmodel.PhaseName(phaseid),
                      detections[detnum, DET_ARID_COL],
                      netmodel.score_event_det(event, phaseid, detnum),
                      timeres, azres, slores))
  
  # store the results
  f, p, r, err = f1_and_error(leb_events, events)

  cursor.execute("update visa_run set run_end=now(), f1=%s, prec=%s, "
                 "recall=%s, error_avg=%s, error_sd=%s, score=%s "
                 "where runid=%s",
                 (f, p, r, err[0], err[1], world_score, runid))

  msg = ("F1=%.2f, Prec=%.2f, Recall=%.2f, Avg Error = %.2f+-%.2f"
         % (f, p, r, err[0], err[1]))
  

  if options.verbose:      
    print "===="
    print "EVAL"
    print "===="  
    true_idx, false_idx, mat_idx = find_true_false_guess(leb_events, events)
    unmat_idx = [x for x in range(len(leb_events))]
    print "Matched:"
    for lebevnum, evnum in mat_idx:
      print "/--"
      print_event(netmodel, earthmodel, leb_events[lebevnum],
                  leb_evlist[lebevnum])
      print "~~ %.1f km ~~>" % dist_km(leb_events[lebevnum,
                                                  [EV_LON_COL, EV_LAT_COL]],
                                       events[evnum, [EV_LON_COL, EV_LAT_COL]])
      print_event(netmodel, earthmodel, events[evnum], ev_detlist[evnum])
      print "\--"
      unmat_idx.remove(lebevnum)
    if len(unmat_idx):
      print "Unmatched:"
    for lebevnum in unmat_idx:
      print "--"
      print_event(netmodel, earthmodel, leb_events[lebevnum],
                  leb_evlist[lebevnum])
      print "--"
    if len(false_idx):
      print "Spurious:"
    for evnum in false_idx:
      print "~~"
      print_event(netmodel, earthmodel, events[evnum], ev_detlist[evnum])
      print "~~"
      
  print(msg)

if __name__ == "__main__":
  main("parameters")
