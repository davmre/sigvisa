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
#checks whether events generated from the proposer are good enough

import os, sys, time
import numpy as np
from optparse import OptionParser
import random

from sigvisa.database.dataset import *
import netvisa, learn
from sigvisa.utils.geog import dist_deg

def print_events_simple(netmodel, earthmodel, detections, events,
                        event_detlists, title, evstart_time, evend_time):
  print "****", title, "****"
  for evnum, event in enumerate(events):
    if event[EV_TIME_COL] > evstart_time and event[EV_TIME_COL] < evend_time:
      print_event_simple(netmodel, earthmodel, detections, event,
                         event_detlists[evnum])
  print "=" * 78

def print_event_simple(netmodel, earthmodel, detections, event, event_detlist):
  print ("%d: lon %4.2f lat %4.2f depth %3.1f mb %1.1f time %.1f"
         % (event[ EV_ORID_COL], event[ EV_LON_COL], event[ EV_LAT_COL],
            event[ EV_DEPTH_COL], event[ EV_MB_COL],
            event[ EV_TIME_COL]))
  print "Detections:",
  detlist = [x for x in event_detlist]
  detlist.sort()
  for phaseid, detid in detlist:
    tres = earthmodel.ArrivalTime(event[EV_LON_COL], event[EV_LAT_COL],
                                  event[EV_DEPTH_COL], event[EV_TIME_COL],
                                  phaseid,
                                  int(detections[detid, DET_SITE_COL]))\
                                  - detections[detid, DET_TIME_COL]

    print "(%s, %d, %s, tres %.1f, sc %.1f)" \
          % (earthmodel.PhaseName(phaseid), detid,
             earthmodel.PhaseName(int(detections[detid, DET_PHASE_COL])),
             tres, netmodel.score_event_det(event, phaseid, detid)),
  print "\nEv Score:", netmodel.score_event(event, event_detlist)

def print_events(netmodel, earthmodel, detections, events, event_detlists,
                 title):
  print "****", title, "****"
  for evnum, event in enumerate(events):
    print_event(netmodel, earthmodel, detections, event,
                event_detlists[evnum])
  print "=" * 78


def print_event(netmodel, earthmodel, detections, event, event_detlist):
  print ("%d: lon %4.2f lat %4.2f depth %3.1f mb %1.1f time %.1f"
         % (event[ EV_ORID_COL], event[ EV_LON_COL], event[ EV_LAT_COL],
            event[ EV_DEPTH_COL], event[ EV_MB_COL],
            event[ EV_TIME_COL]))
  print "Detections:",
  detlist = [x for x in event_detlist]
  detlist.sort()
  for phaseid, detid in detlist:
    inv = netmodel.invert_det(detid, 0)
    if inv is None:
      inv_ev = None
      inv_score = -np.inf
      inv_dist = np.inf
      tres = np.inf
    else:
      inv_ev = event.copy()
      inv_ev[[EV_LON_COL, EV_LAT_COL, EV_DEPTH_COL, EV_TIME_COL]] = inv
      # try inverting as a magnitude 3 or a magnitude 4
      inv_ev[EV_MB_COL] = 3.0
      inv_score3 = netmodel.score_event(inv_ev, event_detlist)
      inv_ev[EV_MB_COL] = 4.0
      inv_score4 = netmodel.score_event(inv_ev, event_detlist)
      inv_score = max(inv_score3, inv_score4)
      inv_dist = dist_deg((event[EV_LON_COL], event[EV_LAT_COL]),
                          (inv_ev[EV_LON_COL], inv_ev[EV_LAT_COL]))
      tres = earthmodel.ArrivalTime(event[EV_LON_COL], event[EV_LAT_COL],
                                    0, event[EV_TIME_COL],
                                    phaseid,
                                    int(detections[detid, DET_SITE_COL]))\
                                    - detections[detid, DET_TIME_COL]

    print "(%s, %d, %s, detsc %.1f, invsc %.1f, invdist %.1f tres %.1f)" \
          % (earthmodel.PhaseName(phaseid), detid,
             earthmodel.PhaseName(int(detections[detid, DET_PHASE_COL])),
             netmodel.score_event_det(event, phaseid, detid), inv_score,
             inv_dist, tres),
  print "\nEv Score:", netmodel.score_event(event, event_detlist)

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-d", "--degree_delta", dest="degree_delta", default=2.5,
                    type="float",
                    help = "degree step (2.5)")
  parser.add_option("-c", "--num_step", dest="num_step", default=2,
                    type="int",
                    help = "number of steps of size degree step (2)")
  parser.add_option("-r", "--hours", dest="hours", default=.7,
                    type="float",
                    help = "inference on HOURS worth of data (0.7)")
  parser.add_option("-k", "--skip", dest="skip", default=0,
                    type="float",
                    help = "skip the first HOURS of data (0)")
  parser.add_option("-w", "--window", dest="window", default=300,
                    type="float",
                    help = "window size in seconds (300)")
  parser.add_option("-s", "--seed", dest="seed", default=123456789,
                    type="int",
                    help = "random number generator seed (123456789)")
  parser.add_option("-v", "--verbose", dest="verbose", default=False,
                    action = "store_true",
                    help = "verbose output (False)")
  parser.add_option("-l", "--label", dest="label", default="validation",
                    help = "training, validation (default), or test")

  (options, args) = parser.parse_args()

  netvisa.srand(options.seed)
  random.seed(options.seed)

  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data(options.label, hours=options.hours, skip=options.skip)

  if (end_time - MAX_TRAVEL_TIME) <= start_time:
    print "Error: too short an interval"
    sys.exit(1)

  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  print_events_simple(netmodel, earthmodel, detections, leb_events, leb_evlist,
                      "LEB", start_time, (start_time + options.window))
  t1 = time.time()
  prop_events, prop_evlist = netmodel.propose(start_time,
                                              start_time + options.window,
                                              0, len(detections),
                                              options.degree_delta,
                                              options.num_step)
  t2 = time.time()

  print "%.1f seconds to propose %d event(s)" % (t2-t1, len(prop_events))

  print_events_simple(netmodel, earthmodel, detections,
                      prop_events, prop_evlist, "PROP",
                      start_time, (start_time + options.window))

  for leb_evnum, leb_event in enumerate(leb_events):

    # we can only predict events for which we have all the detections
    if leb_event[EV_TIME_COL] > (start_time + options.window):
      continue

    print_event(netmodel, earthmodel, detections, leb_event,
                leb_evlist[leb_evnum])

    best_score, best_prop_event, best_prop_evnum = None, None, None
    for prop_evnum, prop_event in enumerate(prop_events):

      if (dist_deg(leb_event[[EV_LON_COL, EV_LAT_COL]],
                   prop_event[[EV_LON_COL, EV_LAT_COL]]) < 5
          and abs(leb_event[EV_TIME_COL] - prop_event[EV_TIME_COL]) < 50):
        score = dist_deg(leb_event[[EV_LON_COL, EV_LAT_COL]],
                         prop_event[[EV_LON_COL, EV_LAT_COL]])/5.0\
                + abs(leb_event[EV_TIME_COL] - prop_event[EV_TIME_COL])/50.0
        if best_score is None or score < best_score:
          best_score, best_prop_event, best_prop_evnum\
                      = score, prop_event, prop_evnum

    if best_score is not None:
      print "--> BEST MATCH"
      print_event_simple(netmodel, earthmodel, detections, best_prop_event,
                         prop_evlist[best_prop_evnum])
    else:
      print "--> MISS\n -"

if __name__ == "__main__":
  main("parameters")
