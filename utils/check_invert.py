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
#checks whether invert detection works correctly

import os, sys, time
import numpy as np
from optparse import OptionParser

from database.dataset import *
from learn import read_datafile_and_sitephase
from database.db import connect
import netvisa, learn
from results.compare import *
from utils.geog import dist_km, dist_deg

def fixup_event_lon(event):
  if event[EV_LON_COL] < - 180:
    event[EV_LON_COL] += 360
  elif event[EV_LON_COL] >= 180:
    event[EV_LON_COL] -= 360

def fixup_event_lat(event):
  if event[EV_LAT_COL] < -90:
    event[EV_LAT_COL] = -180 - event[EV_LAT_COL]
  elif event[EV_LAT_COL] > 90:
    event[EV_LAT_COL] = 180 - event[EV_LAT_COL]
  
def print_event(netmodel, earthmodel, detections, event, event_detlist):
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
  for phaseid, detid in detlist:
    if not earthmodel.IsTimeDefPhase(int(detections[detid, DET_PHASE_COL])):
      phaseid = 0                       # P phase
    else:
      phaseid = int(detections[detid, DET_PHASE_COL])

    tres = detections[detid, DET_TIME_COL] \
           - earthmodel.ArrivalTime(event[EV_LON_COL],
                                    event[EV_LAT_COL],
                                    event[EV_DEPTH_COL],
                                    event[EV_TIME_COL],
                                    phaseid,
                                    int(detections[detid, DET_SITE_COL]))
    
    azres = detections[detid, DET_AZI_COL]\
            - earthmodel.ArrivalAzimuth(event[EV_LON_COL],
                                        event[EV_LAT_COL],
                                        int(detections[detid, DET_SITE_COL]))
    sres = detections[detid, DET_SLO_COL] \
           - earthmodel.ArrivalSlowness(event[EV_LON_COL],
                                        event[EV_LAT_COL],
                                        event[EV_DEPTH_COL],
                                        phaseid,
                                        int(detections[detid, DET_SITE_COL]))

    print "%d: tres %.1f azres %.1f sres %.1f" % (detid, tres, azres, sres)
    
    inv = netmodel.invert_det(detid, 0)
    if inv is None:
      print None
    else:
      print "lon %.2f lat %.2f depth %.1f time %.1f" % inv

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-s", "--seed", dest="seed", default=123456789,
                    type="int",
                    help = "random number generator seed (123456789)")
  parser.add_option("-r", "--hours", dest="hours", default=None,
                    type="float",
                    help = "inference on HOURS worth of data (all)")
  parser.add_option("-k", "--skip", dest="skip", default=0,
                    type="float",
                    help = "skip the first HOURS of data (0)")
  
  parser.add_option("-i", "--runid", dest="runid", default=None,
                    type="int",
                    help = "the run-identifier to treat as LEB (None)")
  parser.add_option("--datafile", dest="datafile", default=None,
                    help = "tar file with data (None)", metavar="FILE")  

  (options, args) = parser.parse_args()

  netvisa.srand(options.seed)

  if options.datafile:
    start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
                sel3_evlist, site_up, sites, phasenames, phasetimedef, \
                sitenames = read_datafile_and_sitephase(options.datafile,
                                                        param_dirname,
                                                        hours = options.hours,
                                                        skip = options.skip)
  else:
    start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
                sel3_evlist, site_up, sites, phasenames, phasetimedef \
                = read_data("validation", hours=options.hours,
                            skip=options.skip)

  # treat a VISA run as LEB
  if options.runid is not None:
    cursor = connect().cursor()
    detections, arid2num = read_detections(cursor, start_time, end_time)
    visa_events, visa_orid2num = read_events(cursor, start_time, end_time,
                                             "visa", options.runid)
    visa_evlist = read_assoc(cursor, start_time, end_time, visa_orid2num,
                             arid2num, "visa", runid=options.runid)
    leb_events, leb_evlist = visa_events, visa_evlist
  
  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  invloc_to_event = []
  event_to_nearest_invloc = []
  invloc_to_event_arr = []
  invloc_to_event_3c = []
  
  t1 = time.time()
  tot_leb, pos_leb, nearby_inv = 0, 0, 0
  for leb_evnum, leb_event in enumerate(leb_events):
    leb_detlist = leb_evlist[leb_evnum]
    tot_leb += 1
    if netmodel.score_event(leb_event, leb_detlist) > 0:
      pos_leb += 1

    has_nearby_inv = False
    
    dist_invlocs = []
    
    for phaseid, detid in leb_detlist:
      inv_ev = netmodel.invert_det(detid, 0)
      if inv_ev is None \
             or (inv_ev[EV_TIME_COL] - leb_event[EV_TIME_COL] > 100):
        continue
      dist = dist_deg((inv_ev[EV_LON_COL], inv_ev[EV_LAT_COL]),
                      (leb_event[EV_LON_COL], leb_event[EV_LAT_COL]))
      if dist > 10:
        continue
      else:
        invloc_to_event.append(dist)
        dist_invlocs.append(dist)
        
        if sites[detections[detid][DET_SITE_COL]][SITE_IS_ARRAY]:
          invloc_to_event_arr.append(dist)
        else:
          invloc_to_event_3c.append(dist)
        
        
        has_nearby_inv = True
        
    if len(dist_invlocs):
      event_to_nearest_invloc.append(min(dist_invlocs))
    
    if has_nearby_inv:
      nearby_inv += 1
    
  t2 = time.time()
  print "%.1f secs elapsed" % (t2 - t1)
  print "%.1f %% LEB events had +ve score"\
        % (pos_leb * 100. / tot_leb)
  print "%.1f %% LEB events had a nearby invert"\
        % (nearby_inv * 100. / tot_leb)
  
  import matplotlib.pyplot as plt
  import numpy as np
  plt.rcParams['text.usetex'] = True    # type 1 fonts for publication
  
  bins = np.arange(0,10,.1)
  plt.figure()
  plt.title("Distance from inverted location to LEB event")
  plt.hist(invloc_to_event, bins)
  plt.xlabel("Distance (degrees)")

  plt.figure()
  plt.title("Distance form LEB event to nearest inverted location")
  plt.hist(event_to_nearest_invloc, bins)
  plt.xlabel("Distance (degrees)")

  plt.figure()
  plt.title("Distance from inverted location to LEB event (Arrays)")
  plt.hist(invloc_to_event_arr, bins)
  plt.xlabel("Distance (degrees)")

  plt.figure()
  plt.title("Distance from inverted location to LEB event (3-C)")
  plt.hist(invloc_to_event_3c, bins)
  plt.xlabel("Distance (degrees)")

  plt.show()
  
if __name__ == "__main__":
  main("parameters")
