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

# statistics on nearby inverts
#  50s  5d -> 80.0%
#  75s  5d -> 81.3%
#  50s  7.5d -> 88.2%
#  75s  7.5d -> 90.1%
# 100s 10d -> 94.7%

# Using an LEB proposer for VISA
# 100s 10d -> 98.4% of the 94.5% recall w/positive score
import os, sys, time, random
import numpy as np
from optparse import OptionParser
# ignore warnings from matplotlib in python 2.4
import warnings
warnings.simplefilter("ignore",DeprecationWarning)
import matplotlib.pyplot as plt

from database.dataset import *
from database.db import connect
import netvisa, learn
from results.compare import *
from utils.geog import dist_km, dist_deg
from utils.draw_earth import draw_events, draw_earth, draw_density

def fixup_event(event):
  fixup_event_lon(event)
  fixup_event_lat(event)
  fixup_event_depth(event)
  fixup_event_mag(event)

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

def fixup_event_depth(event):
  if event[EV_DEPTH_COL] < 0:
    event[EV_DEPTH_COL] = 0
  elif event[EV_DEPTH_COL] > MAX_DEPTH:
    event[EV_DEPTH_COL] = MAX_DEPTH

def fixup_event_mag(event):
  if event[EV_MB_COL] < MIN_MAGNITUDE:
    event[EV_MB_COL] = MIN_MAGNITUDE
  elif event[EV_MB_COL] > MAX_MAGNITUDE:
    event[EV_MB_COL] = MAX_MAGNITUDE

# identify the subset of detections which are good for this event
def good_detections(netmodel, event, event_detlist):
  
  good_detlist = []
  nodet_score = netmodel.score_event(event, []) # score with no detections
  
  for item in event_detlist:
    score = netmodel.score_event(event, [item])
    if score > nodet_score:
      good_detlist.append(item)

  return good_detlist

def opt_score_event(netmodel, event, event_detlist):
  return netmodel.score_event(event, good_detections(netmodel, event,
                                                     event_detlist))

DELTA = 1e-10

# improve the event using gradient ascent
def improve_event_grad(netmodel, earthmodel, detections, event, event_detlist):
  
  best_event = event.copy()
  best_score = opt_score_event(netmodel, best_event, event_detlist)
  
  while 1:
    best_detlist = good_detections(netmodel, best_event, event_detlist)
    grad = gradient(netmodel, earthmodel, detections, best_event,
                    best_detlist)

    scale = 1.
    
    while scale > DELTA:
      new_event = best_event + grad * scale
      fixup_event(new_event)
      new_score = opt_score_event(netmodel, new_event, event_detlist)

      if new_score > best_score:
        best_event, best_score = new_event, new_score
        break
      
      scale /= 10.
      
    else:
      break

  #import pdb
  #pdb.set_trace()

  best_detlist = good_detections(netmodel, best_event, event_detlist)
  best_detlist.sort()
  best_score = netmodel.score_event(best_event, best_detlist)

  return best_event, best_detlist, best_score

def gradient(netmodel, earthmodel, detections, event, event_detlist):
  grad = np.zeros(len(event))
  for dim in range(5):
    # 3 point gradient estimation
    newevent1 = event.copy()
    newevent1[dim] -= DELTA
    fixup_event(newevent1)
    newscore1 = netmodel.score_event(newevent1, event_detlist)

    newevent2 = event.copy()
    newevent2[dim] += DELTA
    fixup_event(newevent2)
    newscore2 = netmodel.score_event(newevent2, event_detlist)
    
    grad[dim] = newscore2 - newscore1

  # normalize gradient
  grad /= np.sqrt((grad ** 2).sum())
  return grad

def improve_event_unif(netmodel, earthmodel, detections, event, event_detlist):
  best_event = event.copy()
  best_score = opt_score_event(netmodel, best_event, event_detlist)
  improved = True
  while improved:
    improved = False
    for cnt in range(100):
      event = best_event.copy()
      scale = [1, 10, 100][random.randrange(3)]
      #scale = 10
      for dim in range(5):
        event[dim] += DELTA[dim] * scale * (2 * random.random() - 1)
      fixup_event(event)
      score = opt_score_event(netmodel, event, event_detlist)
      if score > best_score:
        best_event, best_score = event, score
        improved = True
        break

  best_detlist = good_detections(netmodel, best_event, event_detlist)
  best_detlist.sort()
  best_score = netmodel.score_event(best_event, best_detlist)

  return best_event, best_detlist, best_score

improve_event = improve_event_grad

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
  print "Score:", netmodel.score_event(event, event_detlist)
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

def visualize_posterior(netmodel, event, event_detlist):
  for WINDOW in [5., 1., 1e-2, 1e-4, 1e-8]:
    LON_BUCKET_SIZE = WINDOW / 100
      
    # Z axis is along the earth's axis
    # Z goes from -1 to 1, with the same number of buckets as longitude
    Z_BUCKET_SIZE = (2.0 / 360.0) * LON_BUCKET_SIZE
  
    lon1 = event[EV_LON_COL] - WINDOW
    lon2 = event[EV_LON_COL] + WINDOW
    lat1 = event[EV_LAT_COL] - WINDOW
    lat2 = event[EV_LAT_COL] + WINDOW
  
    new_event = event.copy()

    if new_event[EV_DEPTH_COL] < 0 or new_event[EV_DEPTH_COL] > MAX_DEPTH:
      continue
    
    lon_arr = np.arange(lon1, lon2,
                        LON_BUCKET_SIZE)
    z_arr = np.arange(np.sin(np.radians(lat1)),
                      np.sin(np.radians(lat2)),
                      Z_BUCKET_SIZE)
    lat_arr = np.degrees(np.arcsin(z_arr))

    score = np.zeros((len(lon_arr), len(lat_arr)))
    
    best, worst = -np.inf, np.inf

    for loni, lon in enumerate(lon_arr):
      for lati, lat in enumerate(lat_arr):
        if lon<-180: lon+=360
        if lon>180: lon-=360
        
        tmp = new_event.copy()
        tmp[EV_LON_COL] = lon
        tmp[EV_LAT_COL] = lat
        
        sc = netmodel.score_event(tmp, event_detlist)
        
        score[loni, lati] = sc

        if sc > best: best = sc
        if sc < worst: worst = sc

    if WINDOW == 1:
      levels = [best-(2**i)/100. for i in range(10)][::-1]
      levels = np.round(levels, 2).tolist()

    elif WINDOW == .1:
      levels = [best-(2**i)/1000. for i in range(10)][::-1]
      levels = np.round(levels, 3).tolist()

    elif WINDOW == 1e-2:
      levels = np.linspace(worst, best, 10)
      
    elif WINDOW == 1e-4:
      levels = np.linspace(worst, best, 10)

    elif WINDOW == 1e-8:
      levels = np.linspace(worst, best, 10)
      
    else:
      levels = [best-(2**i)/10. for i in range(10)][::-1]
      levels = np.round(levels, 1).tolist()
      

    bmap = draw_earth("Ball of size %g degrees" % WINDOW,
                      projection="mill",
                      resolution="l",
                      llcrnrlon = lon1, urcrnrlon = lon2,
                      llcrnrlat = lat1, urcrnrlat = lat2,
                      nofillcontinents=True, figsize=(4.5,4))
    
    draw_density(bmap, lon_arr, lat_arr, score, levels = levels, colorbar=True)
    draw_events(bmap, [(event[EV_LON_COL], event[EV_LAT_COL])],
                marker="s", ms=10, mfc="none", mec="blue", mew=2)
  
  
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
  
  parser.add_option("-v", "--verbose", dest="verbose", default=False,
                    action="store_true", help="verbose output")
  
  parser.add_option("-i", "--runid", dest="runid", default=None,
                    type="int",
                    help = "the run-identifier to treat as LEB (None)")

  (options, args) = parser.parse_args()

  netvisa.srand(options.seed)
  
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data("validation", hours=options.hours, skip=options.skip)

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

  vis = True
  t1 = time.time()
  tot_leb, pos_leb, nearby_inv, pos_and_nearby_inv = 0, 0, 0, 0
  for leb_evnum, leb_event in enumerate(leb_events):
    leb_detlist = leb_evlist[leb_evnum]
    tot_leb += 1
    
    if netmodel.score_event(leb_event, leb_detlist) > 0:
      has_pos_score = True
    else:
      has_pos_score = False
      
    has_nearby_inv = False
    
    for phaseid, detid in leb_detlist:
      inv_ev = netmodel.invert_det(detid, 0)
      if (inv_ev is None or (inv_ev[EV_TIME_COL] - leb_event[EV_TIME_COL] > 100)
          or (dist_deg((inv_ev[EV_LON_COL], inv_ev[EV_LAT_COL]),
                      (leb_event[EV_LON_COL], leb_event[EV_LAT_COL])) > 10)):
        continue
      else:
        has_nearby_inv = True
        if options.verbose:
          tmp_ev = leb_event.copy()
          tmp_ev[[EV_LON_COL, EV_LAT_COL, EV_DEPTH_COL, EV_TIME_COL]] = inv_ev
          tmp_ev[EV_MB_COL] = 3.0
          imp_ev, imp_det, imp_sc = improve_event(netmodel, earthmodel,
                                                  detections, tmp_ev,
                                                  leb_detlist)          
          print ("ImpInv %d: lon %4.2f lat %4.2f depth %3.1f mb %1.1f time %.1f"
                 % (detid, imp_ev[ EV_LON_COL], imp_ev[ EV_LAT_COL],
                    imp_ev[ EV_DEPTH_COL], imp_ev[ EV_MB_COL],
                    imp_ev[ EV_TIME_COL]))
          print "Detections:", imp_det
          print "Score:", imp_sc

          if vis:
            base = netmodel.score_event(imp_ev, [])
            for item in leb_detlist:
              print item, netmodel.score_event(imp_ev, [item])-base
            visualize_posterior(netmodel, imp_ev, imp_det)
            #visualize_posterior(netmodel, leb_event, imp_det)

            plt.show()
            vis = False
        break
        
    #evimp, event2, ev2score = improve_event_once(netmodel, earthmodel,
    #                                             detections, leb_event,
    #                                             leb_detlist)
      
    if has_nearby_inv:
      nearby_inv += 1

    if has_pos_score:
      pos_leb += 1

    if has_pos_score and has_nearby_inv:
      pos_and_nearby_inv += 1

    
    if options.verbose:
      print "-" * 78
      print_event(netmodel, earthmodel, detections, leb_event, leb_detlist)
      if has_pos_score:
        print "[Positive Score]",
      if has_nearby_inv:
        print "[Nearby Invert]",
      print
      ## if evimp:
      ##   print ("Imp Event: lon %4.2f lat %4.2f depth %3.1f mb %1.1f time %.1f"
      ##      % (event2[ EV_LON_COL], event2[ EV_LAT_COL],
      ##         event2[ EV_DEPTH_COL], event2[ EV_MB_COL],
      ##         event2[ EV_TIME_COL]))
      ##   print "Score:", ev2score
      print "-" * 78
    
  t2 = time.time()
  print "%.1f secs elapsed" % (t2 - t1)
  if len(leb_events) > 10:
    print "%.1f %% LEB events had a nearby invert"\
          % (nearby_inv * 100. / tot_leb)
    print "%.1f %% LEB events had +ve score"\
          % (pos_leb * 100. / tot_leb)
    print "%.1f %% LEB events had a +ve score *and* a nearby invert"\
          % (pos_and_nearby_inv * 100. / tot_leb)

if __name__ == "__main__":
  main("parameters")
