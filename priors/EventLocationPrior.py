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
import os, random
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, log, pi, degrees, radians, arange, arcsin, zeros, sin
import time

from database.dataset import *
from utils.geog import dist_deg


# we will use these many events to compute the optimal bandwidth
SAMPLE_EVENTS = 1000

UNIFORM_PROB = .001

LON_INTERVAL = 1
# use half as many latitude buckets as longitude buckets
Z_INTERVAL = 2.0 * (2.0 / 360.) * LON_INTERVAL
lon_arr = arange(-180., 180., LON_INTERVAL)
lat_arr = degrees(arcsin(arange(-1.0, 1.0, Z_INTERVAL)))
lat_grid, lon_grid = np.meshgrid(lat_arr, lon_arr)

EARTH_AREA = 4 * pi * AVG_EARTH_RADIUS_KM ** 2  
PATCH_AREA = EARTH_AREA / (len(lon_arr) * len(lat_arr))

def kernel(bandwidth, distance):
  # convert bandwidth and distance to radians
  b, d = radians(bandwidth), radians(distance)
  
  return (1.0 + 1.0/b**2)  / (2 * pi * AVG_EARTH_RADIUS_KM**2) \
         * exp(-d/b) / (1.0 + exp(-pi / b))

def compute_density(bandwidth, events):
  density = zeros((len(lon_arr), len(lat_arr)), float)
  for ev in events:
    density += kernel(bandwidth,
                      dist_deg((lon_grid, lat_grid),
                               ev[[EV_LON_COL, EV_LAT_COL]]))
  density /= len(events)
  
  return density

def leave_oneout_avg_loglike(bandwidth, events):
  print "computing leave one out bw %f" % bandwidth,
  t1 = time.time()
  n = float(len(events))
  density = compute_density(bandwidth, events)
  tot = 0.
  for ev in events:
    evdens = kernel(bandwidth, dist_deg((lon_grid, lat_grid),
                                        ev[[EV_LON_COL, EV_LAT_COL]]))
    netdensity = (n/(n-1)) * density - (1/(n-1)) * evdens
    netdensity = (1 - UNIFORM_PROB) * netdensity + UNIFORM_PROB / EARTH_AREA
    
    loni = int((ev[EV_LON_COL] + 180) // LON_INTERVAL)
    lati = int((sin(radians(ev[EV_LAT_COL])) + 1.) // Z_INTERVAL)
    tot += log(netdensity[loni, lati])
  
  ans = tot / len(events)
  
  print "done (in %.1f secs) -> %f" % (time.time()-t1, ans)
  return ans

def learn(param_fname, options, leb_events):
  ########
  # first we will learn the optimal bandwidth (in degrees)
  ########
  
  # to do so we will pick a random sample of 500 events
  indices = [i for i in range(len(leb_events)) \
             if random.random() < (float(SAMPLE_EVENTS) / len(leb_events))]

  # then we'll use leave-one-out cross-validation to select the best
  # bandwidth

  all_bandw, all_loglike = np.array([]), np.array([])
  for bandw in np.arange(0, 2, .2) + .2:
    all_bandw = np.append(all_bandw, bandw)
    all_loglike = np.append(all_loglike,
                            leave_oneout_avg_loglike(bandw,
                                                     leb_events[indices]))

  best_bandw = all_bandw[all_loglike.argmax()]

  for bandw in np.arange(best_bandw - .2, best_bandw + .2, .05):
    if bandw in all_bandw:
      continue
    all_bandw = np.append(all_bandw, bandw)
    all_loglike = np.append(all_loglike,
                            leave_oneout_avg_loglike(bandw,
                                                     leb_events[indices]))

  best_bandw = all_bandw[all_loglike.argmax()]

  print "best bandwidth", best_bandw

  if options.gui:
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Leave-One-Out Avg. Log Likelihood")
    plt.scatter(all_bandw, all_loglike)
    plt.xlabel("bandwidth (degrees)")
    plt.ylabel("avg. log likelihood")
    plt.xlim(0, 2.)
    
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "EventLocBandwidth")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")

  density = compute_density(best_bandw, leb_events)
  # fold-in a uniform prior
  density = (1 - UNIFORM_PROB) * density + UNIFORM_PROB / EARTH_AREA

  if options.gui:
    if options.type1:
      title = ""
    else:
      title = "Event Location Log Density (b=%.2f)" % best_bandw
    
    from utils.draw_earth import draw_events, draw_earth, draw_density
    bmap = draw_earth(title)
    draw_density(bmap, lon_arr, lat_arr, log(density),
                 colorbar_orientation="horizontal", colorbar_shrink=0.75)
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "EventLocationPrior")
      # a .pdf file would be too huge
      plt.savefig(basename+".png")
    
  # convert the density at the grid points into a probability for each
  # bucket (the grid point is in the lower-left corner of the bucket)
  prob = density * PATCH_AREA
  
  print "Total prob:", prob.sum()

  fp = open(param_fname, "w")

  print >>fp, UNIFORM_PROB
  print >>fp, LON_INTERVAL, Z_INTERVAL
  
  np.savetxt(fp, prob)                 # writes out row by row
  fp.close()

class Dummy:
  pass

def main(param_fname):
  cursor = db.connect().cursor()
  start_time, end_time = read_timerange(cursor, "training", None, 0)
  print "reading events...",
  leb_events = read_events(cursor, start_time, end_time, "leb")[0]
  print "done (%d events)" % len(leb_events)

  options = Dummy()
  
  options.gui = True
  options.type1 = False
  options.writefig = None
  
  learn(param_fname, options, leb_events)
  plt.show()
  
if __name__ == "__main__":
  try:
    main(os.path.join("parameters", "EventLocationPrior.txt"))
  except SystemExit:
    raise
  except:
    import pdb, traceback, sys
    traceback.print_exc(file=sys.stdout)
    pdb.post_mortem(sys.exc_traceback)
    raise
  
