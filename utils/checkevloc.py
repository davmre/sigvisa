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
# compares the performance of VonMises location kernel as well as the
# Arora location kernel
import csv, gzip, sys, time
import numpy as np
from numpy import sin, cos, pi, radians, degrees, log, exp, logaddexp
from sigvisa.database.dataset import *
from sigvisa.utils.geog import dist_deg
from sigvisa.utils.draw_earth import draw_events, draw_earth, draw_density
import matplotlib.pyplot as plt

UNIFORM_PROB = 0.001

LON_INTERVAL = 1
# use half as many latitude buckets as longitude buckets
Z_INTERVAL = 2.0 * (2.0 / 360.) * LON_INTERVAL
lon_arr = np.arange(-180., 180., LON_INTERVAL)
lat_arr = np.degrees(np.arcsin(np.arange(-1.0, 1.0, Z_INTERVAL)))
# in the grid the columns will span latitudes and the rows longitudes
rlat_grid, rlon_grid = np.meshgrid(np.radians(lat_arr), np.radians(lon_arr))
lat_grid, lon_grid = np.meshgrid(lat_arr, lon_arr)

EARTH_AREA = 4 * pi * AVG_EARTH_RADIUS_KM ** 2
PATCH_AREA = EARTH_AREA / (len(lon_arr) * len(lat_arr))

def main():
  if len(sys.argv) != 2:
    print "Usage: python checkevloc.py <origins.csv.gz>"
    sys.exit(1)

  t1 = time.time()

  orig_fname = sys.argv[1]
  if orig_fname.endswith(".gz"):
    orig_fp = gzip.open(orig_fname)
  else:
    orig_fp = open(orig_fname)

  data = csv.reader(orig_fp)
  # the first row is the column names (strip leading spaces
  colnames = [x.strip() for x in data.next()]

  lonlat_data = []

  for row in data:
    event = dict(zip(colnames, row))
    lonlat_data.append((float(event['LON']), float(event['LAT'])))

    if len(lonlat_data) >= 2000:
      break

  lonlat_data = np.array(lonlat_data)

  train = lonlat_data[:int(len(lonlat_data) * .4)] # 40% training
  # 10% validation (for tuning the parameters)
  valid = lonlat_data[len(train):int(len(lonlat_data) * .5)]
  test = lonlat_data[len(valid):]                       # 50% final test

  best_kappa, density = vonmises(train, valid)
  print "Von Mises sum", reduce(logaddexp, density.flat, -np.inf)\
        + log(PATCH_AREA)
  print "Von Mises score", eval_density(density, test)

  best_bw, density2 = arora(train, valid)
  print "Arora sum", reduce(logaddexp, density2.flat, -np.inf)\
        + log(PATCH_AREA)
  print "Arora score", eval_density(density2, test)

  t2 = time.time()

  print "Elapsed %.1f secs" % (t2-t1)

  bmap = draw_earth("Von Mises Kernel (kappa=%.2f)" % best_kappa)
  draw_density(bmap, lon_arr, lat_arr, density)
  #draw_events(bmap, lonlat_data, marker="s", ms=5, mfc="none", mec="yellow",
  #            mew=2)

  bmap = draw_earth("Arora Kernel (b=%.2f)" % best_bw)
  draw_density(bmap, lon_arr, lat_arr, density2)
  #draw_events(bmap, lonlat_data, marker="s", ms=5, mfc="none", mec="yellow",
  #            mew=2)

  plt.show()

def dotproduct(rlon1, rlat1, rlon2, rlat2):
  return cos(rlat1) * cos(rlat2) * cos(rlon1) * cos(rlon2) \
         + cos(rlat1) * cos(rlat2) * sin(rlon1) * sin(rlon2)\
         + sin(rlat1) * sin(rlat2)

def vonmises_kappa(train, kappa):
  weights = -np.inf * np.ones((len(lon_arr), len(lat_arr)), float)

  for lon, lat in train:
    rlon, rlat = np.radians(lon), np.radians(lat)
    weights = logaddexp(weights,
                        kappa * dotproduct(rlon_grid, rlat_grid, rlon, rlat))

  # normalize to make it a probability at each of the grid points
  prob = weights - reduce(logaddexp, weights.flat, -np.inf)

  # now, the density is obtained by dividing the probability by the area of
  # each rectangle corresponding to a grid point
  density = prob - log(PATCH_AREA)

  # account for the uniform probability
  density = logaddexp(log(1 - UNIFORM_PROB) + density,
                      log(UNIFORM_PROB) - log(EARTH_AREA))

  return density


def vonmises(train, valid):
  kappa = grid_search(lambda k: eval_density(vonmises_kappa(train, k), valid))
  print "Von Mises best kappa", kappa
  return kappa, vonmises_kappa(train, kappa)

def arora(train, valid):
  bw = grid_search(lambda b: eval_density(arora_bw(train, b), valid))
  print "Arora best bw", bw
  return bw, arora_bw(train, bw)

def arora_bw(train, bw):
  weights = -np.inf * np.ones((len(lon_arr), len(lat_arr)), float)

  for lon, lat in train:
    weights = logaddexp(weights,
                        - dist_deg((lon_grid, lat_grid), (lon, lat)) / bw)

  # normalize to make it a probability at each of the grid points
  prob = weights - reduce(logaddexp, weights.flat, -np.inf)

  # now, the density is obtained by dividing the probability by the area of
  # each rectangle corresponding to a grid point
  density = prob - log(PATCH_AREA)

  # account for the uniform probability
  density = logaddexp(log(1 - UNIFORM_PROB) + density,
                      log(UNIFORM_PROB) - log(EARTH_AREA))

  return density

def eval_density(density, pts):
  score = 0.
  for lon, lat in pts:
    score += density[int((lon + 180) // LON_INTERVAL),
                     int((sin(radians(lat))+1.) // Z_INTERVAL)]
  return score/len(pts)


def grid_search(func, best_exp=0, max_depth=3):
  """
  search with step size of 1.0
  pow(10, best_exp - step) ... pow(10, best_exp) ... pow(10, best_exp + step)
  then reduce the step size by a factor of 10 and repeat
  21 * max_depth searches are performed
  """
  cache = {}
  def grid_search_lookup(v):
    if v <= 0:
      return -np.inf
    if v not in cache:
      print "Trying parameter", v, "...",
      cache[v] = func(v)
      print cache[v]
    return cache[v]

  for depth in range(max_depth):
    # find the best value at the current depth
    best_exp = max((grid_search_lookup(pow(10,e)), e)
                   for e in np.linspace(best_exp - pow(10, -depth+1),
                                        best_exp + pow(10, -depth+1), 21))[1]
    print "Depth", depth, "best", pow(10, best_exp)

  return pow(10, best_exp)

if __name__ == "__main__":
  try:
    main()
  except SystemExit:
    raise
  except:
    import pdb, traceback, sys
    traceback.print_exc(file=sys.stdout)
    pdb.post_mortem(sys.exc_traceback)
    raise
