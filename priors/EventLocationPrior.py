import os, random
import numpy as np
from numpy import exp, log, pi, degrees, radians
from database.dataset import *
from utils.geog import dist_deg

UNIFORM_PROB = .001

def kernel(bandwidth, distance):
  # convert bandwidth and distance to radians
  b, d = radians(bandwidth), radians(distance)
  
  return (1.0 + 1.0/b**2)  / (2 * pi * AVG_EARTH_RADIUS_KM**2) \
         * exp(-d/b) / (1.0 + exp(-pi / b))

# given a,b,c,... return v s.t. exp(v) = exp(a) + exp(b) + exp(c)
# or, in other words, we need to compute the sum of a bunch of quantities whose
# logarithms are provided and we want to return the logarithm of the answer
def logadd(list):
  m = max(list)
  return m + log(sum(exp(x-m) for x in list))

# computes the log-likelihood at the left out index
def leave_one_loglike_pt(leb_events, bandwidth, left_index):
  log_like = []
  for evidx, ev in enumerate(leb_events):
    if evidx != left_index:
      distance = dist_deg(ev[[EV_LON_COL, EV_LAT_COL]],
                          leb_events[left_index,[EV_LON_COL, EV_LAT_COL]])
      log_like.append(log(kernel(bandwidth, distance)))
  
  kernel_loglike = logadd(log_like) - log(len(leb_events)-1)

  return logadd([log(1-UNIFORM_PROB) + kernel_loglike,
                 log(UNIFORM_PROB) - log(4 * pi * AVG_EARTH_RADIUS_KM ** 2)])

def leave_one_loglike(leb_events, bandwidth):
  return sum(leave_one_loglike_pt(leb_events, bandwidth, idx) \
             for idx in range(len(leb_events))) / len(leb_events)

def learn(param_fname, leb_events):
  ########
  # first we will learn the optimal bandwidth (in degrees)
  ########
  
  # to do so we will pick a random sample of 500 events
  indices = [i for i in range(len(leb_events)) \
             if random.random() < (500. / len(leb_events))]

  # then we'll use leave-one-out cross-validation to select the best
  # bandwidth
  bandwidth, best_loglike = None, None
  for b in [.25, .5, .75, 1, 1.25, 1.5, 2]:
    loglike = leave_one_loglike(leb_events[indices], b)
    print "b=%.1f %.3f" % (b, loglike)
    if best_loglike is None or loglike > best_loglike:
      bandwidth, best_loglike = b, loglike
  
  print "Best bandwidth,", bandwidth

  ########
  # next, we will create a grid
  ########
  LON_INTERVAL = 1
  # use half as many latitude buckets as longitude buckets
  Z_INTERVAL = 2.0 * (2.0 / 360.0) * LON_INTERVAL

  # create longitude and latitude buckets
  lon_arr = np.arange(-180., 180., LON_INTERVAL)
  lat_arr = np.degrees(np.arcsin(np.arange(-1.0, 1.0, Z_INTERVAL)))

  #####
  # then, compute the kernel density at each of the grid points
  # from each of the events, we only look at grid points within 10
  # degrees of the event
  #####
  density = np.zeros((len(lon_arr), len(lat_arr)), float)
  for event in leb_events:
    lon, lat = event[[EV_LON_COL, EV_LAT_COL]]
    lonidx = int((lon + 180) // LON_INTERVAL)
    latidx = int((np.sin(np.radians(lat)) + 1.0) // Z_INTERVAL)
      
    for loni2 in range(-10, 10):
      for lati2 in range(-10, 10):
        
        lonidx2 = (lonidx + loni2) % len(lon_arr)
        latidx2 = latidx + lati2
        
        if latidx2 < 0 or latidx2 >= len(lat_arr):
          continue
        
        lon2 = lon_arr[lonidx2]
        lat2 = lat_arr[latidx2]
        
        density[lonidx2, latidx2] += kernel(bandwidth,
                                            dist_deg((lon,lat), (lon2,lat2)))

  # note, in the above, we simply added the density contribution from each
  # nearby event but didn't divide by the total number of events, so..
  density /= float(len(leb_events))
  
  # convert the density at the grid points into a probability for each
  # bucket (the grid point is in the lower-left corner of the bucket)
  prob = density * 4 * pi * AVG_EARTH_RADIUS_KM ** 2 / density.size

  # fold-in a uniform prior
  prob *= (1-UNIFORM_PROB)
  prob += UNIFORM_PROB / prob.size
  
  print "Total prob:", prob.sum()

  fp = open(param_fname, "w")

  print >>fp, UNIFORM_PROB
  print >>fp, LON_INTERVAL, Z_INTERVAL
  
  np.savetxt(fp, prob)                 # writes out row by row

  fp.close()

def learn2(param_fname, leb_events, LON_INTERVAL=2, DISCOUNT=.5):
  # use as many latitude buckets as longitude buckets
  Z_INTERVAL = (2.0 / 360.0) * LON_INTERVAL
  # create longitude and latitude buckets
  lon_arr = np.arange(-180., 180., LON_INTERVAL)
  lat_arr = np.degrees(np.arcsin(np.arange(-1.0, 1.0, Z_INTERVAL)))

  # create a multi-dim index and score for each space bucket
  lon_idx, lat_idx = np.mgrid[0:len(lon_arr), 0:len(lat_arr)]
  
  count = np.zeros((len(lon_arr), len(lat_arr)), float)

  for event in leb_events:
    lonidx = int((event[EV_LON_COL]+180) // LON_INTERVAL)
    latidx = int((np.sin(np.radians(event[EV_LAT_COL])) + 1.0) // Z_INTERVAL)
    
    count[lonidx, latidx] += 1;

  # remove a DISCOUNT from each bucket with non-zero count and distribute
  # this to all the buckets with zero counts
  redistrib = float((count > 0).sum()) * DISCOUNT / float((count == 0).sum())
  count[count > 0] -= DISCOUNT
  count[count == 0] += redistrib

  # convert to a probability distribution
  count /= count.sum()

  fp = open(param_fname, "w")

  print >>fp, DISCOUNT
  print >>fp, LON_INTERVAL, Z_INTERVAL
  
  np.savetxt(fp, count)                 # writes out row by row

  fp.close()

if __name__ == "__main__":
  print "Learning best LON_INTERVAL and DISCOUNT"
  
