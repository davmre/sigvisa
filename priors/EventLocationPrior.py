import os

from database.dataset import *

def learn(param_fname, leb_events, LON_INTERVAL=2, DISCOUNT=.5):
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
  
