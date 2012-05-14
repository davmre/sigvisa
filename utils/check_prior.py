#checks whether events generated from the prior are good enough

import os, sys
import numpy as np
from optparse import OptionParser
import random

from database.dataset import *
import netvisa, learn
from utils.geog import dist_deg

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-n", "--numsamples", dest="numsamples", default=1,
                    type="int",
                    help = "number of draws from the prior (1000)")
  parser.add_option("-r", "--hours", dest="hours", default=None,
                    type="float",
                    help = "inference on HOURS worth of data (all)")
  parser.add_option("-k", "--skip", dest="skip", default=0,
                    type="float",
                    help = "skip the first HOURS of data (0)")
  parser.add_option("-s", "--seed", dest="seed", default=123456789,
                    type="int",
                    help = "random number generator seed (123456789)")
  parser.add_option("-v", "--verbose", dest="verbose", default=False,
                    action = "store_true",
                    help = "verbose output (False)")

  (options, args) = parser.parse_args()

  netvisa.srand(options.seed)
  random.seed(options.seed)
  
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data("validation", hours=options.hours, skip=options.skip)

  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  prior_events = [netmodel.location_sample() for _ in
                  xrange(options.numsamples)]
  
  dists = []
  for leb_evnum, leb_event in enumerate(leb_events):

    mindist, lon, lat,depth\
             = min((dist_deg(leb_event[[EV_LON_COL, EV_LAT_COL]], (lon, lat)),
                    lon, lat, depth) for lon,lat,depth in prior_events)

    if mindist < 5:
      dists.append(mindist)

  print "%.1f %% matched prior with min dist avg. %.1f s.d. %.1f deg"\
        % (100. * float(len(dists)) / len(leb_events),
           np.average(dists), np.std(dists))
  
if __name__ == "__main__":
  main("parameters")
