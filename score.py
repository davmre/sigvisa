import os

from database.dataset import *
import netvisa, learn

def main(param_dirname):
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data("validation")

  netmodel = learn.load_netvisa(param_dirname, start_time, end_time,
                                detections, site_up)
  
  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  
  print "LEB:"
  netvisa.score_world(earthmodel, netmodel, leb_events, leb_evlist, 1)
  print "SEL3:"
  netvisa.score_world(earthmodel, netmodel, sel3_events, sel3_evlist, 1)

if __name__ == "__main__":
  main("parameters")
