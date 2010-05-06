# visualize the priors
import os, sys, time, math
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt

from database.dataset import *
import netvisa, learn
from results.compare import *

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-t", "--train", dest="train", default=False,
                    action = "store_true",
                    help = "visualize training data (False)")
  (options, args) = parser.parse_args()

  if options.train:
    label = "training"
  else:
    label = "validation"
    
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
              sel3_evlist, site_up, sites, phasenames, phasetimedef \
              = read_data(label)
  
  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  visualize_detection(earthmodel, netmodel, start_time, end_time,
                      detections, leb_events, leb_evlist, site_up)

  plt.show()
  
def visualize_detection(earthmodel, netmodel, start_time, end_time,
                        detections, leb_events, leb_evlist, site_up):
  
  numtimedefphases = earthmodel.NumTimeDefPhases()
  numsites = earthmodel.NumSites()

  # construct a dataset for each phase
  phase_data = dict((i,[]) for i in range(numtimedefphases))  
  for evnum, event in enumerate(leb_events):
    det_phase_site = set((phaseid, detections[detnum, DET_SITE_COL])\
                         for phaseid, detnum in leb_evlist[evnum])
    for siteid in range(numsites):
      dist = earthmodel.Delta(event[EV_LON_COL], event[EV_LAT_COL], siteid)
      for phaseid in range(numtimedefphases):
        arrtime = earthmodel.ArrivalTime(event[EV_LON_COL], event[EV_LAT_COL],
                                         event[EV_DEPTH_COL],
                                         event[EV_TIME_COL], phaseid, siteid)
        
        # check if the site is in the shadow zone of this phase
        if arrtime < 0:
          continue
        
        # check if the site was up at the expected arrival time
        if arrtime < start_time or arrtime >= end_time \
            or not site_up[siteid, int((arrtime - start_time) / UPTIME_QUANT)]:
          continue
        
        isdet = int((phaseid, siteid) in det_phase_site)
        
        phase_data[phaseid].append((siteid, isdet, event[EV_MB_COL],
                                    event[EV_DEPTH_COL], dist))

  SITEID = 6
  plot_phaseid = 0

  plt.figure()
  plt.title("Detection probability at ASAR for P phase of surface event")
  
  # now, bucket the phase data for the P phase at ASAR
  occ = np.ones(18, int)
  det = np.zeros(18, int)
  for siteid, isdet, evmag, evdepth, dist in phase_data[plot_phaseid]:
    if siteid == SITEID and evmag >=3 and evmag <= 4 and evdepth<=50:
      distidx = int(dist/10)
      occ[distidx] += 1
      if isdet:
        occ[distidx] += 1
  prob = det.astype(float) / occ.astype(float)
  x_bucket_pts = range(0, 180, 10)

  plt.bar(left=x_bucket_pts, height=prob, width=10, alpha=.5,
          label="data 3-4 mb")

  x_pts = range(0,181)  
  y_pts = [math.exp(netmodel.detection_logprob(1, 0, 3.5, x, SITEID,
                                               plot_phaseid)) \
           for x in x_pts]
  plt.plot(x_pts, y_pts, label="model 3.5 mb")

  plt.xlim(0,180)
  plt.ylim(0, 1)
  plt.xlabel("Distance (deg)")
  plt.ylabel("Probability")
  plt.legend(loc="upper right")

  
if __name__ == "__main__":
  main("parameters")
