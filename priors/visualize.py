# visualize the priors
import os, sys, time, math
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
# for type 1 fonts
plt.rcParams['ps.useafm'] = True

from utils.draw_earth import draw_events, draw_earth, draw_density
from database.dataset import *
import netvisa, learn
from results.compare import *

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-t", "--train", dest="train", default=False,
                    action = "store_true",
                    help = "visualize training data (False)")
  parser.add_option("-w", "--write", dest="write", default=None,
                    help = "location to write figures to",
                    metavar="DIRNAME")
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

  visualize_arrtime(options, earthmodel, netmodel,
                    detections, leb_events, leb_evlist)

  visualize_location_prior(options, earthmodel, netmodel)
  
  visualize_detection(options, earthmodel, netmodel, start_time, end_time,
                      detections, leb_events, leb_evlist, site_up)

  plt.show()


def visualize_arrtime(options, earthmodel, netmodel,
                      detections, leb_events, leb_evlist):
  SITEID=6                              # ASAR
  PHASEID=0
  residuals = []
  for evnum, event in enumerate(leb_events):
    for phaseid, detnum in leb_evlist[evnum]:
      if phaseid == PHASEID and detections[detnum, DET_SITE_COL] == SITEID:
        evlon, evlat, evdepth, evtime = event[EV_LON_COL], event[EV_LAT_COL],\
                                        event[EV_DEPTH_COL], event[EV_TIME_COL]
        pred_arrtime =  earthmodel.ArrivalTime(evlon, evlat, evdepth, evtime,
                                               PHASEID, SITEID)
        if pred_arrtime > 0:
          residuals.append(detections[detnum, DET_TIME_COL] - pred_arrtime)

  MIN=-7
  STEP=.1
  MAX=7
  bins = np.arange(MIN, MAX+STEP, STEP)
  data_prob = np.zeros(len(bins))
  for r in residuals:
    if r < MAX+STEP:
      data_prob[int((r-MIN)/STEP)] += 1

  data_prob /= sum(data_prob)
  
  prob = [np.exp(netmodel.arrtime_logprob(x+STEP/2, 0, 0, SITEID, PHASEID))
          *STEP for x in bins]
  
  plt.figure()
  plt.title("Time residuals around IASPEI prediction for P phase at ASAR")
  plt.bar(left=bins, height=data_prob, width=STEP, alpha=.1, color="blue",
          linewidth=0,
          label="data")
  plt.plot(bins, prob, color="black", label="model")
  plt.xlabel("Time")
  plt.ylabel("Probability")
  plt.xlim(MIN,MAX)
  plt.ylim(0, .06)
  plt.legend()
  


def visualize_location_prior(options, earthmodel, netmodel):
  LON_BUCKET_SIZE = .5
  # Z axis is along the earth's axis
  # Z goes from -1 to 1 and will have the same number of buckets as longitude
  Z_BUCKET_SIZE = (2.0 / 360.0) * LON_BUCKET_SIZE
  
  lon_arr = np.arange(-180., 180., LON_BUCKET_SIZE)
  z_arr = np.arange(-1.0, 1.0, Z_BUCKET_SIZE)
  lat_arr = np.arcsin(z_arr) * 180. / np.pi
  
  prob = np.zeros((len(lon_arr), len(lat_arr)))

  for loni, lon in enumerate(lon_arr):
    for lati, lat in enumerate(lat_arr):
      prob[loni, lati] = netmodel.location_logprob(lon, lat, 0)

  #import pdb
  #pdb.set_trace()
  
  bmap = draw_earth("Log Prior Density of Events")
  draw_density(bmap, lon_arr, lat_arr, prob,
               #, np.log(prob) / np.log(2),
               colorbar=False)

  
def visualize_detection(options, earthmodel, netmodel, start_time, end_time,
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
  for plot_phaseid in range(numtimedefphases):
    plt.figure()
    plt.title("Detection probability at ASAR for %s phase, surface event"
              % earthmodel.PhaseName(plot_phaseid))
    
    # now, bucket the phase data for the P phase at ASAR
    occ = np.ones(18, int)
    det = np.zeros(18, int)
    for siteid, isdet, evmag, evdepth, dist in phase_data[plot_phaseid]:
      if siteid == SITEID and evmag >=3 and evmag <= 4 and evdepth<=50:
        distidx = int(dist/10)
        occ[distidx] += 1
        if isdet:
          det[distidx] += 1
    prob = det.astype(float) / occ.astype(float)
    x_bucket_pts = range(0, 180, 10)
    
    plt.bar(left=x_bucket_pts, height=prob, width=10, alpha=.1,
            label="data 3--4 mb", color="blue")
    
    mindist, maxdist = earthmodel.PhaseRange(plot_phaseid)
    x_pts = range(int(mindist), int(maxdist)+1)
    y_pts = [math.exp(netmodel.detection_logprob(1, 0, 3.5, x, SITEID,
                                                 plot_phaseid)) \
             for x in x_pts]
    plt.plot(x_pts, y_pts, label="model 3.5 mb", color="black")
    
    plt.xlim(0,180)
    plt.ylim(0, 1)
    plt.xlabel("Distance (deg)")
    plt.ylabel("Probability")
    plt.legend(loc="upper right")
    if options.write is not None:
      pathname = os.path.join(options.write, "detprob_%s.png"
                              % earthmodel.PhaseName(plot_phaseid))
      print "saving fig to %s" % pathname
      plt.savefig(pathname)
  
if __name__ == "__main__":
  main("parameters")
