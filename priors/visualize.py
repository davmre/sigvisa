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
from utils import Laplace
import utils.GMM, utils.LinearModel
import database.db

from utils.LogisticModel import LogisticModel
from priors.ArrivalAmplitudePrior import print_2gmm, print_list,\
     learn_amp_model, predict_amp_model

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-t", "--train", dest="train", default=False,
                    action = "store_true",
                    help = "visualize training data (False)")
  parser.add_option("-a", "--arrival", dest="arrival", default=False,
                    action = "store_true",
                    help = "visualize arrival parameters (False)")
  parser.add_option("-l", "--location", dest="location", default=False,
                    action = "store_true",
                    help = "visualize location prior (False)")
  parser.add_option("-d", "--detection", dest="detection", default=False,
                    action = "store_true",
                    help = "visualize detection prior (False)")
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

  # if no options is selected then select all options
  if not options.arrival and not options.location and not options.detection:
    options.arrival = options.location = options.detection = True
  
  if options.arrival:
    print "visualizing arrival parameters"
    
    visualize_arrtime(options, earthmodel, netmodel,
                      detections, leb_events, leb_evlist)
    
    visualize_arraz(options, earthmodel, netmodel,
                    detections, leb_events, leb_evlist)
    
    visualize_arrslo(options, earthmodel, netmodel,
                     detections, leb_events, leb_evlist)

    visualize_arramp(options, earthmodel, netmodel,
                     detections, leb_events, leb_evlist)

  if options.location:
    print "visualizing location prior"
    
    visualize_location_prior(options, earthmodel, netmodel)

  if options.detection:
    print "visualizing detection prior"
    
    visualize_detection(options, earthmodel, netmodel, start_time, end_time,
                        detections, leb_events, leb_evlist, site_up)

  plt.show()


def visualize_arrtime(options, earthmodel, netmodel,
                      detections, leb_events, leb_evlist):
  MIN=-7
  STEP=.2
  MAX=7

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
          res = detections[detnum, DET_TIME_COL] - pred_arrtime
          if res > MIN and res < MAX:
            residuals.append(res)

  mix_prob, mix_loc, mix_scale = Laplace.estimate_laplace_uniform_dist(
    residuals, MIN, MAX)

  bins = np.arange(MIN, MAX+STEP, STEP)
  data_prob = np.zeros(len(bins))
  for r in residuals:
    if r < MAX+STEP:
      data_prob[int((r-MIN)/STEP)] += 1

  data_prob /= sum(data_prob)
  
  prob = [np.exp(netmodel.arrtime_logprob(x, 0, 0, SITEID, PHASEID))
          *STEP for x in bins]
  mix_prob = [np.exp(Laplace.ldensity_laplace_uniform_dist(
    mix_prob, mix_loc, mix_scale, MIN, MAX, x)) * STEP for x in bins]
  
  plt.figure()
  plt.title("Time residuals around IASPEI prediction for P phase at station 6")
  plt.bar(left=bins, height=data_prob, width=STEP, alpha=1, color="blue",
          linewidth=0,
          label="data")
  plt.plot(bins, prob, color="black", label="Laplace", linewidth=3)
  #plt.plot(bins, mix_prob, color="red", label="Laplace+uniform", linewidth=3)
  plt.xlabel("Time")
  plt.ylabel("Probability")
  plt.xlim(MIN,MAX)
  plt.grid()
  plt.legend()

  # now, visualize travel time residuals from the LEB arrivals
  cursor = database.db.connect().cursor()
  cursor.execute("select lon, lat, timeres from leb_origin join leb_assoc "
                 "using (orid) where sta='ASAR' and phase='P' and time between "
                 "(select start_time from dataset where label='training') and "
                 "(select end_time from dataset where label='training')")
  events = np.array(cursor.fetchall())
  posevs = events[events[:,2] > 0]
  negevs = events[events[:,2] < 0]
  bmap = draw_earth("Distribution of residuals, P phase, station 6")
  draw_events(bmap, posevs[:,[0,1]],
              marker="o", ms=10, mfc="none", mec="blue", mew=1)
  draw_events(bmap, negevs[:,[0,1]],
              marker="o", ms=10, mfc="none", mec="red", mew=1)

def visualize_arraz(options, earthmodel, netmodel,
                    detections, leb_events, leb_evlist):
  MIN=-180
  STEP=.5
  MAX=180

  SITEID=6                              # ASAR
  PHASEID=0
  residuals = []
  for evnum, event in enumerate(leb_events):
    for phaseid, detnum in leb_evlist[evnum]:
      if phaseid == PHASEID and detections[detnum, DET_SITE_COL] == SITEID:
        evlon, evlat = event[EV_LON_COL], event[EV_LAT_COL]
        pred_arraz =  earthmodel.ArrivalAzimuth(evlon, evlat, SITEID)
        res = earthmodel.DiffAzimuth(pred_arraz,
                                     detections[detnum, DET_AZI_COL])
        if res > MIN and res < MAX:
          residuals.append(res)

  mix_prob, mix_loc, mix_scale = Laplace.estimate_laplace_uniform_dist(
    residuals, MIN, MAX)

  bins = np.arange(MIN, MAX+STEP, STEP)
  data_prob = np.zeros(len(bins))
  for r in residuals:
    if r < MAX+STEP:
      data_prob[int((r-MIN)/STEP)] += 1

  data_prob /= sum(data_prob)
  
  prob = [np.exp(netmodel.arraz_logprob(x, 0, 0, SITEID, PHASEID))
          *STEP for x in bins]

  mix_prob = [np.exp(Laplace.ldensity_laplace_uniform_dist(
    mix_prob, mix_loc, mix_scale, MIN, MAX, x)) * STEP for x in bins]
  
  plt.figure()
  plt.title("Azimuth residuals around IASPEI prediction for P phase at "
            "station 6")
  plt.bar(left=bins, height=data_prob, width=STEP, alpha=1, color="blue",
          linewidth=0,
          label="data")
  plt.plot(bins, prob, color="black", label="Laplace", linewidth=3)
  #plt.plot(bins, mix_prob, color="red", label="Laplace+uniform", linewidth=3)
  plt.xlabel("Azimuth")
  plt.ylabel("Probability")
  plt.xlim(-20, 20)
  #plt.ylim(0, .06)
  plt.grid()
  plt.legend()

def visualize_arrslo(options, earthmodel, netmodel,
                     detections, leb_events, leb_evlist):
  MIN=-40
  STEP=.2
  MAX=40

  SITEID=6                              # ASAR
  PHASEID=0
  residuals = []
  for evnum, event in enumerate(leb_events):
    for phaseid, detnum in leb_evlist[evnum]:
      if phaseid == PHASEID and detections[detnum, DET_SITE_COL] == SITEID:
        evlon, evlat, evdepth = event[EV_LON_COL], event[EV_LAT_COL],\
                                event[EV_DEPTH_COL]
        pred_arrslo =  earthmodel.ArrivalSlowness(evlon, evlat, evdepth,
                                                  PHASEID, SITEID)
        res = pred_arrslo - detections[detnum, DET_SLO_COL]
        if res > MIN and res < MAX:
          residuals.append(res)

  mix_prob, mix_loc, mix_scale = Laplace.estimate_laplace_uniform_dist(
    residuals, MIN, MAX)

  bins = np.arange(MIN, MAX+STEP, STEP)
  data_prob = np.zeros(len(bins))
  for r in residuals:
    if r < MAX+STEP:
      data_prob[int((r-MIN)/STEP)] += 1

  data_prob /= sum(data_prob)
  
  prob = [np.exp(netmodel.arrslo_logprob(x, 0, 0, SITEID, PHASEID))
          *STEP for x in bins]

  mix_prob = [np.exp(Laplace.ldensity_laplace_uniform_dist(
    mix_prob, mix_loc, mix_scale, MIN, MAX, x)) * STEP for x in bins]
  
  plt.figure()
  plt.title("Slowness residuals around IASPEI prediction for P phase at "
            "station 6")
  plt.bar(left=bins, height=data_prob, width=STEP, alpha=1, color="blue",
          linewidth=0,
          label="data")
  plt.plot(bins, prob, color="black", label="Laplace", linewidth=3)
  #plt.plot(bins, mix_prob, color="red", label="Laplace+uniform", linewidth=3)
  plt.xlabel("Slowness")
  plt.ylabel("Probability")
  plt.xlim(-10, 10)
  #plt.ylim(0, .06)
  plt.grid()
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

def gtf(val, m, s):
  return math.exp(- float(val - m) ** 2 / (2.0 * float(s) ** 2)) \
         / math.sqrt(2.0 * math.pi * float(s) ** 2)

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
  PHASEID=0
  for plot_phaseid in range(numtimedefphases):
    plt.figure()
    plt.title("Detection probability at station 6 for %s phase, surface event"
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
    # plot the data
    plt.bar(left=x_bucket_pts, height=prob, width=10, alpha=1,
            label="data 3--4 mb", color="blue")
    
    # plot the model
    mindist, maxdist = earthmodel.PhaseRange(plot_phaseid)
    x_pts = range(int(mindist), int(maxdist)+1)
    y_pts = [math.exp(netmodel.detection_logprob(1, 0, 3.5, x, SITEID,
                                                 plot_phaseid)) \
             for x in x_pts]
    plt.plot(x_pts, y_pts, label="model 3.5 mb", color="black", linewidth=3)
    
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

def visualize_arramp(options, earthmodel, netmodel,
                     detections, leb_events, leb_evlist):

  # create a dataset for each site, phase and for all sites, each phase
  false_logamps = []
  all_phase_logamps = []
  phase_logamps = dict((phase, [])
                       for phase in range(earthmodel.NumTimeDefPhases()))
  site_false_logamps = dict((sitenum, []) for sitenum in
                            range(earthmodel.NumSites()))
  site_phase_logamps = dict((sitenum, dict((phase, []) for phase in
                                      range(earthmodel.NumTimeDefPhases())))
                            for sitenum in range(earthmodel.NumSites()))

  false_slo = []
  false_az = []
  site_false_slo = dict((sitenum, []) for sitenum in
                        range(earthmodel.NumSites()))
  site_false_az =  dict((sitenum, []) for sitenum in
                        range(earthmodel.NumSites()))
  
  site_phase_false_slo = dict((sitenum, dict((phase, []) for phase in
                                             range(earthmodel.NumPhases())))
                              for sitenum in range(earthmodel.NumSites()))
  
  false_phase = dict((phase, 0) for phase in range(earthmodel.NumPhases()))
  
  site_false_phase = dict((sitenum, dict((phase, 0) for phase in
                                         range(earthmodel.NumPhases())))
                          for sitenum in range(earthmodel.NumSites()))
  
  # first, the set of true detections
  true_dets = set()
  for evnum, detlist in enumerate(leb_evlist):
    for phase, detnum in detlist:
      true_dets.add(detnum)
      sitenum = int(detections[detnum, DET_SITE_COL])
      datum = (leb_events[evnum, EV_MB_COL], leb_events[evnum, EV_DEPTH_COL],
               earthmodel.Delta(leb_events[evnum, EV_LON_COL],
                                leb_events[evnum, EV_LAT_COL], sitenum),
               np.log(detections[detnum, DET_AMP_COL]))

      all_phase_logamps.append(datum)
      phase_logamps[phase].append(datum)
      site_phase_logamps[sitenum][phase].append(datum)
  
  # next, the false detections
  for detnum in range(len(detections)):
    if detnum not in true_dets:
      sitenum = int(detections[detnum, DET_SITE_COL])
      phasenum = int(detections[detnum, DET_PHASE_COL])
      datum = np.log(detections[detnum, DET_AMP_COL])
      
      false_logamps.append(datum)
      site_false_logamps[sitenum].append(datum)
      
      false_slo.append(detections[detnum, DET_SLO_COL])
      site_false_slo[sitenum].append(detections[detnum, DET_SLO_COL])
      
      false_az.append(detections[detnum, DET_AZI_COL])
      site_false_az[sitenum].append(detections[detnum, DET_AZI_COL])

      site_phase_false_slo[sitenum][phasenum].append(
        detections[detnum, DET_SLO_COL])

      false_phase[phasenum] += 1
      site_false_phase[sitenum][phasenum] += 1
  
  # learn the overall false detection model (for all sites)
  false_wts, false_means, false_stds = utils.GMM.estimate(2, false_logamps)
  
  print "Overall False log(Amp):",
  print_2gmm(false_wts, false_means, false_stds)
  
  STEP = .1
  bins = np.arange(-7, 8, STEP)
  
  plt.figure()
  plt.title("log(amp) for false detections -- all sites")
  plt.hist(false_logamps, bins, label="data", alpha=.5)
  plt.plot(bins, [utils.GMM.evaluate(false_wts, false_means, false_stds,
                                     x+STEP/2)
                  * STEP * len(false_logamps) for x in bins], label="model",
           linewidth=3)
  plt.xlabel("log(amp)")
  plt.ylabel("frequency")
  plt.legend()
  
  SLO_STEP = 1
  slo_bins = np.arange(0, 50, SLO_STEP)
  plt.figure()
  plt.title("Slowness of false detections -- all sites")
  plt.hist(false_slo, slo_bins, label="data", alpha=.5)
  plt.xlabel("slowness")
  plt.ylabel("frequency")
  plt.legend()

  AZ_STEP = 10
  az_bins = np.arange(0, 360, AZ_STEP)
  plt.figure()
  plt.title("Azimuth of false detections -- all sites")
  plt.hist(false_az, az_bins, label="data", alpha=.5)
  plt.xlabel("azimuth")
  plt.ylabel("frequency")
  plt.legend()

  SITEID, PHASEID = 6, 0
  data = site_false_logamps[SITEID]
  wts, means, stds = utils.GMM.estimate(2, data)
  print "Siteid 6 false log(amp)"
  print_2gmm(wts, means, stds)

  plt.figure()
  plt.title("log(amp) for false detections -- site %d" % SITEID)
  plt.hist(data, bins, label="data", alpha=.5)
  plt.plot(bins, [utils.GMM.evaluate(wts, means, stds, x+STEP/2)
                  * STEP * len(data) for x in bins], label="model", linewidth=3)
  plt.xlabel("log(amp)")
  plt.ylabel("frequency")
  plt.legend()
  
  plt.figure()
  plt.title("Slowness of false detections -- site %d" % SITEID )
  plt.hist(site_false_slo[SITEID], slo_bins, label="data", alpha=.5)
  plt.xlabel("slowness")
  plt.ylabel("frequency")
  plt.legend()

  for phasenum in range(2):
    plt.figure()
    plt.title("Slowness of false detections -- site %d, phase %s"
              % (SITEID, earthmodel.PhaseName(phasenum)))
    plt.hist(site_phase_false_slo[SITEID][phasenum], slo_bins,
             label="data", alpha=.5)
    plt.xlabel("slowness")
    plt.ylabel("frequency")
    plt.legend()
  
  plt.figure()
  plt.title("Azimuth of false detections -- site %d" % SITEID)
  plt.hist(site_false_az[SITEID], az_bins, label="data", alpha=.5)
  plt.xlabel("azimuth")
  plt.ylabel("frequency")
  plt.legend()

  plt.figure()
  plt.title("Phase of false detections -- all sites")
  plt.bar(left=range(earthmodel.NumPhases()),
          height=[false_phase[p] for p in range(earthmodel.NumPhases())],
          color="blue", label="data")
  plt.xlabel("phase index")
  plt.ylabel("frequency")
  
  plt.figure()
  plt.title("Phase of false detections -- site %d" % SITEID)
  plt.bar(left=range(earthmodel.NumPhases()),
          height=[site_false_phase[SITEID][p]
                  for p in range(earthmodel.NumPhases())],
          color="blue", label="data")
  plt.xlabel("phase index")
  plt.ylabel("frequency")
  
  data = site_phase_logamps[SITEID][PHASEID]
  coeffs = learn_amp_model(data)
  print "Amp model coefficients (intercept, mb, depth, dist)"
  print_list(sys.stdout, coeffs)
  
  x_pts = range(0, 180)
  y_pts = np.array([predict_amp_model(coeffs, 3.5, 0, x) for x in x_pts])

  dist_sum = np.zeros(18, float)
  dist_count = np.ones(18, float) * 1e-6
  
  for (mb, depth, dist, logamp) in data:
    if mb >=3 and mb <=4 and depth <=50:
      idx = int(dist/10)
      dist_sum[idx] += logamp
      dist_count[idx] += 1

  dist_avg = dist_sum / dist_count
  
  plt.figure()
  plt.title("Log Amplitude at station %d for %s phase, surface event"
            % (SITEID, earthmodel.PhaseName(PHASEID)))
  plt.bar(range(0, 180, 10), dist_avg, width=10,
          label="data 3--4 mb", color="blue", linewidth=1)
  plt.plot(x_pts, y_pts, label="model 3.5 mb", color = "black", linewidth=3)
  
  plt.xlim(0,180)
  plt.xlabel("Distance (deg)")
  plt.ylabel("Avg. Log Amplitude")
  plt.legend(loc="upper right")
  if options.write is not None:
    pathname = os.path.join(options.write, "logamp_%d_%s.png"
                            % (SITEID, earthmodel.PhaseName(PHASEID)))
    print "saving fig to %s" % pathname
    plt.savefig(pathname)
  
if __name__ == "__main__":
  main("parameters")
