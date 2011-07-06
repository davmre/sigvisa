# visualize the priors
import os, sys, time, math
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
# for type 1 fonts
#plt.rcParams['ps.useafm'] = True

from utils.draw_earth import draw_events, draw_earth, draw_density
from database.dataset import *
import netvisa, learn
from results.compare import *
from utils import Laplace
import utils.GMM, utils.LinearModel, utils.gamma
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
  parser.add_option("-n", "--noise", dest="noise", default=False,
                    action = "store_true",
                    help = "visualize noise parameters (False)")
  parser.add_option("-c", "--correlate", dest="correlate", default=False,
                    action = "store_true",
                    help = "visualize correlations (False)")
  parser.add_option("-l", "--location", dest="location", default=False,
                    action = "store_true",
                    help = "visualize location prior (False)")
  parser.add_option("-d", "--detection", dest="detection", default=False,
                    action = "store_true",
                    help = "visualize detection prior (False)")
  parser.add_option("-w", "--write", dest="write", default=None,
                    help = "location to write figures to",
                    metavar="DIRNAME")
  parser.add_option("-i", "--visa_leb_runid", dest="visa_leb_runid",
                    default=None, help = "Visa runid to be treated as leb",
                    metavar="RUNID")
  parser.add_option("-1", "--type1", dest="type1", default=False,
                    action = "store_true",
                    help = "Type 1 fonts (False)")
  (options, args) = parser.parse_args()

  if options.train:
    label = "training"
  else:
    label = "validation"
    
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
              sel3_evlist, site_up, sites, phasenames, phasetimedef \
              = read_data(label, visa_leb_runid=options.visa_leb_runid)
  
  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  if options.type1:
    plt.rcParams['text.usetex'] = True

  # if no options is selected then select all options
  if (not options.arrival and not options.location and not options.detection
      and not options.noise and not options.correlate):
    options.arrival = options.location = options.detection = \
                      options.noise = options.correlate = True

  if options.correlate:
    print "visualize correlations of P and S residuals"
    visualize_p_s_res_corr(options, earthmodel, netmodel,
                           detections, leb_events, leb_evlist)
  if options.noise:
    print "visualize noise"
    visualize_noise(options, earthmodel, netmodel,
                    detections, leb_events, leb_evlist)
    
    visualize_arramp(options, earthmodel, netmodel,
                     detections, leb_events, leb_evlist)    
    
    
  if options.arrival:
    print "visualizing arrival parameters"

    visualize_arr_slo_az(options, earthmodel, netmodel,
                         detections, leb_events, leb_evlist)
    
    visualize_arrtime(options, earthmodel, netmodel,
                      detections, leb_events, leb_evlist)
    
    visualize_corr_ttime(options, earthmodel, netmodel,
                         detections, leb_events, leb_evlist)
    
    visualize_arrphase(options, earthmodel, netmodel,
                       detections, leb_events, leb_evlist)
    
    visualize_arraz(options, earthmodel, netmodel,
                    detections, leb_events, leb_evlist)
    
    visualize_arrslo(options, earthmodel, netmodel,
                     detections, leb_events, leb_evlist)


  if options.location:
    print "visualizing location prior"
    
    visualize_location_prior(options, earthmodel, netmodel)

  if options.detection:
    print "visualizing detection prior"
    
    visualize_detection(options, earthmodel, netmodel, start_time, end_time,
                        detections, leb_events, leb_evlist, site_up)

  plt.show()

def visualize_p_s_res_corr(options, earthmodel, netmodel,
                           detections, leb_events, leb_evlist):
  all_p_res, all_s_res = [], []
  
  # for each phase the list of p travel time and phase travel time
  all_ttime_phase_p = dict((i, ([], [])) for i in
                           range(earthmodel.NumTimeDefPhases()))
  
  for evnum, event in enumerate(leb_events):
    
    site_p_res = dict((i, None) for i in range(earthmodel.NumSites()))
    site_s_res = dict((i, None) for i in range(earthmodel.NumSites()))
    
    site_p_ttime = dict((i, None) for i in range(earthmodel.NumSites()))

    for phaseid, detnum in leb_evlist[evnum]:
      evlon, evlat, evdepth, evtime = event[EV_LON_COL], event[EV_LAT_COL],\
                                      event[EV_DEPTH_COL], event[EV_TIME_COL]
      siteid = int(detections[detnum, DET_SITE_COL])
      pred_arrtime =  earthmodel.ArrivalTime(evlon, evlat, evdepth, evtime,
                                             phaseid, siteid)
      if pred_arrtime > 0:
        res = detections[detnum, DET_TIME_COL] - pred_arrtime
        ttime = detections[detnum, DET_TIME_COL] - evtime

        if earthmodel.PhaseName(phaseid) == 'P':
          site_p_res[siteid] = res
          site_p_ttime[siteid] = ttime
          
        elif earthmodel.PhaseName(phaseid) == 'S':
          site_s_res[siteid] = res

        if site_p_res[siteid] is not None:
          all_ttime_phase_p[phaseid][0].append(ttime)
          all_ttime_phase_p[phaseid][1].append(site_p_ttime[siteid])

    for siteid in range(earthmodel.NumSites()):
      if site_p_res[siteid] is not None and site_s_res[siteid] is not None:
        all_p_res.append(site_p_res[siteid])
        all_s_res.append(site_s_res[siteid])

  plt.figure()
  plt.title("P and S phase residuals, all stations")
  plt.scatter(all_p_res, all_s_res)
  plt.xlabel("P phase residual (time in secs)")
  plt.ylabel("S phase residual (time in secs)")
  plt.grid()

  for phaseid in range(earthmodel.NumTimeDefPhases()):
    if len(all_ttime_phase_p[phaseid][1]):  
      plt.figure()
      plt.title("Phase %s ttime vs P ttime" % earthmodel.PhaseName(phaseid))
      plt.scatter(all_ttime_phase_p[phaseid][1], all_ttime_phase_p[phaseid][0])
      plt.xlabel("P phase ttime (secs)")
      plt.ylabel("%s phase ttime (secs)" % earthmodel.PhaseName(phaseid))
  
def visualize_corr_ttime(options, earthmodel, netmodel,
                         detections, leb_events, leb_evlist):
  ratios = []
  p_ttimes = []
  s_ttimes = []
  s_iaspei_ttimes = []
  for evnum, event in enumerate(leb_events):
    p_ttime, s_ttime, s_iaspei_ttime = {}, {}, {}
    for phaseid, detnum in leb_evlist[evnum]:
      ttime = detections[detnum, DET_TIME_COL] - event[EV_TIME_COL]
      if earthmodel.PhaseName(phaseid) == 'P':
        p_ttime[detections[detnum, DET_SITE_COL]] = ttime
      elif earthmodel.PhaseName(phaseid) == 'S':
        s_ttime[detections[detnum, DET_SITE_COL]] = ttime
        iaspei_arrtime = earthmodel.ArrivalTime(event[EV_LON_COL],
                                                event[EV_LAT_COL],
                                                event[EV_DEPTH_COL],
                                                event[EV_TIME_COL], phaseid,
                                         int(detections[detnum,DET_SITE_COL]))
        assert(iaspei_arrtime > 0)
        s_iaspei_ttime[detections[detnum, DET_SITE_COL]] = iaspei_arrtime \
                                                           - event[EV_TIME_COL]
    for sta in p_ttime.iterkeys():
      if sta in s_ttime:
        ratios.append(s_ttime[sta]/p_ttime[sta])
        p_ttimes.append(p_ttime[sta])
        s_ttimes.append(s_ttime[sta])
        s_iaspei_ttimes.append(s_iaspei_ttime[sta])

  
  plt.figure()
  plt.title("IASPEI residuals -- S phase")
  plt.hist(np.array(s_ttimes) - np.array(s_iaspei_ttimes),
           np.arange(-25, 25, .25))
  plt.xlabel("time (sec)")
  plt.ylabel("frequency")
  plt.xlim(-25,25)
  plt.ylim(0,75)

  plt.figure()
  plt.title("S and P travel times")
  plt.scatter(p_ttimes, s_ttimes)
  plt.xlabel("P Travel Time")
  plt.ylabel("S Travel Time")

  
  s_model = utils.LinearModel.LinearModel("s_ttime", ["p_ttime"],
                                          [p_ttimes], s_ttimes)
  print "S model", s_model.coeffs
  s_pred_ttimes = [s_model[a] for a in p_ttimes]
  
  plt.figure()
  plt.title("Predicted residuals -- S phase")
  plt.hist(np.array(s_ttimes) - np.array(s_pred_ttimes),
           np.arange(-25, 25, .25))
  plt.xlabel("time (sec)")
  plt.ylabel("frequency")
  plt.xlim(-25,25)
  plt.ylim(0,75)
  
def visualize_arrphase(options, earthmodel, netmodel,
                       detections, leb_events, leb_evlist):
  true_false_phases = np.zeros((earthmodel.NumTimeDefPhases(),
                                earthmodel.NumPhases()))
  for evnum, event in enumerate(leb_events):
    for phaseid, detnum in leb_evlist[evnum]:
      true_false_phases[phaseid, int(detections[detnum, DET_PHASE_COL])] += 1
  
  # add-one smoothing
  true_false_phases += 1;
  
  true_false_phases = (true_false_phases.T
                       / true_false_phases.sum(axis=1).astype(float)).T

  #for i in range(earthmodel.NumTimeDefPhases()):
  #  print "[%d]:" % i, true_false_phases[i]
  
  x = np.arange(0, earthmodel.NumPhases(), .1)
  y = np.arange(0, earthmodel.NumTimeDefPhases(), .1)
  X,Y = np.meshgrid(x,y)
  
  Z = np.zeros((len(y), len(x)))
  
  for xi, x_val in enumerate(x):
    for yi, y_val in enumerate(y):
      Z[yi, xi] = true_false_phases[int(y_val), int(x_val)]

  x_ticks = np.arange(earthmodel.NumPhases())
  y_ticks = np.arange(earthmodel.NumTimeDefPhases())
  
  plt.figure()
  plt.title("Phase confusion matrix -- all stations")
  plt.contourf(X, Y, Z)
  plt.plot([0,13.9], [0,13.9], color="black", linewidth=3)
  plt.xticks(x_ticks + .5, [earthmodel.PhaseName(i) for i in x_ticks],
             rotation="vertical")
  plt.yticks(y_ticks + .5, [earthmodel.PhaseName(i) for i in y_ticks])
  plt.xlabel("Arrival Phase")
  plt.ylabel("True Phase")
  plt.subplots_adjust(bottom=.17)        # extra space for the ticks and label
  if options.write is not None:
    pathname = os.path.join(options.write, "phase_confusion_all.png")
    print "saving fig to %s" % pathname
    plt.savefig(pathname)


  x = np.arange(earthmodel.NumPhases())
  y = np.arange(earthmodel.NumTimeDefPhases())
  X,Y = np.meshgrid(x,y)

  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  plt.title("Phase confusion matrix -- all stations")
  ax = fig.add_subplot(111, projection='3d')
  ax.bar3d(list(X.flat), list(Y.flat), list((true_false_phases * 0).flat),
           1, 1, list(true_false_phases.flat), alpha=1)
  #ax.set_xticks(x, [earthmodel.PhaseName(i) for i in x])
  #ax.set_yticks(y, [earthmodel.PhaseName(i) for i in y])
  ax.set_xlabel("Arrival Phase")
  ax.set_ylabel("True Phase")
  ax.set_zlabel("Probability")

def plot_ttime_residuals(netmodel,residuals, title, SITEID, PHASEID, MIN, MAX,
                         STEP):
  loc, scale = Laplace.estimate(residuals)
  mix_prob, mix_loc, mix_scale = Laplace.estimate_laplace_uniform_dist(
    residuals, MIN, MAX)

  bins = np.arange(MIN, MAX+STEP, STEP)
  data_prob = np.zeros(len(bins))
  for r in residuals:
    if r < MAX+STEP:
      data_prob[int((r-MIN)/STEP)] += 1

  data_prob /= sum(data_prob)

  x_vals = np.arange(MIN, MAX+STEP, STEP/10.)
  curve = [np.exp(Laplace.ldensity(loc, scale, x))
          *(STEP) for x in x_vals]
  mix_curve = [np.exp(Laplace.ldensity_laplace_uniform_dist(
    mix_prob, mix_loc, mix_scale, MIN, MAX, x)) * (STEP) for x in x_vals]
  
  plt.figure()
  plt.title(title)
  plt.bar(left=bins, height=data_prob, width=STEP, alpha=1, color="blue",
          linewidth=0,
          label="data")
  plt.plot(x_vals, curve, color="black", label="Laplace", linewidth=3)
  plt.plot(x_vals, mix_curve, color="red", label="Laplace+uniform", linewidth=3)
  plt.xlabel("Time")
  plt.ylabel("Probability")
  plt.xlim(MIN,MAX)
  plt.ylim(0,.15)
  plt.grid()
  plt.legend()
  
def visualize_arrtime(options, earthmodel, netmodel,
                      detections, leb_events, leb_evlist):
  MIN=-7
  STEP=.2
  MAX=7

  SITEID=6                              # ASAR
  PHASEID=0
  residuals = []
  residuals1, residuals2, residuals3, residuals4 = [], [], [], []
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
            dist = earthmodel.Delta(event[EV_LON_COL], event[EV_LAT_COL],
                                    SITEID)
            if dist < 40:
              residuals1.append(res)
            elif dist < 60:
              residuals2.append(res)
            elif dist < 80:
              residuals3.append(res)
            else:
              residuals4.append(res)
            

  plot_ttime_residuals(netmodel,residuals,
                       "TTime Residuals, P phase, site 6, all distances",
                       SITEID, PHASEID, MIN, MAX, STEP)
  plot_ttime_residuals(netmodel,residuals1,
                       "TTime Residuals, P phase, site 6, distances < 40",
                       SITEID, PHASEID, MIN, MAX, STEP)
  plot_ttime_residuals(netmodel,residuals2,
                       "TTime Residuals, P phase, site 6, distances 40 -- 60",
                       SITEID, PHASEID, MIN, MAX, STEP)
  plot_ttime_residuals(netmodel,residuals3,
                       "TTime Residuals, P phase, site 6, distances 60 -- 80",
                       SITEID, PHASEID, MIN, MAX, STEP)
  plot_ttime_residuals(netmodel,residuals4,
                       "TTime Residuals, P phase, site 6, distances > 80",
                       SITEID, PHASEID, MIN, MAX, STEP)

  return

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

def visualize_noise(options, earthmodel, netmodel,
                     detections, leb_events, leb_evlist):
  true_detections = set()
  for evnum, event in enumerate(leb_events):
    for phaseid, detnum in leb_evlist[evnum]:
      true_detections.add(detnum)

  plt.figure()
  plt.title("Overall noise distribution")
  plt.hist([detections[detnum, DET_TIME_COL] for detnum in
            range(len(detections)) if detnum not in true_detections])
  plt.xlabel("Time")
  plt.ylabel("Count")

  plt.figure()
  plt.title("Site 0 noise distribution")
  plt.hist([detections[detnum, DET_TIME_COL] for detnum in
            range(len(detections)) if detnum not in true_detections and
            detections[detnum, DET_SITE_COL] == 0])
  plt.xlabel("Time")
  plt.ylabel("Count")

  plt.figure()
  plt.title("Site 6 noise distribution")
  plt.hist([detections[detnum, DET_TIME_COL] for detnum in
            range(len(detections)) if detnum not in true_detections and
            detections[detnum, DET_SITE_COL] == 6])
  plt.xlabel("Time")
  plt.ylabel("Count")
  

def visualize_arr_slo_az(options, earthmodel, netmodel,
                         detections, leb_events, leb_evlist):
  SITEID=6                              # ASAR
  PHASEID=0
  
  true_detections = set()
  for evnum, event in enumerate(leb_events):
    for phaseid, detnum in leb_evlist[evnum]:
      true_detections.add(detnum)

  plt.figure()
  plt.title("Azimuth and Slowness of false detections -- all sites")
  plt.scatter([detections[detnum, DET_AZI_COL] for detnum in
               range(len(detections)) if detnum not in true_detections],
              [detections[detnum, DET_SLO_COL] for detnum in
               range(len(detections)) if detnum not in true_detections],
              s=1)
  plt.xlabel("Azimuth")
  plt.ylabel("Slowness")

  plt.figure()
  plt.title("Azimuth and Slowness of false detections -- site 6")
  plt.scatter([detections[detnum, DET_AZI_COL] for detnum in
               range(len(detections)) if detnum not in true_detections
               and detections[detnum, DET_SITE_COL] == 6],
              [detections[detnum, DET_SLO_COL] for detnum in
               range(len(detections)) if detnum not in true_detections
               and detections[detnum, DET_SITE_COL] == 6],
              s=1)
  plt.xlabel("Azimuth")
  plt.ylabel("Slowness")

  plt.figure()
  plt.title("Azimuth and Slowness of false detections -- site 1")
  plt.scatter([detections[detnum, DET_AZI_COL] for detnum in
               range(len(detections)) if detnum not in true_detections
               and detections[detnum, DET_SITE_COL] == 1],
              [detections[detnum, DET_SLO_COL] for detnum in
               range(len(detections)) if detnum not in true_detections
               and detections[detnum, DET_SITE_COL] == 1],
              s=1)
  plt.xlabel("Azimuth")
  plt.ylabel("Slowness")
  
  
def visualize_arrslo(options, earthmodel, netmodel,
                     detections, leb_events, leb_evlist):
  MIN=-40
  STEP=.2
  MAX=40

  SITEID=6                              # ASAR
  PHASEID=0
  residuals = []
  true_detections = set()
  for evnum, event in enumerate(leb_events):
    for phaseid, detnum in leb_evlist[evnum]:
      true_detections.add(detnum)
      if phaseid == PHASEID and detections[detnum, DET_SITE_COL] == SITEID:
        evlon, evlat, evdepth = event[EV_LON_COL], event[EV_LAT_COL],\
                                event[EV_DEPTH_COL]
        pred_arrslo =  earthmodel.ArrivalSlowness(evlon, evlat, evdepth,
                                                  PHASEID, SITEID)
        res = pred_arrslo - detections[detnum, DET_SLO_COL]
        if res > MIN and res < MAX:
          residuals.append(res)
  
  true_det_list = list(true_detections)
  
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

  plt.figure()
  plt.title("Slowness of true and false detections at all stations")
  plt.hist([detections[detnum, DET_SLO_COL] for detnum in range(len(detections))
            if detnum not in true_detections], label="false")
  plt.hist([detections[detnum, DET_SLO_COL] for detnum in true_detections],
           label="true")
  plt.xlabel("Slowness")
  plt.ylabel("Count")
  plt.legend()

  plt.figure()
  plt.title("Slowness of true and false detections at site 0")
  plt.hist([detections[detnum, DET_SLO_COL] for detnum in range(len(detections))
            if detnum not in true_detections
            and detections[detnum, DET_SITE_COL] == 0], label="false")
  plt.hist([detections[detnum, DET_SLO_COL] for detnum in true_det_list
            if detections[detnum, DET_SITE_COL] == 0],
           label="true")
  plt.xlabel("Slowness")
  plt.ylabel("Count")
  plt.legend()

  plt.figure()
  plt.title("Slowness of true and false detections at site 6")
  plt.hist([detections[detnum, DET_SLO_COL] for detnum in range(len(detections))
            if detnum not in true_detections
            and detections[detnum, DET_SITE_COL] == 6], label="false")
  plt.hist([detections[detnum, DET_SLO_COL] for detnum in true_det_list
            if detections[detnum, DET_SITE_COL] == 6],
           label="true")
  plt.xlabel("Slowness")
  plt.ylabel("Count")
  plt.legend()

  plt.figure()
  plt.title("Slowness of false detections by site")
  plt.scatter([detections[detnum, DET_SLO_COL] for detnum in
               range(len(detections)) if detnum not in true_detections],
              [detections[detnum, DET_SITE_COL] for detnum in
               range(len(detections)) if detnum not in true_detections], s=1)
  plt.xlabel("Slowness")
  plt.ylabel("Site index")

  plt.figure()
  plt.title("Slowness of true detections by site")
  plt.scatter([detections[detnum, DET_SLO_COL] for detnum in true_det_list],
              [detections[detnum, DET_SITE_COL] for detnum in true_det_list],
              s=1)
  plt.xlabel("Slowness")
  plt.ylabel("Site index")
  

def visualize_location_prior(options, earthmodel, netmodel):
  LON_BUCKET_SIZE = .25
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

  # for publications don't add a title
  if options.type1:
    bmap = draw_earth("")
  else:
    bmap = draw_earth("Log Prior Density of Events")
    
  draw_density(bmap, lon_arr, lat_arr, prob, levels=30, nolines=True,
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
  # TODO: learn the false arrival amplitude of each site as Gamma distribution
  #
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
               #earthmodel.Delta(leb_events[evnum, EV_LON_COL],
               #                 leb_events[evnum, EV_LAT_COL], sitenum),
               detections[detnum, DET_TIME_COL] - leb_events[evnum,EV_TIME_COL],
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
  false_params = utils.gamma.estimate(false_logamps)
  
  print "Overall False log(Amp):",
  print_2gmm(false_wts, false_means, false_stds)
  print "OR as Gamma: location, shape, scale:", false_params
  
  STEP = .1
  bins = np.arange(MIN_LOGAMP, MAX_LOGAMP, STEP)
  
  plt.figure()
  plt.title("log(amp) for false detections -- all sites")
  plt.hist(false_logamps, bins, label="data", alpha=.5)
  plt.plot(bins, [utils.GMM.evaluate(false_wts, false_means, false_stds,
                                     x+STEP/2)
                  * STEP * len(false_logamps) for x in bins], label="model",
           linewidth=3)
  plt.plot(bins, [np.exp(utils.gamma.ldensity(false_params, x + STEP/2))
                  * STEP * len(false_logamps) for x in bins], label="gamma",
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
  params = utils.gamma.estimate(data)
  print "Siteid 6 false log(amp)"
  print_2gmm(wts, means, stds)
  print "OR Gamma:", params

  plt.figure()
  plt.title("log(amp) for false detections -- site %d" % SITEID)
  plt.hist(data, bins, label="data", alpha=.5)
  plt.plot(bins, [utils.GMM.evaluate(wts, means, stds, x+STEP/2)
                  * STEP * len(data) for x in bins], label="model",linewidth=3)
  plt.plot(bins, [np.exp(utils.gamma.ldensity(params, x+STEP/2))
                  * STEP * len(data) for x in bins], label="gamma",linewidth=3)
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
  plt.xticks(np.arange(earthmodel.NumPhases()),
             [earthmodel.PhaseName(i)
              for i in np.arange(earthmodel.NumPhases())], rotation="vertical")
  plt.xlabel("Arrival Phase")
  plt.ylabel("frequency")
  plt.subplots_adjust(bottom=.17)        # extra space for the ticks and label
  
  plt.figure()
  plt.title("Phase of false detections -- site %d" % SITEID)
  plt.bar(left=range(earthmodel.NumPhases()),
          height=[site_false_phase[SITEID][p]
                  for p in range(earthmodel.NumPhases())],
          color="blue", label="data")
  plt.xticks(np.arange(earthmodel.NumPhases()),
             [earthmodel.PhaseName(i)
              for i in np.arange(earthmodel.NumPhases())], rotation="vertical")
  plt.xlabel("Arrival Phase")
  plt.ylabel("frequency")
  plt.subplots_adjust(bottom=.17)        # extra space for the ticks and label
  
  data = site_phase_logamps[SITEID][PHASEID]
  coeffs = learn_amp_model(data)
  print "Amp model coefficients (intercept, mb, depth, ttime)"
  print_list(sys.stdout, coeffs)
  print
  
  x_pts = range(200, 800, 10)
  y_pts = np.array([predict_amp_model(coeffs, 3.5, 0, x) for x in x_pts])

  dist_sum = np.zeros(90, float)
  dist_count = np.ones(90, float) * 1e-6
  
  for (mb, depth, ttime, logamp) in data:
    if mb >=3 and mb <=4 and depth <=50:
      idx = int(ttime/10)
      dist_sum[idx] += logamp
      dist_count[idx] += 1

  dist_avg = dist_sum / dist_count
  
  plt.figure()
  plt.title("Log Amplitude at station %d for %s phase, surface event"
            % (SITEID, earthmodel.PhaseName(PHASEID)))
  plt.bar(range(0, 900, 10), dist_avg, width=10,
          label="data 3--4 mb", color="blue", linewidth=1)
  plt.plot(x_pts, y_pts, label="model 3.5 mb", color = "black", linewidth=3)
  
  plt.xlim(0,900)
  #plt.xlabel("Distance (deg)")
  plt.xlabel("Travel Time (sec)")
  plt.ylabel("Avg. Log Amplitude")
  plt.legend(loc="lower left")
  if options.write is not None:
    pathname = os.path.join(options.write, "logamp_%d_%s_3_4.png"
                            % (SITEID, earthmodel.PhaseName(PHASEID)))
    print "saving fig to %s" % pathname
    plt.savefig(pathname)

  x_pts = range(15, 95)
  y_pts = np.array([predict_amp_model(coeffs, 4.5, 0, x) for x in x_pts])

  dist_sum = np.zeros(90, float)
  dist_count = np.ones(90, float) * 1e-6
  
  for (mb, depth, ttime, logamp) in data:
    if mb >=4 and mb <=5 and depth <=50:
      idx = int(ttime/10)
      dist_sum[idx] += logamp
      dist_count[idx] += 1

  dist_avg = dist_sum / dist_count
  
  plt.figure()
  plt.title("Log Amplitude at station %d for %s phase, surface event"
            % (SITEID, earthmodel.PhaseName(PHASEID)))
  plt.bar(range(0, 900, 10), dist_avg, width=10,
          label="data 4--5 mb", color="blue", linewidth=1)
  plt.plot(x_pts, y_pts, label="model 4.5 mb", color = "black", linewidth=3)
  
  plt.xlim(0,900)
  plt.xlabel("Distance (deg)")
  plt.ylabel("Avg. Log Amplitude")
  plt.legend(loc="upper right")
  if options.write is not None:
    pathname = os.path.join(options.write, "logamp_%d_%s_4_5.png"
                            % (SITEID, earthmodel.PhaseName(PHASEID)))
    print "saving fig to %s" % pathname
    plt.savefig(pathname)
  
if __name__ == "__main__":
  main("parameters")
