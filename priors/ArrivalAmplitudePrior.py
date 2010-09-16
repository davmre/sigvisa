import matplotlib.pyplot as plt
import numpy as np
from database.dataset import *
import utils.GMM, utils.LinearModel
import math

NUM_PRIOR = 100                          # number of prior points

def gtf(val, m, s):
  return math.exp(- float(val - m) ** 2 / (2.0 * float(s) ** 2)) \
         / math.sqrt(2.0 * math.pi * float(s) ** 2)

def print_2gmm(wts, means, stds):
  print "w1=%.2f m1=%.2f s1=%.2f w2=%.2f m2=%.2f s2=%.2f"\
        % (wts[0], means[0], stds[0],
           wts[1], means[1], stds[1])

def print_list(fp, list):
  for x in list:
    print >>fp, x,
  
def learn_amp_model(data):
  mb_list = []
  depth_list = []
  ttime_list = []
  ttime0_list = []
  ttime50_list = []
  ttime100_list = []
  mb_ttime_list = []
  logamp_list = []
  
  for (mb, depth, ttime, logamp) in data:
    mb_list.append(mb)
    depth_list.append(depth)
    ttime_list.append(ttime)
    ttime0_list.append(gtf(ttime, 10, 20))
    ttime50_list.append(gtf(ttime, 50, 20))
    ttime100_list.append(gtf(ttime, 90, 20))
    mb_ttime_list.append((7-mb)*ttime)
    logamp_list.append(logamp)
  
  model = utils.LinearModel.LinearModel("logamp", ["mb", "depth", "ttime"],
#                                                   "ttime0", "ttime50",
#                                                   "ttime100", "mb_ttime"],
                                        [mb_list, depth_list, ttime_list],
#                                         ttime0_list, ttime50_list,
#                                         ttime100_list, mb_ttime_list],
                                        logamp_list)
  logamp_pred = [model[a,b,c] for a,b,c \
                 in zip(mb_list, depth_list,ttime_list)
                 #, ttime0_list, ttime50_list,
                 #       ttime100_list, mb_ttime_list)
                 ]
  
  std = np.sqrt(float(sum((y1-y2) ** 2 for y1, y2
                          in zip(logamp_list, logamp_pred))) /len(logamp_pred))
  
  return model.coeffs[-1], model.coeffs[0], model.coeffs[1],\
         model.coeffs[2], std
#, model.coeffs[3], model.coeffs[4],\
#         model.coeffs[5], model.coeffs[6], std

def predict_amp_model(coeffs, mb, depth, ttime):
  return coeffs[0] + coeffs[1] * mb + coeffs[2] * depth \
         + coeffs[3] * ttime
#+ coeffs[4] * gtf(ttime, 10, 20) \
#         + coeffs[5] * gtf(ttime, 50, 20)\
#         + coeffs[6] * gtf(ttime, 90, 20)\
#         + coeffs[7] * (7 - mb) * ttime

def learn(param_filename, options, earthmodel, detections, leb_events,
          leb_evlist):

  fp = open(param_filename, "w")

  print >>fp, earthmodel.NumSites(), earthmodel.NumTimeDefPhases()
  
  
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
  
  # first, the set of true detections
  true_dets = set()
  for evnum, detlist in enumerate(leb_evlist):
    for phase, detnum in detlist:
      true_dets.add(detnum)
      sitenum = int(detections[detnum, DET_SITE_COL])
      datum = (leb_events[evnum, EV_MB_COL], leb_events[evnum, EV_DEPTH_COL],
               detections[detnum, DET_TIME_COL] - leb_events[evnum,EV_TIME_COL],
               np.log(detections[detnum, DET_AMP_COL]))

      all_phase_logamps.append(datum)
      phase_logamps[phase].append(datum)
      site_phase_logamps[sitenum][phase].append(datum)

  # next, the false detections
  for detnum in range(len(detections)):
    if detnum not in true_dets:
      sitenum = detections[detnum, DET_SITE_COL]
      datum = np.log(detections[detnum, DET_AMP_COL])

      false_logamps.append(datum)
      site_false_logamps[sitenum].append(datum)
    
  # learn the overall false detection model (for all sites)
  false_wts, false_means, false_stds = utils.GMM.estimate(2, false_logamps)

  #print_list(fp, false_wts)
  #print_list(fp, false_means)
  #print_list(fp, false_stds)
  #print >> fp
  
  if options.verbose:
    print "Overall False log(Amp):",
    print_2gmm(false_wts, false_means, false_stds)

  STEP = .1
  bins = np.arange(-7, 8, STEP)
  if options.gui:  
    plt.figure()
    plt.title("log(amp) for false detections -- all sites")
    plt.hist(false_logamps, bins, label="data", alpha=.5)
    plt.plot(bins, [utils.GMM.evaluate(false_wts, false_means, false_stds,
                                       x+STEP/2)
                    * STEP * len(false_logamps) for x in bins], label="model")

    plt.xlabel("log(amp)")
    plt.ylabel("frequency")
    plt.legend()

  # sample some points from the overal false detection empirical distribution
  false_prior = [false_logamps[np.random.randint(len(false_logamps))] for
                 i in range(NUM_PRIOR)]

  # learn the overall log(amp) model for each phase
  all_phase_prior = [all_phase_logamps[
    np.random.randint(len(all_phase_logamps))] for i in range(NUM_PRIOR)]
  phase_prior = {}
  for phase in range(earthmodel.NumTimeDefPhases()):
    data = phase_logamps[phase] + all_phase_prior
    phase_prior[phase] = [data[np.random.randint(len(data))]
                          for i in range(NUM_PRIOR)]
    coeffs = learn_amp_model(data)
    #print_list(fp, coeffs)
    #print >> fp

    if options.verbose:
      print "Overall phase %d amplitude coeffs:\n" % phase, coeffs
  
  # now, learn the site-specific false detection model and the amplitude model
  for sitenum in range(earthmodel.NumSites()):
    data = site_false_logamps[sitenum] + false_prior
    wts, means, stds = utils.GMM.estimate(2, data)
    
    print_list(fp, wts)
    print_list(fp, means)
    print_list(fp, stds)
    print >> fp

    if options.verbose:
      print "[%d]: False" % (sitenum,),
      
      print_2gmm(wts, means, stds)
    
    if sitenum == 113 and options.gui:
      plt.figure()
      plt.title("log(amp) for false detections -- WRA")
      plt.hist(data, bins, label="data", alpha=.5)
      plt.plot(bins, [utils.GMM.evaluate(wts, means, stds, x+STEP/2)
                      * STEP * len(data) for x in bins], label="model")
      plt.xlabel("log(amp)")
      plt.ylabel("frequency")
      plt.legend()

    for phase in range(earthmodel.NumTimeDefPhases()):
      data = site_phase_logamps[sitenum][phase] + phase_prior[phase]
      coeffs = learn_amp_model(data)
      print_list(fp, coeffs)
      print >> fp
      
      if options.verbose:
        print "[%d]: phase %d amplitude coeffs:\n" % (sitenum, phase), coeffs
      
  fp.close()
  return
