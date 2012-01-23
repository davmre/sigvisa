import csv
import matplotlib.pyplot as plt
import numpy as np
from database.dataset import *
import utils.GMM, utils.LinearModel
import math

NUM_PRIOR = 100                         # number of prior points

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
    ttime0_list.append(np.exp(-ttime/50.))
    ttime50_list.append(gtf(ttime, 50, 20))
    ttime100_list.append(gtf(ttime, 90, 20))
    mb_ttime_list.append((7-mb)*ttime)
    logamp_list.append(logamp)
  
  model = utils.LinearModel.LinearModel("logamp", ["mb", "depth", "ttime",
                                                   "ttime0"],
#                                                   "ttime0", "ttime50",
#                                                   "ttime100", "mb_ttime",
                                        [mb_list, depth_list, ttime_list,
                                         ttime0_list],
#                                         ttime0_list, ttime50_list,
#                                         ttime100_list, mb_ttime_list,
                                        logamp_list)
  logamp_pred = [model[a,b,c,d] for a,b,c,d \
                 in zip(mb_list, depth_list,ttime_list, ttime0_list)
                 #, ttime0_list, ttime50_list,
                 #       ttime100_list, mb_ttime_list
                 ]
  
  std = np.sqrt(float(sum((y1-y2) ** 2 for y1, y2
                          in zip(logamp_list, logamp_pred))) /len(logamp_pred))
  
  return model.coeffs[-1], model.coeffs[0], model.coeffs[1],\
         model.coeffs[2], model.coeffs[3], std
#, model.coeffs[3], model.coeffs[4],\
#         model.coeffs[5], model.coeffs[6], std

def predict_amp_model(coeffs, mb, depth, ttime):
  return coeffs[0] + coeffs[1] * mb + coeffs[2] * depth \
         + coeffs[3] * ttime + coeffs[4] * np.exp(-ttime / 50.)
#+ coeffs[4] * gtf(ttime, 10, 20) \
#         + coeffs[5] * gtf(ttime, 50, 20)\
#         + coeffs[6] * gtf(ttime, 90, 20)\
#         + coeffs[7] * (7 - mb) * ttime

def learn(param_filename, options, earthmodel, detections, leb_events,
          leb_evlist, false_dets):
  np.seterr(divide = 'raise')
  
  fp = open(param_filename, "w")

  print >>fp, earthmodel.NumSites(), earthmodel.NumTimeDefPhases()
  
  if options.datadir:
    fname = os.path.join(options.datadir, "ArrivalAmplitude.csv")
    writer = csv.writer(open(fname, "wb"))
    writer.writerow(["SITEID", "PHASEID", "MB", "DEPTH", "TTIME", "LOGAMP"])
  # create a dataset for each site, phase and for all sites, each phase
  false_logamps = []
  phase_logamps = dict((phase, [])
                       for phase in range(earthmodel.NumTimeDefPhases()))
  site_false_logamps = dict((sitenum, []) for sitenum in
                            range(earthmodel.NumSites()))
  site_phase_logamps = dict((sitenum, dict((phase, []) for phase in
                                      range(earthmodel.NumTimeDefPhases())))
                            for sitenum in range(earthmodel.NumSites()))
  
  # first, the set of true detections
  for evnum, detlist in enumerate(leb_evlist):
    for phase, detnum in detlist:
      
      # -1 => amplitude not observed
      if -1 == detections[detnum, DET_AMP_COL]:
        continue
      
      sitenum = int(detections[detnum, DET_SITE_COL])
      dist = earthmodel.Delta(leb_events[evnum, EV_LON_COL],
                              leb_events[evnum, EV_LAT_COL], sitenum)
      datum = (leb_events[evnum, EV_MB_COL], leb_events[evnum, EV_DEPTH_COL],
               detections[detnum, DET_TIME_COL] - leb_events[evnum,EV_TIME_COL],
               np.log(detections[detnum, DET_AMP_COL]))

      phase_logamps[phase].append(datum)
      site_phase_logamps[sitenum][phase].append(datum)

      if options.datadir:
        writer.writerow([sitenum, phase, leb_events[evnum, EV_MB_COL],
                         leb_events[evnum, EV_DEPTH_COL],
                         detections[detnum, DET_TIME_COL]
                         - leb_events[evnum,EV_TIME_COL],
                         np.log(detections[detnum, DET_AMP_COL])])
  
  if options.datadir:
    del writer
  
  # next, the false detections
  for detnum in false_dets:
    
    if -1 == detections[detnum, DET_AMP_COL]:
      continue
    
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
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("log(amp) for false detections -- all sites")
    plt.hist(false_logamps, bins, label="data", alpha=.5)
    plt.plot(bins, [utils.GMM.evaluate(false_wts, false_means, false_stds,
                                       x+STEP/2)
                    * STEP * len(false_logamps) for x in bins], label="model",
             linewidth=3, color="black")
    
    plt.xlabel("log(amp)")
    plt.ylabel("frequency")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "ArrivalAmplitudeFalseAllSites")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")

  # sample some points from the overall false detection empirical distribution
  false_prior = [false_logamps[np.random.randint(len(false_logamps))] for
                 i in range(NUM_PRIOR)]
  
  # learn the overall log(amp) model for each phase
  phase_coeffs = dict((phase, learn_amp_model(phase_logamps[phase]))
                       for phase in range(earthmodel.NumTimeDefPhases()))
  
  if options.verbose:
    for phase in range(earthmodel.NumTimeDefPhases()):
      print "Overall phase %d amplitude coeffs:\n" % phase, phase_coeffs[phase]
  
  if options.gui:
    all_pred, all_res = [], []
  
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
    
    if sitenum in [6, 113] and options.gui:
      plt.figure(figsize=(8,4.8))
      if not options.type1:
        plt.title("log(amp) for false detections -- %d" % sitenum)
      plt.hist(data, bins, label="data", alpha=.5)
      plt.plot(bins, [utils.GMM.evaluate(wts, means, stds, x+STEP/2)
                      * STEP * len(data) for x in bins], label="model",
               linewidth=3, color="black")
      plt.xlabel("log(amp)")
      plt.ylabel("frequency")
      plt.legend(loc="upper left")
      if options.writefig is not None:
        basename = os.path.join(options.writefig, "ArrivalAmplitudeFalseSite%d"
                                % sitenum)
        if options.type1:
          plt.savefig(basename+".pdf")
        else:
          plt.savefig(basename+".png")

    for phase in range(earthmodel.NumTimeDefPhases()):
      data = site_phase_logamps[sitenum][phase]
      if len(data) < 10:
        coeffs = phase_coeffs[phase]
      else:
        coeffs = learn_amp_model(data)
      print_list(fp, coeffs)
      print >> fp
      
      if options.verbose:
        print "[%d]: phase %d amplitude coeffs:\n" % (sitenum, phase), coeffs
        
      if options.gui:
        for (mb, depth, ttime, logamp) in site_phase_logamps[sitenum][phase]:
          pred = predict_amp_model(coeffs, mb, depth, ttime)
          res = pred - logamp
          all_pred.append(pred)
          all_res.append(res)
  fp.close()

  if options.gui:
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("All sites and phases arrival amplitude residuals")
    plt.scatter(all_pred, all_res, s=1)
    plt.xlim(-10, 10)
    plt.ylim(-6, 6)
    plt.xlabel("Predictions")
    plt.ylabel("Residuals")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "ArrivalAmplitudePredRes")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
  
  return
