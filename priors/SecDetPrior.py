import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import laplace

from utils.geog import degdiff
from database.dataset import *

# maximum time that a secondary detection can occur after the primary
MAX_SECDET_DELAY = 30.0
MIN_SECDET_DELAY = 4.0

# the maximum number of primary + secondary detections for any phase
# keep consistent with ** netvisa.h **
MAX_PHASE_DET = 15

def learn(param_fname, options, earthmodel, detections, leb_events,
          leb_seclist):
  """
  The secondary detection model consists of multiple models which predict:
  - The probability of a secondary detection
  - The time delay of the secondary detection
  - The azimuth diff
  - The slowness diff
  - The log-amp diff
  - The secondary phase
  """
  time_data, az_data, slo_data, logamp_data, snr_data = [], [], [], [], []
  sec_phase = np.zeros(earthmodel.NumPhases())

  HIGH_AMP, STEP_AMP = 500, 20
  amp_det, amp_tot = np.zeros(HIGH_AMP/STEP_AMP), np.zeros(HIGH_AMP/STEP_AMP)

  HIGH_SNR, STEP_SNR = 200, 1
  snr_det, snr_tot = np.zeros(HIGH_SNR/STEP_SNR), np.zeros(HIGH_SNR/STEP_SNR)
  
  
  # we will compute the probability of a secondary detection at each position
  # i.e. first secondary, second secondary etc.
  pos_det, pos_tot = np.zeros(MAX_PHASE_DET), np.zeros(MAX_PHASE_DET)

  detcnt, totcnt = 0, 0
  
  for seclist in leb_seclist:
    for phase_detlist in seclist:
      detlist = phase_detlist[1:]
      for pos, detnum in enumerate(detlist):
        det = detections[detnum]
        
        if det[DET_AMP_COL] < HIGH_AMP:
          amp_tot[ det[DET_AMP_COL] // STEP_AMP ] += 1

        if det[DET_SNR_COL] < HIGH_SNR:
          snr_tot[ det[DET_SNR_COL] // STEP_SNR ] += 1

        pos_tot[pos] += 1
        totcnt += 1
        
        if (pos+1) < len(detlist):
          secdet = detections[detlist[pos+1]]

          delta = secdet[DET_TIME_COL] - det[DET_TIME_COL]
          assert(delta >= MIN_SECDET_DELAY and delta <= MAX_SECDET_DELAY)
          time_data.append(delta)
          az_data.append(degdiff(det[DET_AZI_COL], secdet[DET_AZI_COL]))
          slo_data.append(secdet[DET_SLO_COL] - det[DET_SLO_COL])
          logamp_data.append(np.log(secdet[DET_AMP_COL])
                             - np.log(det[DET_AMP_COL]))
          snr_data.append(secdet[DET_SNR_COL] - det[DET_SNR_COL])
          sec_phase[int(secdet[DET_PHASE_COL])] += 1

          if det[DET_AMP_COL] < HIGH_AMP:
            amp_det[ det[DET_AMP_COL] // STEP_AMP ] += 1

          if det[DET_SNR_COL] < HIGH_SNR:
            snr_det[ det[DET_SNR_COL] // STEP_SNR ] += 1

          pos_det[pos] += 1
          detcnt += 1

  detprob = float(detcnt) / totcnt
  time_shape, _, time_scale = gamma.fit(time_data, loc=MIN_SECDET_DELAY-.1)
  az_loc, az_scale = laplace.fit(az_data)
  slo_loc, slo_scale = laplace.fit(slo_data)
  logamp_loc, logamp_scale = laplace.fit(logamp_data)

  # fit the SNR as two exponential distribution around 0
  pos_snr_data = [x for x in snr_data if x >= 0 and x < 20]
  neg_snr_data = [-x for x in snr_data if x < 0 and x > -20]
  snr_prob_plus = float(len(pos_snr_data)) / (len(pos_snr_data)
                                              + len(neg_snr_data))
  snr_lambda_plus = 1.0 / np.average(pos_snr_data)
  snr_lambda_minus = 1.0 / np.average(neg_snr_data)
  
  sec_phase += 1.0
  sec_phase /= sec_phase.sum()
  
  amp_tot += .0001

  snr_tot += .0001
  
  pos_tot += .0001

  print "Secondary Detection :"
  print "Probability", detprob
  print "Time", time_shape, MIN_SECDET_DELAY, time_scale, MAX_SECDET_DELAY
  print "Time lost probability mass:", \
        gamma.sf(MAX_SECDET_DELAY, time_shape, MIN_SECDET_DELAY, time_scale)
  print "Azimuth", az_loc, az_scale
  print "Slowness", slo_loc, slo_scale
  print "Logamp", logamp_loc, logamp_scale
  print "SNR", snr_prob_plus, snr_lambda_plus, snr_lambda_minus
  print "Phase:",
  for s in sec_phase:
    print s,
  print
  
  if options.gui:
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Secondary detection probability vs position")
    plt.bar(np.arange(MAX_PHASE_DET), pos_det / pos_tot, 1)
    plt.ylabel("probability")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "SecDetProbPos")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Secondary detection probability vs primary detection amp")
    plt.bar(np.arange(0, HIGH_AMP, STEP_AMP), amp_det / amp_tot, STEP_AMP)
    plt.ylabel("probability")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "SecDetProbAmp")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")

    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Secondary detection probability vs primary detection SNR")
    plt.bar(np.arange(0, HIGH_SNR, STEP_SNR), snr_det / snr_tot, STEP_SNR)
    plt.ylabel("probability")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "SecDetProbSNR")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
        
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Secondary detection time delay")
    _,xpts,_ = plt.hist(time_data, 100, normed=True, label="data", alpha=.5)
    plt.plot(xpts, gamma.pdf(xpts, time_shape, MIN_SECDET_DELAY, time_scale),
             linewidth=3, label="model")
    plt.xlabel("seconds")
    plt.ylabel("frequency")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "SecDetTime")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")

    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Secondary detection azimuth difference")
    _,xpts,_ = plt.hist(az_data, 100, normed=True, label="data", alpha=.5)
    plt.plot(xpts, laplace.pdf(xpts, az_loc, az_scale),
             linewidth=3, label="model")
    plt.xlabel("degrees")
    plt.ylabel("frequency")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "SecDetAzimuth")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")

    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Secondary detection slowness difference")
    _,xpts,_ = plt.hist(slo_data, 100, normed=True, label="data", alpha=.5)
    plt.plot(xpts, laplace.pdf(xpts, slo_loc, slo_scale),
             linewidth=3, label="model")
    plt.xlabel("seconds per degree")
    plt.ylabel("frequency")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "SecDetSlowness")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Secondary detection logamp difference")
    _,xpts,_ = plt.hist(logamp_data, 100, normed=True, label="data", alpha=.5)
    plt.plot(xpts, laplace.pdf(xpts, logamp_loc, logamp_scale), linewidth=3,
             label="model")
    plt.ylabel("frequency")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "SecDetLogamp")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")

    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Secondary detection SNR difference")
    _,xpts,_ = plt.hist(snr_data, np.arange(- 4/snr_lambda_minus,
                                            + 4/snr_lambda_plus, STEP_SNR),
                        normed=True, label="data", alpha=.5)
    def double_exp_pdf(x, prob_plus, lambda_plus, lambda_minus):
      if x >= 0:
        return prob_plus * lambda_plus * np.exp(- lambda_plus * x)
      else:
        return (1-prob_plus) * lambda_minus * np.exp(lambda_minus * x)
    
    plt.plot(xpts, [double_exp_pdf(x, snr_prob_plus, snr_lambda_plus,
                                   snr_lambda_minus) for x in xpts],
             linewidth=3, label="model")
    plt.ylabel("frequency")
    plt.legend(loc="upper left")    
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "SecDetSNR")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")


    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Secondary detection phase")
    phaserange = np.arange(earthmodel.NumPhases())
    phasenames = [earthmodel.PhaseName(i) for i in phaserange]
    plt.bar(phaserange, sec_phase, 1)
    plt.ylabel("probability")
    plt.xticks(phaserange+.5, phasenames)
    plt.xlabel("phase")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "SecDetPhase")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    
  
  fp = open(param_fname, "w")
  
  print >> fp, detprob
  # we need the location of the Gamma to not be at exactly zero
  print >> fp, time_shape, MIN_SECDET_DELAY-.1, time_scale, MAX_SECDET_DELAY
  print >> fp, az_loc, az_scale
  print >> fp, slo_loc, slo_scale
  print >> fp, logamp_loc, logamp_scale
  print >> fp, snr_prob_plus, snr_lambda_plus, snr_lambda_minus
  print >> fp, len(sec_phase)
  for s in sec_phase:
    print >> fp, s,
  print
  fp.close()

def compute_secondary_dets(earthmodel, detections, leb_events, leb_evlist):
  # first create a copy of the leb_evlist and map each associated
  # detection to its phase_detlist
  leb_seclist = []
  detnum_map = {}
  for detlist in leb_evlist:
    seclist = []
    for phase, detnum in detlist:
      phase_detlist = [phase, detnum]
      seclist.append(phase_detlist)
      detnum_map[detnum] = phase_detlist
    leb_seclist.append(seclist)
  
  for detnum, det in enumerate(detections):
    # only an associated detection can have a secondary
    if detnum not in detnum_map or len(detnum_map[detnum]) == (MAX_PHASE_DET+1):
      continue
    
    for secdetnum in xrange(detnum+1, len(detections)):
      
      secdet = detections[secdetnum]
      
      if secdet[DET_TIME_COL] > (det[DET_TIME_COL] + MAX_SECDET_DELAY):
        break
      
      if secdet[DET_SITE_COL] != det[DET_SITE_COL]:
        continue
      
      # if we find an associated detection at this site then that one
      # owns the future secondary detections
      if secdetnum in detnum_map:
        break
      
      if secdet[DET_TIME_COL] < (det[DET_TIME_COL] + MIN_SECDET_DELAY):
        continue
      
      # we have  a secondary detection
      phase_detlist = detnum_map[detnum]
      assert(phase_detlist[-1] == detnum)
      
      phase_detlist.append(secdetnum)
      detnum_map[secdetnum] = phase_detlist
      break

  # convert the lists back to tuples
  return [[tuple(phase_detlist) for phase_detlist in seclist]
          for seclist in leb_seclist]
