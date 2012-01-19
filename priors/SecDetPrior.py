import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import laplace
import sys

from utils.geog import degdiff
from database.dataset import *

# maximum time that a secondary detection can occur after the primary
MAX_SECDET_DELAY = 30.0
MIN_SECDET_DELAY = 4.0

# the maximum number of primary + secondary detections for any phase
# keep consistent with ** netvisa.h **
MAX_PHASE_DET = 15

LOW_LOGAMP, HIGH_LOGAMP, STEP_LOGAMP = -4, 10, .25
HIGH_SNR, STEP_SNR = 200, 1

def learn(param_fname, options, earthmodel, detections, leb_events,
          leb_evlist):
  """
  The secondary detection model consists of multiple models which predict:
  - The probability of a secondary detection
  - The time delay of the secondary detection
  - The azimuth diff
  - The slowness diff
  - The log-amp diff
  - The secondary phase
  """
  # first identify the set of associated arrivals
  assoc_dets = set()
  for evlist in leb_evlist:
    for ph,detnum in evlist:
      assoc_dets.add(detnum)
  # now identify two types of adjacent arrivals
  assoc_coda_detpair = []               # assoc det -> coda det
  coda_coda_detpair = []                # coda det -> coda det
  for detnum, det in enumerate(detections):
    sec_found = False
    for secdetnum in xrange(detnum+1, len(detections)):
      secdet = detections[secdetnum]
      # no coda found
      if secdet[DET_TIME_COL] > (det[DET_TIME_COL] + MAX_SECDET_DELAY):
        break
      # different site
      if secdet[DET_SITE_COL] != det[DET_SITE_COL]:
        continue
      # we don't want coda -> assoc det
      if secdetnum in assoc_dets:
        break
      # too close, this is a rare occurrence and we are ignoring for now
      if secdet[DET_TIME_COL] < (det[DET_TIME_COL] + MIN_SECDET_DELAY):
        break
      # if the azimuth is more than 90 degrees off or the slowness doesn't
      # match then discount it
      if (abs(degdiff(det[DET_AZI_COL], secdet[DET_AZI_COL])) > 50
          or abs(secdet[DET_SLO_COL] - det[DET_SLO_COL]) > 10):
        break

      sec_found = True
      if detnum in assoc_dets:
        assoc_coda_detpair.append((detnum, secdetnum))
      else:
        coda_coda_detpair.append((detnum, secdetnum))
      break

    if not sec_found:
      if detnum in assoc_dets:
        assoc_coda_detpair.append((detnum, None))
      else:
        coda_coda_detpair.append((detnum, None))

  # the prefix "pc" stands for phase -> coda
  pc_time_data, pc_az_data, pc_slo_data, pc_logamp_data, pc_snr_data \
                = [], [], [], [], []
  pc_sec_phase = np.zeros(earthmodel.NumPhases())

  pc_logamp_det = np.zeros((HIGH_LOGAMP - LOW_LOGAMP)/STEP_LOGAMP)
  pc_logamp_tot = np.zeros((HIGH_LOGAMP - LOW_LOGAMP)/STEP_LOGAMP)

  for detnum, secdetnum in assoc_coda_detpair:
    det = detections[detnum]
    if (np.log(det[DET_AMP_COL]) < HIGH_LOGAMP and
        np.log(det[DET_AMP_COL]) >= LOW_LOGAMP):
      pc_logamp_tot[(np.log(det[DET_AMP_COL])-LOW_LOGAMP) // STEP_LOGAMP] += 1

    if secdetnum is None:
      continue

    secdet = detections[secdetnum]
    
    delta = secdet[DET_TIME_COL] - det[DET_TIME_COL]
    assert(delta >= MIN_SECDET_DELAY and delta <= MAX_SECDET_DELAY)
    pc_time_data.append(delta)
    pc_az_data.append(degdiff(det[DET_AZI_COL], secdet[DET_AZI_COL]))
    pc_slo_data.append(secdet[DET_SLO_COL] - det[DET_SLO_COL])
    pc_logamp_data.append(np.log(secdet[DET_AMP_COL]) - np.log(det[DET_AMP_COL]))
    pc_snr_data.append(secdet[DET_SNR_COL] - det[DET_SNR_COL])
    pc_sec_phase[int(secdet[DET_PHASE_COL])] += 1
    
    if (np.log(det[DET_AMP_COL]) < HIGH_LOGAMP and
        np.log(det[DET_AMP_COL]) >= LOW_LOGAMP):
      pc_logamp_det[(np.log(det[DET_AMP_COL]) - LOW_LOGAMP) // STEP_LOGAMP] += 1

  pc_time_shape, _, pc_time_scale = gamma.fit(pc_time_data,
                                              loc=MIN_SECDET_DELAY - .1)
  pc_az_loc, pc_az_scale = laplace.fit(pc_az_data)
  pc_slo_loc, pc_slo_scale = laplace.fit(pc_slo_data)
  pc_logamp_loc, pc_logamp_scale = laplace.fit(pc_logamp_data)

  # fit the SNR as two exponential distributions around 0
  pc_pos_snr_data = [x for x in pc_snr_data if x >= 0 and x < 20]
  pc_neg_snr_data = [-x for x in pc_snr_data if x < 0 and x > -20]
  pc_snr_prob_plus = float(len(pc_pos_snr_data)) / (len(pc_pos_snr_data)
                                                    + len(pc_neg_snr_data))
  pc_snr_lambda_plus = 1.0 / np.average(pc_pos_snr_data)
  pc_snr_lambda_minus = 1.0 / np.average(pc_neg_snr_data)
  
  pc_sec_phase += 1.0                   # add one smoothing
  pc_sec_phase /= pc_sec_phase.sum()

  pc_logamp_det += .000001
  pc_logamp_tot += .001
  
  print "Phase -> Coda Arrival :"
  print "DetProb(logamp)", pc_logamp_det / pc_logamp_tot
  print "Time", pc_time_shape, MIN_SECDET_DELAY-.1, pc_time_scale,\
        MAX_SECDET_DELAY
  print "Time lost probability mass:", \
        gamma.sf(MAX_SECDET_DELAY, pc_time_shape, MIN_SECDET_DELAY,
                 pc_time_scale)
  print "Azimuth", pc_az_loc, pc_az_scale
  print "Slowness", pc_slo_loc, pc_slo_scale
  print "Logamp", pc_logamp_loc, pc_logamp_scale
  print "SNR", pc_snr_prob_plus, pc_snr_lambda_plus, pc_snr_lambda_minus
  print "Phase:",
  print_list(sys.stdout, pc_sec_phase)

  # the prefix "cc" stands for coda -> coda
  cc_time_data, cc_az_data, cc_slo_data, cc_logamp_data, cc_snr_data \
                = [], [], [], [], []
  cc_sec_phase = np.zeros(earthmodel.NumPhases())

  cc_logamp_det = np.zeros((HIGH_LOGAMP - LOW_LOGAMP)/STEP_LOGAMP)
  cc_logamp_tot = np.zeros((HIGH_LOGAMP - LOW_LOGAMP)/STEP_LOGAMP)

  for detnum, secdetnum in coda_coda_detpair:
    det = detections[detnum]
    if (np.log(det[DET_AMP_COL]) < HIGH_LOGAMP and
        np.log(det[DET_AMP_COL]) >= LOW_LOGAMP):
      cc_logamp_tot[(np.log(det[DET_AMP_COL])-LOW_LOGAMP) // STEP_LOGAMP] += 1

    if secdetnum is None:
      continue

    secdet = detections[secdetnum]
    
    delta = secdet[DET_TIME_COL] - det[DET_TIME_COL]
    assert(delta >= MIN_SECDET_DELAY and delta <= MAX_SECDET_DELAY)
    cc_time_data.append(delta)
    cc_az_data.append(degdiff(det[DET_AZI_COL], secdet[DET_AZI_COL]))
    cc_slo_data.append(secdet[DET_SLO_COL] - det[DET_SLO_COL])
    cc_logamp_data.append(np.log(secdet[DET_AMP_COL]) - np.log(det[DET_AMP_COL]))
    cc_snr_data.append(secdet[DET_SNR_COL] - det[DET_SNR_COL])
    cc_sec_phase[int(secdet[DET_PHASE_COL])] += 1
    
    if (np.log(det[DET_AMP_COL]) < HIGH_LOGAMP and
        np.log(det[DET_AMP_COL]) >= LOW_LOGAMP):
      cc_logamp_det[(np.log(det[DET_AMP_COL]) - LOW_LOGAMP) // STEP_LOGAMP] += 1

  cc_time_shape, _, cc_time_scale = gamma.fit(cc_time_data,
                                              loc=MIN_SECDET_DELAY - .1)
  cc_az_loc, cc_az_scale = laplace.fit(cc_az_data)
  cc_slo_loc, cc_slo_scale = laplace.fit(cc_slo_data)
  cc_logamp_loc, cc_logamp_scale = laplace.fit(cc_logamp_data)

  # fit the SNR as two exponential distributions around 0
  cc_pos_snr_data = [x for x in cc_snr_data if x >= 0 and x < 20]
  cc_neg_snr_data = [-x for x in cc_snr_data if x < 0 and x > -20]
  cc_snr_prob_plus = float(len(cc_pos_snr_data)) / (len(cc_pos_snr_data)
                                                    + len(cc_neg_snr_data))
  cc_snr_lambda_plus = 1.0 / np.average(cc_pos_snr_data)
  cc_snr_lambda_minus = 1.0 / np.average(cc_neg_snr_data)
  
  cc_sec_phase += 1.0                   # add one smoothing
  cc_sec_phase /= cc_sec_phase.sum()

  cc_logamp_det += .000001
  cc_logamp_tot += .001
  
  print "Coda -> Coda Arrival :"
  print "DetProb(logamp)", cc_logamp_det / cc_logamp_tot
  print "Time", cc_time_shape, MIN_SECDET_DELAY-.1, cc_time_scale,\
        MAX_SECDET_DELAY
  print "Time lost probability mass:", \
        gamma.sf(MAX_SECDET_DELAY, cc_time_shape, MIN_SECDET_DELAY,
                 cc_time_scale)
  print "Azimuth", cc_az_loc, cc_az_scale
  print "Slowness", cc_slo_loc, cc_slo_scale
  print "Logamp", cc_logamp_loc, cc_logamp_scale
  print "SNR", cc_snr_prob_plus, cc_snr_lambda_plus, cc_snr_lambda_minus
  print "Phase:",
  print_list(sys.stdout, cc_sec_phase)
  
  if options.gui:
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Probability of coda arrival vs phase arrival amp")
    plt.bar(np.arange(LOW_LOGAMP, HIGH_LOGAMP, STEP_LOGAMP),
            pc_logamp_det / pc_logamp_tot, STEP_LOGAMP)
    plt.xlabel("log(amp)")
    plt.ylabel("probability")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "ProbCodaGivenPhaseLogAmp")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()
    
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Probability of coda arrival vs coda arrival amp")
    plt.bar(np.arange(LOW_LOGAMP, HIGH_LOGAMP, STEP_LOGAMP),
            cc_logamp_det / cc_logamp_tot, STEP_LOGAMP)
    plt.xlabel("log(amp)")
    plt.ylabel("probability")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "ProbCodaGivenCodaLogAmp")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()
    
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Phase to Coda time delay")
    _,xpts,_ = plt.hist(pc_time_data, 100, normed=True, label="data", alpha=.5)
    plt.plot(xpts, gamma.pdf(xpts, pc_time_shape, MIN_SECDET_DELAY,
                             pc_time_scale),
             linewidth=3, label="model")
    plt.xlabel("seconds")
    plt.ylabel("probability density")
    plt.ylim(0,.15)
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "PhaseCodaTime")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()

    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Coda to Coda time delay")
    _,xpts,_ = plt.hist(cc_time_data, 100, normed=True, label="data", alpha=.5)
    plt.plot(xpts, gamma.pdf(xpts, cc_time_shape, MIN_SECDET_DELAY,
                             cc_time_scale),
             linewidth=3, label="model")
    plt.xlabel("seconds")
    plt.ylabel("probability density")
    plt.ylim(0,.15)
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "CodaCodaTime")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()
      
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Phase to Coda azimuth difference")
    _,xpts,_ = plt.hist(pc_az_data, 100, normed=True, label="data", alpha=.5)
    plt.plot(xpts, laplace.pdf(xpts, pc_az_loc, pc_az_scale),
             linewidth=3, label="model")
    plt.xlabel("degrees")
    plt.ylabel("probability density")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "PhaseCodaAzimuth")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()

    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Coda to Coda azimuth difference")
    _,xpts,_ = plt.hist(cc_az_data, 100, normed=True, label="data", alpha=.5)
    plt.plot(xpts, laplace.pdf(xpts, cc_az_loc, cc_az_scale),
             linewidth=3, label="model")
    plt.xlabel("degrees")
    plt.ylabel("probability density")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "CodaCodaAzimuth")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()
    
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Phase to Coda slowness difference")
    _,xpts,_ = plt.hist(pc_slo_data, 100, normed=True, label="data", alpha=.5)
    plt.plot(xpts, laplace.pdf(xpts, pc_slo_loc, pc_slo_scale),
             linewidth=3, label="model")
    plt.xlabel("seconds per degree")
    plt.ylabel("probability density")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "PhaseCodaSlowness")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()
    
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Coda to Coda slowness difference")
    _,xpts,_ = plt.hist(cc_slo_data, 100, normed=True, label="data", alpha=.5)
    plt.plot(xpts, laplace.pdf(xpts, cc_slo_loc, cc_slo_scale),
             linewidth=3, label="model")
    plt.xlabel("seconds per degree")
    plt.ylabel("probability density")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "CodaCodaSlowness")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()
    
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Phase to Coda logamp difference")
    _,xpts,_ = plt.hist(pc_logamp_data, 100, normed=True, label="data", alpha=.5)
    plt.plot(xpts, laplace.pdf(xpts, pc_logamp_loc, pc_logamp_scale),linewidth=3,
             label="model")
    plt.ylabel("probability density")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "PhaseCodaLogamp")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()

    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Coda to Coda logamp difference")
    _,xpts,_ = plt.hist(cc_logamp_data, 100, normed=True, label="data", alpha=.5)
    plt.plot(xpts, laplace.pdf(xpts, cc_logamp_loc, cc_logamp_scale),linewidth=3,
             label="model")
    plt.ylabel("probability density")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "CodaCodaLogamp")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()

    def double_exp_pdf(x, prob_plus, lambda_plus, lambda_minus):
      if x >= 0:
        return prob_plus * lambda_plus * np.exp(- lambda_plus * x)
      else:
        return (1-prob_plus) * lambda_minus * np.exp(lambda_minus * x)
    
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Phase to Coda SNR difference")
    _,xpts,_ = plt.hist(pc_snr_data, np.arange(- 20, 20, .25),
                        normed=True, label="data", alpha=.5)
    plt.plot(xpts, [double_exp_pdf(x, pc_snr_prob_plus, pc_snr_lambda_plus,
                                   pc_snr_lambda_minus) for x in xpts],
             linewidth=3, label="model")
    plt.ylabel("probability density")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "PhaseCodaSNR")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()

    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Coda to Coda SNR difference")
    _,xpts,_ = plt.hist(cc_snr_data, np.arange(- 20, 20, .25),
                        normed=True, label="data", alpha=.5)
    plt.plot(xpts, [double_exp_pdf(x, cc_snr_prob_plus, cc_snr_lambda_plus,
                                   cc_snr_lambda_minus) for x in xpts],
             linewidth=3, label="model")
    plt.ylabel("probability density")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "CodaCodaSNR")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()

    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Phase to Coda phase")
    phaserange = np.arange(earthmodel.NumPhases())
    phasenames = [earthmodel.PhaseName(i) for i in phaserange]
    plt.bar(phaserange, pc_sec_phase, 1)
    plt.ylabel("probability")
    plt.xticks(phaserange+.5, phasenames)
    plt.xlabel("phase")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "PhaseCodaPhase")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()

    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Coda to Coda phase")
    phaserange = np.arange(earthmodel.NumPhases())
    phasenames = [earthmodel.PhaseName(i) for i in phaserange]
    plt.bar(phaserange, cc_sec_phase, 1)
    plt.ylabel("probability")
    plt.xticks(phaserange+.5, phasenames)
    plt.xlabel("phase")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "CodaCodaPhase")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()
  
  fp = open(param_fname, "w")
  print >> fp, len(pc_logamp_det), LOW_LOGAMP, STEP_LOGAMP
  print_list(fp, pc_logamp_det / pc_logamp_tot)
  print_list(fp, cc_logamp_det / cc_logamp_tot)
  # the location of the Gamma can't be at precisely zero!
  print >> fp, pc_time_shape, MIN_SECDET_DELAY-.1, pc_time_scale, \
        MAX_SECDET_DELAY
  print >> fp, cc_time_shape, MIN_SECDET_DELAY-.1, cc_time_scale, \
        MAX_SECDET_DELAY
  print >> fp, pc_az_loc, pc_az_scale
  print >> fp, cc_az_loc, cc_az_scale
  print >> fp, pc_slo_loc, pc_slo_scale
  print >> fp, cc_slo_loc, cc_slo_scale
  print >> fp, pc_logamp_loc, pc_logamp_scale
  print >> fp, cc_logamp_loc, cc_logamp_scale
  print >> fp, pc_snr_prob_plus, pc_snr_lambda_plus, pc_snr_lambda_minus
  print >> fp, cc_snr_prob_plus, cc_snr_lambda_plus, cc_snr_lambda_minus
  print >> fp, len(cc_sec_phase)
  print_list(fp, pc_sec_phase)
  print_list(fp, cc_sec_phase)
  
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

def print_list(fp, list):
  for x in list:
    print >>fp, x,
  print >> fp
  
