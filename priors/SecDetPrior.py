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
  assoc_coda_detpair, coda_coda_detpair = compute_coda_pairs(detections,
                                                             leb_evlist)
  coda_detpair = assoc_coda_detpair + coda_coda_detpair
  
  # the prefix "pc" stands for phase -> coda
  time_data, az_data, slo_data, logamp_data, snr_data \
                = [], [], [], [], []
  sec_phase = np.zeros(earthmodel.NumPhases())

  logamp_det = np.zeros((HIGH_LOGAMP - LOW_LOGAMP)/STEP_LOGAMP)
  logamp_tot = np.zeros((HIGH_LOGAMP - LOW_LOGAMP)/STEP_LOGAMP)

  for detnum, secdetnum in coda_detpair:
    det = detections[detnum]
    if (np.log(det[DET_AMP_COL]) < HIGH_LOGAMP and
        np.log(det[DET_AMP_COL]) >= LOW_LOGAMP):
      logamp_tot[(np.log(det[DET_AMP_COL])-LOW_LOGAMP) // STEP_LOGAMP] += 1

    if secdetnum is None:
      continue

    secdet = detections[secdetnum]
    
    delta = secdet[DET_TIME_COL] - det[DET_TIME_COL]
    assert(delta >= MIN_SECDET_DELAY and delta <= MAX_SECDET_DELAY)
    time_data.append(delta)
    az_data.append(degdiff(det[DET_AZI_COL], secdet[DET_AZI_COL]))
    slo_data.append(secdet[DET_SLO_COL] - det[DET_SLO_COL])
    logamp_data.append(np.log(secdet[DET_AMP_COL]) - np.log(det[DET_AMP_COL]))
    snr_data.append(secdet[DET_SNR_COL] - det[DET_SNR_COL])
    sec_phase[int(secdet[DET_PHASE_COL])] += 1
    
    if (np.log(det[DET_AMP_COL]) < HIGH_LOGAMP and
        np.log(det[DET_AMP_COL]) >= LOW_LOGAMP):
      logamp_det[(np.log(det[DET_AMP_COL]) - LOW_LOGAMP) // STEP_LOGAMP] += 1

  time_shape, _, time_scale = gamma.fit(time_data,
                                              loc=MIN_SECDET_DELAY - .1)
  az_loc, az_scale = laplace.fit(az_data)
  slo_loc, slo_scale = laplace.fit(slo_data)
  logamp_loc, logamp_scale = laplace.fit(logamp_data)

  # fit the SNR as two exponential distributions around 0
  pos_snr_data = [x for x in snr_data if x >= 0 and x < 20]
  neg_snr_data = [-x for x in snr_data if x < 0 and x > -20]
  snr_prob_plus = float(len(pos_snr_data)) / (len(pos_snr_data)
                                                    + len(neg_snr_data))
  snr_lambda_plus = 1.0 / np.average(pos_snr_data)
  snr_lambda_minus = 1.0 / np.average(neg_snr_data)
  
  sec_phase += 1.0                   # add one smoothing
  sec_phase /= sec_phase.sum()

  logamp_det += .000001
  logamp_tot += .001
  
  print "Phase -> Coda Arrival :"
  print "DetProb(logamp)", logamp_det / logamp_tot
  print "Time", time_shape, MIN_SECDET_DELAY-.1, time_scale,\
        MAX_SECDET_DELAY
  print "Time lost probability mass:", \
        gamma.sf(MAX_SECDET_DELAY, time_shape, MIN_SECDET_DELAY,
                 time_scale)
  print "Azimuth", az_loc, az_scale
  print "Slowness", slo_loc, slo_scale
  print "Logamp", logamp_loc, logamp_scale
  print "SNR", snr_prob_plus, snr_lambda_plus, snr_lambda_minus
  print "Phase:",
  print_list(sys.stdout, sec_phase)

  
  if options.gui:
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Probability of coda arrival vs arrival amp")
    else:
      plt.title("Coda detection probability")
    plt.bar(np.arange(LOW_LOGAMP, HIGH_LOGAMP, STEP_LOGAMP),
            logamp_det / logamp_tot, STEP_LOGAMP)
    plt.xlabel("log(amp)")
    plt.ylabel("Probability")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "CodaDetProb")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()
    
    plt.figure(figsize=(8,4.8))
    plt.title("Coda time delay")
    _,xpts,_ = plt.hist(time_data, 100, normed=True, label="data", alpha=1)
    plt.plot(xpts, gamma.pdf(xpts, time_shape, MIN_SECDET_DELAY,
                             time_scale),
             linewidth=3, label="model", color="black")
    plt.xlabel("Time (s)")
    plt.ylabel("Probability density")
    plt.ylim(0,.15)
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "CodaTime")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()

    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Coda azimuth vs previous detection's azimuth")
    else:
      plt.title("Coda azimuth")
    _,xpts,_ = plt.hist(az_data, 100, normed=True, label="data", alpha=1,
                        facecolor="blue")
    plt.plot(xpts, laplace.pdf(xpts, az_loc, az_scale),
             linewidth=3, label="model", color="black")
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Probability density")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "CodaAzimuth")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()

    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Coda slowness vs previous detection's slowness")
    else:
      plt.title("Coda slowness")
    _,xpts,_ = plt.hist(slo_data, 100, normed=True, label="data", alpha=1,
                        facecolor="blue")
    plt.plot(xpts, laplace.pdf(xpts, slo_loc, slo_scale),
             linewidth=3, label="model", color="black")
    plt.xlabel("Slowness (s/deg)")
    plt.ylabel("Probability density")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "CodaSlowness")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()
    
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Coda amplitude vs previous detection's amplitude")
    else:
      plt.title("Coda amplitude")
    _,xpts,_ = plt.hist(logamp_data, 100, normed=True, label="data", alpha=1,
                        facecolor="blue")
    plt.plot(xpts, laplace.pdf(xpts, logamp_loc, logamp_scale),linewidth=3,
             label="model", color="black")
    plt.ylabel("Probability density")
    plt.xlabel("log(amp)")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "CodaAmp")
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
      plt.title("Coda SNR vs previous detection's SNR")
    else:
      plt.title("Coda SNR")
    _,xpts,_ = plt.hist(snr_data, np.arange(- 20, 20, .25),
                        normed=True, label="data", alpha=1, facecolor="blue")
    plt.plot(xpts, [double_exp_pdf(x, snr_prob_plus, snr_lambda_plus,
                                   snr_lambda_minus) for x in xpts],
             linewidth=3, label="model", color="black")
    plt.ylabel("Probability density")
    plt.legend(loc="upper left")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "CodaSNR")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()

    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Coda Phase Distribution")
    phaserange = np.arange(earthmodel.NumPhases())
    phasenames = [earthmodel.PhaseName(i) for i in phaserange]
    plt.bar(phaserange, sec_phase, 1)
    plt.ylabel("Probability")
    plt.xticks(phaserange+.5, phasenames, rotation="vertical")
    plt.subplots_adjust(bottom=.17)
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "CodaPhase")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    if options.pdf:
      options.pdf.savefig()

  fp = open(param_fname, "w")
  print >> fp, len(logamp_det), LOW_LOGAMP, STEP_LOGAMP
  print_list(fp, logamp_det / logamp_tot)
  # the location of the Gamma can't be at precisely zero!
  print >> fp, time_shape, MIN_SECDET_DELAY-.1, time_scale, \
        MAX_SECDET_DELAY
  print >> fp, az_loc, az_scale
  print >> fp, slo_loc, slo_scale
  print >> fp, logamp_loc, logamp_scale
  print >> fp, snr_prob_plus, snr_lambda_plus, snr_lambda_minus
  print >> fp, len(sec_phase)
  print_list(fp, sec_phase)
  
  fp.close()

def compute_secondary_dets(earthmodel, detections, leb_events, leb_evlist):
  assoc_coda_detpair, coda_coda_detpair = compute_coda_pairs(detections,
                                                             leb_evlist)
  det_coda = dict(assoc_coda_detpair + coda_coda_detpair)
  
  leb_seclist = []
  for detlist in leb_evlist:
    seclist = []
    for ph, detnum in detlist:
      codalist = [ph, detnum]
      while (len(codalist) < (MAX_PHASE_DET+1)
             and det_coda[codalist[-1]] is not None):
        codalist.append(det_coda[codalist[-1]])
      seclist.append(tuple(codalist))
    leb_seclist.append(seclist)
  return leb_seclist

def print_list(fp, list):
  for x in list:
    print >>fp, x,
  print >> fp

def compute_coda_pairs(detections, leb_evlist):
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

  return assoc_coda_detpair, coda_coda_detpair

# return a set of detnums which are *independent* noise arrivals, i.e. they
# are not associated to any event or prior arrival
def compute_false_detections(detections, leb_evlist):
  phase_dets, coda_dets = set(), set()
  assoc_coda_detpair, coda_coda_detpair = compute_coda_pairs(detections,
                                                             leb_evlist)

  for detnum, secdetnum in assoc_coda_detpair:
    phase_dets.add(detnum)
    coda_dets.add(secdetnum)
    
  for detnum, secdetnum in coda_coda_detpair:
    coda_dets.add(secdetnum)
    
  return set(detnum for detnum in xrange(len(detections))
             if detnum not in phase_dets and detnum not in coda_dets)
