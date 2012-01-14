import numpy as np
import matplotlib.pyplot as plt

from utils import Laplace
from utils.geog import degdiff
from database.dataset import *

# maximum time that a secondary detection can occur after the primary
MAX_SECDET_DELAY = 10.0
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
  time_data, az_data, slo_data, logamp_data = [], [], [], []
  sec_phase = np.zeros(earthmodel.NumPhases())

  HIGH_AMP, STEP_AMP = 500, 20
  amp_det, amp_tot = np.zeros(HIGH_AMP/STEP_AMP), np.zeros(HIGH_AMP/STEP_AMP)

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
          sec_phase[int(secdet[DET_PHASE_COL])] += 1

          if det[DET_AMP_COL] < HIGH_AMP:
            amp_det[ det[DET_AMP_COL] // STEP_AMP ] += 1

          pos_det[pos] += 1
          detcnt += 1

  detprob = float(detcnt) / totcnt
  az_loc, az_scale = Laplace.estimate(az_data)
  slo_loc, slo_scale = Laplace.estimate(slo_data)
  logamp_loc, logamp_scale = Laplace.estimate(logamp_data)

  sec_phase += 1.0
  sec_phase /= sec_phase.sum()
  
  amp_tot += .0001

  pos_tot += .0001
  
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
      plt.title("Secondary detection time delay")
    plt.hist(time_data, 100)
    plt.xlabel("seconds")
    plt.ylabel("frequency")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "SecDetTime")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")

    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Secondary detection azimuth difference")
    plt.hist(az_data, 100)
    plt.xlabel("degrees")
    plt.ylabel("frequency")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "SecDetAzimuth")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")

    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Secondary detection slowness difference")
    plt.hist(slo_data, 100)
    plt.xlabel("seconds per degree")
    plt.ylabel("frequency")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "SecDetSlowness")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Secondary detection logamp difference")
    plt.hist(slo_data, 100)
    plt.ylabel("frequency")
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "SecDetLogamp")
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
  print >> fp, MIN_SECDET_DELAY, MAX_SECDET_DELAY
  print >> fp, az_loc, az_scale
  print >> fp, slo_loc, slo_scale
  print >> fp, logamp_loc, logamp_scale
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
