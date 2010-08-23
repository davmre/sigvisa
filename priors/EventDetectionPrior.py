import numpy as np
import math, random

from utils.LogisticModel import LogisticModel
from database.dataset import *

def gtf(val, m, s):
  return math.exp(- float(val - m) ** 2 / (2.0 * float(s) ** 2)) \
         / math.sqrt(2.0 * math.pi * float(s) ** 2)

def learn_phase_site(phasename, true_data, fake_data):
  output, mag_feat, depth_feat, dist_feat = [], [], [], []
  dist0_feat, dist35_feat, dist40_feat, dist12520_feat = [], [], [], []
  dist12540_feat, dist145_feat, dist170_feat, dist175_feat = [], [], [], []
  mag6_feat, mag68_feat, md_feat, md145_feat = [], [], [], []
  
  for (isdet, mag, depth, dist) in true_data + fake_data:
    output.append(isdet)
    
    if phasename not in ("PcP", "ScP", "PKKPbc", "PKPab"):
      mag_feat.append(mag)
    else:
      mag_feat.append(0)

    if phasename not in ("Lg", "Pg", "Rg"):
      depth_feat.append(depth)
    else:
      depth_feat.append(0)

    if phasename not in ("PcP", "ScP", "PKKPbc", "PKP", "PKPbc", "PKPab"):
      dist_feat.append(dist)
    else:
      dist_feat.append(0)
    
    if phasename in ("P", "S"):
      dist0_feat.append(gtf(dist, 0, 5))
    else:
      dist0_feat.append(0)
      
    if phasename in ("ScP",):
      dist35_feat.append(gtf(dist, 35, 20))
    else:
      dist35_feat.append(0)
      
    if phasename in ("PcP",):
      dist40_feat.append(gtf(dist, 40, 20))
    else:
      dist40_feat.append(0)
      
    if phasename in ("PKP",):
      dist12520_feat.append(gtf(dist, 125, 20))
    else:
      dist12520_feat.append(0)
      
    if phasename in ("PKKPbc",):
      dist12540_feat.append(gtf(dist, 125, 40))
    else:
      dist12540_feat.append(0)
      
    if phasename in ("PKPbc",):
      dist145_feat.append(gtf(dist, 145, 10))
    else:
      dist145_feat.append(0)
      
    if phasename in ("PKP",):
      dist170_feat.append(gtf(dist, 170, 20))
    else:
      dist170_feat.append(0)
      
    if phasename in ("PKPab",):
      dist175_feat.append(gtf(dist, 175, 30))
    else:
      dist175_feat.append(0)
      
    if phasename in ("PcP", "ScP", "PKPab"):
      mag6_feat.append(gtf(mag, 6, 5.5))
    else:
      mag6_feat.append(0)
      
    if phasename in ("PKKPbc",):
      mag68_feat.append(gtf(mag, 6, 8))
    else:
      mag68_feat.append(0)
      
    if phasename in ("P", "S", "PKP", "Sn", "Pn"):
      md_feat.append((7-mag) * dist)
    else:
      md_feat.append(0)

  feature_names = ["mag", "depth", "dist", "dist0", "dist35",
                   "dist40", "dist12520", "dist12540", "dist145",
                   "dist170", "dist175", "mag6", "mag68", "md"]
  feature_values = [mag_feat, depth_feat, dist_feat, dist0_feat,
                    dist35_feat, dist40_feat, dist12520_feat,
                    dist12540_feat, dist145_feat, dist170_feat,
                    dist175_feat, mag6_feat, mag68_feat, md_feat]
  
  model = LogisticModel("y", feature_names, feature_values, output,
                        alpha=1e-5)

  for i in range(len(model.coeffs)):
    # for any feature which is not being used, set its coefficients to zero
    if i < len(feature_values) and sum(feature_values[i]) == 0:
      model.coeffs[i] = 0
    # the other coefficients are not allowed to be NaN
    if np.isnan(model.coeffs[i]):
      raise ValueError("nan coeffs")

  # score the model on the original dataset just for a sanity check
  sumlogprob = 0.
  cnt = 0
  for isdet, f in zip(output, zip(*feature_values)):
    cnt += 1
    p = model[f]
    if isdet:
      sumlogprob += math.log(p)
    else:
      sumlogprob += math.log(1-p)

    if cnt >= len(true_data):
      break
    
  if cnt:
    print "Avg. log like:", sumlogprob / cnt
  else:
    print
  
  return model.coeffs
  
def learn(param_fname, earthmodel, start_time, end_time,
          detections, leb_events, leb_evlist, site_up):

  # assume that the time-defining phases precede the non-time-defining ones
  numtimedefphases = earthmodel.NumTimeDefPhases()
  numsites = earthmodel.NumSites()
  
  fp = open(param_fname, "w")
  
  print >>fp, "%d %d" % (numtimedefphases, numsites)
  print >>fp, "Phase, Siteid, (Intercept), mag, depth, dist, dist0, dist35,"\
        "dist40, dist12520, dist12540, dist145, dist170, dist175, mag6,"\
        "mag68, md"

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


  # learn for each phase, site
  NUM_FAKE_EMPIRICAL = 40
  NUM_FAKE_UNIFORM = 40
  for phaseid in range(numtimedefphases):
    true_data = phase_data[phaseid]
    # create some perturbed fake data drawn from the empirical distribution
    fake_data = []
    for (snum, isdet, mag, dep, dist) in random.sample(true_data,
                                                       NUM_FAKE_EMPIRICAL):
      # perturb some misdetections
      if random.random() < .25:
        isdet = 1 - isdet
      # perturb the magnitude depth and distance as well
      mag += random.random()/10.0
      dep += random.random()
      dist += random.random()
      
      fake_data.append((isdet, mag, dep, dist))

    # create some fake data drawn uniformly
    mindist, maxdist = earthmodel.PhaseRange(phaseid)
    for _ in xrange(NUM_FAKE_UNIFORM):
      isdet = int(random.random() < .25)
      mag = random.randrange(2, 7)
      dep = random.randrange(0, 700)
      dist = random.randrange(mindist, maxdist)

      fake_data.append((isdet, mag, dep, dist))
    
    for siteid in range(numsites):
      print "[%s]: (%d):" % (earthmodel.PhaseName(phaseid), siteid),
      
      site_data = [(isdet, mag, dep, dist) for (snum, isdet, mag, dep, dist) \
                   in true_data if snum == siteid]
      
      coeffs = learn_phase_site(earthmodel.PhaseName(phaseid), site_data,
                                fake_data)

      buf = "%s,%d,%s," % (earthmodel.PhaseName(phaseid), siteid,
                           str(coeffs[-1]))

      buf += ",".join([str(coeffs[i]) for i in range(14)])

      print >>fp, buf
      
  fp.close()
  
  
