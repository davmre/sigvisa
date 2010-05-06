import numpy as np
import math, random

from utils.LogisticModel import LogisticModel
from database.dataset import *

def gtf(val, m, s):
  return math.exp(- float(val - m) ** 2 / (2.0 * s ** 2)) \
         / math.sqrt(2.0 * math.pi * s ** 2)

def learn_phase_site(true_data, fake_data):
  output, mag_feat, depth_feat, dist_feat = [], [], [], []
  dist0_feat = []
  for (isdet, mag, depth, dist) in true_data + fake_data:
    output.append(isdet)
    mag_feat.append(mag)
    depth_feat.append(depth)
    dist_feat.append(dist)
    dist0_feat.append(gtf(dist, 0, 5))
    
  model = LogisticModel("y", ["mag", "depth", "dist", "dist0"],
                        [mag_feat, depth_feat, dist_feat, dist0_feat], output)

  for i in range(len(model.coeffs)):
    if np.isnan(model.coeffs[i]):
      raise ValueError("nan coeffs")

  # score the model on the original dataset just for a sanity check
  sumlogprob = 0.
  cnt = 0
  for (isdet, mag, depth, dist) in true_data:
    cnt += 1
    p = model[(mag, depth, dist, gtf(dist, 0, 5))]
    if isdet:
      sumlogprob += math.log(p)
    else:
      sumlogprob += math.log(1-p)
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
  print >>fp, "Phase, Siteid, (Intercept), mag, depth, dist, dist0, ..."

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
  NUM_FAKE_EMPIRICAL = 50
  for phaseid in range(numtimedefphases):
    true_data = phase_data[phaseid]
    # create some perturbed true data
    fake_data = []
    for (snum, isdet, mag, dep, dist) in random.sample(true_data,
                                                       NUM_FAKE_EMPIRICAL):
      # perturb some misdetections
      if not(isdet) and random.random() < .25:
        isdet = 1
      # perturb the magnitude depth and distance as well
      mag += random.random()/10.0
      dep += random.random()
      dist += random.random()
      
      fake_data.append((isdet, mag, dep, dist))

    
    for siteid in range(numsites):
      print "[%s]: (%d):" % (earthmodel.PhaseName(phaseid), siteid),
      
      site_data = [(isdet, mag, dep, dist) for (snum, isdet, mag, dep, dist) \
                   in true_data if snum == siteid]
      
      coeffs = learn_phase_site(site_data, fake_data)

      buf = "%s,%d,%s," % (earthmodel.PhaseName(phaseid), siteid,
                           str(coeffs[-1]))

      for i in range(4):
        buf += "%s," % (str(coeffs[i]),)

      buf += ",".join(["0" for _ in range(10)])

      print >>fp, buf
      
  fp.close()
  
  
