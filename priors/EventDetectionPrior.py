import numpy as np
import math, random

from utils.LogisticModel import LogisticModel
from database.dataset import *

def learn_phase_site(true_data, fake_data):
  output, mag_feat, depth_feat, dist_feat = [], [], [], []
  for (isdet, mag, depth, dist) in true_data + fake_data:
    output.append(isdet)
    mag_feat.append(mag)
    depth_feat.append(depth)
    dist_feat.append(dist)
    
  model = LogisticModel("y", ["mag", "depth", "dist"],
                        [mag_feat, depth_feat, dist_feat], output)

  for i in range(len(model.coeffs)):
    if np.isnan(model.coeffs[i]):
      raise ValueError("nan coeffs")

  # score the model on the original dataset just for a sanity check
  sumlogprob = 0.
  cnt = 0
  for (isdet, mag, depth, dist) in true_data:
    cnt += 1
    p = model[(mag, depth, dist)]
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
  print >>fp, "Phase, Siteid, (Intercept), mag, depth, dist, ..."

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
  NUM_FAKE_EMPIRICAL = 30
  NUM_FAKE_UNIFORM = 20
  for phaseid in range(numtimedefphases):
    true_data = phase_data[phaseid]
    fake_data = [(isdet, mag, dep, dist) for (snum, isdet, mag, dep, dist) \
                 in random.sample(true_data, NUM_FAKE_EMPIRICAL)]\
                 + [(int(random.random() < .5), random.randrange(3,6),
                     random.randrange(0,700), random.randrange(0,180)) \
                    for _ in xrange(NUM_FAKE_UNIFORM)]
    
    for siteid in range(numsites):
      print "[%s]: (%d):" % (earthmodel.PhaseName(phaseid), siteid),
      
      site_data = [(isdet, mag, dep, dist) for (snum, isdet, mag, dep, dist) \
                   in true_data if snum == siteid]
      
      coeffs = learn_phase_site(site_data, fake_data)

      buf = "%s,%d,%s," % (earthmodel.PhaseName(phaseid), siteid,
                           str(coeffs[-1]))

      for i in range(3):
        buf += "%s," % (str(coeffs[i]),)

      buf += ",".join(["0" for _ in range(11)])

      print >>fp, buf
      
  fp.close()
  
  
