import numpy as np
import math, random

from utils.LogisticModel import LogisticModel
from database.dataset import *

def learn_site(earthmodel, start_time, end_time, detections, leb_events,
               leb_evlist, site_up, numtimedefphases, siteid):
  # initialize feature vectors
  mag_feat = []
  dist_feat = []
  phaseid_feat = [[] for _ in range(numtimedefphases)]
  output = []

  for evnum, event in enumerate(leb_events):
    # compute the list of phases detected for this event at this site
    det_phase = np.zeros(numtimedefphases, int)
    for true_phaseid, detnum in leb_evlist[evnum]:
      det = detections[detnum]
      if det[DET_SITE_COL] == siteid:
        det_phase[true_phaseid] = 1

    # check if the site was up at the time of the event and 1 hour later
    if event[EV_TIME_COL] < start_time \
       or event[EV_TIME_COL] + UPTIME_QUANT > end_time \
       or not site_up[siteid,
                      int((event[EV_TIME_COL] - start_time) / UPTIME_QUANT)]\
       or not site_up[siteid,
                      int((event[EV_TIME_COL] + UPTIME_QUANT - start_time)
                          / UPTIME_QUANT)]:
      continue

    dist = earthmodel.Delta(event[EV_LON_COL], event[EV_LAT_COL], siteid)
    
    # we assume that only time-defining phases are detected
    for pnum in range(numtimedefphases):
      # check if the site is in the shadow zone of this phase
      arrtime = earthmodel.ArrivalTime(event[EV_LON_COL], event[EV_LAT_COL],
                                       event[EV_DEPTH_COL],
                                       event[EV_TIME_COL], pnum, siteid)
      if arrtime < 0:
        continue
      
      output.append(det_phase[pnum])
      mag_feat.append(event[EV_MB_COL])
      dist_feat.append(dist)
      # construct the features one per phase
      for i in range(numtimedefphases):
        phaseid_feat[i].append(int(i == pnum))

  print "%d event-phases detected out of %d" % (sum(output), len(output))
  
  # copy the original dataset
  mag_feat2 = [x for x in mag_feat]
  dist_feat2 = [x for x in dist_feat]
  phaseid_feat2 = [[x for x in y] for y in phaseid_feat]
  output2 = [x for x in output]

  # add some random fake detections/mis-detections
  NUM_FAKE = 50
  output2.extend([int(random.random() > .5) for _ in range(NUM_FAKE)])
  mag_feat2.extend([random.randrange(3,6) for _ in range(NUM_FAKE)])
  dist_feat2.extend([random.randrange(0, 180) for _ in range(NUM_FAKE)])
  for i in range(numtimedefphases):
    phaseid_feat2[i].extend([int(random.random() > .5) \
                             for _ in range(NUM_FAKE)])

  # train the model
  model = LogisticModel("y", ["mag", "dist"] +
                        ["phase%d" % i for i in range(numtimedefphases)],
                        [mag_feat2, dist_feat2] + phaseid_feat2, output2)
  
  
  for i in range(len(model.coeffs)):
    if np.isnan(model.coeffs[i]):
      raise ValueError("nan coeffs")
  
  # score the model on the original dataset just for a sanity check
  sumlogprob = 0.
  cnt = 0
  for o, f in zip(output, zip(*([mag_feat, dist_feat] + phaseid_feat))):
    cnt += 1
    p = model[f]
    if o:
      sumlogprob += math.log(p)
    else:
      sumlogprob += math.log(1-p)
  if cnt:
    print "Avg. log like all phases:", sumlogprob / cnt
  
  sumlogprob = 0.
  cnt = 0
  for o, f in zip(output, zip(*([mag_feat, dist_feat] + phaseid_feat))):
    # look for P phase which is the first phase feature
    if not f[2]:
      continue
    cnt += 1
    p = model[f]
    if o:
      sumlogprob += math.log(p)
    else:
      sumlogprob += math.log(1-p)
  if cnt:
    print "Avg. log like P phase:", sumlogprob / cnt

  return model

def learn(param_fname, earthmodel, start_time, end_time,
          detections, leb_events, leb_evlist,
          site_up, sites, phasenames, phasetimedef):

  # assume that the time-defining phases precede the non-time-defining ones
  numtimedefphases = earthmodel.NumTimeDefPhases()
  
  fp = open(param_fname, "w")
  
  print >>fp, "%d %d" % (len(sites), numtimedefphases)
  
  for siteid in range(len(sites)):
    model = learn_site(earthmodel, start_time, end_time,
                       detections, leb_events, leb_evlist, site_up,
                       numtimedefphases, siteid)
    
    for x in model.coeffs:
      print >>fp, x,
    print >>fp
    print "learnt site id", siteid

    del model
    
  fp.close()
  
  
