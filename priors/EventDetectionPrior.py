import numpy as np
import math

from utils.LogisticModel import LogisticModel
from database.dataset import *

def learn_phase(detections, leb_events, leb_evlist,
                site_up, sites, phasenames, phaseid):
  
  # initialize feature vectors
  siteid_feat = [[] for _ in range(len(sites))]
  output = []

  for evnum, event in enumerate(leb_events):
    # compute the list of sites where the phase was detected for this event
    det_site = np.zeros(len(sites), int)
    for detnum in leb_evlist[evnum]:
      det = detections[detnum]
      if det[DET_PHASE_COL] == phaseid:
        det_site[det[DET_SITE_COL]] = 1
    
    for snum in range(len(sites)):
      # TODO: check if the site is up
      # TODO: check if the phase is in range
      output.append(det_site[snum])
      for i in range(len(sites)):
        siteid_feat[i].append(int(i == snum))

  model = LogisticModel("y", ["site%d" % i for i in range(len(sites))],
                        siteid_feat, output)

  return model

def learn_site(detections, leb_events, leb_evlist,
               site_up, sites, phasenames, siteid):
  
  # initialize feature vectors
  phaseid_feat = [[] for _ in range(len(phasenames))]
  output = []

  for evnum, event in enumerate(leb_events):
    # compute the list of phases detected for this event at this site
    det_phase = np.zeros(len(phasenames), int)
    for true_phaseid, detnum in leb_evlist[evnum]:
      det = detections[detnum]
      if det[DET_SITE_COL] == siteid:
        det_phase[true_phaseid] = 1
    
    for pnum in range(len(phasenames)):
      # TODO: check if the site is up
      # TODO: check if the phase is in range
      output.append(det_phase[pnum])
      # construct the features one per phase
      for i in range(len(phasenames)):
        phaseid_feat[i].append(int(i == pnum))

  # add one fake detection with all features on just to prevent
  # underflow for any feature value
  phaseid_feat2 = [x + [1] for x in phaseid_feat]
  output2 = output + [1]
  model = LogisticModel("y", ["phase%d" % i for i in range(len(phasenames))],
                        phaseid_feat2, output2)

  sumlogprob = 0.
  cnt = 0
  for o, f in zip(output, zip(*phaseid_feat)):
    cnt += 1
    p = model[f]
    if o:
      sumlogprob += math.log(p)
    else:
      sumlogprob += math.log(1-p)
  print "Avg. log like all phases:", sumlogprob / cnt
  
  sumlogprob = 0.
  cnt = 0
  for o, f in zip(output, zip(*phaseid_feat)):
    if not f[0]:
      continue
    cnt += 1
    p = model[f]
    if o:
      sumlogprob += math.log(p)
    else:
      sumlogprob += math.log(1-p)
  print "Avg. log like P phase:", sumlogprob / len(output)

  return model

def learn(param_fname, detections, leb_events, leb_evlist,
          site_up, sites, phasenames):

  fp = open(param_fname, "w")

  print >>fp, "%d %d" % (len(sites), len(phasenames))
  
  for siteid in range(len(sites)):
    model = learn_site(detections, leb_events, leb_evlist, site_up, sites,
                       phasenames, siteid)
    for x in model.coeffs:
      print >>fp, x,
    print >>fp
    print "learnt site id", siteid

    del model
    
  fp.close()
  
  
