import numpy as np
import math

from utils.LogisticModel import LogisticModel
from database.dataset import *

def learn_site(earthmodel, detections, leb_events, leb_evlist,
               site_up, sites, phasenames, siteid):
  
  # initialize feature vectors
  mag_feat = []
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
      # check if the phase is in range
      if earthmodel.InRange(event[EV_LON_COL], event[EV_LAT_COL],
                            event[EV_DEPTH_COL], pnum, siteid):
        output.append(det_phase[pnum])
        mag_feat.append(event[EV_MB_COL])
        # construct the features one per phase
        for i in range(len(phasenames)):
          phaseid_feat[i].append(int(i == pnum))

  print "%d event-phases detected" % sum(output)
  
  # copy the original dataset
  mag_feat2 = [x for x in mag_feat]
  phaseid_feat2 = [[x for x in y] for y in phaseid_feat]
  output2 = [x for x in output]

  # add one fake detection
  output2.append(1)
  mag_feat2.append(4.0)
  for i in range(len(phasenames)):
    phaseid_feat2[i].append(1)

  # and one fake mis-detection
  output2.append(0)
  mag_feat2.append(4.0)
  for i in range(len(phasenames)):
    phaseid_feat2[i].append(1)
  
  # train the model
  model = LogisticModel("y", ["mag"] +
                        ["phase%d" % i for i in range(len(phasenames))],
                        [mag_feat2] + phaseid_feat2, output2)
  
  # replace any NaN with 0 TODO: needs further investigation
  nan_coeffs = 0
  for i in range(len(model.coeffs)):
    if np.isnan(model.coeffs[i]):
      model.coeffs[i] = 0.0
      nan_coeffs += 1
  
  if nan_coeffs:
    print "%d NaN coefficients" % nan_coeffs
  
  # score the model on the original dataset just for a sanity check
  sumlogprob = 0.
  cnt = 0
  for o, f in zip(output, zip(*([mag_feat2] + phaseid_feat))):
    cnt += 1
    p = model[f]
    if o:
      sumlogprob += math.log(p)
    else:
      sumlogprob += math.log(1-p)
  print "Avg. log like all phases:", sumlogprob / cnt
  
  sumlogprob = 0.
  cnt = 0
  for o, f in zip(output, zip(*([mag_feat2] + phaseid_feat))):
    # look for P phase which is the phaseid 0
    if not f[1]:
      continue
    cnt += 1
    p = model[f]
    if o:
      sumlogprob += math.log(p)
    else:
      sumlogprob += math.log(1-p)
  print "Avg. log like P phase:", sumlogprob / cnt

  return model

def learn(param_fname, earthmodel, detections, leb_events, leb_evlist,
          site_up, sites, phasenames):

  fp = open(param_fname, "w")

  print >>fp, "%d %d" % (len(sites), len(phasenames))
  
  for siteid in range(len(sites)):
    model = learn_site(earthmodel, detections, leb_events, leb_evlist, site_up,
                       sites, phasenames, siteid)
    for x in model.coeffs:
      print >>fp, x,
    print >>fp
    print "learnt site id", siteid

    del model
    
  fp.close()
  
  
