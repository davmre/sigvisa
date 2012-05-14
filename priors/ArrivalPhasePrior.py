import numpy as np

from database.dataset import DET_PHASE_COL

from NumFalseDetPrior import get_false_detnums

def learn(filename, options, earthmodel, detections, leb_events, leb_evlist):
  numtimedefphases = earthmodel.NumTimeDefPhases()
  numphases = earthmodel.NumPhases()

  # add-one smoothing
  ph2ph = np.ones((numtimedefphases, numphases), float)
  falseph = np.ones(numphases, float)

  for evnum, ph_detlist in enumerate(leb_evlist):
    for ph, detnum in ph_detlist:
      ph2ph[ph, int(detections[detnum, DET_PHASE_COL])] += 1.0

  for detnum in get_false_detnums(leb_evlist, detections):
    falseph[int(detections[detnum, DET_PHASE_COL])] += 1.0

  # normalize
  for i in range(numtimedefphases):
    ph2ph[i] /= ph2ph[i].sum()

  falseph /= falseph.sum()
    
  fp = open(filename, "w")
  
  print >>fp, numtimedefphases, numphases

  for i in range(numtimedefphases):
    for j in range(numphases):
      print >>fp, ph2ph[i,j],
    print >> fp

  for j in range(numphases):
    print >>fp, falseph[j],
  print >>fp

  fp.close()
  
  if options.verbose:
    print "Phase Emission Probabilities"
    for i in range(numtimedefphases):
      print "phase[%2d]:" % i,
      for j in range(numphases):
        print ph2ph[i,j],
      print
    print "False:",
    for j in range(numphases):
      print falseph[j],
    print
  
    
