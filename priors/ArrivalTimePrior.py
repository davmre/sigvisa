from utils import Laplace

from database.dataset import EV_LON_COL, EV_LAT_COL, EV_DEPTH_COL,\
     EV_TIME_COL, DET_SITE_COL, DET_TIME_COL

NUM_PRIOR = 10                          # number of prior points

def learn(param_filename, earthmodel, detections, leb_events, leb_evlist):
  overall_res = []
  
  # keep a list of arrival time residuals for each site
  site_res = dict((siteid, []) for siteid in range(earthmodel.NumSites()))
  
  # keep residuals for each phase
  phase_res = dict((phaseid, []) for phaseid in range(earthmodel.NumPhases()))
  
  # keep a residual for each site and phase
  site_phase_res = dict(((siteid, phaseid), [])
                        for phaseid in range(earthmodel.NumPhases())
                        for siteid in range(earthmodel.NumSites()))
  
  for evnum, event in enumerate(leb_events):
    for phaseid, detnum in leb_evlist[evnum]:
      siteid = int(detections[detnum, DET_SITE_COL])
      arrtime = detections[detnum, DET_TIME_COL]
      
      pred_arrtime = earthmodel.ArrivalTime(event[EV_LON_COL],
                                            event[EV_LAT_COL],
                                            event[EV_DEPTH_COL],
                                            event[EV_TIME_COL], phaseid,
                                            siteid)
      if pred_arrtime < 0:
        continue
      
      res = arrtime - pred_arrtime

      if abs(res) > 30:
        continue
      
      if res > 1000:
        raise ValueError("residual too large")
      
      overall_res.append(res)
      site_res[siteid].append(res)
      phase_res[phaseid].append(res)
      site_phase_res[siteid, phaseid].append(res)

  # learn the overall residual for and sample some points
  # for an empirical Bayes prior
  overall_params = Laplace.estimate_trunc(overall_res)
  overall_prior = [Laplace.sample_trunc(*overall_params) for _ in
                   xrange(NUM_PRIOR)]

  print "Overall Arrival Time Parameters:", overall_params

  # learn the overall site residual for each site and sample some points
  # for an empirical Bayes prior
  site_prior = {}
  print "Site Arrival Time Parameters:"
  for siteid in range(earthmodel.NumSites()):
    reslist = site_res[siteid]
    if len(reslist) < NUM_PRIOR:
      site_prior[siteid] = []
    else:
      site_params = Laplace.estimate_trunc(reslist)
      site_prior[siteid] = [Laplace.sample_trunc(*site_params)
                            for _ in xrange(NUM_PRIOR)]
      print siteid, ":", site_params
  
  # learn the overall phase residual for each phase and sample some points
  # for an empirical Bayes prior
  phase_prior = {}
  print "Phase Arrival Time Residuals: Location and Scale"
  for phaseid in range(earthmodel.NumPhases()):
    reslist = phase_res[phaseid]
    if len(reslist) < NUM_PRIOR:
      phase_prior[phaseid] = []
    else:
      phase_params = Laplace.estimate_trunc(reslist)
      phase_prior[phaseid] = [Laplace.sample_trunc(*phase_params)
                              for _ in xrange(NUM_PRIOR)]
      print phaseid, ":", phase_params
  
  fp = open(param_filename, "w")
  
  print >>fp, earthmodel.NumSites(), earthmodel.NumPhases()
  
  for siteid in range(earthmodel.NumSites()):
    for phaseid in range(earthmodel.NumPhases()):
      reslist = site_phase_res[siteid, phaseid] + site_prior[siteid]\
                + phase_prior[phaseid] + overall_prior
      loc, scale, minval, maxval = Laplace.estimate_trunc(reslist)
      print >>fp, loc, scale, minval, maxval
  
  fp.close()
  
