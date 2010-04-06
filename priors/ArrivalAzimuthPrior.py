from utils import Laplace

from database.dataset import EV_LON_COL, EV_LAT_COL, EV_DEPTH_COL,\
     EV_TIME_COL, DET_SITE_COL, DET_AZI_COL

NUM_PRIOR = 10                          # number of prior points

def learn(param_filename, earthmodel, detections, leb_events, leb_evlist):
  overall_res = []
  
  # keep a list of arrival azimuth residuals for each site
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
      arraz = detections[detnum, DET_AZI_COL]
      
      pred_az = earthmodel.ArrivalAzimuth(event[EV_LON_COL], event[EV_LAT_COL],
                                          siteid)
      
      res = earthmodel.DiffAzimuth(pred_az, arraz)

      if res > 1000:
        raise ValueError("residual too large")
      
      overall_res.append(res)
      site_res[siteid].append(res)
      phase_res[phaseid].append(res)
      site_phase_res[siteid, phaseid].append(res)

  # learn the overall residual for and sample some points
  # for an empirical Bayes prior
  overall_loc, overall_scale = Laplace.estimate(overall_res)
  overall_prior = [Laplace.sample(overall_loc, overall_scale) for _ in
                   xrange(NUM_PRIOR)]

  print "Overall Arrival Azimuth location and scale:",\
        overall_loc, overall_scale

  # learn the overall site residual for each site and sample some points
  # for an empirical Bayes prior
  site_prior = {}
  print "Site Arrival Azimuth Residuals: Location and Scale"
  for siteid in range(earthmodel.NumSites()):
    reslist = site_res[siteid]
    if len(reslist) < NUM_PRIOR:
      site_prior[siteid] = []
    else:
      site_loc, site_scale = Laplace.estimate(reslist)
      site_prior[siteid] = [Laplace.sample(site_loc, site_scale)
                            for _ in xrange(NUM_PRIOR)]
      print siteid, ":", site_loc, site_scale
  
  # learn the overall phase residual for each phase and sample some points
  # for an empirical Bayes prior
  phase_prior = {}
  print "Phase Arrival Azimuth Residuals: Location and Scale"
  for phaseid in range(earthmodel.NumPhases()):
    reslist = phase_res[phaseid]
    if len(reslist) < NUM_PRIOR:
      phase_prior[phaseid] = []
    else:
      phase_loc, phase_scale = Laplace.estimate(reslist)
      phase_prior[phaseid] = [Laplace.sample(phase_loc, phase_scale)
                              for _ in xrange(NUM_PRIOR)]
      print phaseid, ":", phase_loc, phase_scale
  
  fp = open(param_filename, "w")
  
  print >>fp, earthmodel.NumSites(), earthmodel.NumPhases()
  
  for siteid in range(earthmodel.NumSites()):
    for phaseid in range(earthmodel.NumPhases()):
      reslist = site_phase_res[siteid, phaseid] + site_prior[siteid]\
                + phase_prior[phaseid] + overall_prior
      loc, scale = Laplace.estimate(reslist)
      print >>fp, loc, scale
      
  fp.close()
  
