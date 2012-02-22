from utils import Laplace

from database.dataset import EV_LON_COL, EV_LAT_COL, EV_DEPTH_COL,\
     EV_TIME_COL, DET_SITE_COL, DET_TIME_COL

def learn(param_filename, earthmodel, detections, leb_events, leb_evlist):
  # keep a residual for each site and phase
  phase_site_res = [[[] for siteid in range(earthmodel.NumSites())]
                        for phaseid in range(earthmodel.NumTimeDefPhases())]
                        
  
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
      
      if res > 1000:
        raise ValueError("residual too large")
      
      phase_site_res[phaseid][siteid].append(res)
  
  phase_site_param = []
  # for each phase
  print "Arrival Time:"
  for phaseid in range(earthmodel.NumTimeDefPhases()):
    # we will learn each site's location and scale
    print earthmodel.PhaseName(phaseid),":"
    site_param, loc, scale, beta=Laplace.hier_estimate(phase_site_res[phaseid])
    print loc, scale, beta
    phase_site_param.append(site_param)
  
  fp = open(param_filename, "w")
  
  print >>fp, earthmodel.NumSites(), earthmodel.NumTimeDefPhases()
  
  for siteid in range(earthmodel.NumSites()):
    for phaseid in range(earthmodel.NumTimeDefPhases()):
      loc, scale = phase_site_param[phaseid][siteid]
      print >>fp, loc, scale
  
  fp.close()
