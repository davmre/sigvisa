from database.dataset import DET_SITE_COL, UPTIME_QUANT

def learn(param_filename, earthmodel, start_time, end_time,
          detections, leb_events, leb_evlist,
          site_up):

  # assume all detections are false
  false_det = [1 for _ in detections]
  for ph_det_list in leb_evlist:
    for ph, detid in ph_det_list:
      false_det[detid] = 0

  # add one smoothing, add one false detection for each site
  site_falsecnt = [1 for _ in range(len(site_up))]

  for detid, isfalse in enumerate(false_det):
    if isfalse:
      site_falsecnt[int(detections[detid, DET_SITE_COL])] += 1
  
  fp = open(param_filename, "w")

  print >>fp, len(site_up)              # number of sites

  for siteid in range(len(site_up)):
    # note: we are adding one extra hour for each site
    print >>fp, float(site_falsecnt[siteid]) / ((1+sum(site_up[siteid]))
                                                * UPTIME_QUANT)

  fp.close()

