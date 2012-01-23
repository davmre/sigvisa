from database.dataset import DET_SITE_COL, UPTIME_QUANT

def learn(param_filename, detections, false_dets, site_up):

  # add one smoothing, add one false detection for each site
  site_falsecnt = [1 for _ in range(len(site_up))]

  for detnum in false_dets:
    site_falsecnt[int(detections[detnum, DET_SITE_COL])] += 1
  
  fp = open(param_filename, "w")

  print >>fp, len(site_up)              # number of sites

  for siteid in range(len(site_up)):
    # note: we are adding one extra hour for each site
    print >>fp, float(site_falsecnt[siteid]) / ((1+sum(site_up[siteid]))
                                                * UPTIME_QUANT)

  fp.close()

