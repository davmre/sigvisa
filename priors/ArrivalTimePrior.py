
def learn(param_filename, earthmodel, start_time, end_time,
          detections, leb_events, leb_evlist, site_up):
  
  fp = open(param_filename, "w")
  
  print >>fp, len(site_up), earthmodel.NumPhases()
  
  fp.close()
  
