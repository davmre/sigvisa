import os

from database.dataset import *

import priors.NumEventPrior
import priors.EventLocationPrior
import priors.EventMagPrior
import priors.EventDetectionPrior

import netvisa

def load_earth(param_dirname, sites, phasenames, phasetimedef):
  model = netvisa.EarthModel(sites, phasenames, phasetimedef,
                             os.path.join(param_dirname, "ttime", "iasp91."))
  return model

def load_netvisa(param_dirname, start_time, end_time, detections, site_up):
  
  model = netvisa.NetModel(start_time, end_time, detections, site_up,
                           os.path.join(param_dirname, "NumEventPrior.txt"),
                           os.path.join(param_dirname,
                                        "EventLocationPrior.txt"),
                           os.path.join(param_dirname, "EventMagPrior.txt"),
                           os.path.join(param_dirname,
                                        "EventDetectionPrior.txt")
                           )
  
  return model

def main(param_dirname):
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef = read_data(hours=100)

  earthmodel = load_earth(param_dirname, sites, phasenames, phasetimedef)
  
  priors.NumEventPrior.learn(os.path.join(param_dirname, "NumEventPrior.txt"),
                             start_time, end_time, leb_events)
  
  priors.EventLocationPrior.learn(os.path.join(param_dirname,
                                               "EventLocationPrior.txt"),
                                  leb_events)
  
  priors.EventMagPrior.learn(os.path.join(param_dirname,
                                          "EventMagPrior.txt"),
                             leb_events)

  priors.EventDetectionPrior.learn(os.path.join(param_dirname,
                                                "EventDetectionPrior.txt"),
                                   earthmodel,
                                   detections, leb_events, leb_evlist,
                                   site_up, sites, phasenames)
if __name__ == "__main__":
  try:
    main("parameters")
  except SystemExit:
    raise
  except:
    import pdb, traceback, sys
    traceback.print_exc(file=sys.stdout)
    pdb.post_mortem(sys.exc_traceback)
    raise
  
  
