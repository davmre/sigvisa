import os
from optparse import OptionParser

from database.dataset import *

import priors.NumEventPrior
import priors.EventLocationPrior
import priors.EventMagPrior
import priors.EventDetectionPrior
import priors.NumFalseDetPrior
import priors.ArrivalTimePrior
import priors.ArrivalAzimuthPrior
import priors.ArrivalSlownessPrior
import priors.ArrivalPhasePrior
import priors.ArrivalSNR

import netvisa

def load_earth(param_dirname, sites, phasenames, phasetimedef):
  model = netvisa.EarthModel(sites, phasenames, phasetimedef,
                             os.path.join(param_dirname, "ttime", "iasp91."))
  return model

def load_netvisa(param_dirname, start_time, end_time, detections, site_up,
                 sites, phasenames, phasetimedef):

  earthmodel = load_earth(param_dirname, sites, phasenames, phasetimedef)
    
  model = netvisa.NetModel(earthmodel,
                           start_time, end_time, detections, site_up,
                           os.path.join(param_dirname, "NumEventPrior.txt"),
                           os.path.join(param_dirname,
                                        "EventLocationPrior.txt"),
                           os.path.join(param_dirname, "EventMagPrior.txt"),
                           os.path.join(param_dirname,
                                        "EventDetectionPrior.txt"),
                           os.path.join(param_dirname,
                                        "ArrivalTimePrior.txt"),
                           os.path.join(param_dirname,
                                        "NumFalseDetPrior.txt"),
                           os.path.join(param_dirname,
                                        "ArrivalAzimuthPrior.txt"),
                           os.path.join(param_dirname,
                                        "ArrivalSlownessPrior.txt"),
                           os.path.join(param_dirname,
                                        "ArrivalPhasePrior.txt"),
                           os.path.join(param_dirname,
                                        "ArrivalSNR.txt")
                           )
  
  return model

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-q", "--quick", dest="quick", default=False,
                    action = "store_true",
                    help = "quick training on a subset of data (False)")
  parser.add_option("-n", "--nodet", dest="nodet", default=False,
                    action = "store_true",
                    help = "no detection prior training (False)")
  parser.add_option("-x", "--text", dest="gui", default=True,
                    action = "store_false",
                    help = "text only output (False)")
  parser.add_option("-s", "--silent", dest="verbose", default=True,
                    action = "store_false",
                    help = "silent, i.e. no output (False)")
  (options, args) = parser.parse_args()

  if options.quick:
    hours = 100
  else:
    hours = None
  
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data(hours=hours)

  earthmodel = load_earth(param_dirname, sites, phasenames, phasetimedef)
  
  priors.NumEventPrior.learn(os.path.join(param_dirname, "NumEventPrior.txt"),
                             start_time, end_time, leb_events)
  
  priors.EventLocationPrior.learn(os.path.join(param_dirname,
                                               "EventLocationPrior.txt"),
                                  leb_events)
  
  priors.EventMagPrior.learn(os.path.join(param_dirname,
                                          "EventMagPrior.txt"),
                             leb_events)

  if not options.nodet:
    priors.EventDetectionPrior.learn(os.path.join(param_dirname,
                                                  "EventDetectionPrior.txt"),
                                     earthmodel, start_time, end_time,
                                     detections, leb_events, leb_evlist,
                                     site_up)

  priors.NumFalseDetPrior.learn(os.path.join(param_dirname,
                                             "NumFalseDetPrior.txt"),
                                earthmodel, start_time, end_time,
                                detections, leb_events, leb_evlist,
                                site_up)

  priors.ArrivalTimePrior.learn(os.path.join(param_dirname,
                                             "ArrivalTimePrior.txt"),
                                earthmodel, detections, leb_events, leb_evlist)

  priors.ArrivalAzimuthPrior.learn(os.path.join(param_dirname,
                                                "ArrivalAzimuthPrior.txt"),
                                   earthmodel, detections, leb_events,
                                   leb_evlist)
  
  priors.ArrivalSlownessPrior.learn(os.path.join(param_dirname,
                                                 "ArrivalSlownessPrior.txt"),
                                    earthmodel, detections, leb_events,
                                    leb_evlist)

  priors.ArrivalPhasePrior.learn(os.path.join(param_dirname,
                                              "ArrivalPhasePrior.txt"),
                                 options, earthmodel, detections, leb_events,
                                 leb_evlist)

  priors.ArrivalSNR.learn(os.path.join(param_dirname,
                                       "ArrivalSNR.txt"),
                          options, earthmodel, detections, leb_events,
                          leb_evlist)
  
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
  
  
