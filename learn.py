import os
import numpy as np
import matplotlib.pyplot as plt
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
import priors.ArrivalAmplitudePrior

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
                                        "ArrivalSNR.txt"),
                           os.path.join(param_dirname,
                                        "ArrivalAmplitudePrior.txt")
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
  parser.add_option("-i", "--visa_leb_runid", dest="visa_leb_runid",
                    default=None, help = "Visa runid to be treated as leb",
                    metavar="RUNID")
  (options, args) = parser.parse_args()

  if options.quick:
    hours = 100
  else:
    hours = None
  
  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data(hours=hours, visa_leb_runid=options.visa_leb_runid)

  earthmodel = load_earth(param_dirname, sites, phasenames, phasetimedef)

  leb_seclist = compute_secondary_dets(earthmodel, detections, leb_events,
                                       leb_evlist)
  
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
                                earthmodel, detections, leb_events,
                                leb_evlist)

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

  priors.ArrivalAmplitudePrior.learn(os.path.join(param_dirname,
                                                  "ArrivalAmplitudePrior.txt"),
                                     options, earthmodel, detections,
                                     leb_events, leb_evlist)

  if options.gui:
    plt.show()

def compute_secondary_dets(earthmodel, detections, leb_events, leb_evlist):
  """
  Add secondary detections to LEB
  """
  # compute the set of detections which are off-limits
  detused = [False for detnum in xrange(len(detections))]
  for detlist in leb_evlist:
    for phaseid, detnum in detlist:
      detused[detnum] = True
  
  # assign each unassigned detection to the first event-phase within 10 seconds
  leb_seclist = []
  low_detnum = 0
  for event,detlist in zip(leb_events, leb_evlist):
    phase_site = [[[] for siteid in xrange(earthmodel.NumSites())]
                  for phaseid in xrange(earthmodel.NumPhases())]

    for phaseid, detnum in detlist:
      siteid = int(detections[detnum, DET_SITE_COL])
      prev_dets = phase_site[phaseid][siteid]
      prev_dets.append(detnum)
    
    for detnum in xrange(low_detnum, len(detections)):
      det = detections[detnum]

      if det[DET_TIME_COL] < event[EV_TIME_COL]:
        low_detnum += 1
        continue

      if det[DET_TIME_COL] > (event[EV_TIME_COL] + MAX_TRAVEL_TIME):
        break

      if detused[detnum]:
        continue
      
      siteid = int(det[DET_SITE_COL])
      
      for phaseid in xrange(earthmodel.NumTimeDefPhases()):
        prev_dets = phase_site[phaseid][siteid]
        if not len(prev_dets):
          continue

        last_detnum = prev_dets[-1]
        last_det_time = detections[last_detnum, DET_TIME_COL]
        if det[DET_TIME_COL] > last_det_time \
               and det[DET_TIME_COL] < (last_det_time + 10):
          prev_dets.append(detnum)
          detused[detnum]
          break
        
    seclist = []
    for siteid in xrange(earthmodel.NumSites()):
      for phaseid in xrange(earthmodel.NumPhases()):
        detnums = phase_site[phaseid][siteid]
        if len(detnums):
          seclist.append(tuple([phaseid] + detnums))
    
    leb_seclist.append(seclist)
            
  return leb_seclist

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
  
  
