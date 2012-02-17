import os, cPickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from optparse import OptionParser

from database.dataset import *

import priors.SecDetPrior
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
                            os.path.join(param_dirname, "ttime", "iasp91."),
                            os.path.join(param_dirname,"GA_dist_depth_ranges"),
                            os.path.join(param_dirname,"qfvc.mb"))
  return model

def load_netvisa(param_dirname, start_time, end_time, detections, site_up,
                 sites, phasenames, phasetimedef):

  earthmodel = load_earth(param_dirname, sites, phasenames, phasetimedef)
    
  model = netvisa.NetModel(earthmodel,
                           start_time, end_time, detections, site_up,
                           os.path.join(param_dirname, "SecDetPrior.txt"),
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
                                        "ArrivalSNRPrior.txt"),
                           os.path.join(param_dirname,
                                        "ArrivalAmplitudePrior.txt")
                           )
  
  return model

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-q", "--quick", dest="quick", default=False,
                    action = "store_true",
                    help = "quick training on a subset of data (False)")
  parser.add_option("-g", "--gui", dest="gui", default=False,
                    action = "store_true",
                    help = "display the model  (False)")
  parser.add_option("-s", "--silent", dest="verbose", default=True,
                    action = "store_false",
                    help = "silent, i.e. no output (False)")
  parser.add_option("-1", "--type1", dest="type1", default=False,
                    action = "store_true",
                    help = "Type 1 fonts (False)")
  parser.add_option("-i", "--visa_leb_runid", dest="visa_leb_runid",
                    default=None, help = "Visa runid to be treated as leb",
                    metavar="RUNID", type="int")
  parser.add_option("-m", "--model", dest="model",
                    default=None, help = "Which model(s) to learn (all)")
  parser.add_option("-w", "--writefig", dest="writefig",
                    default=None, help = "Directory to save figures (None)",
                    metavar="DIR")
  parser.add_option("-p", "--pdf", dest="pdf",
                    default=None, help = "pdf file to save figures (None)",
                    metavar="FILE")
  parser.add_option("-d", "--datadir", dest="datadir",
                    default=None, help = "Directory to save data (None)",
                    metavar="DIR")
  parser.add_option("-c", "--cache", dest="cache", default=False,
                    action = "store_true",
                    help = "write data to cache and read from cache (False)")
  
  (options, args) = parser.parse_args()

  # use Type 1 fonts by invoking latex
  if options.type1:
    plt.rcParams['text.usetex'] = True
    
  if options.pdf:
    options.pdf = PdfPages(options.pdf)
    
  if options.gui:
    options.plt = plt
    showgui = True
  # if the user has not request a GUI but he does want the pictures then we
  # will set the gui flag for the subcomponents, however, we won't display
  # the generated images
  elif options.pdf or options.writefig:
    options.plt = plt
    options.gui = True
    showgui = False
  else:
    showgui = False

  if options.datadir:
    param_dirname = options.datadir
  
  if options.quick:
    hours = 100
  else:
    hours = None

  if options.model is not None:
    options.model = set(m for m in options.model.split(","))

  cache_fname = "cache-training-hours-%s-visa-%s.pic1" \
                % (str(hours), str(options.visa_leb_runid))
  
  if options.cache and os.path.exists(cache_fname):
    start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
                sel3_evlist, site_up, sites, phasenames, phasetimedef \
                = cPickle.load(file(cache_fname, "rb"))
  else:
    start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
                sel3_evlist, site_up, sites, phasenames, phasetimedef \
                = read_data(hours=hours, visa_leb_runid=options.visa_leb_runid)

  if options.cache and not os.path.exists(cache_fname):
    cPickle.dump((start_time, end_time, detections, leb_events, leb_evlist,
                  sel3_events, sel3_evlist, site_up, sites, phasenames,
                  phasetimedef), file(cache_fname, "wb"), protocol=1)

  earthmodel = load_earth(param_dirname, sites, phasenames, phasetimedef)

  false_dets = priors.SecDetPrior.compute_false_detections(detections,
                                                           leb_evlist)

  if options.model is None or "SecDet" in options.model:
    priors.SecDetPrior.learn(os.path.join(param_dirname, "SecDetPrior.txt"),
                             options, earthmodel, detections, leb_events,
                             leb_evlist)

  if options.model is None or "NumEvent" in options.model:
    priors.NumEventPrior.learn(os.path.join(param_dirname, "NumEventPrior.txt"),
                               options, start_time, end_time, leb_events)
  
  if options.model is None or "EventLocation" in options.model:
    priors.EventLocationPrior.learn(os.path.join(param_dirname,
                                                 "EventLocationPrior.txt"),
                                    options, leb_events)
  
  if options.model is None or "EventMag" in options.model:
    priors.EventMagPrior.learn(os.path.join(param_dirname,
                                            "EventMagPrior.txt"),
                               options, leb_events)

  if options.model is None or "EventDetection" in options.model:
    priors.EventDetectionPrior.learn(os.path.join(param_dirname,
                                                  "EventDetectionPrior.txt"),
                                     earthmodel, start_time, end_time,
                                     detections, leb_events, leb_evlist,
                                     site_up)

  if options.model is None or "NumFalseDet" in options.model:
    priors.NumFalseDetPrior.learn(os.path.join(param_dirname,
                                               "NumFalseDetPrior.txt"),
                                  detections, false_dets, site_up)
  
  if options.model is None or "ArrivalTime" in options.model:
    priors.ArrivalTimePrior.learn(os.path.join(param_dirname,
                                               "ArrivalTimePrior.txt"),
                                  earthmodel, detections, leb_events,
                                  leb_evlist)

  if options.model is None or "ArrivalAzimuth" in options.model:
    priors.ArrivalAzimuthPrior.learn(os.path.join(param_dirname,
                                                  "ArrivalAzimuthPrior.txt"),
                                     earthmodel, detections, leb_events,
                                     leb_evlist)
  
  if options.model is None or "ArrivalSlowness" in options.model:
    priors.ArrivalSlownessPrior.learn(os.path.join(param_dirname,
                                                   "ArrivalSlownessPrior.txt"),
                                      earthmodel, detections, leb_events,
                                      leb_evlist)

  if options.model is None or "ArrivalPhase" in options.model:
    priors.ArrivalPhasePrior.learn(os.path.join(param_dirname,
                                                "ArrivalPhasePrior.txt"),
                                   options, earthmodel, detections, leb_events,
                                   leb_evlist, false_dets)

  if options.model is None or "ArrivalSNR" in options.model:
    priors.ArrivalSNR.learn(os.path.join(param_dirname,
                                         "ArrivalSNRPrior.txt"),
                            options, earthmodel, detections, leb_events,
                            leb_evlist, false_dets)

  if options.model is None or "ArrivalAmplitude" in options.model:
    priors.ArrivalAmplitudePrior.learn(
      os.path.join(param_dirname, "ArrivalAmplitudePrior.txt"),
      options, earthmodel, detections, leb_events, leb_evlist, false_dets)

  if options.pdf:
    options.pdf.close()
  
  if showgui:
    plt.show()

if __name__ == "__main__":
  np.seterr(divide = 'raise', invalid='raise', over='raise')
  try:
    main("parameters")
  except SystemExit:
    raise
  except:
    import pdb, traceback, sys
    traceback.print_exc(file=sys.stdout)
    pdb.post_mortem(sys.exc_traceback)
    raise
  
  
