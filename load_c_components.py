import os
import numpy as np


import sigvisa_c


def load_earth(param_dirname, sitenames, allsites, phasenames, phasetimedef):
  model = sigvisa_c.EarthModel(sitenames, allsites, phasenames, phasetimedef,
                             os.path.join(param_dirname, "ttime", "iasp91."),
                             "",
                             os.path.join(param_dirname,"GA_dist_depth_ranges"),
                             os.path.join(param_dirname,"qfvc.mb"))

  return model


def load_sigvisa(earthmodel, param_dirname, site_up,
                 sites, phasenames, phasetimedef):
  sigmodel = sigvisa_c.SigModel(earthmodel,
                              os.path.join(param_dirname, "NumEventPrior.txt"),
                              os.path.join(param_dirname, "EventLocationPrior.txt"),
                              os.path.join(param_dirname, "EventMagPrior.txt"),
                              os.path.join(param_dirname, "ArrivalTimePrior.txt"),
                              os.path.join(param_dirname, "ArrivalAzimuthPrior.txt"),
                              os.path.join(param_dirname, "ArrivalSlownessPrior.txt"),
                              os.path.join(param_dirname, "ArrivalAmplitudePrior.txt"),
                              )

#  if (load_signal_params):
#    sigmodel.set_all_signal_params(priors.SignalPrior.read_params(os.path.join(param_dirname, "EnvelopeSignalModel.txt")))

  return sigmodel
