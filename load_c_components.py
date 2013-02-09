import os, cPickle, sys, tarfile, csv
import numpy as np
from optparse import OptionParser

from database.dataset import *

from datetime import datetime
from database.az_slow_corr import load_az_slow_corr

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

import sigvisa_c, sigvisa

def log_trace(trc, filename, format):

  print "WARNING: ignoring log_trace from c code"
  return True

def load_earth(param_dirname, sites, phasenames, phasetimedef):
  model = sigvisa_c.EarthModel(sites, phasenames, phasetimedef,
                             os.path.join(param_dirname, "ttime", "iasp91."),
                             "",
                             os.path.join(param_dirname,"GA_dist_depth_ranges"),
                             os.path.join(param_dirname,"qfvc.mb"))

  return model


def load_sigvisa(param_dirname, start_time, end_time, signal_model_name, site_up,
                 sites, phasenames, phasetimedef, load_signal_params = True):
  earthmodel = load_earth(param_dirname, sites, phasenames, phasetimedef)

  sigmodel = sigvisa_c.SigModel(earthmodel, start_time, end_time, signal_model_name,
                              os.path.join(param_dirname, "NumEventPrior.txt"),
                              os.path.join(param_dirname, "EventLocationPrior.txt"),
                              os.path.join(param_dirname, "EventMagPrior.txt"),
                              os.path.join(param_dirname, "ArrivalTimePrior.txt"),
                              os.path.join(param_dirname, "ArrivalAzimuthPrior.txt"),
                              os.path.join(param_dirname, "ArrivalSlownessPrior.txt"),
                              os.path.join(param_dirname, "ArrivalAmplitudePrior.txt"),
                              log_trace)

#  if (load_signal_params):
#    sigmodel.set_all_signal_params(priors.SignalPrior.read_params(os.path.join(param_dirname, "EnvelopeSignalModel.txt")))

  return sigmodel
