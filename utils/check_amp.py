# Copyright (c) 2012, Bayesian Logic, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Bayesian Logic, Inc. nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# Bayesian Logic, Inc. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
import os, sys, time
import numpy as np
from optparse import OptionParser
from scipy.interpolate import RectBivariateSpline
from scipy.stats import probplot, norm, laplace

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from sigvisa.database.dataset import *
from sigvisa.database.db import connect
import netvisa, learn
import sigvisa.utils.Laplace as Laplace

class NoTextPlot:
  def __init__(self, real_plot, type1):
    self._plot = real_plot
    self._type1 = type1

  def plot(self, *args, **kwargs):
    self._plot.plot(*args, **kwargs)

  def title(self, *args, **kwargs):
    if not self._type1:
      self._plot.title(*args, **kwargs)

  def xlabel(self, *args, **kwargs):
    self._plot.xlabel(*args, **kwargs)

  def ylabel(self, *args, **kwargs):
    self._plot.ylabel(*args, **kwargs)

  def text(self, *args, **kwargs):
    # latex requires a $ around mathematical formulas
    if self._type1 and len(args) >= 3:
      args = args[:2] + (r'$' + args[2] + r'$',) + args[3:]

    self._plot.text(*args, **kwargs)

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-1", "--type1", dest="type1", default=False,
                    action = "store_true",
                    help = "Type 1 fonts (False)")
  parser.add_option("-n", "--numbins", dest="numbins", default=50,
                    type = "int",
                    help = "number of histogram bins (50)")
  parser.add_option("-p", "--phase", dest="phase", default="P",
                    help = "data on the specified phase or "" for all (P)")

  (options, args) = parser.parse_args()

  # use Type 1 fonts by invoking latex
  if options.type1:
    plt.rcParams['text.usetex'] = True
    file_suffix = ".pdf"
  else:
    file_suffix = ".png"

  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data("validation", hours=None, skip=0)

  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  all_zvals = []

  for evnum, event in enumerate(leb_events):
    evlist = leb_evlist[evnum]

    for phaseid, detnum in evlist:
      # restrict to the requested phase
      if len(options.phase) and options.phase != phasenames[phaseid]:
        continue

      det = detections[detnum]
      siteid = int(det[DET_SITE_COL])

      ttime = earthmodel.ArrivalTime(event[EV_LON_COL], event[EV_LAT_COL],
                                     event[EV_DEPTH_COL], 0, phaseid, siteid)

      if ttime > 0:
        zval, logprob, cdf = netmodel.arramp_zval_logprob_cdf(event[EV_MB_COL],
                                         event[EV_DEPTH_COL],
                                         ttime, siteid, phaseid,
                                         det[DET_AMP_COL])

        all_zvals.append(zval)

  # histogram of Z vals
  plt.figure(figsize=(8,4.8))
  if not options.type1:
    if len(options.phase):
      plt.title("%s phase amplitude z vals, all sites" % options.phase)
    else:
      plt.title("all phase amplitude z vals, all sites")

  n, bins, patches = plt.hist(all_zvals, options.numbins,
                              normed=True, alpha=0.5,
                              facecolor="blue",
                              label="validation z vals")
  stdnorm_pdf = mlab.normpdf(bins, 0, 1)
  plt.plot(bins, stdnorm_pdf, 'k--', linewidth=3, label="Standard Normal")
  plt.ylim(0, .6)
  plt.xlabel("z val")
  plt.ylabel("probability")
  plt.legend(loc="upper left")
  plt.savefig(os.path.join("output", "amp-%s-zvals%s"
                           % (options.phase, file_suffix)))


  # Q-Q plot of Z vals
  plt.figure(figsize=(8,4.8))
  probplot(all_zvals, (0, 1), dist='norm', fit=False,
           plot=NoTextPlot(plt, options.type1))
  if not options.type1:
    plt.title("Amplitude Z-vals phase=%s Probability Plot" % options.phase)
  plt.savefig(os.path.join("output", "amp-%s-zvals-qq%s"
                           % (options.phase, file_suffix)))


  plt.show()

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
