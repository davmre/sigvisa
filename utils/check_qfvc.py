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

def open_word_iter(filename):
  """
  Iterate a file word by word, ignoring whitespace and comments
  """
  filep = open(filename)
  lines = filep.readlines()
  for line in lines:
    for word in line.split():
      if word.isspace():
        continue
      elif word[0] == '#':
        break
      else:
        yield word
  filep.close()

def load_qfvc(filename):
  qfvc_file = open_word_iter(filename)
  qfvc_file.next()                      # skip first word

  numdep = int(qfvc_file.next())
  depths = np.array([float(qfvc_file.next()) for _ in range(numdep)])

  numdist = int(qfvc_file.next())
  dists = np.array([float(qfvc_file.next()) for _ in range(numdist)])

  table = np.ndarray((numdep, numdist), float)

  for dep in range(numdep):
    for dist in range(numdist):
      table[dep, dist] = float(qfvc_file.next())

  return RectBivariateSpline(depths, dists, table)

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-r", "--hours", dest="hours", default=None,
                    type="float",
                    help = "inference on HOURS worth of data (all)")
  parser.add_option("-k", "--skip", dest="skip", default=0,
                    type="float",
                    help = "skip the first HOURS of data (0)")
  parser.add_option("-1", "--type1", dest="type1", default=False,
                    action = "store_true",
                    help = "Type 1 fonts (False)")
  parser.add_option("-p", "--phase", dest="phase", default="P",
                    help = "data on the specified phase or "" for all (P)")

  (options, args) = parser.parse_args()

  # use Type 1 fonts by invoking latex
  if options.type1:
    plt.rcParams['text.usetex'] = True

  start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
         sel3_evlist, site_up, sites, phasenames, phasetimedef \
         = read_data("training", hours=options.hours, skip=options.skip)

  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)

  qfvc = load_qfvc(os.path.join(param_dirname, "qfvc.mb"))


  overall_q_err = []
  overall_snr = []
  overall_mb = []
  sta_q_err = dict((s,[]) for s in range(len(sites)))

  for evnum, event in enumerate(leb_events):
    if event[EV_MB_COL] <= 2:
      continue

    evlist = leb_evlist[evnum]

    for phaseid, detnum in evlist:
      if len(options.phase) and options.phase != phasenames[phaseid]:
        continue

      det = detections[detnum]
      siteid = int(det[DET_SITE_COL])
      dist = earthmodel.Delta(event[EV_LON_COL], event[EV_LAT_COL], siteid)

      if len(leb_events) < 10:
        print "mb %f log(a/t) %f qf %f" % (event[EV_MB_COL],
                                    np.log10(det[DET_AMP_COL]/det[DET_PER_COL]),
                                    float(qfvc(event[EV_DEPTH_COL], dist)))

      obs_q = event[EV_MB_COL] - np.log10(det[DET_AMP_COL] / det[DET_PER_COL])
      calc_q = float(qfvc(event[EV_DEPTH_COL], dist))

      overall_q_err.append(obs_q - calc_q)
      sta_q_err[siteid].append(obs_q - calc_q)

      overall_snr.append(det[DET_SNR_COL])
      overall_mb.append(event[EV_MB_COL])

  def plot_q_factor(title, data):
    plt.figure()
    plt.title(title)
    n, bins, patches = plt.hist(data, 100, normed=True, facecolor='blue',
                                alpha=1.0, label="data")

    mu, sigma = np.average(data), np.std(data)
    gauss_pdf = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, gauss_pdf, 'r--', linewidth=3, label="Gauss")

    print title, "mu", mu, "sigma", sigma, "avg. logpdf", \
          norm.logpdf(data, mu, sigma).sum()/len(data)

    loc, scale = Laplace.estimate(data)
    lap_pdf = np.exp(Laplace.ldensity(loc, scale, bins))
    plt.plot(bins, lap_pdf, 'k-', linewidth=3, label="Laplace")

    print title, "loc", loc, "scale", scale, "avg. logpdf", \
          laplace.logpdf(data, loc, scale).sum()/len(data)

    ## prob, loc, scale = Laplace.estimate_laplace_uniform_dist(data, -8, 4)
    ## lapu_pdf = np.exp(Laplace.ldensity_laplace_uniform_dist(prob, loc, scale,
    ##                                                         -8, 4, bins))
    ## plt.plot(bins, lapu_pdf, 'g-', linewidth=3, label="Laplace+Unif")

    plt.xlabel("Obs Q - Calc Q")
    plt.legend()

    # draw Q-Q plots of the residuals
    plt.figure()
    probplot(data, (mu, sigma), dist='norm', fit=False, plot=plt)
    plt.title("Normal distribution Probability Plot")

    plt.figure()
    probplot(data, (loc, scale), dist='laplace', fit=False, plot=plt)
    plt.title("Laplace distribution Probability Plot")

  # plot overall Q factors
  plot_q_factor("Error in Q Factor, %s phase, all sites" % options.phase,
                overall_q_err)

  # array vs 3-C Q factors
  plot_q_factor("Error in Q Factor, %s phase, all arrays" % options.phase,
                sum([sta_q_err[siteid] for siteid, site in enumerate(sites)
                     if site[SITE_IS_ARRAY]], []))

  plot_q_factor("Error in Q Factor, %s phase, all 3-C" % options.phase,
                sum([sta_q_err[siteid] for siteid, site in enumerate(sites)
                     if not site[SITE_IS_ARRAY]], []))

  # top sites
  sorted_siteids = range(len(sites))
  sorted_siteids.sort(cmp=lambda x,y: cmp(len(sta_q_err[x]), len(sta_q_err[y])),
                      reverse=True)

  for siteid in sorted_siteids[:4]:
    plot_q_factor("Error in Q Factor, %s phase, site %d" % (options.phase,
                                                            siteid),
                  sta_q_err[siteid])

  plt.figure()
  plt.title("%s phase" % options.phase)
  plt.scatter(overall_q_err, overall_snr, s=1)
  plt.xlabel("Obs Q - Calc Q")
  plt.ylabel("SNR")

  plt.figure()
  plt.title("%s phase" % options.phase)
  plt.scatter(overall_q_err, overall_mb, s=1)
  plt.xlabel("Obs Q - Calc Q")
  plt.ylabel(r'$m_b$')

  plt.figure()
  plt.title("%s phase" % options.phase)
  plt.scatter(overall_mb, overall_snr, s=1)
  plt.xlabel(r'$m_b$')
  plt.ylabel("SNR")

  plt.show()

if __name__ == "__main__":
  main("parameters")
