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
import numpy as np
import math, random

from sigvisa.utils.LogisticModel import LogisticModel
from sigvisa.database.dataset import *

def gtf(val, m, s):
  return math.exp(- float(val - m) ** 2 / (2.0 * float(s) ** 2)) \
         / math.sqrt(2.0 * math.pi * float(s) ** 2)

FEATURE_NAMES = ["mag", "depth", "dist", "dist0", "dist35",
                 "dist40", "dist12520", "dist12540",
                 "mag6", "mag68", "md"]

COEFF_PRIOR_MEAN = 0
COEFF_PRIOR_PREC = 1e-4
COEFF_PREC_PRIOR_SHAPE = 1e-2
COEFF_PREC_PRIOR_SCALE = 1e2

def extract_features(raw):
  output, mag_feat, depth_feat, dist_feat = [], [], [], []
  dist0_feat, dist35_feat, dist40_feat, dist12520_feat = [], [], [], []
  dist12540_feat = []
  mag6_feat, mag68_feat, md_feat, md145_feat = [], [], [], []

  for (isdet, mag, depth, dist) in raw:

    output.append(isdet)

    mag_feat.append(mag)
    depth_feat.append(depth)
    dist_feat.append(dist)
    dist0_feat.append(gtf(dist, 0, 5))
    dist35_feat.append(gtf(dist, 35, 20))
    dist40_feat.append(gtf(dist, 40, 20))
    dist12520_feat.append(gtf(dist, 125, 20))
    dist12540_feat.append(gtf(dist, 125, 40))
    mag6_feat.append(gtf(mag, 6, 5.5))
    mag68_feat.append(gtf(mag, 6, 8))
    md_feat.append((7-mag) * dist)

  feature_values = [mag_feat, depth_feat, dist_feat, dist0_feat,
                    dist35_feat, dist40_feat, dist12520_feat,
                    dist12540_feat,
                    mag6_feat, mag68_feat, md_feat]

  return feature_values, output

def learn(param_fname, earthmodel, start_time, end_time,
          detections, leb_events, leb_evlist, site_up):

  # assume that the time-defining phases precede the non-time-defining ones
  numtimedefphases = earthmodel.NumTimeDefPhases()
  numsites = earthmodel.NumSites()

  site_phase_coeffs = [[None for phaseid in xrange(numtimedefphases)]
                       for siteid in xrange(numsites)]

  # we will learn the model for each phase at a time
  for phaseid in xrange(numtimedefphases):

    # separate out the data for each site
    site_raw = [[] for site in xrange(numsites)]
    for evnum, event in enumerate(leb_events):
      det_site = set(detections[detnum, DET_SITE_COL]\
                     for ph, detnum in leb_evlist[evnum] if ph==phaseid)
      for siteid in range(numsites):
        dist = earthmodel.Delta(event[EV_LON_COL], event[EV_LAT_COL], siteid)
        arrtime = earthmodel.ArrivalTime(event[EV_LON_COL], event[EV_LAT_COL],
                                         event[EV_DEPTH_COL],
                                         event[EV_TIME_COL], phaseid, siteid)

        # check if the site is in the shadow zone of this phase
        if arrtime < 0:
          continue

        # check if the site was up at the expected arrival time
        if arrtime < start_time or arrtime >= end_time \
            or not site_up[siteid, int((arrtime - start_time) / UPTIME_QUANT)]:
          continue

        isdet = int(siteid in det_site)

        site_raw[siteid].append((isdet, event[EV_MB_COL],
                                 event[EV_DEPTH_COL], dist))

    # extract the features for each site and phase and construct a dataset
    site_data = [extract_features(site_raw[siteid])
                 for siteid in xrange(numsites)]

    # we initialize all the means to zero and precisions to 1
    feat_mean = np.zeros(len(FEATURE_NAMES) + 1) # add one for intercept
    feat_prec = np.ones(len(FEATURE_NAMES) + 1)

    print "Phase", earthmodel.PhaseName(phaseid),

    while True:
      # first we will learn the coefficients for each site using
      # the phase global coefficients
      for siteid in xrange(numsites):
        predictors, output = site_data[siteid]
        if len(output) == 0:
          site_phase_coeffs[siteid][phaseid] = feat_mean.copy()
        else:
          model =  LogisticModel("EventDetection", FEATURE_NAMES,
                                 predictors, output,
                                 prior_means = feat_mean,
                                 prior_precisions = feat_prec)

          site_phase_coeffs[siteid][phaseid] = model.coeffs

      prev_feat_mean = feat_mean.copy()
      prev_feat_prec = feat_prec.copy()

      # next we will update the phase global coefficients using the site
      # specific coefficients and the prior
      for fnum in xrange(len(FEATURE_NAMES) + 1):
        feat_mean[fnum] = (feat_prec[fnum]
                           * sum(site_phase_coeffs[siteid][phaseid][fnum]
                                 for siteid in xrange(numsites))
                           + COEFF_PRIOR_PREC * COEFF_PRIOR_MEAN) / \
                           ( numsites * feat_prec[fnum] + COEFF_PRIOR_PREC)

        feat_prec[fnum] = ((.5 * numsites + COEFF_PREC_PRIOR_SHAPE-1)
                           / (0.5 *sum((site_phase_coeffs[siteid][phaseid][fnum]
                                         - feat_mean[fnum])**2
                                        for siteid in xrange(numsites))
                              + (1.0/COEFF_PREC_PRIOR_SCALE)))

      # for each feature we want to measure the square of the relative change
      # in the mean.
      delta_mean = (2.0 * (prev_feat_mean - feat_mean)
                     / (prev_feat_mean + feat_mean + 1e-6)) ** 2

      if delta_mean.max() < .1:
        print
        break
      else:
        print "%.0e" % delta_mean.max(),

    print feat_mean
    print feat_prec

  # write the parameters to the file
  fp = open(param_fname, "w")

  print >>fp, "%d %d" % (numtimedefphases, numsites)
  print >>fp, "Phase, Siteid, (Intercept),", ", ".join(FEATURE_NAMES)

  for phaseid in xrange(numtimedefphases):
    for siteid in xrange(numsites):
      coeffs = site_phase_coeffs[siteid][phaseid]
      # we want to write out the intercept first
      buf = "%s,%d,%s," % (earthmodel.PhaseName(phaseid), siteid,
                           str(coeffs[-1]))

      buf += ",".join([str(coeffs[i]) for i in range(len(FEATURE_NAMES))])

      print >>fp, buf

  fp.close()

def create_featureset(earthmodel,start_time, end_time, detections, leb_events,
                      leb_evlist, sel3_events, sel3_evlist, site_up, sites,
                      phasenames, phasetimedef):
  phase_raw = [[] for phaseid in xrange(phasetimedef.sum())]

  for evnum, event in enumerate(leb_events):
    for phaseid in xrange(phasetimedef.sum()):
      det_site = set(detections[detnum, DET_SITE_COL]\
                     for ph, detnum in leb_evlist[evnum] if ph==phaseid)
      for siteid in [6,]:
        dist = earthmodel.Delta(event[EV_LON_COL], event[EV_LAT_COL], siteid)
        arrtime = earthmodel.ArrivalTime(event[EV_LON_COL], event[EV_LAT_COL],
                                         event[EV_DEPTH_COL],
                                         event[EV_TIME_COL], phaseid, siteid)

        # check if the site is in the shadow zone of this phase
        if arrtime < 0:
          continue

        # check if the site was up at the expected arrival time
        if arrtime < start_time or arrtime >= end_time \
            or not site_up[siteid, int((arrtime - start_time) / UPTIME_QUANT)]:
          continue

        isdet = int(siteid in det_site)

        phase_raw[phaseid].append((isdet, event[EV_MB_COL],
                                    event[EV_DEPTH_COL], dist))

  # extract the features for each phase and construct a dataset
  phase_data = [extract_features(phase_raw[phaseid])
                for phaseid in xrange(phasetimedef.sum())]

  return phase_data

def test_model(earthmodel, train, test):
  train_feat = create_featureset(earthmodel, *train)
  test_feat = create_featureset(earthmodel, *test)
  # we will evaluate with increasing number of features
  for numfeatures in range(len(FEATURE_NAMES)+1):

    print "Evaluating features:", FEATURE_NAMES[:numfeatures]

    # train a model for each phase
    phase_model = []
    for phaseid, (predictors, output) in enumerate(train_feat):
      model = LogisticModel("EventDetection", FEATURE_NAMES[:numfeatures],
                            predictors[:numfeatures], output, verbose=False)

      phase_model.append(model.coeffs)

    # now predict on the test data
    totcnt = 0
    tot_logprob = 0.
    for phaseid, (predictors, output) in enumerate(test_feat):
      model = phase_model[phaseid]
      for idx in range(len(output)):
        logodds = sum(predictors[f][idx] * model[f]
                      for f in xrange(numfeatures)) + model[-1]
        if output[idx]:
          if logodds < -40:
            logprob = logodds
          else:
            logprob = - np.log(1 + np.exp(-logodds))
        else:
          if logodds > 40:
            logprob = -logodds
          else:
            logprob = - np.log(1 + np.exp(logodds))

        tot_logprob += logprob
        totcnt += 1

    print "Avg. log likelihood", tot_logprob / totcnt
