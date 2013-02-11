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
from sigvisa.utils import Laplace
import numpy as np

from sigvisa.database.dataset import EV_LON_COL, EV_LAT_COL, EV_DEPTH_COL,\
     EV_TIME_COL, DET_SITE_COL, DET_TIME_COL, DET_AZI_COL, DET_SLO_COL

def learn(param_filename, earthmodel, detections, leb_events, leb_evlist):
  # keep a residual for each site and phase
  phase_site_res = [[[] for siteid in range(earthmodel.NumSites())]
                        for phaseid in range(earthmodel.NumTimeDefPhases())]

  for evnum, event in enumerate(leb_events):
    for phaseid, detnum in leb_evlist[evnum]:
      siteid = int(detections[detnum, DET_SITE_COL])
      arrtime = detections[detnum, DET_TIME_COL]

      pred_arrtime = earthmodel.ArrivalTime(event[EV_LON_COL],
                                            event[EV_LAT_COL],
                                            event[EV_DEPTH_COL],
                                            event[EV_TIME_COL], phaseid,
                                            siteid)
      if pred_arrtime < 0:
        continue

      res = arrtime - pred_arrtime
      if res > 100:
        print "Skipping time residual %d" % res
        continue

      phase_site_res[phaseid][siteid].append(res)

  phase_site_param = []
  # for each phase
  print "Arrival Time:"
  for phaseid in range(earthmodel.NumTimeDefPhases()):
    # we will learn each site's location and scale
    print earthmodel.PhaseName(phaseid),":"
    site_param, loc, scale, beta=Laplace.hier_estimate(phase_site_res[phaseid])
    print loc, scale, beta
    phase_site_param.append(site_param)

  fp = open(param_filename, "w")

  print >>fp, earthmodel.NumSites(), earthmodel.NumTimeDefPhases()

  for siteid in range(earthmodel.NumSites()):
    for phaseid in range(earthmodel.NumTimeDefPhases()):
      loc, scale = phase_site_param[phaseid][siteid]
      print >>fp, loc, scale

  fp.close()

def create_featureset(earthmodel,start_time, end_time, detections, leb_events,
                      leb_evlist, sel3_events, sel3_evlist, site_up, sites,
                      phasenames, phasetimedef):

  phase_data = [([], [], []) for phaseid in xrange(phasetimedef.sum())]

  for evnum, event in enumerate(leb_events):
    for phaseid, detnum in leb_evlist[evnum]:
      siteid = int(detections[detnum, DET_SITE_COL])
      if siteid != 6:
        continue

      arrtime = detections[detnum, DET_TIME_COL]

      pred_arrtime = earthmodel.ArrivalTime(event[EV_LON_COL],
                                            event[EV_LAT_COL],
                                            event[EV_DEPTH_COL],
                                            event[EV_TIME_COL], phaseid,
                                            siteid)
      if pred_arrtime < 0:
        continue

      restime = arrtime - pred_arrtime
      if restime > 100:
        continue

      arraz = detections[detnum, DET_AZI_COL]

      pred_az = earthmodel.ArrivalAzimuth(event[EV_LON_COL], event[EV_LAT_COL],
                                          siteid)

      resaz = earthmodel.DiffAzimuth(pred_az, arraz)

      arrslo = detections[detnum, DET_SLO_COL]

      pred_slo = earthmodel.ArrivalSlowness(event[EV_LON_COL],
                                            event[EV_LAT_COL],
                                            event[EV_DEPTH_COL],
                                            phaseid, siteid)

      resslo = arrslo - pred_slo


      phase_data[phaseid][0].append(restime)
      phase_data[phaseid][1].append(resaz)
      phase_data[phaseid][2].append(resslo)

  return phase_data

def compare_laplace_gaussian(trainres, testres):
  loc, scale = Laplace.estimate(trainres)

  laploglike = sum(Laplace.ldensity(loc, scale, v) for v in testres)

  mean, std = np.mean(trainres), np.std(trainres)

  gausslike = sum(-0.5 * np.log(2*np.pi) - np.log(std) \
                  - 0.5 * (v - mean)**2 / std ** 2 for v in testres)

  print "Laplace Avg Log Like", laploglike/len(testres),
  print "Gaussian Avg Log Like", gausslike/len(testres)

def corrcoeff(xvals, yvals):
  xmean = np.mean(xvals)
  ymean = np.mean(yvals)

  return sum((x - xmean) * (y-ymean) for x,y in zip(xvals, yvals)) \
         / (np.sqrt(sum((x-xmean)**2 for x in xvals))
            * np.sqrt(sum((y-ymean)**2 for y in yvals)))

def test_model(options, earthmodel, train, test):
  train_feat = create_featureset(earthmodel, *train)
  test_feat = create_featureset(earthmodel, *test)

  for phaseid, (timeres, azres, slores) in enumerate(train_feat):

    if len(timeres) < 2 or len(test_feat[phaseid][0]) < 2:
      continue

    phname = train[9][phaseid]
    print "Phase", phname, "traincnt", len(timeres), "testcnt", \
          len(test_feat[phaseid][0])

    print "Time residual"
    compare_laplace_gaussian(timeres, test_feat[phaseid][0])

    print "Azimuth residual"
    compare_laplace_gaussian(timeres, test_feat[phaseid][1])

    print "Slowness residual"
    compare_laplace_gaussian(timeres, test_feat[phaseid][2])

    print "correlation time az", corrcoeff(timeres, azres)
    print "correlation time slo", corrcoeff(timeres, slores)
    print "correlation slo az", corrcoeff(slores, azres)

    if options.gui and phname in ('P', 'S'):
      plt = options.plt
      plt.figure()
      plt.title(phname)
      plt.scatter(timeres, azres, s=1)
      plt.xlabel("Time Residual (s)")
      plt.ylabel("Azimuth Residual (s/deg)")

      plt.figure()
      plt.title(phname)
      plt.scatter(timeres, slores, s=1)
      plt.xlabel("Time Residual (s)")
      plt.ylabel("Slowness Residual (s/deg)")

      plt.figure()
      plt.title(phname)
      plt.scatter(azres, slores, s=1)
      plt.xlabel("Azimuth Residual (deg)")
      plt.ylabel("Slowness Residual (s/deg)")
