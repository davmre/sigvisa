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

from sigvisa.database.dataset import EV_LON_COL, EV_LAT_COL, EV_DEPTH_COL,\
    EV_TIME_COL, DET_SITE_COL, DET_AZI_COL


def learn(param_filename, earthmodel, detections, leb_events, leb_evlist):
    # keep a residual for each site and phase
    phase_site_res = [[[] for siteid in range(earthmodel.NumSites())]
                      for phaseid in range(earthmodel.NumTimeDefPhases())]

    for evnum, event in enumerate(leb_events):
        for phaseid, detnum in leb_evlist[evnum]:
            siteid = int(detections[detnum, DET_SITE_COL])
            arraz = detections[detnum, DET_AZI_COL]

            pred_az = earthmodel.ArrivalAzimuth(event[EV_LON_COL], event[EV_LAT_COL],
                                                siteid)

            res = earthmodel.DiffAzimuth(pred_az, arraz)
            phase_site_res[phaseid][siteid].append(res)

    phase_site_param = []
    # for each phase
    print "Arrival Azimuth:"
    for phaseid in range(earthmodel.NumTimeDefPhases()):
        # we will learn each site's location and scale
        print earthmodel.PhaseName(phaseid), ":"
        site_param, loc, scale, beta = Laplace.hier_estimate(phase_site_res[phaseid])
        print loc, scale, beta
        phase_site_param.append(site_param)

    fp = open(param_filename, "w")

    print >>fp, earthmodel.NumSites(), earthmodel.NumTimeDefPhases()

    for siteid in range(earthmodel.NumSites()):
        for phaseid in range(earthmodel.NumTimeDefPhases()):
            loc, scale = phase_site_param[phaseid][siteid]
            print >>fp, loc, scale

    fp.close()
