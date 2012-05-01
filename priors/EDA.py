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
import rpy2.robjects as robjects

from database.dataset import *

def learn_site(earthmodel, start_time, end_time, detections, leb_events,
               leb_evlist, site_up, numtimedefphases, siteid):
  # initialize feature vectors
  mag_feat = []
  dist_feat = []
  dep_feat = []
  phaseid_feat = [[] for _ in range(numtimedefphases)]
  output = []

  for evnum, event in enumerate(leb_events):
    # compute the list of phases detected for this event at this site
    det_phase = np.zeros(numtimedefphases, int)
    for true_phaseid, detnum in leb_evlist[evnum]:
      det = detections[detnum]
      if det[DET_SITE_COL] == siteid:
        det_phase[true_phaseid] = 1

    dist = earthmodel.Delta(event[EV_LON_COL], event[EV_LAT_COL], siteid)
    
    # we assume that only time-defining phases are detected
    for pnum in range(numtimedefphases):
      arrtime = earthmodel.ArrivalTime(event[EV_LON_COL], event[EV_LAT_COL],
                                       event[EV_DEPTH_COL],
                                       event[EV_TIME_COL], pnum, siteid)
      # check if the site is in the shadow zone of this phase
      if arrtime < 0:
        continue
      
      # check if the site was up at the expected arrival time
      if arrtime < start_time or arrtime >= end_time \
         or not site_up[siteid, int((arrtime - start_time) / UPTIME_QUANT)]:
        continue
      
      output.append(det_phase[pnum])
      mag_feat.append(event[EV_MB_COL])
      dep_feat.append(event[EV_DEPTH_COL])
      dist_feat.append(dist)
      # construct the features, one per phase
      for i in range(numtimedefphases):
        phaseid_feat[i].append(int(i == pnum))
    
  print "%d event-phases detected out of %d" % (sum(output), len(output))
  
  # copy the original dataset
  mag_feat2 = [x for x in mag_feat]
  dep_feat2 = [x for x in dep_feat]
  dist_feat2 = [x for x in dist_feat]
  phaseid_feat2 = [[x for x in y] for y in phaseid_feat]
  output2 = [x for x in output]

  
  print "y",len(output2),sum(output2)
  
 
  return mag_feat2,dep_feat2,dist_feat2,phaseid_feat2,output2

def learn(param_dirname,earthmodel, start_time, end_time,
          detections, leb_events, leb_evlist,
          site_up, sites, phasenames, phasetimedef):

  param_fname = "utils/EventDetectionTEMP.txt"
  param_fout = param_dirname + "/EventDetectionPrior.txt"
  param_rfile = "utils/empBayesLogistic_4py.R"

  fp = open(param_fname, "w")

  # assume that the time-defining phases precede the non-time-defining ones
  numtimedefphases = earthmodel.NumTimeDefPhases()
 
  print >>fp,"site det mag dep dist",
  for item in phasenames[:-4]:
      print >>fp,item,
  print >>fp,"\r"
  
  for siteid in range(len(sites)):
    a,b,c,d,e = learn_site(earthmodel, start_time, end_time,
                       detections, leb_events, leb_evlist, site_up,
                       numtimedefphases, siteid)

    for i in range(len(a)):
        print >>fp,siteid,e[i],a[i],b[i],c[i],
        for j in range(len(d)):
            print >>fp,d[j][i],
        print >>fp,'\r'
    print "learnt site id", siteid

    del a,b,c,d,e
    
  fp.close()

  #use R script to read in file and build phase-site specific logistic regression
 ### R CMD BATCH empBayesLogistic_4py.R param_fname param_fout
  
  robjects.r("source('" + param_rfile + "')")
  robjects.r.FitLogistic(param_fname,outfile=param_fout,
                         pwd = os.path.curdir, ossep=os.path.sep);
  os.system("rm " + param_fname)


