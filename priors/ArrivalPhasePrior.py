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

from database.dataset import DET_PHASE_COL

def learn(filename, options, earthmodel, detections, leb_events, leb_evlist,
          false_dets):
  numtimedefphases = earthmodel.NumTimeDefPhases()
  numphases = earthmodel.NumPhases()

  # add-one smoothing
  ph2ph = np.ones((numtimedefphases, numphases), float)
  falseph = np.ones(numphases, float)

  for evnum, ph_detlist in enumerate(leb_evlist):
    for ph, detnum in ph_detlist:
      ph2ph[ph, int(detections[detnum, DET_PHASE_COL])] += 1.0

  for detnum in false_dets:
    falseph[int(detections[detnum, DET_PHASE_COL])] += 1.0

  # normalize
  for i in range(numtimedefphases):
    ph2ph[i] /= ph2ph[i].sum()

  falseph /= falseph.sum()
    
  fp = open(filename, "w")
  
  print >>fp, numtimedefphases, numphases

  for i in range(numtimedefphases):
    for j in range(numphases):
      print >>fp, ph2ph[i,j],
    print >> fp

  for j in range(numphases):
    print >>fp, falseph[j],
  print >>fp

  fp.close()
  
  if options.verbose:
    print "Phase Emission Probabilities"
    for i in range(numtimedefphases):
      print "phase[%2d]:" % i,
      for j in range(numphases):
        print ph2ph[i,j],
      print
    print "False:",
    for j in range(numphases):
      print falseph[j],
    print
  
    
