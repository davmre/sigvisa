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
# kernel density
import numpy as np

def kdensity_estimate_bw(min, max, step, vals, densfn_bw, bwrange):
  """
  Returns the density at each of the points min through max (not including)
  in steps of step plus the optimal bandwidth
  """
  # hold out 10% of the data to tune the bandwidth
  perm = np.random.permutation(len(vals))
  valid = vals[perm[:len(vals)/10]]
  train = vals[perm[len(vals)/10:]]
  
  best_loglike, best_bw = None, None
  for bw in bwrange:
    dens = kdensity_estimate(min, max, step, train, lambda x: densfn_bw(bw,x))
    loglike = sum(np.log(kdensity_pdf(min, max, step, dens, x)) for x in valid)

    if best_loglike is None or loglike > best_loglike:
      best_loglike, best_bw = loglike, bw

  return kdensity_estimate(min, max, step, vals,
                           lambda x : densfn_bw(best_bw, x)), best_bw

def kdensity_estimate(min, max, step, vals, densfn):
  """
  Returns the density at each of the points min through max (not including)
  in steps of step
  """
  pts = np.arange(min, max, step)
  dens = np.zeros(len(pts))

  # for each point spread its density to all the points
  for v in vals:
    dens2 = densfn(pts-v)      # compute the density at all the points
    dens2 /= step * dens2.sum()         # normalize density
    dens += dens2
  
  return dens / len(vals)

def kdensity_pdf(min, max, step, dens, val):
  assert(val >= min and val < max)
  idx = int((val - min) // step)
  return dens[idx]

def kdensity_cdf(min, max, step, dens, val):
  idx = int((val - min) // step)
  return dens[:idx+1].sum() * step
