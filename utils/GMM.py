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

def test_gmm():
  seed = np.random.randint(10000)+1
  print "seed", seed
  np.random.seed(seed)

  # generate some data from two distributions
  data = np.hstack([np.random.standard_normal(1000)*.1,
                    np.random.standard_normal(500)*.2 + 0.4])
  probs, means, stds = estimate(2, data)

  print "probs", probs
  print "means", means
  print "stds", stds
  
# estimate a gaussian mixture model, 3 arrays:
# -- array of mixture probabilities
# -- array of means
# -- array of standard deviations
def estimate(n, data, iters=10, tol = 1e-3):
  # convert the data to a numpy array if needed
  if type(data) is not np.ndarray:
    data = np.array(data)
  # TODO: in future handle multivariate data as well
  assert(len(data.shape) == 1)
  means = np.ndarray(n, float)
  stds = np.ndarray(n, float)
  
  # create an index for the cluster parameters and the data
  paridx, datidx = np.mgrid[0:n, 0:len(data)]
  
  # initialize the weights uniformly at random
  wts = np.random.uniform(size=(n, len(data)))
  # compute the probability that each data point belongs to a given cluster
  dataprobs = wts / wts.sum(axis=0)
  # and the probability of each cluster
  clusprobs = dataprobs.sum(axis=1) / dataprobs.shape[1]

  old_wts_sum = 0
  wts_sum = tol
  
  while abs(wts_sum - old_wts_sum) >= tol:
    old_wts_sum = wts_sum
    # compute the cluster parameters
    means = (data[datidx] * dataprobs).sum(axis=1) / dataprobs.sum(axis=1)
    stds = np.sqrt(((data[datidx] - means[paridx])**2 * dataprobs).sum(axis=1)
                   / dataprobs.sum(axis=1))
    # re-compute the weights
    # TODO: handle under-flow by using log weights
    #logwts = np.log(clusprobs[paridx]) - np.log(stds[paridx]) \
    #         - 0.5 * ((data[datidx] - means[paridx])**2 \
    #                  / stds[paridx] ** 2)
    wts = (clusprobs[paridx] / stds[paridx] ) \
          * np.exp(- 0.5 * ((data[datidx] - means[paridx])**2 \
                            / stds[paridx] ** 2))
    wts_sum = wts.sum()
    
    # compute the probability that each data point belongs to a given cluster
    dataprobs = wts / wts.sum(axis=0)
    # and the probability of each cluster
    clusprobs = dataprobs.sum(axis=1) / dataprobs.shape[1]
                                       
  return clusprobs, means, stds

def gaussian(m, s, val):
  return np.exp(- float(val - m) ** 2 / (2.0 * float(s) ** 2)) \
         / np.sqrt(2.0 * np.pi * float(s) ** 2)

def evaluate(wts, means, stds, val):
  return sum(wts[j] * gaussian(means[j], stds[j], val)
             for j in xrange(len(wts)))

def sample(wts, means, stds):
  idx = np.where(np.random.multinomial(1, wts) == 1)[0][0]
  return means[idx] + stds[idx] * np.random.normal()

def _test():
  np.seterr(divide = 'raise')
  test_gmm()

if __name__ == "__main__":
  try:
    _test()
  except SystemExit:
    raise
  except:
    import pdb, traceback, sys
    traceback.print_exc(file=sys.stdout)
    pdb.post_mortem(sys.exc_traceback)
    raise
