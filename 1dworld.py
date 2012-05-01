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
# The 1-d seismic model in BLOG:
#
# guaranteed Event ev;
# EventLocation(Event e) ~ Uniform[0,1];
# EventTime(Event e) ~ Uniform[0, 1];
#
# guaranteed Station sta[3];
# StationLocation(sta[1]) = 0;
# StationLocation(sta[2]) = .5;
# StationLocation(sta[3]) = 1;
#
# Distance(Event e, Station s) = abs( EventLocation(e) - StationLocation(s) );
#
# IsDetected(Event e, Station s) ~ Bernoulli( exp (-.1 - 2 * Distance(e,s) ));
#
# #Detection(Source = Event e, Dest = Station s)
#   if IsDetected(e,s) = 1 else = 0;
#
# ArrivalTime(Detection d)
#   ~ Gaussian(EventTime(Source(d)) + Distance(Source(d), Dest(d)), .05);
#
# The 1d model in english:
#
# Exactly one event occurs in the world uniformly at random in [0, 1]
# and at time uniformly at random in [0, 1].
# There are three stations in the world at 0, .5, and 1.
# The distance between the event and the station is the Euclidean distance.
# At event is detected at a station with probability exp(-.1 - 2 * distance).
# The seismic waves travel with a speed of 1 in both directions from the event.
# The time at which the waves arrive at a station has a Gaussian error around
# the expected time with standard deviation .05.
# (NOTE: this model has no false detections)
#
# The data for the model consists of all the detection times, for example:
#
# obs {Detection d} = {d1, d2}
# Dest(d1) = sta[2];
# ArrivalTime(d1) = .75;
# Dest(d2) = sta[3];
# ArrivalTime(d2) = .6;
#
# We are interested in querying the location and time of the event;
#
# query EventLocation(ev);
# query EventTime(ev);
#
# ==================================================
#
# The following Python code generates an event and detections as per the
# above model and plots the posterior location of the event
#
import random, time, optparse, math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 5 stations, equally spaced between 0 and 1
StationLocations = [0., .2, .4, .6, .8, 1.]
ArrivalTimeSD = .005

def generate_world_n_obs():
  # as the name suggests this creates the
  #
  # world:
  #
  # - EventLocation(ev)
  # - EventTime(ev)
  # - an array for each of the three stations
  #   o IsDetected
  #   o ArrivalTime
  #
  # and observation:
  #
  # - an array of ArrivalTimes one per station, the ArrivalTime is None if the
  #   station doesn't have a detection
  evloc = random.random()
  evtime = random.random()
  dets = []
  obs = []
  for snum, staloc in enumerate(StationLocations):
    dist = abs(evloc - staloc)
    detprob = math.exp(-.1 - 2*dist)
    if random.random() < detprob:
      arrtime = random.gauss(evtime + dist, ArrivalTimeSD)
      dets.append((True, arrtime))
      obs.append(arrtime)
    else:
      dets.append((False, None))
      obs.append(None)

  return (evloc, evtime, dets), obs

def event_log_density(evloc, evtime, dets):
  """
  computes the probability density of a hypothesized event and its
  observed arrival times
  """
  world = (evloc, evtime, dets)
  return world_log_density(world)
  
def world_log_density(world):
  """
  computes the log probability density of a hypothesized world
  """
  evloc, evtime, dets = world

  if evloc < 0 or evloc > 1 or evtime < 0 or evtime > 1:
    return -np.inf
  
  # note: event location and time have density 1 and hence log density 0
  logprob = 0.
  for staloc, (isdet, arrtime) in zip(StationLocations, dets):
    dist = abs(evloc - staloc)
    detprob = math.exp(-.1 - 2*dist)

    if isdet:
      logprob += math.log(detprob)
      logprob += gaussian_log_density(val = arrtime, mean = evtime + dist,
                                      sd = ArrivalTimeSD)
    else:
      logprob += math.log(1 - detprob)
      
  return logprob

def compute_median_levels(array, n):
  vals = [x for x in array.flat]
  vals.sort()
  levs = [array.min()]
  for i in range(n):
    if not len(vals):
      break
    lev = vals[len(vals)/2]
    
    levs.extend(np.linspace(levs[-1], lev, 5)[1:])
    
    vals = vals[len(vals)/2+1:]
  return levs

def plot_posterior(true_world, obs):
  dets = []
  for arrtime in obs:
    if arrtime is None:
      dets.append((False, None))
    else:
      dets.append((True, arrtime))

  grad_locs1 = gradient_ascent(lambda x: event_log_density(x[0], x[1], dets),
                               np.array([0.1, 0.1]),
                               lambda x: event_gradient(x, dets))
  
  grad_locs2 = gradient_ascent(lambda x: event_log_density(x[0], x[1], dets),
                               np.array([0.1, .9]),
                               lambda x: event_gradient(x, dets))

  grad_locs3 = gradient_ascent(lambda x: event_log_density(x[0], x[1], dets),
                               np.array([.9, .9]),
                               lambda x: event_gradient(x, dets))

  grad_locs4 = gradient_ascent(lambda x: event_log_density(x[0], x[1], dets),
                               np.array([.9, 0.1]),
                               lambda x: event_gradient(x, dets))

  evloc_arr = np.linspace(0, 1, 1000)
  evtime_arr = np.linspace(0, 1, 1000)
  logprob = np.zeros((len(evloc_arr), len(evtime_arr)))
  
  for loci, evloc in enumerate(evloc_arr):
    for timei, evtime in enumerate(evtime_arr):
      logprob[loci, timei] = event_log_density(evloc, evtime, dets)

  X, Y = np.meshgrid(evloc_arr, evtime_arr)
  
  plt.figure()

  levels = compute_median_levels(logprob, 100)

  plt.contourf(X, Y, logprob.T, levels=levels, cmap=matplotlib.cm.jet)
  
  plt.plot([x for x,y in grad_locs1], [y for x,y in grad_locs1],
           marker="s", ms=4, mfc="none",
           mec="blue", label="Gradient Ascent From (0,0)")

  plt.plot([x for x,y in grad_locs2], [y for x,y in grad_locs2],
           marker="s", ms=4, mfc="none",
           mec="blue", label="Gradient Ascent From (0,1)")
  
  plt.plot([x for x,y in grad_locs3], [y for x,y in grad_locs3],
           marker="s", ms=4, mfc="none",
           mec="blue", label="Gradient Ascent From (1,1)")
  
  plt.plot([x for x,y in grad_locs4], [y for x,y in grad_locs4],
           marker="s", ms=4, mfc="none",
           mec="blue", label="Gradient Ascent From (1,0)")

  plt.plot([true_world[0]], [true_world[1]], marker="*", ms=10, mfc="yellow",
           label="True Location")
  
  
  plt.title("Log posterior density")
  plt.xlim(0,1)
  plt.ylim(0,1)
  plt.xlabel("Location")
  plt.ylabel("Time")

def main():
  parser = optparse.OptionParser()
  parser.add_option("-s", "--seed", dest="seed", default=None,
                    type = "int",
                    help = "random number generator seed")
  (options, args) = parser.parse_args()

  if options.seed is None:
    options.seed = int(time.time())
  random.seed(options.seed)
  print "Random number generator seed = %d" % options.seed

  true_world, obs = generate_world_n_obs()

  print "True World :", true_world
  print "True World log prob. density :", world_log_density(true_world)
  print "Observation :", obs

  plot_posterior(true_world, obs)
  plt.show()

def gaussian_log_density(val, mean, sd):
  return -0.5 * math.log(2 * math.pi) - math.log(sd) \
         - 0.5 * (val - mean)**2 / sd**2

def event_gradient(ev_loc_time, dets, delta=1e-6):
  ev_loc, ev_time = ev_loc_time
  val1 = event_log_density(ev_loc, ev_time, dets)
  val2 = event_log_density(ev_loc + delta, ev_time, dets)
  val3 = event_log_density(ev_loc, ev_time + delta, dets)

  grad = np.array([val2 - val1, val3 - val1]) / delta
  
  return grad
  
def gradient_ascent(obj_fn, init_loc, grad_fn, init_step=1e-6):
  
  curr_loc = init_loc
  curr_val = obj_fn(curr_loc)
  curr_step = init_step

  locs = [curr_loc]
  
  while True:
    grad = grad_fn(curr_loc)
    new_loc = curr_loc + grad * curr_step
    new_val = obj_fn(new_loc)

    if new_val > curr_val:
      curr_loc, curr_val, curr_step = new_loc, new_val, curr_step*2
      locs.append(curr_loc)
      
    elif curr_step > init_step:
      curr_step /= 2.

    else:
      break

  return locs
    
if __name__ == "__main__":
  try:
    main()
  except SystemExit:
    raise
  except:
    import pdb, traceback, sys
    traceback.print_exc(file=sys.stdout)
    pdb.post_mortem(sys.exc_traceback)
    raise
