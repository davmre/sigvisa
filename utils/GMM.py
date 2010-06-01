import numpy as np
# estimate a gaussian mixture model, 3 arrays:
# -- array of mixture weights (probabilities)
# -- array of means
# -- array of standard deviations
def estimate(n, vals, iters = 10):
  # convert values to a numpy array
  vals = np.array(vals)
  # indicator variable for each value
  ind = np.random.randint(0, n, len(vals))
  # initialize wts, means and std.
  wts = np.ndarray(n, float)
  means = np.ndarray(n, float)
  stds = np.ndarray(n, float)
  
  for i in range(iters):
    for j in range(n):
      means[j] = np.average(vals[ind == j])
      stds[j] = np.std(vals[ind == j])

    for k in range(len(vals)):
      ind[k] = np.argmin((vals[k] - means) ** 2 / stds)

    # if none of the points are assigned to a cluster then randomly
    # select a point for that cluster
    for j in range(n):
      if sum(ind == j) == 0:
        ind[np.random.randint(len(ind))] = j
        
  for j in range(n):
    wts[j] = float(sum(ind == j)) / len(ind)

  return wts, means, stds

def gaussian(m, s, val):
  return np.exp(- float(val - m) ** 2 / (2.0 * float(s) ** 2)) \
         / np.sqrt(2.0 * np.pi * float(s) ** 2)

def evaluate(wts, means, stds, val):
  return sum(wts[j] * gaussian(means[j], stds[j], val)
             for j in xrange(len(wts)))

def sample(wts, means, stds):
  idx = np.where(np.random.multinomial(1, wts) == 1)[0][0]
  return means[idx] + stds[idx] * np.random.normal()

