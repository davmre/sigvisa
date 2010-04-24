import numpy as np

def estimate(vals):
  logvals = np.log(vals)
  
  return np.average(logvals), np.std(logvals)

def sample(mean, sigma):
  return np.random.lognormal(mean, sigma)
