import numpy as np

# Laplace distribution
# laplacian density
def estimate(vals):
  location = np.median(vals)
  scale = np.average(abs(vals - location))
  
  return location, scale
  
def ldensity(location, scale, val):
  return - np.log(2 * scale) - abs(val-location) / scale

def cdf(location, scale, val):
  return .5 * (1 + np.sign(val-location)*(1 - exp(-abs(val-location)/scale)))

def lprob(location, scale, val1, val2):
  return log(cdf(location, scale, val2)
             - cdf(location, scale, val1))

def sample(location, scale):
  return np.random.laplace(location, scale)

def ldensity_laplace_uniform_dist(prob_laplace, location, scale, low, high,
                                  val):
  """
  A mixture of a Laplace and a uniform distribution
  """
  return log((prob_laplace * exp(-abs(val-location)/scale) / (2*scale))
             + ((1-prob_laplace) / (high - low)))

def lprob_laplace_uniform_dist(prob_laplace, location, scale, low, high,
                               val1, val2):
  return log((prob_laplace * (cdf(location, scale, val2)
                              - cdf(location, scale, val1)))
             + (1-prob_laplace) * (val2-val1) / (high - low))
