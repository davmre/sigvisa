import numpy as np
from scipy.optimize import fmin_ncg, fmin

# Laplace distribution
# laplacian density
def estimate(vals):
  location = np.median(vals)
  scale = np.average(abs(vals - location))
  
  return location, scale

def estimate_laplace_uniform_dist(vals, low, high):
  laplace_vals = [x for x in vals]
  location, scale = estimate(laplace_vals)

  unif_ldens = ldensity_uniform(low, high)

  flips = len(laplace_vals)
  while flips > 0:
    flips = 0
    new_laplace_vals = []
    # is the point better explained by the Laplacian or the uniform distrib?
    for val in laplace_vals:
      if ldensity(location, scale, val) > unif_ldens:
        new_laplace_vals.append(val)
      else:
        flips += 1
    laplace_vals = new_laplace_vals
    location, scale = estimate(laplace_vals)

  prob_laplace = float(len(laplace_vals)) / len(vals)
                  
  return prob_laplace, location, scale

def ldensity(location, scale, val):
  return - np.log(2 * scale) - abs(val-location) / scale

def ldensity_uniform(low, high):
  return - np.log(high - low)

def cdf(location, scale, val):
  return .5 * (1 + np.sign(val-location)*(1 - exp(-abs(val-location)/scale)))

def lprob(location, scale, val1, val2):
  return log(cdf(location, scale, val2)
             - cdf(location, scale, val1))

# alternate version: sample alpha from Unif(0,1)
# if alpha == .5 return location
#   alpha < .5 return location + scale * log(2*alpha)
#   alpha > .5 return location - scale * log(2*(1-alpha))
def sample(location, scale):
  return np.random.laplace(location, scale)


def ldensity_laplace_uniform_dist(prob_laplace, location, scale, low, high,
                                  val):
  """
  A mixture of a Laplace and a uniform distribution
  """
  return np.log((prob_laplace * np.exp(-abs(val-location)/scale) / (2*scale))
             + ((1-prob_laplace) / (high - low)))

def lprob_laplace_uniform_dist(prob_laplace, location, scale, low, high,
                               val1, val2):
  return log((prob_laplace * (cdf(location, scale, val2)
                              - cdf(location, scale, val1)))
             + (1-prob_laplace) * (val2-val1) / (high - low))



def sample_trunc(loc, scale, minval, maxval):
  # note the normalization constant is scale * (lower_norm + upper_norm)
  # below loc the area is scale * lower_norm and above it is scale * upper_norm
  lower_norm = 1 - np.exp(- (loc - minval) / scale)
  upper_norm = 1 - np.exp(- (maxval - loc) / scale)
  
  tot_norm = lower_norm + upper_norm
  
  mid_norm = lower_norm / tot_norm

  tgt = np.random.random()

  if tgt < mid_norm:
    return loc + scale * np.log(1 - lower_norm + tgt * tot_norm)
  elif tgt > mid_norm:
    return loc - scale * np.log(1 + lower_norm - tgt * tot_norm)
  else:
    return loc

def logprob_trunc(loc, scale, minval, maxval, val):
  if loc < minval or loc > maxval:
    return -np.inf
  
  normconst = scale * (2 - np.exp(-(loc - minval)/scale)\
                       - np.exp(-(maxval-loc)/scale))

  return - np.log(normconst) - abs(val-loc) / scale

def estimate_trunc(vals):
  init_location = np.median(vals)
  init_scale = np.average(abs(vals - init_location))
  minval = min(vals)
  maxval = max(vals)
  
  def neg_log_lik(param):
    loc, sc = param
    if loc <= minval or loc >= maxval:
      return np.inf
    normconst = sc * (2 - np.exp(-(loc - minval)/sc) - np.exp(-(maxval-loc)/sc))
    return np.log(normconst)*len(vals)+sum(abs(val - loc)/sc for val in vals)
  

  params = fmin(neg_log_lik, np.array((init_location, init_scale)),
                xtol = .01, ftol=.01, disp = 0)
  
  location, scale = params
  
  return location, scale, minval, maxval


