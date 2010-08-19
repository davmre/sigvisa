import numpy as np

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
