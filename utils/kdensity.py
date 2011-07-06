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
