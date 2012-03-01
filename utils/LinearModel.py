import math, random
import numpy as np
from scipy.optimize import fmin_ncg

def test_linear():
  random.seed(10)
  x = [random.random() for i in range(10000)]
  w = [random.random() for i in range(10000)]

  y = [random.normalvariate(2*a - 3*b + 5, .1) for a,b in zip(x,w)]

  model = LinearModel("y", ["x", "w"], [x, w], y)

  ypred = [model[a,b] for a,b in zip(x,w)]

  print "linear"
  print "coeffs", model.coeffs
  
  err = np.sqrt(float(sum((y1-y2) ** 2 for y1, y2 in zip(y, ypred))) / len(y))

  print "RMSE", err
  assert(err < .11)

class LinearModel:
  def __init__(self, name, dim_names, dim_vals, samples):
    """
    Learns a linear function of the input dimensions
    """
    assert(len(dim_names) == len(dim_vals))
    self.name, self.dim_names = name, dim_names

    datalen = len(dim_vals[0])
    
    dim_vals = [d for d in dim_vals]              # make a copy
    dim_vals.append(np.ones(datalen))             # add intercept
    
    data = np.vstack([np.array(d) for d in dim_vals]).T
    
    self.coeffs, residues, rank, sing  = np.linalg.lstsq(data, samples, 1e-6)
    self.converged = True
    #print "Rank:", rank
    #print "Singular values:", sing
    if rank < len(dim_vals):
      print "Warning: LinearModel: predictors are not independent"
      
      
  def __init2__(self, name, dim_names, dim_vals, samples):
    """
    Learns a linear function of the input dimensions
    """
    import rpy2.robjects as robjects
    self.name, self.dim_names = name, dim_names
    
    num_dims = len(dim_names)
    param_names = list ("x"+str(i) for i in range(num_dims))
    
    fmla = robjects.RFormula("y ~ " + "+".join(param_names))
    fmla.environment["y"] = robjects.FloatVector(samples)
    for i in range(num_dims):
      fmla.environment[param_names[i]] = robjects.FloatVector(dim_vals[i])
      
    fit = robjects.r.glm(formula=fmla, family = robjects.r.gaussian())
    
    coeffs = fit.subset("coefficients")[0]
    
    self.coeffs = [coeffs[1+i] for i in range(num_dims)]
    self.coeffs.append(coeffs[0])
    
    self.converged = fit.subset("converged")[0][0]
    
  def __getitem__(self, index):
    if type(index) is not tuple:
      index = (index,)
    
    if len(index) != len(self.dim_names):
      raise ValueError("%d dimensions expected got %d" % (len(self.dim_names),
                                                          len(index)))

    val = self.coeffs[-1] + sum(self.coeffs[i] * idx \
                                for i,idx in enumerate(index))
    
    return val


def linearreg(data, samples, init_coeffs = None,
              noise_prec=1.0, coeff_mean=None, coeff_prec=None):
  """
  data is an n x d array
  samples is a length n array

  noise_prec -> precision of gaussian noise around linear model (float)
  coeff_mean -> prior mean of the coefficients (length d array)
  coeff_prec -> prior precision of coefficients (length d array)
  
  returns a length d array of coefficients
  """
  numfeats = data.shape[1]
  numsamples = data.shape[0]
  assert(numsamples == len(samples))

  if init_coeffs is None:
    init_coeffs = np.zeros(numfeats)
  
  if coeff_mean is None:
    coeff_mean = np.zeros(numfeats)
  
  if coeff_prec is None:
    coeff_prec = np.zeros(numfeats)

  # the gradient of the negative log likelihood
  def grad_neg_log_lik(coeffs):
    res = np.dot(data, coeffs) - samples
    return noise_prec * np.dot(res, data) + coeff_prec * (coeffs - coeff_mean)

  # the Hessian of the negative log likelihood
  hess = noise_prec * sum(np.outer(data[i], data[i])
                              for i in xrange(numsamples))\
       + coeff_prec * np.eye(numfeats)
  
  # since this is a second order function we can directly compute its minima
  coeffs = init_coeffs - np.dot(np.linalg.inv(hess),
                                grad_neg_log_lik(init_coeffs))
  
  return coeffs

# site_data: [([data x_k^i i=1..n_k], [samples y_k^i i=1..n_k]) k=1..K sites]
#
# where x_k is a vector of length F
#
# Returns [coeffs c_k k=1..K]   where c_k is a vector of length F
#
# y_k^i        ~ Normal(x_k^i . c_k, sigma_k)
# c_k^f        ~ Normal(mean^f, sigma^f)
# mean^f       ~ Normal(mean0, sigma0)
# sigma^f^{-2} ~ Gamma(alpha0, beta0)
# sigma_k^{-2} ~ Gamma(alpha1, beta)
# beta^{-1}    ~ Gamma(alpha2, beta2)
#
# We use the following update rules
# c_k = argmin_{c_k} .5 * (1/sigma_k)^2 * \sum_i (y_k^i - x_k^i . c_k)^2
#                  + .5 * \sum_f (1/sigma^f)^2 * (c_k^f - mean^f)^2
#        --> computed in linearreg above
# mean^f  = ((1/sigma^f)^2 * \sum_k c_k + (1/sigma0)^2 * mean0)
#            / ( (1/sigma^f)^2 *K + (1/sigma0)^2 )
# sigma^f = sqrt[(\sum_k (c_k^f-mean^f)^2 + 2/beta0) / (K + 2*alpha0 -2) ]
# sigma_k = sqrt[(\sum_i=1..n_k (y_k^i - x_k^i.c_k)^2 + 2/beta)
#                / (n_k + 2 * alpha1 - 2)]
# beta = \frac{\sum_{k=1..K} (1/sigma_k)^2 + 1/beta2}{K alpha1 + alpha2 - 1}
#
# We iterate till mean, sigma, and beta and converged till the given tolerance
def hier_linearreg(site_data, init_mean=None, init_sigma=None, init_beta = 1,
                   mean0=0., sigma0=100., alpha0=0.01, beta0=100,
                   alpha1=100., alpha2=0.01, beta2=100, tolerance=.01,
                   maxiters=100):
  numsites = len(site_data)
  assert numsites > 1                    # there should be more than one site
  
  for (data, samples) in site_data:
    if len(data):
      numfeats = data.shape[1]
      break
  else:
    raise ValueError("No data for regression")
  
  if init_mean is None:
    init_mean = np.zeros(numfeats)
  
  if init_sigma is None:
    init_prec = np.ones(numfeats)
  else:
    init_prec = (1./init_sigma)**2

  mean, prec, beta = init_mean, init_prec, init_beta
  site_coeffs = np.ndarray((numsites, numfeats))
  site_coeffs[:] = init_mean
  site_prec = np.ndarray(numsites)
  if alpha1 > 1:
    site_prec[:] = (alpha1-1) * beta
  else:
    site_prec[:] = beta
  
  for iternum in xrange(maxiters):
    prev_mean, prev_prec, prev_beta = mean, prec, beta
    
    # update coefficients and precision for each site
    for siteid, (data, samples) in enumerate(site_data):
      if not len(data):
        this_coeffs = mean
        if alpha1 > 1:
          this_prec = beta * (alpha1-1)
        else:
          # we pick the mean since the mode is undefined for alpha1<=1
          this_prec = beta * alpha1
      else:
        this_coeffs = linearreg(data, samples,
                                init_coeffs = site_coeffs[siteid],
                                noise_prec = site_prec[siteid],
                                coeff_mean = mean, coeff_prec = prec)
        this_prec = (len(data) + 2 *alpha1 - 2) \
                    / (((samples - np.dot(data, this_coeffs))**2).sum()
                              + (2./beta))
      
      site_coeffs[siteid] = this_coeffs
      site_prec[siteid] = this_prec
    
    # update global mean, prec, and beta
    mean = (site_coeffs.sum(axis=0) * prec + mean0 * (1./sigma0)**2) \
           / (numsites * prec + (1./sigma0)**2)
    
    prec = (numsites + 2*alpha0 - 2 ) \
           / ( ((site_coeffs - mean)**2).sum(axis=0)  + (2./beta0))
    
    beta = ( site_prec.sum() + (1./beta2)) / ( numsites * alpha1 + alpha2 - 1.)
    
    if abs(prev_mean - mean).max() < tolerance \
           and abs(prev_prec - prec).max() < tolerance\
           and abs(prev_beta - beta) < tolerance:
      break

  return site_coeffs, np.sqrt(1./site_prec), mean, np.sqrt(1/prec), beta



def _test():
  test_linear()

if __name__ == "__main__":
  _test()
