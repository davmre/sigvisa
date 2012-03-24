import math, random
import numpy as np
from scipy.optimize import fmin_ncg
import time

def logistic(a): return 1.0 / (1.0 + np.exp(-a))

def test_logistic():
  NUM_DATA=10000
  random.seed(10)
  def generate(a,b,c):
    return int(random.random() < logistic(2*a - 3*b + 5*c - 2))
  
  x = [random.random() for i in range(NUM_DATA)]
  w = [random.random() for i in range(NUM_DATA)]
  z = [random.random() for i in range(NUM_DATA)]

  y = [generate(a,b,c) for a,b,c in zip(x,w,z)]

  t1 = time.time()
  model = LogisticModel("y", ["x", "w", "z"], [x, w, z], y)
  t2 = time.time()

  ypred = [model[a,b,c] for a,b,c in zip(x,w,z)]

  err = math.sqrt(float(sum((a-b) ** 2 for a, b in zip(y, ypred)))/ len(y))

  x2 = [random.random() for i in range(NUM_DATA)]
  w2 = [random.random() for i in range(NUM_DATA)]
  z2 = [random.random() for i in range(NUM_DATA)]

  y2 = [generate(a,b,c) for a,b,c in zip(x2,w2,z2)]

  ypred2 = [model[a,b,c] for a,b,c in zip(x2,w2,z2)]

  err2 = math.sqrt(float(sum((a-b) ** 2 for a, b in zip(y2, ypred2)))/ len(y2))

  print "logistic in %.2f secs" % (t2-t1)
  print "coeffs", model.coeffs
  print "rmse on training data", err
  print "rmse on test data", err2
  assert (err < .5)
  
class LogisticModel:
  def __init__(self, name, dim_names, dim_vals, samples, weights=None,
               alpha = 0, prior_means = None, prior_precisions = None):
    """
    Logistic Regression Model. Learns to predict a probability based
    on specified inputs
    
    Note: samples must be an array of  0 and 1 values

    alpha/2 is the L2 regularization term
    Note: the intercept coefficient is not regularized
    """
    assert(len(dim_names) == len(dim_vals))
    self.dim_names = dim_names

    dim_vals = [x for x in dim_vals]              # make a copy
    dim_vals.append(np.ones_like(dim_vals[0])) # add intercept

    if weights is None:
      weights = np.ones_like(dim_vals[0])
    else:
      weights = np.array(weights)
    
    features = np.column_stack(dim_vals)
    output = 2 * np.array(samples) - 1  # convert 0 to -1 and 1 to 1

    feat_outer = np.array([np.outer(x,x) for x in features])

    if prior_means is None or prior_precisions is None:
      prior_means = np.zeros(len(dim_vals))
      prior_precisions = alpha * np.ones(len(dim_vals))
    else:
      prior_means = np.array(prior_means)
      prior_precisions = np.array(prior_precisions)

    def log_logistic(x):
      return -np.log(1 + np.exp(-x))
    
    def neg_log_lik(coeffs):
      return - (log_logistic(output * (features * coeffs).sum(axis=1))
                * weights).sum()\
                + (prior_precisions * (coeffs - prior_means) ** 2).sum() / 2

    def grad_neg_log_lik(coeffs):
      return - ((output * logistic( -output * (features * coeffs).sum(axis=1))
                 * weights) * features.T).sum(axis=1)\
                 + prior_precisions * (coeffs - prior_means)
    
    def hess_neg_log_lik(coeffs):
      sgn_log_odds = output * (features * coeffs).sum(axis=1)
      return ((logistic(-sgn_log_odds) * logistic(sgn_log_odds) * weights)
              * feat_outer.T).sum(axis=2)\
              + prior_precisions * np.eye(len(coeffs))
    
    self.coeffs, fopt, fcalls, gcalls, hcalls, warnflag \
                 = fmin_ncg(neg_log_lik, np.zeros(features.shape[1]),
                            fprime = grad_neg_log_lik, fhess=hess_neg_log_lik,
                            disp = 0, full_output=1, avextol=1e-12)
    if warnflag:
      self.converged = False
      print "LogisticModel: Warning(%d): regression did not converge" % warnflag
      print "coeffs:", self.coeffs
      print "fopt %.1f fcalls=%d gcalls=%d hcalls=%d" % (fopt, fcalls, gcalls,
                                                         hcalls)
    else:
      self.converged = True
    
  def __init2__(self, name, dim_names, dim_vals, samples, weights=None):
    """
    Logistic Regression Model. Learns to predict a probability based
    on specified inputs
    
    Note: samples must be an array of values between 0 and 1
    """
    assert(len(dim_names) == len(dim_vals))
    import rpy2.robjects as robjects

    if weights is None:
      weights = [1 for i in range(len(samples))]
      
    self.name, self.dim_names = name, dim_names

    num_dims = len(dim_names)
    param_names = list ("x"+str(i) for i in range(num_dims))

    fmla = robjects.RFormula("y ~ " + "+".join(param_names))
    fmla.environment["y"] = robjects.BoolVector(samples)
    for i in range(num_dims):
      fmla.environment[param_names[i]] = robjects.FloatVector(dim_vals[i])
      
    fit = robjects.r.glm(formula=fmla, weights = robjects.FloatVector(weights),
                         family = robjects.r.binomial(link="logit"))

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
    
    return 1.0 / (1.0 + np.exp(-val))

def _test():
  test_logistic()



if __name__ == "__main__":
  _test()
