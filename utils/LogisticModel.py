import math, random
import rpy2.robjects as robjects
import numpy as np

def test_logistic():
  def logistic(a): return 1.0 / (1.0 + math.exp(-a))
  
  random.seed(10)
  x = [random.random() for i in range(10000)]
  w = [random.random() for i in range(10000)]

  y = [int(random.random() < logistic(2*a - 3*b)) for a,b in zip(x,w)]

  model = LogisticModel("y", ["x", "w"], [x, w], y)

  ypred = [model[a,b] for a,b in zip(x,w)]

  err = math.sqrt(float(sum((y1-y2) ** 2 for y1, y2 in zip(y, ypred)))/ len(y))

  print "logistic"
  print "coeffs", model.coeffs
  print "rmse", err
  assert(err < .45)
  
class LogisticModel:
  def __init__(self, name, dim_names, dim_vals, samples, weights=None):
    """
    Logistic Regression Model. Learns to predict a probability based
    on specified inputs
    
    Note: samples must be an array of values between 0 and 1
    """
    assert(len(dim_names) == len(dim_vals))

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
