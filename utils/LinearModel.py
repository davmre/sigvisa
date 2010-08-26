import math, random
import rpy2.robjects as robjects
import numpy as np

def test_linear():
  random.seed(10)
  x = [random.random() for i in range(10000)]
  w = [random.random() for i in range(10000)]

  y = [random.normalvariate(2*a - 3*b + 5, .1) for a,b in zip(x,w)]

  model = LinearModel("y", ["x", "w"], [x, w], y)

  ypred = [model[a,b] for a,b in zip(x,w)]

  err = np.sqrt(float(sum((y1-y2) ** 2 for y1, y2 in zip(y, ypred))) / len(y))

  print "linear"
  print "coeffs", model.coeffs
  print "RMSE", err
  assert(err < .11)

class LinearModel:
  def __init__(self, name, dim_names, dim_vals, samples):
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


def _test():
  test_linear()

if __name__ == "__main__":
  _test()
