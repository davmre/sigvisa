# Kolmogorov-Smirnov goodness of fit test

import numpy as np

def kstest(values, cdf):
  values = [v for v in values]          # copy the values
  values.sort()                         # .. before sorting them
  
  #import pdb
  #pdb.set_trace()

  stat = -np.inf
  for c,v in enumerate(values):
    tp = cdf(v)
    ep = (c+1.) / len(values)
    
    diff = abs(tp - ep)
    
    if diff > stat:
      stat = diff
  
  stat *= np.sqrt(len(values))
  
  # 5% level
  return stat <= 1.358, stat

