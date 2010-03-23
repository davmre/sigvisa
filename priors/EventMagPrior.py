from database.dataset import *

def learn(param_fname, leb_events):
  # MLE of an exponential distribution is the inverse of the mean
  rate = len(leb_events) / (leb_events[:, EV_MB_COL] - MIN_MAGNITUDE).sum()
  
  print "magnitude rate is %f" % rate
  
  fp = open(param_fname, "w")
  
  print >>fp, "%f %f" % (MIN_MAGNITUDE, rate)
  
  fp.close()

