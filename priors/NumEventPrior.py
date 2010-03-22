import os

def learn(param_fname, start_time, end_time, leb_events):
  rate = float(len(leb_events)) / (end_time - start_time)
  print ("event rate is %f per second or %.1f per hour"
         % (rate, rate * 60 * 60))
  fp = open(os.path.join(param_fname,), "w")
  
  print >>fp, rate
  
  fp.close()

