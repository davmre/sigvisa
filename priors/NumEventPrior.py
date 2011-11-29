from database.dataset import *
import numpy as np
import os

def learn(param_fname, options, start_time, end_time, leb_events):
  rate = float(len(leb_events)) / (end_time - start_time)
  print ("event rate is %f per second or %.1f per hour"
         % (rate, rate * 60 * 60))

  if options.gui:
    hrly_rate = rate * 60 * 60
    num_hrs = int((end_time - start_time) / (60 * 60))
    bins = np.arange(0, num_hrs+1)
    options.plt.figure(figsize=(8,4.8))
    if not options.type1:
      options.plt.title("Event Rate per hour")
    options.plt.hist((leb_events[:, EV_TIME_COL] - start_time) / (60*60),
                     bins, label="data", alpha=1.0, edgecolor="none",
                     facecolor="blue")
    options.plt.plot([0, num_hrs],
                     [hrly_rate + np.sqrt(hrly_rate),
                      hrly_rate + np.sqrt(hrly_rate)],
                     label = "model +std", linewidth=3, linestyle=":",
                     color="black")
    options.plt.plot([0, num_hrs], [hrly_rate, hrly_rate],
                     label = "model", linewidth=3, linestyle="-", color="black")
    options.plt.plot([0, num_hrs],
                     [hrly_rate - np.sqrt(hrly_rate),
                      hrly_rate - np.sqrt(hrly_rate)],
                     label = "model -std", linewidth=3,
                     linestyle="-.", color="black")
    options.plt.xlabel("Hour index")
    options.plt.ylabel("Frequency")
    options.plt.legend(loc="upper left")
    options.plt.xlim(0, num_hrs)
    
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "NumEventPrior")
      if options.type1:
        options.plt.savefig(basename+".pdf")
      else:
        options.plt.savefig(basename+".png")
  
  fp = open(param_fname, "w")
  print >>fp, rate
  fp.close()

