import matplotlib.pyplot as plt
import numpy as np

from database.dataset import *

def learn(param_fname, options, leb_events):

  mbs = []

  # events with the minimum mb actually value have unknown mb, best to
  # leave them out for estimation
  for mb in leb_events[:, EV_MB_COL]:
    if mb > MIN_MAGNITUDE:
      mbs.append(mb)
  
  mbs = np.array(mbs)
  
  # MLE of an exponential distribution is the inverse of the mean
  rate = len(mbs) / (mbs - MIN_MAGNITUDE).sum()
  
  print "magnitude rate is %f" % rate
  
  STEP=.1
  bins = np.arange(MIN_MAGNITUDE, MAX_MAGNITUDE, STEP)
  counts = np.zeros(len(bins), float)
  for m in mbs:
    counts[int((m - MIN_MAGNITUDE) // STEP)] += 1
  peak_mb = bins[np.argmax(counts)]
  print "Peak mb", peak_mb
  
  mbs_trunc = []
  for mb in mbs:
    if mb > peak_mb:
      mbs_trunc.append(mb)
  
  mbs_trunc = np.array(mbs_trunc)
  
  # MLE of an exponential distribution is the inverse of the mean
  rate_trunc = len(mbs_trunc) / (mbs_trunc - peak_mb).sum()
  
  # the probability of the first truncated bucket
  bins_trunc = np.arange(peak_mb, MAX_MAGNITUDE, STEP)
  counts_trunc = np.zeros(len(bins_trunc), float)
  for m in mbs_trunc:
    counts_trunc[int((m - peak_mb) // STEP)] += 1

  print "Truncated rate", rate_trunc
  
  if options.gui:
    plt.figure()
    plt.title("Event mb")
    plt.xlim(MIN_MAGNITUDE, MAX_MAGNITUDE)
    plt.bar(bins, counts / counts.sum(), STEP, color="blue", label="data")
    plt.plot(bins, [rate * np.exp(- rate * (m-MIN_MAGNITUDE)) * STEP
                    for m in bins],
             "black", label="MLE", linewidth=3, linestyle=":")
    plt.legend()

    plt.figure()
    plt.title("Event mb")
    plt.xlim(MIN_MAGNITUDE, MAX_MAGNITUDE)
    plt.bar(bins_trunc, counts_trunc / counts_trunc.sum(), STEP,
            color="blue", label="data trunc")
    plt.plot(bins_trunc, [rate_trunc * np.exp(- rate_trunc * (m-peak_mb))
                          * STEP for m in bins_trunc],
             "black", label="MLE trunc", linewidth=3, linestyle="-")
    plt.legend()
  
  fp = open(param_fname, "w")
  
  print >>fp, "%f %f" % (MIN_MAGNITUDE, rate_trunc)
  
  fp.close()

