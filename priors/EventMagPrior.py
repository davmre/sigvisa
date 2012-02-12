import matplotlib.pyplot as plt
import numpy as np

from database.dataset import *

def learn(param_fname, options, leb_events):
  # Gutenberg Richter dictates that the there are 10 times as many events
  # of magnitude >= k than are >= k+1
  mb_rate = np.log(10)

  fp = open(param_fname, "w")
  
  print >>fp, "%f %f" % (MIN_MAGNITUDE, mb_rate)
  
  fp.close()
  
  if options.gui:
    mbs = []
    
    # events with the minimum mb actually value have unknown mb, best to
    # leave them out for estimation
    for mb in leb_events[:, EV_MB_COL]:
      if mb > MIN_MAGNITUDE:
        mbs.append(mb)
    
    mbs = np.array(mbs)
    
    plt.figure(figsize=(8,4.8))
    if not options.type1:    
      plt.title("Event mb")
    plt.xlim(MIN_MAGNITUDE, MAX_MAGNITUDE)
    xpts = np.arange(MIN_MAGNITUDE, MAX_MAGNITUDE, .1)
    plt.hist(mbs, xpts, facecolor="blue", edgecolor="none", normed=True,
             label="data", alpha=0.5)
    plt.plot(xpts, [mb_rate * np.exp(-mb_rate * (x-MIN_MAGNITUDE))
                    for x in xpts], color="blue", label="data")
    plt.legend(loc="upper left")
    if options.type1:
      plt.xlabel(r'$m_{b}$')
    else:
      plt.xlabel('mb')
    plt.ylabel("probability density")
    # save the figure
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "EventMagPrior")
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    
  

