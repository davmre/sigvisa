import matplotlib.pyplot as plt

from obspy.core import Trace, Stream, UTCDateTime

from database.dataset import *
import database.db
import learn, netvisa
from utils.geog import dist_deg, azimuth

def plot_det_times(trc, all_det_times, all_det_labels):
  if trc == None or all_det_times == None:
    return

  if all_det_times is not None:
    maxtrc, mintrc = float(max(trc.data)), float(min(trc.data))
    plt.bar(left=all_det_times, height=[maxtrc-mintrc for _ in all_det_times],
            width=.25, bottom=mintrc, color="red", linewidth=0, alpha=.5)
    if all_det_labels is not None:
      for (t, lbl) in zip(all_det_times, all_det_labels):
        plt.text(t+3, maxtrc - (maxtrc-mintrc)*0.1, lbl, color="red", fontsize=4)

  

# does not save for you - you need to call savefig() yourself!
def plot_segment(chan_dict, title=None, all_det_times=None, all_det_labels=None, band="broadband_envelope", format = "k-"):
  plt.figure()
  plt.xlabel("Time (s)")

  for chidx, chan in enumerate(sorted(chan_dict.keys())):
    if chidx == 0:
      axes = plt.subplot(len(chan_dict), 1, 1)
      if title is not None:
        plt.title(title)
    else:
      plt.subplot(len(chan_dict), 1, chidx+1, sharex=axes)
      
    plt.ylabel(chan)

    trc = chan_dict[chan][band]
    srate = trc.stats.sampling_rate
    npts = trc.stats.npts
    stime = trc.stats["starttime_unix"]
    timevals = np.arange(stime, stime + npts/srate, 1.0 /srate)[0:npts]

    plt.plot(timevals, trc, format)
    plot_det_times(trc, all_det_times, all_det_labels)

def plot_trace(trc, title=None, all_det_times=None, all_det_labels=None, format="k-"):
  plt.figure()
  plt.xlabel("Time (s)")

  if title is not None:
    plt.title(title)
      
  chan_name = trc.stats["channel"]

  plt.ylabel(chan_name)

  srate = trc.stats.sampling_rate
  npts = trc.stats.npts
  stime = trc.stats["starttime_unix"]
  timevals = np.arange(stime, stime + npts/srate, 1.0 /srate)[0:npts]

  plt.plot(timevals, trc, format)
  plot_det_times(trc, all_det_times, all_det_labels)


def plot_traces(traces, title=None, all_det_times=None, all_det_labels=None, formats=None, linewidths=None):

  if formats is None:
    if len(traces) == 1:
      formats = ["k-",]
    elif len(traces) == 2:
      formats = ["k-", "r-"]
    else:
      formats = ["k-" for t in traces]

  if linewidths is None:
      linewidths = [1 for t in traces]


  plt.figure()
  plt.xlabel("Time (s)")

  if title is not None:
    plt.title(title, fontsize=12)
      
  chan_name = traces[0].stats["channel"]

  plt.ylabel(chan_name)

  for (i,trc) in enumerate(traces):
    srate = trc.stats.sampling_rate
    npts = len(trc.data)
    stime = trc.stats["starttime_unix"]
    timevals = np.arange(stime, stime + npts/srate, 1.0 /srate)[0:npts]

    plt.plot(timevals, trc, formats[i], linewidth = linewidths[i])

  plot_det_times(traces[0], all_det_times, all_det_labels)


# does not save for you - you need to call savefig() yourself!
def plot_bands(bands_dict, title=None, all_det_times=None, all_det_labels=None):
  format = "k-"
    
  plt.figure(figsize=(12, 30))
  plt.xlabel("Time (s)")

  for (bidx, band) in enumerate(sorted(bands_dict.keys())):
    if bidx == 0:
      axes = plt.subplot(len(bands_dict), 1, 1)
      if title is not None:
        plt.title(title)
    else:
      plt.subplot(len(bands_dict), 1, bidx+1, sharex=axes)
      
    if band.startswith("narrow_logenvelope_"):
        yl = band[19:]
    else:
        yl = band
    plt.ylabel(yl)

    trc = bands_dict[band]
    npts = trc.stats["npts"]
    srate = trc.stats["sampling_rate"]
    stime = trc.stats["starttime_unix"]
    timevals = np.arange(stime, stime + npts/srate, 1.0 /srate)[0:npts]

    
    for (i, n) in enumerate(trc.data):
        if np.isnan(n) or not np.isfinite(n):
            trc.data[i] = 0

#    if np.isnan(s) or not np.isfinite(s):
#        print "invalid data in band %s, not plotting..." % (band)
#        continue

    plt.plot(timevals, trc, format)

    plot_det_times(trc, all_det_times, all_det_labels)

