import matplotlib.pyplot as plt

from obspy.core import Trace, Stream, UTCDateTime

from database.dataset import *
import database.db
import learn, netvisa
from utils.geog import dist_deg, azimuth



# does not save for you - you need to call savefig() yourself!
def plot_segment(chan_dict, title=None, all_det_times=None, band="broadband_envelope", format = "k:"):
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

    if all_det_times is not None:
      maxtrc, mintrc = float(max(trc.data)), float(min(trc.data))
      plt.bar(left=all_det_times, height=[maxtrc-mintrc for _ in all_det_times],
              width=.25, bottom=mintrc, color="red", linewidth=0, alpha=.5)

def plot_trace(trc, title=None, all_det_times=None, format="k:"):
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

  if all_det_times is not None:
    maxtrc, mintrc = float(max(trc.data)), float(min(trc.data))
    plt.bar(left=all_det_times, height=[maxtrc-mintrc for _ in all_det_times],
            width=.25, bottom=mintrc, color="red", linewidth=0, alpha=.5)

def plot_traces(trc1, trc2, title=None, all_det_times=None, format1="k:", format2="r-"):
  plt.figure()
  plt.xlabel("Time (s)")

  if title is not None:
    plt.title(title)
      
  chan_name = trc1.stats["channel"]

  plt.ylabel(chan_name)

  srate = trc1.stats.sampling_rate
  npts = trc1.stats.npts
  stime = trc1.stats["starttime_unix"]
  timevals = np.arange(stime, stime + npts/srate, 1.0 /srate)[0:npts]
  plt.plot(timevals, trc1, format1)

  srate = trc2.stats.sampling_rate
  npts = trc2.stats.npts
  stime = trc2.stats["starttime_unix"]
  timevals = np.arange(stime, stime + npts/srate, 1.0 /srate)[0:npts]
  plt.plot(timevals, trc2, format2)

  if all_det_times is not None:
    maxtrc, mintrc = float(max(trc1.data)), float(min(trc1.data))
    plt.bar(left=all_det_times, height=[maxtrc-mintrc for _ in all_det_times],
            width=.25, bottom=mintrc, color="red", linewidth=0, alpha=.5)


# does not save for you - you need to call savefig() yourself!
def plot_bands(bands_dict, title=None, all_det_times=None):
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

    if all_det_times is not None:
      maxtrc, mintrc = float(max(trc.data)), float(min(trc.data))
      plt.bar(left=all_det_times, height=[maxtrc-mintrc for _ in all_det_times],
              width=.25, bottom=mintrc, color="red", linewidth=0, alpha=.5)

