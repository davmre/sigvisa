import matplotlib.pyplot as plt

from obspy.core import Trace, Stream, UTCDateTime

from database.dataset import *
import database.db
import learn
from utils.geog import dist_deg, azimuth

def plot_det_times(wave, axes=None, logscale=False):
  if wave == None
    return

  if axes=None:
    axes = plt.subplot(1, 1, 1)

  arrivals = wave['arrivals']
  all_det_times = arrivals[:, DET_TIME_COL]
  all_det_labels = Sigvisa().phasenames(arrivals[:, DET_PHASE_COL])


  if all_det_times is not None:
    maxwave, minwave = float(max(wave.data)), float(min(wave.data))
    if logscale:
      (maxwave, minwave) = (np.log(maxwave), np.log(minwave))
    axes.bar(left=all_det_times, height=[maxwave-minwave for _ in all_det_times],
            width=.25, bottom=minwave, color="red", linewidth=0, alpha=.5)
    if all_det_labels is not None:
      for (t, lbl) in zip(all_det_times, all_det_labels):
        axes.text(t+3, maxwave - (maxwave-minwave)*0.1, lbl, color="red", fontsize=4)



# does not save for you - you need to call savefig() yourself!
def plot_segment(segment, title=None, format = "k-", chans=None, logscale=False):
  plt.figure()
  plt.xlabel("Time (s)")

  chans = segment.chans.keys() if chans is None else chans
  for chidx, chan in enumerate(sorted(chans)):
    if chidx == 0:
      axes = plt.subplot(len(chan_dict), 1, 1)
      if title is not None:
        plt.title(title)
    else:
      plt.subplot(len(chan_dict), 1, chidx+1, sharex=axes)

    plt.ylabel(chan)

    wave = segment[chan]
    srate = wave['srate']
    npts = wave['npts']
    stime = wave['stime']
    timevals = np.arange(stime, stime + npts/srate, 1.0 /srate)[0:npts]

    wave_data = np.log(wave.data) if logscale else wave.data

    plt.plot(timevals, wave_data, format)
    plot_det_times(wave, logscale=logscale)

def plot_waveform(wave, title=None,  format="k-"):
  fig = plt.figure()
  plt.xlabel("Time (s)")

  if title is not None:
    plt.title(title)

  chan_name = waves["chan"]

  plt.ylabel(chan_name)

  srate = wave['srate']
  npts = wave['npts']
  stime = wave['stime']
  timevals = np.arange(stime, stime + npts/srate, 1.0 /srate)[0:npts]

  plt.plot(timevals, wave.data, format)
  plot_det_times(wave)
  return fig

def plot_waveforms(waves, title=None, formats=None, linewidths=None):
  plt.figure()
  if title is not None:
    plt.title(title, fontsize=12)

  plot_waves_subplot(plt.subplot(1,1,1), waves, formats, linewidths)

def plot_waves_subplot(axes, waves, formats=None, linewidths=None, logscale=False):

  if formats is None:
    if len(traces) == 1:
      formats = ["k-",]
    elif len(traces) == 2:
      formats = ["k-", "r-"]
    else:
      formats = ["k-" for t in traces]

  if linewidths is None:
      linewidths = [1 for t in traces]

  axes.set_xlabel("Time (s)")

  chan_name = traces[0].stats["channel"]

  axes.set_ylabel(chan_name)

  for (i,wave) in enumerate(waves):
    srate = wave['srate']
    npts = wave['npts']
    stime = wave['stime']
    timevals = np.arange(stime, stime + npts/srate, 1.0 /srate)[0:npts]
    tdata = np.log(wave.data) if logscale else wave.data

    axes.plot(timevals, tdata, formats[i], linewidth = linewidths[i])

  plot_det_times(waves[0], axes)


# does not save for you - you need to call savefig() yourself!
def plot_bands(wave, bands=None, title=None):
  format = "k-"

  if bands is None:
    bands=Sigvisa().bands

  plt.figure(figsize=(12, 30))
  plt.xlabel("Time (s)")

  for (bidx, band) in enumerate(sorted(bands)):
    if bidx == 0:
      axes = plt.subplot(len(bands), 1, 1)
      if title is not None:
        plt.title(title)
    else:
      plt.subplot(len(bands), 1, bidx+1, sharex=axes)

    plt.ylabel(yl)

    npts = wave["npts"]
    srate = wave["srate"]
    stime = wave["stime"]
    timevals = np.arange(stime, stime + npts/srate, 1.0 /srate)[0:npts]

    nwave = wave.filter(band)

    plt.plot(timevals, nwave.data, format)

    plot_det_times(wave)

