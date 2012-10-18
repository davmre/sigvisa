import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from obspy.core import Trace, Stream, UTCDateTime

from database.dataset import *
import database.db
import learn
from sigvisa import Sigvisa
from utils.geog import dist_deg, azimuth

from source.event import Event

def plot_det_times(wave, axes=None, logscale=False):
  if wave is None:
    return

  if axes is None:
    axes = plt.subplot(1, 1, 1)

  try:
    arrivals = wave['event_arrivals']
  except:
    arrivals = wave['arrivals']

  if arrivals.shape[0] == 0:
    return

  all_det_times = arrivals[:, DET_TIME_COL]
  all_det_labels = [Sigvisa().phasenames[arrivals[i, DET_PHASE_COL]] for i in range(arrivals.shape[0])]

  if all_det_times is not None:
    maxwave, minwave = float(max(wave.data)), float(min(wave.data))
    if logscale:
      (maxwave, minwave) = (np.log(maxwave), np.log(minwave))
    axes.bar(left=all_det_times, height=[maxwave-minwave for _ in all_det_times],
            width=.25, bottom=minwave, edgecolor="red", color="red", linewidth=1, alpha=.5)
    if all_det_labels is not None:
      for (t, lbl) in zip(all_det_times, all_det_labels):
        axes.text(t+3, maxwave - (maxwave-minwave)*0.1, lbl, color="red", fontsize=10)



# does not save for you - you need to call savefig() yourself!
def plot_segment(segment, title=None, format = "k-", chans=None, logscale=False):
  fig = plt.figure(figsize=(10,8), dpi=250)
  plt.xlabel("Time (s)")

  # if no title is given, try to generate a simple one based on the event being plotted
  if title is None:
    evid=None
    try:
      evid = segment['evid']
      title = "Event %d at %s" % (evid, segment['sta'])
    except KeyError:
      pass

  if title is not None:
    plt.suptitle(title, fontsize=20)

  # loop over channels to plot each channel
  chans = segment.get_chans() if chans is None else chans
  n = len(chans)
  gs = gridspec.GridSpec(n*3+1, 1)
  gs.update(left=0.1, right=0.95, hspace=1)
  for chidx, chan in enumerate(sorted(chans)):
    if chidx == 0:
      axes = plt.subplot(gs[chidx*3:chidx*3+3, 0])
    else:
      axes = plt.subplot(gs[chidx*3:chidx*3+3, 0], sharex=axes)

    plt.ylabel(chan)

    wave = segment[chan]
    srate = wave['srate']
    npts = wave['npts']
    stime = wave['stime']
    timevals = np.arange(stime, stime + npts/srate, 1.0 /srate)[0:npts]

    wave_data = np.log(wave.data) if logscale else wave.data

    plt.plot(timevals, wave_data, format)
    plot_det_times(wave, axes=axes, logscale=logscale)

  axes = plt.subplot(gs[n*3, 0])
  axes.axis('off')
#  axes.get_xaxis().set_visible(False)
#  axes.get_yaxis().set_visible(False)
  descr = "Segment: " + str(segment)
  try:
    evid = segment['evid']
    e = Event(evid)
    descr = descr + "\n\n" + "Event: " + str(e)
  except KeyError as e:
    pass
  axes.text(0.5, 0, descr, fontsize=10, color="black", horizontalalignment='center', verticalalignment='center')

#  fig.canvas.mpl_connect('draw_event', on_draw)



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



















#################################################################
#################################################################

"""
matplotlib text wrapping code from
http://stackoverflow.com/questions/4018860/text-box-in-matplotlib
"""

def on_draw(event):
    """Auto-wraps all text objects in a figure at draw-time"""
    import matplotlib as mpl
    fig = event.canvas.figure

    # Cycle through all artists in all the axes in the figure
    for ax in fig.axes:
        for artist in ax.get_children():
            # If it's a text artist, wrap it...
            if isinstance(artist, mpl.text.Text):
                autowrap_text(artist, event.renderer)

    # Temporarily disconnect any callbacks to the draw event...
    # (To avoid recursion)
    func_handles = fig.canvas.callbacks.callbacks[event.name]
    fig.canvas.callbacks.callbacks[event.name] = {}
    # Re-draw the figure..
    fig.canvas.draw()
    # Reset the draw event callbacks
    fig.canvas.callbacks.callbacks[event.name] = func_handles

def autowrap_text(textobj, renderer):
    """Wraps the given matplotlib text object so that it exceed the boundaries
    of the axis it is plotted in."""
    import textwrap
    # Get the starting position of the text in pixels...
    x0, y0 = textobj.get_transform().transform(textobj.get_position())
    # Get the extents of the current axis in pixels...
    clip = textobj.get_axes().get_window_extent()
    # Set the text to rotate about the left edge (doesn't make sense otherwise)
    textobj.set_rotation_mode('anchor')

    # Get the amount of space in the direction of rotation to the left and
    # right of x0, y0 (left and right are relative to the rotation, as well)
    rotation = textobj.get_rotation()
    right_space = min_dist_inside((x0, y0), rotation, clip)
    left_space = min_dist_inside((x0, y0), rotation - 180, clip)

    # Use either the left or right distance depending on the horiz alignment.
    alignment = textobj.get_horizontalalignment()
    if alignment is 'left':
        new_width = right_space
    elif alignment is 'right':
        new_width = left_space
    else:
        new_width = 2 * min(left_space, right_space)

    # Estimate the width of the new size in characters...
    aspect_ratio = 0.5 # This varies with the font!!
    fontsize = textobj.get_size()
    pixels_per_char = aspect_ratio * renderer.points_to_pixels(fontsize)

    # If wrap_width is < 1, just make it 1 character
    wrap_width = max(1, new_width // pixels_per_char)
    try:
        wrapped_text = textwrap.fill(textobj.get_text(), wrap_width)
    except TypeError:
        # This appears to be a single word
        wrapped_text = textobj.get_text()
    textobj.set_text(wrapped_text)

def min_dist_inside(point, rotation, box):
    """Gets the space in a given direction from "point" to the boundaries of
    "box" (where box is an object with x0, y0, x1, & y1 attributes, point is a
    tuple of x,y, and rotation is the angle in degrees)"""
    from math import sin, cos, radians
    x0, y0 = point
    rotation = radians(rotation)
    distances = []
    threshold = 0.0001
    if cos(rotation) > threshold:
        # Intersects the right axis
        distances.append((box.x1 - x0) / cos(rotation))
    if cos(rotation) < -threshold:
        # Intersects the left axis
        distances.append((box.x0 - x0) / cos(rotation))
    if sin(rotation) > threshold:
        # Intersects the top axis
        distances.append((box.y1 - y0) / sin(rotation))
    if sin(rotation) < -threshold:
        # Intersects the bottom axis
        distances.append((box.y0 - y0) / sin(rotation))
    return min(distances)
