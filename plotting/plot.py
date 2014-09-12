from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm

import scipy.stats

from obspy.core import Trace, Stream, UTCDateTime

from sigvisa.database.dataset import *
import sigvisa.database.db
from sigvisa import Sigvisa
from sigvisa.utils.geog import dist_deg, azimuth
from sigvisa.source.event import get_event


def savefig(fname, fig, **kwargs):
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(fname, **kwargs)

def basic_plot_to_file(fname, data, data2=None, data3=None, xvals=None, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, **kwargs):

    fig = Figure(figsize=(8, 5), dpi=300)
    axes = fig.add_subplot(111)

    def plotdata(d):
        if xvals is None:
            axes.plot(d, **kwargs)
        else:
            axes.plot(xvals, d, **kwargs)

    plotdata(data)
    if data2 is not None:
        plotdata(data2)
    if data3 is not None:
        plotdata(data3)

    if title is not None:
        axes.set_title(title)

    if xlabel is not None:
        axes.set_xlabel(xlabel)

    if ylabel is not None:
        axes.set_ylabel(ylabel)

    if xlim is not None:
        axes.set_xlim(xlim)

    if ylim is not None:
        axes.set_ylim(ylim)

    savefig(fname, fig, bbox_inches="tight", dpi=300)

def bounds_without_outliers(data, coverage=99.99, epsilon=0.05):
    """

    Given a 1D array, find the min and the max excluding extreme
    outliers. Intended to be used as min/max values in plotting, to
    ensure that most of the data is visible on the plot.

    """

    # if data is a masked array, ignore the masked entries
    try:
        data = data.compressed()
    except:
        pass

    min_bound = scipy.stats.scoreatpercentile(data, per=(100 - coverage) / 2.0)
    max_bound = scipy.stats.scoreatpercentile(data, per=100 - (100 - coverage) / 2.0)
    padding = (max_bound - min_bound) * epsilon

    return min_bound - padding, max_bound + padding


def plot_with_fit(fname, wn, title="",
                  ymin=None, ymax=None,
                  unass_lw = 1,
                  ev_lw=2,
                  model_lw = 1,
                  tmpl_lw = None,
                  highlight_eid=None,
                  fig=None,
                  **kwargs):

    if fig is None:
        fig = Figure(figsize=(8, 5), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    axes.set_title(title)
    axes.set_xlabel("Time (s)", fontsize=8)

    if ymin is not None and ymax is not None:
        axes.set_ylim(ymin, ymax)

    import copy
    mutable = copy.copy(wn._mutable)
    wave = wn.get_wave()
    wn.unfix_value()
    wn.parent_predict(include_wiggles=True)
    pred = wn.get_wave()

    unass_tmpls = []
    ev_tmpls = []
    highlight_tmpl = None
    wn.parent_predict(include_wiggles=False)
    tmpl = wn.get_wave()

    for arrival in wn.arrivals():
        wn.parent_predict(include_wiggles=False, arrivals=[arrival,])
        if arrival[0] < 0:
            unass_tmpls.append(wn.get_wave())
        elif arrival[0] != highlight_eid:
            ev_tmpls.append(wn.get_wave())
        else:
            highlight_tmpl = wn.get_wave()

    subplot_waveform(wave, axes, color='black', linewidth=1.5, **kwargs)
    if model_lw is not None:
        subplot_waveform(pred, axes, color="green",
                              linewidth=model_lw, alpha = 1,
                              plot_dets=False, **kwargs)
    if tmpl_lw is not None:
        subplot_waveform(tmpl, axes, color="red",
                         linewidth=tmpl_lw, alpha = 1,
                         plot_dets=False, **kwargs)
    if unass_lw is not None:
        for unass_tmpl in unass_tmpls:
            subplot_waveform(unass_tmpl, axes, color="purple",
                             linewidth=unass_lw, alpha = 1,
                             plot_dets=False, **kwargs)
    if ev_lw is not None:
        for ev_tmpl in ev_tmpls:
            subplot_waveform(ev_tmpl, axes, color="blue",
                             linewidth=ev_lw, alpha = 1,
                             plot_dets=False, **kwargs)
    if highlight_tmpl is not None:
        subplot_waveform(highlight_tmpl, axes, color="orange",
                         linewidth=2, alpha = 1,
                         plot_dets=False, **kwargs)

    wn.set_value(wave.data)
    wn._mutable = mutable
    wn._update_mutable_cache()

    if fname is None:
        return fig
    else:
        savefig(fname, fig, bbox_inches="tight", dpi=300)

def plot_with_fit_shapes(fname, wn, title="",
                         ymin=None, ymax=None,
                         unass_lw = 1,
                         ev_lw=2,
                         model_lw = 1,
                         highlight_eid=None,
                         fig=None,
                         axes=None,
                         plot_dets="leb",
                         plot_wave=True,
                         shape_colors=None,
                         alpha=1.0,
                         **kwargs):

    if axes is None:
        if fig is None:
            fig = Figure(figsize=(8, 5), dpi=144)
        fig.patch.set_facecolor('white')
        axes = fig.add_subplot(111)


    axes.set_title(title)
    axes.set_xlabel("Time (s)", fontsize=8)

    if ymin is not None and ymax is not None:
        axes.set_ylim(ymin, ymax)

    if shape_colors is None:
        shape_colors = dict()

    import copy
    mutable = copy.copy(wn._mutable)
    wave = wn.get_wave()
    wn.unfix_value()
    wn.parent_predict(include_wiggles=True)
    pred = wn.get_wave()

    unass_tmpls = []
    ev_tmpls = []
    highlight_tmpl = None
    wn.parent_predict(include_wiggles=False)
    tmpl = wn.get_wave()

    for arrival in wn.arrivals():
        wn.parent_predict(include_wiggles=False, arrivals=[arrival,])
        eid = arrival[0]
        if eid < 0:
            if eid not in shape_colors:
                shape_colors[eid] = cm.get_cmap('jet')(np.random.rand()*.5)
            unass_tmpls.append((wn.get_wave(), shape_colors[eid]))
            print "plotting uatemplate", eid, "as color", shape_colors[eid]
        elif arrival[0] != highlight_eid:
            if eid not in shape_colors:
                shape_colors[eid] = cm.get_cmap('jet')(np.random.rand()*.5+.5)
            ev_tmpls.append((wn.get_wave(), shape_colors[eid]))
        else:
            highlight_tmpl = wn.get_wave()

    if plot_wave:
        subplot_waveform(wave, axes, color='black', linewidth=1.5, plot_dets=plot_dets, **kwargs)
    if model_lw is not None:
        subplot_waveform(pred, axes, color="purple",
                              linewidth=4, alpha = 1*alpha, linestyle='-',
                              plot_dets=False, **kwargs)

    if unass_lw is not None:
        for (unass_tmpl, color) in unass_tmpls:
            subplot_waveform(unass_tmpl, axes, color=color,
                             linewidth=unass_lw, alpha = 0.5*alpha,
                             plot_dets=False, fill_y2=wn.nm.c,
                             **kwargs)
    if ev_lw is not None:
        evcolors = cm.get_cmap('jet')(np.linspace(0.5,1.0, len(ev_tmpls)))
        for (ev_tmpl,color) in ev_tmpls:
            subplot_waveform(ev_tmpl, axes, color=color,
                             linewidth=ev_lw, alpha = 0.8*alpha,
                             plot_dets=False, fill_y2=wn.nm.c,
                             **kwargs)

    wn.set_value(wave.data)
    wn._mutable = mutable
    wn._update_mutable_cache()

    if fname is not None:
        savefig(fname, fig, bbox_inches="tight", dpi=300)
    #else:
    #    return fig
    return shape_colors


def plot_det_times(wave, axes=None, logscale=False, stime=None, etime=None, color="red", all_arrivals=False):
    if wave is None:
        return

    if axes is None:
        axes = plt.subplot(1, 1, 1)

    try:
        arrivals = wave['event_arrivals']
    except:
        arrivals = wave['arrivals']
    if all_arrivals:
        arrivals = wave['arrivals_idcx']

    if arrivals.shape[0] == 0:
        return

    all_det_times = arrivals[:, DET_TIME_COL]
    all_det_labels = [Sigvisa().phasenames[arrivals[i, DET_PHASE_COL]] for i in range(arrivals.shape[0])]

    if etime is not None:
        adt = [all_det_times[i] for i in range(len(all_det_times)) if stime <= all_det_times[i] <= etime]
        adl = [all_det_labels[i] for i in range(len(all_det_times)) if stime <= all_det_times[i] <= etime]
        all_det_times = adt
        all_det_labels = adl

    if all_det_times is not None:

        start_idx = 0
        end_idx = len(wave.data)
        if stime:
            start_idx = max(int(wave['srate']*(stime-wave['stime'])), start_idx)
        if etime:
            end_idx = min(int(wave['srate']*(etime-wave['stime'])), end_idx)

        maxwave, minwave = float(np.max(wave.data[start_idx:end_idx])), float(np.min(wave.data[start_idx:end_idx]))

        if logscale:
            (maxwave, minwave) = (np.log(maxwave), np.log(minwave))
        axes.bar(left=all_det_times, height=[maxwave - minwave for _ in all_det_times],
                 width=.25, bottom=minwave, edgecolor=color, color=color, linewidth=1, alpha=.5)
        if all_det_labels is not None:
            for (t, lbl) in zip(all_det_times, all_det_labels):
                axes.text(t + .5, maxwave - (maxwave - minwave) * 0.1, lbl, color=color, fontsize=10)

# does not save for you - you need to call savefig() yourself!

def plot_pred_atimes(predictions, wave, axes=None, logscale=False, stime=None, etime=None, color="green", draw_bars=True, draw_text=True, alpha=0.5, **kwargs):
    if predictions is None or wave is None:
        return

    if axes is None:
        axes = plt.subplot(1, 1, 1)

    if etime is not None:
        old_predictions = predictions
        predictions = dict([p for p in predictions.items() if stime <= p[1] <= etime])
    else:
        old_predictions = predictions

    if len(predictions) < 1:
        if stime is None:
            stime = wave['stime']
        if etime is None:
            etime = wave['etime']
        print "warning: atimes %s not within plot window (%f, %f)!" % (old_predictions, stime, etime)
        return

    pred_labels, pred_times = zip(*predictions.items())


    start_idx = 0
    end_idx = len(wave.data)
    if stime:
        start_idx = max(int(wave['srate']*(stime-wave['stime'])), start_idx)
    if etime:
        end_idx = min(int(wave['srate']*(etime-wave['stime'])), end_idx)

    maxwave, minwave = float(np.max(wave.data[start_idx:end_idx])), float(np.min(wave.data[start_idx:end_idx]))


    if logscale:
        (maxwave, minwave) = (np.log(maxwave), np.log(minwave))

    if draw_bars:
        axes.bar(left=pred_times, height=[maxwave - minwave for _ in pred_times],
                 width=.25, bottom=minwave, edgecolor=color, color=color, linewidth=1, alpha=alpha, **kwargs)

    if draw_text:
        for (lbl, t) in predictions.items():
            axes.text(t + .5, maxwave - (maxwave - minwave) * 0.1, lbl, color=color, fontsize=10)

# does not save for you - you need to call savefig() yourself!


def plot_segment(segment, title=None, chans=None, logscale=False):
    fig = Figure(figsize=(10, 8), dpi=250)

    # if no title is given, try to generate a simple one based on the event being plotted
    if title is None:
        evid = None
        try:
            evid = segment['evid']
            title = "Event %d at %s" % (evid, segment['sta'])
        except KeyError:
            pass

    if title is not None:
        fig.suptitle(title, fontsize=20)

    # loop over channels to plot each channel
    chans = segment.get_chans() if chans is None else chans
    n = len(chans)
    gs = gridspec.GridSpec(n * 3 + 1, 1)
    gs.update(left=0.1, right=0.95, hspace=1)
    axes = None
    for chidx, chan in enumerate(sorted(chans)):
        axes = fig.add_subplot(gs[chidx * 3:chidx * 3 + 3, 0], sharex=axes)
        axes.set_xlabel("Time (s)")
        wave = segment[chan]
        subplot_waveform(wave, logscale=logscale, axes=axes)

    axes = fig.add_subplot(gs[n * 3, 0])
    axes.axis('off')
#  axes.get_xaxis().set_visible(False)
#  axes.get_yaxis().set_visible(False)
    descr = "Segment: " + str(segment)
    try:
        evid = segment['evid']
        e = get_event(evid)
        descr = descr + "\n\n" + "Event: " + str(e)
    except KeyError as e:
        pass
    axes.text(0.5, 0, descr, fontsize=8, color="black", horizontalalignment='center', verticalalignment='center')

    return fig


def plot_waveform(wave, title=None, logscale=False):
    fig = Figure()

    if title is not None:
        fig.suptitle(title)

    axes = fig.add_subplot(1, 1, 1)
    axes.set_xlabel("Time (s)")
    subplot_waveform(wave, axes, logscale=logscale)
    return fig


def subplot_waveform(wave, axes, logscale=False, stime=None, etime=None, plot_dets=True, plot_predictions=None, fill_y2=None,  **kwargs):
    srate = wave['srate']
    wave_data = np.log(wave.data) if logscale else wave.data
    if stime is None:
        npts = wave['npts']
        stime = wave['stime']
        timevals = np.arange(stime, stime + npts / srate, 1.0 / srate)[0:npts]
    else:
        sidx = max(0, int((stime - wave['stime']) * srate))
        eidx = max(0, min(int((etime - wave['stime']) * srate), wave['npts']))
        npts = eidx-sidx
        effective_stime = float(sidx) / srate + wave['stime']
        effective_etime = effective_stime + float(npts) / srate
        timevals = np.arange(effective_stime, effective_etime, 1.0 / srate)[0:npts]
        wave_data=wave_data[sidx:eidx]

    axes.set_ylabel(wave['chan'])
    if fill_y2 is None:
        axes.plot(timevals, wave_data, **kwargs)
    else:
        axes.fill_between(timevals, wave_data, y2=fill_y2, **kwargs)
    if plot_dets:
        all_arrivals = (plot_dets=="idcx")
        plot_det_times(wave, axes=axes, logscale=logscale, stime=stime, etime=etime, all_arrivals=all_arrivals)

    if plot_predictions:
        plot_pred_atimes(predictions=plot_predictions, wave=wave,
                         axes=axes, logscale=logscale,
                         stime=stime, etime=etime)

# does not save for you - you need to call savefig() yourself!
def plot_bands(wave, bands=None, title=None):
    format = "k-"

    if bands is None:
        bands = Sigvisa().bands

    Figure(figsize=(12, 30))
    fig.xlabel("Time (s)")

    for (bidx, band) in enumerate(sorted(bands)):
        if bidx == 0:
            axes = fig.subplot(len(bands), 1, 1)
            if title is not None:
                fig.title(title)
        else:
            fig.subplot(len(bands), 1, bidx + 1, sharex=axes)

        fig.ylabel(yl)

        npts = wave["npts"]
        srate = wave["srate"]
        stime = wave["stime"]
        timevals = np.arange(stime, stime + npts / srate, 1.0 / srate)[0:npts]

        nwave = wave.filter(band)

        fig.plot(timevals, nwave.data, format)

        plot_det_times(wave)


#################################################################
#################################################################
"""
matplotlib text wrapping code from
http://stackoverflow.com/questions/4018860/text-box-in-matplotlib

to use: insert the line
fig.canvas.mpl_connect('draw_event', on_draw)
at the very end of constructing the figure

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
    aspect_ratio = 0.5  # This varies with the font!!
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
