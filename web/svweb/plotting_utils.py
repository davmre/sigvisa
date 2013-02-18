import numpy as np
import scipy

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import sigvisa.plotting.plot as plot

from django.http import HttpResponse, HttpResponseRedirect

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

def view_wave(request, wave, ratio=1.6, dpi=144, **kwargs):
    """

    Given a Waveform object, return a django HttpResponse visualizing
    that object, with plot parameters optionally set by the GET
    request string.

    """

    fig = Figure(figsize=(5 * ratio,5), dpi=dpi)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    axes.set_xlabel("Time (s)", fontsize=8)
    plot.subplot_waveform(wave, axes, **kwargs)
    process_plot_args(request, axes)
    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response

def process_plot_args(request, axes):
    xmin = request.GET.get("xmin", "auto")
    xmax = request.GET.get("xmax", "auto")
    ymin = request.GET.get("ymin", "auto")
    ymax = request.GET.get("ymax", "auto")

    if xmin != "auto" and xmax != "auto":
        axes.set_xlim(float(xmin), float(xmax))
    if ymin != "auto" and ymax != "auto":
        axes.set_ylim(float(ymin), float(ymax))

    if xmin == "auto" and xmax == "auto":
        xmin, xmax = axes.get_xlim()
        xlen = float(request.GET.get('len', "-1"))
        if xlen > 0:
            axes.set_xlim(xmin, xmin + xlen)

