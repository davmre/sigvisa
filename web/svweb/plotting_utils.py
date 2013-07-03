import numpy as np
import scipy

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import sigvisa.plotting.plot as plot

from django.http import HttpResponse, HttpResponseRedirect


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
