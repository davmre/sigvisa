import django
import django.views.generic
from django.shortcuts import render_to_response, get_object_or_404
from django.views.decorators.cache import cache_page
from django.template import RequestContext
from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse

import numpy as np
import sys
from signals.io import *
from sigvisa import *

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from datetime import datetime
import django.db

import plotting.plot as plot
import textwrap

# detail view for a particular fit
def WaveSelectView(request):
    return render_to_response('waves/select.html', {}, context_instance=RequestContext(request))


@cache_page(60*60*24*365)
def WaveImageView(request):
    quote_name = django.db.connection.ops.quote_name

    sta = quote_name(request.GET.get("sta", ""))[1:-1]
    chan = quote_name(request.GET.get("chan", ""))[1:-1]
    stime = float(request.GET.get("stime", ""))
    etime = float(request.GET.get("etime", ""))
    logscale = request.GET.get("logscale", "false").lower().startswith('t')
    filter_str = quote_name(request.GET.get("filter_str", ""))[1:-1].lower()

    print sta

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    try:
        wave = fetch_waveform(sta, chan, stime, etime)

        fig = plt.figure(figsize=(8,5), dpi=144)
        fig.patch.set_facecolor('white')
        plt.xlabel("Time (s)")
        axes = plt.gca()
        plot.subplot_waveform(wave.filter(filter_str), axes, color='black', linewidth=1.5, logscale=logscale)
        matplotlib.rcParams.update({'font.size': 8})

    except Exception as e:
        error_text = 'Error plotting waveform: \"%s\"' % str(e)
        fig = plt.figure(figsize=(5,3), dpi=144)
        fig.patch.set_facecolor('white')
        axes = plt.gca()
        plt.text(.5, .5, "\n".join(textwrap.wrap(error_text, 60)), horizontalalignment='center', verticalalignment='center', transform = axes.transAxes)
        matplotlib.rcParams.update({'font.size': 8})

    canvas=FigureCanvas(fig)
    response=django.http.HttpResponse(content_type='image/png')
    plt.tight_layout()
    canvas.print_png(response)
    plt.close(fig)
    return response