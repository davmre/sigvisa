import django
import django.views.generic
from django.shortcuts import render_to_response, get_object_or_404
from django.views.decorators.cache import cache_page
from django.template import RequestContext
from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse

import numpy as np
import sys
from sigvisa.signals.io import *
from sigvisa import *

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
from matplotlib.figure import Figure
from datetime import datetime
import django.db

import sigvisa.plotting.plot as plot
import textwrap

from svweb.plotting_utils import process_plot_args, view_wave, bounds_without_outliers


# detail view for a particular fit
def WaveSelectView(request):
    return render_to_response('svweb/select_wave.html', {}, context_instance=RequestContext(request))


@cache_page(60*60*24*365)
def WaveImageView(request):
    quote_name = django.db.connection.ops.quote_name

    sta = str(quote_name(request.GET.get("sta", ""))[1:-1])
    chan = str(quote_name(request.GET.get("chan", ""))[1:-1])
    stime = float(request.GET.get("stime", ""))
    etime = float(request.GET.get("etime", ""))
    logscale = request.GET.get("logscale", "false").lower().startswith('t')
    filter_str = str(quote_name(request.GET.get("filter_str", ""))[1:-1].lower())
    ratio = float(request.GET.get("ratio", "1.6"))
    dpi = int(request.GET.get("dpi", "144"))

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    wave = fetch_waveform(sta, chan, stime, etime).filter(filter_str)

    Figure(figsize=(5 * ratio,5), dpi=dpi)


    return view_wave(request, wave, color='black', linewidth=1.5, logscale=logscale)
