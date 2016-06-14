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

from svweb.plotting_utils import process_plot_args, view_wave
from sigvisa.plotting.plot import bounds_without_outliers

# detail view for a particular fit
def WaveSelectView(request):
    sta = request.GET.get("sta", "")
    chan = request.GET.get("chan", "BHZ")
    stime = float(request.GET.get("stime", "0"))
    etime = float(request.GET.get("etime", "0"))
    filter_str = request.GET.get("filter_str", "freq_0.8_4.5;env")

    return render_to_response('svweb/select_wave.html', \
                              {'sta': sta, \
                               'chan': chan, \
                               'stime': stime, \
                               'etime': etime, \
                               'filter_str': filter_str}, \
                              context_instance=RequestContext(request))


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




    return view_wave(request, wave, ratio=ratio, dpi=dpi, color='black', linewidth=1.5, logscale=logscale)

def signal_availability_view(request):

    s = Sigvisa()
    def get_signal_file(sta, t):
        query = "select fname from llnl_wfdisc where sta = '%s' and stime < %f and etime > %f and hz > 9.0 " % (sta, t, t)
        r = s.sql(query)
        return [f[0] for f in r]


    from datetime import datetime
    from pytz import timezone
    time_format = "%b %d %Y, %H:%M:%S"
    def timestr(unix_timestamp):
        return datetime.fromtimestamp(unix_timestamp, timezone('UTC')).strftime(time_format)

    sta = request.GET.get("sta", "")
    stime = float(request.GET.get("stime", "1203646562"))
    etime = float(request.GET.get("etime", "1204416000"))
    interval = float(request.GET.get("interval", "3600"))

    query_times = np.linspace(stime, etime, (etime-stime)/float(interval)+1)

    files = [(timestr(t), get_signal_file(sta, t)) for t in query_times]

    return render_to_response('svweb/signal_availability.html', \
                              {'sta': sta, \
                               'stime': stime, \
                               'etime': etime, \
                               'interval': interval, \
                               'files': files}, \
                              context_instance=RequestContext(request))
