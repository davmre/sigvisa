import django
import django.views.generic
from django.shortcuts import render_to_response, get_object_or_404
from django.views.decorators.cache import cache_page
from django.core.cache import cache
from django.template import RequestContext
from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.core.paginator import Paginator
from django_easyfilters import FilterSet
from django_easyfilters.filters import NumericRangeFilter

import numpy as np
import sys
from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *
from sigvisa import *
from sigvisa.models.noise.armodel.learner import ARLearner

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datetime import datetime
from pytz import timezone
import hashlib

from svweb.models import SigvisaNoiseModel
from svweb.views import filterset_GET_string
from svweb.plotting_utils import process_plot_args, view_wave
from sigvisa.plotting.plot import bounds_without_outliers

class NMFilterSet(FilterSet):
    fields = [
        'sta',
        'chan',
        'band',
        'hz',
        ('window_stime', {}, NumericRangeFilter),
        'model_type',
        'nparams',
    ]


def nm_list_view(request):
    nms = SigvisaNoiseModel.objects.all()
    nm_filter = NMFilterSet(nms, request.GET)
    return render_to_response("svweb/nm_list.html",
                              {'nm_list': nm_filter.qs,
                               'nm_filter': nm_filter,
                               }, context_instance=RequestContext(request))


def nm_detail_view(request, nmid):

    nm = SigvisaNoiseModel.objects.get(nmid=nmid)

    model = nm.load()
    wave = nm.get_data()
    sample = model.sample(500)
    c = np.concatenate([sample, wave.data.compressed()])
    ymin, ymax = bounds_without_outliers(c)

    time_format = "%b %d %Y, %H:%M:%S"
    timestr = datetime.fromtimestamp(nm.window_stime, timezone('UTC')).strftime(time_format)

    return render_to_response('svweb/nm_detail.html', {
        'nm': nm,
        'ymin': ymin,
        'ymax': ymax,
        'timestr': timestr,
        'filter_str': wave['filter_str'].replace(';', "%3B")
    }, context_instance=RequestContext(request))


def nm_param_plot(request, nmid):
    nm = SigvisaNoiseModel.objects.get(nmid=nmid)

    fig = Figure(figsize=(8, 5), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    axes.set_xlabel("t", fontsize=8)


    ind = np.arange(nm.nparams)
    width = 1

    model = nm.load()

    axes.bar(ind, model.params[::-1])
    axes.set_xticks(ind+1)
    axes.set_xticklabels( [str(i) for i in ind[::-1]] )

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response

def nm_sample(request, nmid):
    nm = SigvisaNoiseModel.objects.get(nmid=nmid)
    model = nm.load()

    len_s = float(request.GET.get('len', nm.window_len))
    n = int(len_s * nm.hz)
    d = model.sample(n)

    fig = Figure(figsize=(8, 5), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    axes.set_xlabel("seconds", fontsize=8)

    x = np.linspace(0, len_s, n)
    axes.plot(x, d)

    axes.plot(x, nm.mean * np.ones((len(x),)))

    process_plot_args(request, axes)

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response



def nm_spectrum(request, nmid):


    def empirical_psd(x, hz):
        y, x = matplotlib.mlab.psd(x, NFFT=len(x), Fs=1)
        return (x * hz, np.log(y))


    nm = SigvisaNoiseModel.objects.get(nmid=nmid)
    model = nm.load()
    wave = nm.get_data()

    fig = Figure(figsize=(5, 3), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)


    if nm.model_type == "ar":
        x, y = model.psd(size=len(wave.data))
        l1 = axes.plot(x, y, linewidth=2, zorder=100)
        psd_label = "ideal AR PSD"
    else:
        n=60 * nm.hz
        s = model.sample(n=n)
        s -= model.predict(n=n)
        x, y = empirical_psd(s, hz=nm.hz)
        l1 = axes.plot(x, y, linewidth=1, zorder=100)
        psd_label='model PSD (empirical)'

    x2, y2 = empirical_psd(wave.data.compressed(), hz=nm.hz)
    l2 = axes.plot(x2, y2)
    axes.set_xlabel('Frequency (Hz)', fontsize=8)
    axes.set_ylabel('Power (natural log scale)', fontsize=8)
    fig.legend(l1+l2, (psd_label, 'training data PSD'), loc=1, fontsize=8)

    process_plot_args(request, axes)

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response

def nm_crossval(request, nmid):
    nm = SigvisaNoiseModel.objects.get(nmid=nmid)

    if nm.model_type != "ar":
        raise NotImplementedError("cross-validation is only implemented for AR models")

    model = nm.load()
    model.sf = nm.hz
    wave = nm.get_data()

    max_order = int(request.GET.get('max_order', 50))

    x = np.array(range(2, max_order)) + 1

    lnr = ARLearner(wave.data.compressed(), sf = wave['srate'])
    y = [lnr.crossval(p) for p in x]

    fig = Figure(figsize=(5, 3), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    axes.plot(x, y)
    axes.set_xlabel('degree of model', fontsize=8)
    axes.set_ylabel('lklhood (sum of log prob)', fontsize=8)

    maxi = np.argmax(y)
    axes.text(.95, .05, "max: p=%d, ll=%.1f" % (x[maxi], y[maxi]), horizontalalignment='right', verticalalignment='bottom', transform=axes.transAxes, fontsize=8)


    process_plot_args(request, axes)

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response
