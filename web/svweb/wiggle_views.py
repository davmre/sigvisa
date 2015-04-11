import django
import django.views.generic
from django.shortcuts import render_to_response, get_object_or_404
from django.views.decorators.cache import cache_page
from django.template import RequestContext
from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.core.paginator import Paginator

import numpy as np
import sys
from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *

from sigvisa import *

import pyublas
from sigvisa.models.wiggles.wavelets import parse_wavelet_basis_str, construct_implicit_basis_C
#from sigvisa.models.statespace.compact_support import CompactSupportSSM
from sigvisa.ssms_c import CompactSupportSSM
from sigvisa.signals.common import Waveform

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import calendar
from pytz import timezone

import sigvisa.plotting.plot as plot
import textwrap

from svweb.models import SigvisaCodaFit, SigvisaCodaFitPhase, SigvisaCodaFittingRun, SigvisaWiggle
from svweb.plotting_utils import process_plot_args, view_wave
from sigvisa.plotting.plot import bounds_without_outliers
from sigvisa.signals.common import load_waveform_from_file
from sigvisa.signals.io import *
from sigvisa.source.event import get_event
from sigvisa.graph.load_sigvisa_graph import load_sg_from_db_fit

# detail view for a particular fit


def wiggle_detail_view(request, fpid):
    fphase = get_object_or_404(SigvisaCodaFitPhase, pk=fpid)

    if fphase.wiggle_family != "iid":
        family, res, levels, len_s = parse_wavelet_basis_str(fphase.wiggle_family)
    else:
        len_s = 120

    messages = read_messages(fphase.message_fname, fphase.fitid.runid.runid)

    try:
        pmeans, pvars = messages[fphase.wiggle_family +"_posterior"]
        print "posterior vars", pvars
    except KeyError:
        pmeans, pvars = np.zeros((0,)), np.ones((0,))

    xmin = fphase.arrival_time - 10
    xmax = fphase.arrival_time + len_s + 20

    return render_to_response('svweb/wiggle.html', {
                              #        'wiggle': wiggle,
                              'messages': messages,
                              'message_repr': repr(messages),
                              'pmeans': pmeans,
                              'pvars': pvars,
                              'n_coefs': len(pmeans),
                              'fphase': fphase,
                               'xmin': xmin,
                               'xmax': xmax,
                              }, context_instance=RequestContext(request))


def FitWiggleView(request, fpid):
    # plot the posterior wiggle
    # (mean plus samples?)
    # also show the wiggle coefs

    n_samples = int(request.GET.get("samples", "2"))
    srate = float(request.GET.get("srate", "-1"))

    fphase = get_object_or_404(SigvisaCodaFitPhase, pk=fpid)


    family, res, levels, len_s = parse_wavelet_basis_str(fphase.wiggle_family)
    if srate < 0:
        srate = fphase.fitid.hz

    messages = read_messages(fphase.message_fname, fphase.fitid.runid.runid)
    pmeans, pvars = messages[fphase.wiggle_family +"_posterior"]
    print "posterior vars", pvars

    (sts, ets, ids, prototypes, N) = construct_implicit_basis_C(srate, fphase.wiggle_family, sort=False)
    cssm = CompactSupportSSM(sts, ets, ids, prototypes, pmeans, pvars, 0.0, 1.0)

    wiggle_mean = cssm.mean_obs(N)
    wiggle_var = cssm.obs_var(N)
    samples = [cssm.prior_sample(N) for i in range(n_samples)]

    import sigvisa.models.statespace.compact_support as cs
    from sigvisa.models.wiggles.wavelets import construct_implicit_basis

    #stspm, etsp, idsp, ptsp, Np = construct_implicit_basis(srate, fphase.wiggle_family, sort=False)
    #ic = cs.ImplicitCompactSupportSSM(stspm, etsp, idsp, ptsp, pmeans, pvars, bias=1.0, obs_noise=0.0)
    #wm = ic.mean_obs(N)
    #wv = ic.obs_var(N)
    #samples = [ic.prior_sample(N) for i in range(n_samples)]

    #wiggle_mean = wm
    #wiggle_var = wv


    fig = Figure(figsize=(10, 5), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    axes.set_xlabel("Time (s)", fontsize=8)

    x = np.linspace(0, len_s, N)

    axes.plot(x, wiggle_mean, c="black", lw=2)
    axes.plot(x, wiggle_mean + 2*np.sqrt(wiggle_var), c="red", lw=2)
    axes.plot(x, wiggle_mean - 2*np.sqrt(wiggle_var), c="red", lw=2)
    for s in samples:
        axes.plot(x, s)

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response
