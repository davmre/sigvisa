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


from sigvisa.models.wiggles.wavelets import parse_wavelet_basis_str, construct_wavelet_basis
from sigvisa.models.statespace.compact_support import CompactSupportSSM
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

    family, res, levels, len_s = parse_wavelet_basis_str(fphase.wiggle_family)

    messages = read_messages(fphase.message_fname, fphase.fitid.runid.runid)
    pmeans, pvars = messages[fphase.wiggle_family +"_posterior"]
    print "posterior vars", pvars

    return render_to_response('svweb/wiggle.html', {
                              #        'wiggle': wiggle,
                              'messages': messages,
                              'message_repr': repr(messages),
                              'pmeans': pmeans,
                              'pvars': pvars,
                              'n_coefs': len(pmeans),
                              'fphase': fphase,
                              }, context_instance=RequestContext(request))


def FitWiggleView(request, fpid):
    # plot the posterior wiggle
    # (mean plus samples?)
    # also show the wiggle coefs

    n_samples = int(request.GET.get("samples", "20"))
    srate = float(request.GET.get("srate", "-1"))

    fphase = get_object_or_404(SigvisaCodaFitPhase, pk=fpid)


    family, res, levels, len_s = parse_wavelet_basis_str(fphase.wiggle_family)
    if srate < 0:
        srate = fphase.fitid.hz

    messages = read_messages(fphase.message_fname, fphase.fitid.runid.runid)
    pmeans, pvars = messages[fphase.wiggle_family +"_posterior"]
    print "posterior vars", pvars

    basis = construct_wavelet_basis(srate, fphase.wiggle_family)
    cssm = CompactSupportSSM(basis, coef_prior_means=pmeans, coef_prior_vars=pvars, obs_noise=0.0, bias=1.0)

    N = basis.shape[1]
    wiggle_mean = cssm.mean_obs(N)
    wiggle_var = cssm.obs_var(N)
    print "priors", cssm.prior_vars()
    samples = [cssm.prior_sample(N) for i in range(n_samples)]

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
