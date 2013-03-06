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

from sigvisa.models.templates.load_by_name import load_template_model
from sigvisa.models.wiggles.wiggle import create_wiggled_phase
from sigvisa.models.wiggles.fourier_features import FourierFeatures
from sigvisa.signals.common import Waveform

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import calendar
from pytz import timezone

import sigvisa.plotting.plot as plot
import textwrap

from svweb.models import SigvisaCodaFit, SigvisaCodaFitPhase, SigvisaCodaFittingRun, SigvisaWiggle
from svweb.plotting_utils import process_plot_args, view_wave, bounds_without_outliers
from sigvisa.signals.common import load_waveform_from_file
from sigvisa.signals.io import *
from sigvisa.source.event import get_event
from sigvisa.models.sigvisa_graph import load_sg_from_db_fit

# detail view for a particular fit


def wiggle_detail_view(request, fpid):


    s = Sigvisa()

    phase = get_object_or_404(SigvisaCodaFitPhase, pk=fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)
    run = get_object_or_404(SigvisaCodaFittingRun, pk=fit.runid.runid)

    cursor = s.dbconn.cursor()
    env_wave = load_event_station_chan(fit.evid, str(fit.sta), str(fit.chan), cursor=cursor).filter(str(fit.band) + ";env")

    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")
    wiggle_wave = load_waveform_from_file(os.path.join(wiggle_dir, phase.wiggle_fname))

    env_ymin, env_ymax = bounds_without_outliers(env_wave.data)
    wiggle_ymin, wiggle_ymax = bounds_without_outliers(wiggle_wave.data)

    wiggle_len = float(request.GET.get('wiggle_len', '-1'))
    if wiggle_len == -1:
        #if phase.sigvisawiggle_set.count() > 0:
        #    wiggle_len = np.max([(w.etime - w.stime) * 1.2 for w in phase.sigvisawiggle_set.all()])
        #else:
        wiggle_len = wiggle_wave['len']

    xmin = wiggle_wave['stime'] - 4
    xmax = xmin + wiggle_len + 8

    return render_to_response('svweb/wiggle.html', {
                              #        'wiggle': wiggle,
                              'phase': phase,
                              'fit': fit,
                              'run': run,
                              'env_ymin': env_ymin,
                              'env_ymax': env_ymax,
                              'wiggle_ymin': wiggle_ymin,
                              'wiggle_ymax': wiggle_ymax,
                              'xmax': xmax,
                              'xmin': xmin,
                              }, context_instance=RequestContext(request))




def raw_wiggle_view(request, fpid):
    phase = get_object_or_404(SigvisaCodaFitPhase, pk=fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)

    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")
    wave = load_waveform_from_file(os.path.join(wiggle_dir, phase.wiggle_fname))
    return view_wave(request, wave, color='black', linewidth=1.5, logscale=False)


def template_wiggle_view(request, fpid):

    phase = get_object_or_404(SigvisaCodaFitPhase, pk=fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)
    ev = get_event(evid=fit.evid)
    skip_initial_s = phase.wiggle_stime - phase.param1

    s = Sigvisa()

    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")
    wiggle = load_waveform_from_file(os.path.join(wiggle_dir, phase.wiggle_fname))

    sg = load_sg_from_db_fit(fit.fitid)
    wave_node = sg.leaf_nodes[0]
    tm_node = sg.get_template_node(ev=ev, wave=wave_node.mw, phase=phase.phase)
    wiggled = create_wiggled_phase(tm_node=tm_node, wave_node=wave_node,
                                   wiggle_data=wiggle.data, skip_initial_s=skip_initial_s)
    wave_node.set_value(wiggled)
    wiggled_wave = wave_node.get_wave()

    return view_wave(request, wiggled_wave, color='black', linewidth=1.5, logscale=False)


def reconstruct_wiggle_data(wiggle):
    if wiggle.type == "fourier":
        min_freq = wiggle.meta1
        max_freq = wiggle.meta2
        fundamental = wiggle.meta0
        srate = wiggle.srate

        f = FourierFeatures(fundamental=fundamental, min_freq=min_freq, max_freq=max_freq, srate=srate)
        reconstructed_wiggle = f.signal_from_encoded_params(
            wiggle.params, srate=wiggle.srate, len_seconds=wiggle.etime - wiggle.stime)
    else:
        raise Exception("unrecognized wiggle type %s" % wiggle.type)
    if wiggle.log == '1':
        reconstructed_wiggle = np.exp(reconstructed_wiggle)

    return reconstructed_wiggle


def reconstructed_wiggle_view(request, wiggleid):

    wiggle = get_object_or_404(SigvisaWiggle, pk=wiggleid)
    phase = get_object_or_404(SigvisaCodaFitPhase, pk=wiggle.fpid.fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)
    ev = get_event(evid=fit.evid)


    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")
    extracted_wave = load_waveform_from_file(os.path.join(wiggle_dir, phase.wiggle_fname))

    sg = load_sg_from_db_fit(fit.fitid)

    wave_node = sg.leaf_nodes[0]
    wm_node = sg.get_wiggle_node(ev=ev, wave=wave_node.mw, phase=phase.phase)
    wiggle_wave = Waveform(my_stats = extracted_wave.my_stats.copy(), segment_stats = extracted_wave.segment_stats.copy(), data = wm_node.get_wiggle(npts=extracted_wave['npts']))

    return view_wave(request, wiggle_wave, color='black', linewidth=1.5, logscale=False)


def reconstructed_template_wiggle_view(request, wiggleid):

    wiggle = get_object_or_404(SigvisaWiggle, pk=wiggleid)
    phase = get_object_or_404(SigvisaCodaFitPhase, pk=wiggle.fpid.fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)

    sg = load_sg_from_db_fit(fit.fitid)
    wave_node = sg.leaf_nodes[0]
    wave_node.fixed_value = False
    wave_node.prior_predict()

    return view_wave(request, wave_node.get_wave(), color='black', linewidth=1.5, logscale=False)
