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
from svweb.views import process_plot_args, error_wave
from sigvisa.signals.common import load_waveform_from_file
from sigvisa.signals.io import *


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
        if phase.sigvisawiggle_set.count() > 0:
            wiggle_len = np.max([(w.etime - w.stime) * 1.2 for w in phase.sigvisawiggle_set.all()])
        else:
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


def view_wave(request, wave, **kwargs):
    """

    Given a Waveform object, return a django HttpResponse visualizing
    that object, with plot parameters optionally set by the GET
    request string.

    """

    fig = Figure(figsize=(5, 3), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    axes.set_xlabel("Time (s)", fontsize=8)
    plot.subplot_waveform(wave, axes, **kwargs)
    process_plot_args(request, axes)
    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response


def raw_wiggle_view(request, fpid):

    phase = get_object_or_404(SigvisaCodaFitPhase, pk=fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    tm = load_template_model(phase.template_model, run_name=None, run_iter=0, model_type="dummy")

    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")

    wave = load_waveform_from_file(os.path.join(wiggle_dir, phase.wiggle_fname))
    return view_wave(request, wave, color='black', linewidth=1.5, logscale=False)


def template_wiggle_view(request, fpid):

    phase = get_object_or_404(SigvisaCodaFitPhase, pk=fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)

    s = Sigvisa()

    tm = load_template_model(phase.template_model, run_name=None, run_iter=0, model_type="dummy")

    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")

    wave = load_waveform_from_file(os.path.join(wiggle_dir, phase.wiggle_fname))

    fit_phases = fit.sigvisacodafitphase_set.all()
    fit_params = np.asfarray([(p.param1, p.param2, p.param3, p.param4) for p in fit_phases])
    phases = tuple([str(p.phase) for p in fit_phases])
    (phases, vals) = filter_and_sort_template_params(phases, fit_params, filter_list=s.phases)

    st = calendar.timegm(fit.stime.timetuple())
    et = calendar.timegm(fit.etime.timetuple())
    wave = create_wiggled_phase((phases, vals), tm, phase.phase, wiggle=wave.data, wiggle_stime=phase.wiggle_stime, st=st,
        npts=((et - st) * wave['srate']), srate = wave['srate'], chan=str(fit.chan), sta=str(fit.sta), band=str(fit.band))

    return view_wave(request, wave, color='black', linewidth=1.5, logscale=False)


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

    tm = load_template_model(phase.template_model, run_name=None, run_iter=0, model_type="dummy")
    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")

    #    try:
    reconstructed_wiggle = reconstruct_wiggle_data(wiggle)
    wiggle_wave = Waveform(data=reconstructed_wiggle, srate=wiggle.srate, stime=wiggle.stime, sta=str(fit.sta),
                           evid=fit.evid, filter_str="wiggle", chan=str(fit.chan))
    return view_wave(request, wiggle_wave, color='black', linewidth=1.5, logscale=False)
#    except Exception as e:
#        return error_wave(e)


def reconstructed_template_wiggle_view(request, wiggleid):

    wiggle = get_object_or_404(SigvisaWiggle, pk=wiggleid)
    phase = get_object_or_404(SigvisaCodaFitPhase, pk=wiggle.fpid.fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)

    tm = load_template_model(phase.template_model, run_name=None, run_iter=0, model_type="dummy")
    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")

    s = Sigvisa()

    reconstructed_wiggle_data = reconstruct_wiggle_data(wiggle)

    fit_phases = fit.sigvisacodafitphase_set.all()
    fit_params = np.asfarray([(p.param1, p.param2, p.param3, p.param4) for p in fit_phases])
    phases = tuple([str(p.phase) for p in fit_phases])
    (phases, vals) = filter_and_sort_template_params(phases, fit_params, filter_list=s.phases)

    st = calendar.timegm(fit.stime.timetuple())
    et = calendar.timegm(fit.etime.timetuple())
    wave = create_wiggled_phase((phases, vals), tm, phase.phase, wiggle=reconstructed_wiggle_data, 
                                wiggle_stime=phase.wiggle_stime, st=st, npts=((et - st) * wiggle.srate), 
                                srate = wiggle.srate, chan=str(fit.chan), sta=str(fit.sta), 
                                band=str(fit.band))
    return view_wave(request, wave, color='black', linewidth=1.5, logscale=False)
