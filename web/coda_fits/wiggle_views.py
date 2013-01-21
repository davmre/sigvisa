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
from database.dataset import *
from database.signal_data import *

from sigvisa import *

from signals.template_models.load_by_name import load_template_model
from learn.train_wiggles import create_wiggled_phase
from signals.waveform_matching.fourier_features import FourierFeatures
from signals.common import Waveform

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import calendar
from pytz import timezone

import plotting.plot as plot
import textwrap

from coda_fits.models import SigvisaCodaFit, SigvisaCodaFitPhase, SigvisaCodaFittingRun, SigvisaCodaFitPhaseWiggle
from coda_fits.views import process_plot_args, error_wave


# detail view for a particular fit
def wiggle_detail_view(request, wiggleid):

    # get the fit corresponding to the given pageid for this run


    s = Sigvisa()

    wiggle = get_object_or_404(SigvisaCodaFitPhaseWiggle, pk=wiggleid)
    phase = get_object_or_404(SigvisaCodaFitPhase, pk=wiggle.fpid.fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)
    run = get_object_or_404(SigvisaCodaFittingRun, pk=fit.runid.runid)

    return render_to_response('coda_fits/wiggle.html', {
        'wiggle': wiggle,
        'phase': phase,
        'fit': fit,
        'run': run,
        }, context_instance=RequestContext(request))

def view_wave(request, wave, **kwargs):
    #    try:
    fig = Figure(figsize=(5,3), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    axes.set_xlabel("Time (s)", fontsize=8)
    plot.subplot_waveform(wave, axes, **kwargs)
    #matplotlib.rcParams.update({'font.size': 8})
    canvas=FigureCanvas(fig)
    response=django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response
#    except Exception as e:
#        return error_wave(e)
    
   

def raw_wiggle_view(request, wiggleid):

    wiggle = get_object_or_404(SigvisaCodaFitPhaseWiggle, pk=wiggleid)
    phase = get_object_or_404(SigvisaCodaFitPhase, pk=wiggle.fpid.fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    tm = load_template_model(phase.template_model, run_name=None, run_iter=0, model_type="dummy")

    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggles")

    #    try:
    raw_wiggle = np.loadtxt(os.path.join(wiggle_dir, wiggle.filename))
    wiggle_wave = Waveform(data=raw_wiggle, srate=wiggle.srate, stime=wiggle.stime, sta=str(fit.sta), evid=fit.evid, filter_str="wiggle", chan=str(fit.chan))
    return view_wave(request, wiggle_wave, color='black', linewidth=1.5, logscale=False)
#    except Exception as e:
#        return error_wave(e)


def template_wiggle_view(request, wiggleid):

    wiggle = get_object_or_404(SigvisaCodaFitPhaseWiggle, pk=wiggleid)
    phase = get_object_or_404(SigvisaCodaFitPhase, pk=wiggle.fpid.fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)

    s = Sigvisa()

    tm = load_template_model(phase.template_model, run_name=None, run_iter=0, model_type="dummy")

    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggles")

    #try:
    raw_wiggle = np.loadtxt(os.path.join(wiggle_dir, wiggle.filename))

    fit_phases = fit.sigvisacodafitphase_set.all()
    fit_params =np.asfarray([(p.param1, p.param2, p.param3, p.param4) for p in fit_phases])
    phases = tuple([str(p.phase) for p in fit_phases])
    (phases, vals) = filter_and_sort_template_params(phases, fit_params, filter_list=s.phases)

    st = calendar.timegm(fit.stime.timetuple())
    et = calendar.timegm(fit.etime.timetuple())
    wiggled_phase_data = create_wiggled_phase((phases, vals), tm, phase.phase, raw_wiggle, st, npts=((et-st) * wiggle.srate), srate = wiggle.srate, chan=str(fit.chan), sta=str(fit.sta))
    wiggle_wave = Waveform(data=wiggled_phase_data, srate=wiggle.srate, stime=st, sta=str(fit.sta), evid=fit.evid, filter_str="wiggle", chan=str(fit.chan))
    return view_wave(request, wiggle_wave, color='black', linewidth=1.5, logscale=False)
    #except Exception as e:
    #    return error_wave(e)


def reconstruct_wiggle_data(wiggle):
    if wiggle.type == "fourier":
        min_freq = wiggle.meta1
        max_freq = wiggle.meta2
        fundamental = wiggle.fundamental
        srate = wiggle.srate
        params = np.array(eval(wiggle.params))

        f = FourierFeatures(fundamental=fundamental, min_freq=min_freq, max_freq=max_freq, srate=srate)
        reconstructed_wiggle = f.signal_from_features(params, len_seconds=wiggle.etime-wiggle.stime)
    else:
        raise Exception("unrecognized wiggle type %s" % wiggle.type)
    return reconstructed_wiggle


def reconstructed_wiggle_view(request, wiggleid):

    wiggle = get_object_or_404(SigvisaCodaFitPhaseWiggle, pk=wiggleid)
    phase = get_object_or_404(SigvisaCodaFitPhase, pk=wiggle.fpid.fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)

    tm = load_template_model(phase.template_model, run_name=None, run_iter=0, model_type="dummy")
    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggles")

    #    try:
    reconstructed_wiggle = reconstruct_wiggle_data(wiggle)
    wiggle_wave = Waveform(data=reconstructed_wiggle, srate=wiggle.srate, stime=wiggle.stime, sta=str(fit.sta), evid=fit.evid, filter_str="wiggle", chan=str(fit.chan))
    return view_wave(request, wiggle_wave, color='black', linewidth=1.5, logscale=False)
#    except Exception as e:
#        return error_wave(e)

def reconstructed_template_wiggle_view(request, wiggleid):

    wiggle = get_object_or_404(SigvisaCodaFitPhaseWiggle, pk=wiggleid)
    phase = get_object_or_404(SigvisaCodaFitPhase, pk=wiggle.fpid.fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)

    tm = load_template_model(phase.template_model, run_name=None, run_iter=0, model_type="dummy")
    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggles")

    s = Sigvisa()

    try:
        reconstructed_wiggle = reconstruct_wiggle_data(wiggle)

        fit_phases = fit.sigvisacodafitphase_set.all()
        fit_params =np.asfarray([(p.param1, p.param2, p.param3, p.param4) for p in fit_phases])
        phases = tuple([str(p.phase) for p in fit_phases])
        (phases, vals) = filter_and_sort_template_params(phases, fit_params, filter_list=s.phases)

        st = calendar.timegm(fit.stime.timetuple())
        et = calendar.timegm(fit.etime.timetuple())
        wiggled_phase_data = create_wiggled_phase((phases, vals), tm, phase.phase, reconstructed_wiggle, st, npts=((et-st) * wiggle.srate), srate = wiggle.srate, chan=str(fit.chan), sta=str(fit.sta))

        wiggle_wave = Waveform(data=wiggled_phase_data, srate=wiggle.srate, stime=st, sta=str(fit.sta), evid=fit.evid, filter_str="wiggle", chan=str(fit.chan))
        return view_wave(request, wiggle_wave, color='black', linewidth=1.5, logscale=False)
    except Exception as e:
        return error_wave(e)
