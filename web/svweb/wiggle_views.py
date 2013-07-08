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

from sigvisa.models.wiggles.wiggle import create_wiggled_phase
from sigvisa.signals.common import Waveform

import matplotlib
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
from sigvisa.graph.load_sigvisa_graph import load_sg_from_db_fit

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


def psd_of_wave(request, wave):

    def empirical_psd(x, hz):
        y, x = matplotlib.mlab.psd(x, NFFT=len(x), Fs=1)
        return (x * hz, np.log(y))

    fig = Figure(figsize=(8, 5), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)

    x, y = empirical_psd(wave.data.compressed(), hz=wave['srate'])
    axes.plot(x, y)
    axes.set_xlabel('Frequency (Hz)', fontsize=8)
    axes.set_ylabel('Power (natural log scale)', fontsize=8)

    process_plot_args(request, axes)

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response


def wiggle_spectrum_view(request, fpid):
    phase = get_object_or_404(SigvisaCodaFitPhase, pk=fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)

    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")
    wave = load_waveform_from_file(os.path.join(wiggle_dir, phase.wiggle_fname))
    return psd_of_wave(request, wave)


def template_wiggle_view(request, fpid):

    phase = get_object_or_404(SigvisaCodaFitPhase, pk=fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)
    ev = get_event(evid=fit.evid)

    s = Sigvisa()

    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")
    wiggle = load_waveform_from_file(os.path.join(wiggle_dir, phase.wiggle_fname))

    sg = load_sg_from_db_fit(fit.fitid)
    wave_node = list(sg.leaf_nodes)[0]
    wiggled = create_wiggled_phase(arrival=(ev.eid, phase.phase), wave_node=wave_node,
                                   wiggle_data=wiggle.data)
    wave_node.set_value(wiggled)
    wiggled_wave = wave_node.get_wave()

    np.savetxt('extract_envelope_%d.txt' % int(fpid), wave_node.get_wave().data)

    actual_wave = fetch_waveform(station=wiggle['sta'], chan=wiggle['chan'], stime=wiggle['stime'], etime=wiggle['etime']).filter('%s;env;hz_%.2f' % (fit.band, fit.hz))
    np.savetxt('actual_envelope.txt', actual_wave.data)

    return view_wave(request, wiggled_wave, color='black', linewidth=1.5, logscale=False)


def reconstruct_wiggle_wave(request, wiggleid):

    wiggle = get_object_or_404(SigvisaWiggle, pk=wiggleid)
    phase = get_object_or_404(SigvisaCodaFitPhase, pk=wiggle.fpid.fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)
    ev = get_event(evid=fit.evid)

    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")
    extracted_wave = load_waveform_from_file(os.path.join(wiggle_dir, phase.wiggle_fname))

    sg = load_sg_from_db_fit(fit.fitid)

    wave_node = list(sg.leaf_nodes)[0]
    wiggle_wave = Waveform(data = wave_node.get_wiggle_for_arrival(eid=ev.eid, phase=phase.phase), stime=extracted_wave['stime'], srate=wave_node.srate, sta=fit.sta, chan=fit.chan, filter_str='%s;env;hz_%.2f' % (fit.band, wave_node.srate))
    return wiggle_wave

def reconstructed_wiggle_spectrum_view(request, wiggleid):
    wiggle_wave = reconstruct_wiggle_wave(request, wiggleid)
    return psd_of_wave(request, wiggle_wave)

def reconstructed_wiggle_view(request, wiggleid):
    wiggle_wave = reconstruct_wiggle_wave(request, wiggleid)
    return view_wave(request, wiggle_wave, color='black', linewidth=1.5, logscale=False)


def reconstructed_template_wiggle_view(request, wiggleid):

    wiggle = get_object_or_404(SigvisaWiggle, pk=wiggleid)
    phase = get_object_or_404(SigvisaCodaFitPhase, pk=wiggle.fpid.fpid)
    fit = get_object_or_404(SigvisaCodaFit, pk=phase.fitid.fitid)

    sg = load_sg_from_db_fit(fit.fitid)
    wave_node = list(sg.leaf_nodes)[0]
    wave_node.unfix_value()
    wave_node.parent_predict()

    np.savetxt('param_envelope_%d.txt' % phase.fpid, wave_node.get_wave().data)
    return view_wave(request, wave_node.get_wave(), color='black', linewidth=1.5, logscale=False)
