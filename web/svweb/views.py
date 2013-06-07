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
from sigvisa.signals.io import *
from sigvisa import *
from sigvisa.models.noise.noise_util import get_noise_model
from sigvisa.models.templates.load_by_name import load_template_model
from sigvisa.source.event import get_event, EventNotFound
from sigvisa.models.noise.armodel.model import ARModel, ErrorModel
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.graph.load_sigvisa_graph import load_sg_from_db_fit
import sigvisa.utils.geog

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datetime import datetime
from pytz import timezone
import hashlib

import sigvisa.plotting.plot as plot
import sigvisa.plotting.histogram as histogram
import textwrap

from svweb.models import SigvisaCodaFit, SigvisaCodaFitPhase, SigvisaCodaFittingRun, view_options
from svweb.plotting_utils import process_plot_args, view_wave, bounds_without_outliers

def filterset_GET_string(filterset):
    """

    Given a django_easyfilters FilterSet object, return a dictionary
    of field->field_value mappings (where the value could be a string
    indicating a range), and a string containing these mappings that
    can be used in a new GET request to recreate the FilterSet.

    """

    def field_names(self):
        field_names = []
        for field in self.fields:
            if isinstance(field, basestring):
                field_names.append(field)
            else:
                field_names.append(field[0])
        return field_names

    def dict_to_GET(d):
        return ';'.join(['%s=%s' % (k, d[k]) for k in d if d[k]])

    field_vals = {}
    for field in field_names(filterset):
        field_vals[field] = filterset.params.get(field, "")
    filter_GET_params = dict_to_GET(field_vals)
    return filter_GET_params, field_vals


def main_view(request):
    return render_to_response("svweb/main.html",
                              {}, context_instance=RequestContext(request))


#@cache_page(60 * 60)
def fit_list_view(request, runid):
    run = SigvisaCodaFittingRun.objects.get(pk=runid)
    fits = SigvisaCodaFit.objects.filter(runid=runid)
    fits_filter = FitsFilterSet(fits, request.GET)

    mean_acost = np.mean([fit.acost for fit in fits_filter.qs])
    mean_time = np.mean([fit.elapsed for fit in fits_filter.qs])
    total_fits = fits_filter.qs.count

    filter_GET_params, filter_args = filterset_GET_string(fits_filter)
    filter_args['runid'] = runid

    return render_to_response("svweb/fits.html",
                              {'fit_list': fits_filter.qs,
                               'fits_filter': fits_filter,
                               'filter_args': filter_args,
                               'filter_GET_params': filter_GET_params,
                               'runid': runid,
                               'mean_acost': mean_acost,
                               'mean_time': mean_time,
                               'total_fits': total_fits,
                               'run': run,
                               }, context_instance=RequestContext(request))


class FitsFilterSet(FilterSet):
    fields = [
        ('evid', {}, NumericRangeFilter),
        'sta',
        'chan',
        'band',
        'human_approved',
        'tmpl_optim_method',
        'dist',
        'azi',
    ]


# detail view for a particular fit
def fit_detail(request, fitid):

    fit = SigvisaCodaFit.objects.get(fitid=fitid)

    # use a FilterSet to determine links to next/previous fits
    fits = SigvisaCodaFit.objects.filter(runid=fit.runid)
    fits_filter = FitsFilterSet(fits, request.GET)
    filter_GET_params, filter_args = filterset_GET_string(fits_filter)

    next = fits_filter.qs.filter(fitid__gt=fitid)
    next = next[0] if next else False
    prev = fits_filter.qs.filter(fitid__lt=fitid).order_by('-fitid')
    prev = prev[0] if prev else False

    time_format = "%b %d %Y, %H:%M:%S"

    fit_time_str = fit.timestamp.strftime(time_format)

    # get the current display options; initialize to defaults if there are none
    try:
        fit_view_options = view_options.objects.get(id=1)
    except:
        fit_view_options = view_options()
        fit_view_options.id = 1
        fit_view_options.smoothing = 8
        fit_view_options.logscale = False
        fit_view_options.wiggle = False
        fit_view_options.save()

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    # load the waveform so that we can display data about it
    try:
        wave = load_event_station_chan(fit.evid, str(fit.sta), str(fit.chan), cursor=cursor).filter(str(fit.band) + ";env;hz_%.2f" % fit.hz)
        cursor.close()

        wave_stime_str = datetime.fromtimestamp(wave['stime'], timezone('UTC')).strftime(time_format)
        wave_etime_str = datetime.fromtimestamp(wave['etime'], timezone('UTC')).strftime(time_format)

    except Exception as e:
        wave = Waveform()
        wave_stime_str = str(e)
        wave_etime_str = str(e)
        nm = ARModel([], ErrorModel(0, 1), c=0)

    # load the event so that we can display data about it
    try:
        ev = get_event(evid=fit.evid)

        station_location = tuple(s.earthmodel.site_info(str(fit.sta), ev.time)[0:2])
        dist = utils.geog.dist_km((ev.lon, ev.lat), station_location)
        azi = utils.geog.azimuth(station_location, (ev.lon, ev.lat))

        ev_time_str = datetime.fromtimestamp(ev.time, timezone('UTC')).strftime(time_format)
        loc_str = utils.geog.lonlatstr(ev.lon, ev.lat)
    except EventNotFound as e:
        ev = Event()
        dist = None
        azi = None
        loc_str = ""
        ev_time_str = ""

    # get other fits of this same arrival
    other_fits = SigvisaCodaFit.objects.filter(sta=fit.sta, evid=fit.evid, chan=fit.chan, band=fit.band)


    return render_to_response('svweb/detail.html', {
        'fit': fit,
        'fit_time_str': fit_time_str,
        'fit_view_options': fit_view_options,
        'filter_GET_params': filter_GET_params,
        'next': next,
        'prev': prev,
        'wave': wave,
        'wave_stime_str': wave_stime_str,
        'wave_etime_str': wave_etime_str,
        'ev': ev,
        'ev_time_str': ev_time_str,
        'loc_str': loc_str,
        'dist': dist,
        'azi': azi,
        'fits_filter': fits_filter,
        'other_fits': other_fits,
    }, context_instance=RequestContext(request))


def wave_plus_template_view(wave, template, logscale=True, smoothing=0, request=None,
                            ratio=1.6, dpi=144, tmpl_alpha=1, tmpl_width=3):

    fig = Figure(figsize=(ratio*5, 5), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    axes.set_xlabel("Time (s)", fontsize=8)

    if wave is not None:
        plot.subplot_waveform(wave.filter(
            "smooth_%d" % smoothing) if smoothing > 0 else wave, axes, color='black', linewidth=1.5, logscale=logscale)
    if template is not None:
        plot.subplot_waveform(template.filter("smooth_%d" % smoothing), axes, color="green",
                              linewidth=tmpl_width, alpha = tmpl_alpha,
                              logscale=logscale, plot_dets=False)

    if request is not None:
        process_plot_args(request, axes)

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response

def custom_wave_plus_template_view(evid, sta, chan, band, phases, vals, nmid, param_type, **kwargs):
    cursor = Sigvisa().dbconn.cursor()
    wave = load_event_station_chan(int(evid), str(sta), str(chan), cursor=cursor).filter(str(band) + ";env")
    cursor.close()

    sg = setup_sigvisa_graph(evid=evid, wave=wave, phases=phases, vals=vals)
    wave_node = sg.get_wave_node(wave=wave)
    wave_node.set_noise_model(nmid=nmid)
    wave_node.unfix_value()
    wave_node.prior_predict()
    synth_wave = wave_node.get_wave()
    synth_wave.data.mask = wave.data.mask

    return wave_plus_template_view(wave=wave, template=synth_wave, **kwargs)


def phases_from_fit(fit):
    fit_phases = fit.sigvisacodafitphase_set.all()
    fit_params = np.asfarray([(p.arrival_time, p.peak_offset, p.coda_height, p.coda_decay) for p in fit_phases])
    phases = tuple([str(p.phase) for p in fit_phases])
    (phases, vals) = filter_and_sort_template_params(phases, fit_params, filter_list=Sigvisa().phases)
    return phases, vals

@cache_page(60 * 60)
def FitImageView(request, fitid):

    fit = get_object_or_404(SigvisaCodaFit, pk=fitid)

    logscale = request.GET.get("logscale", "False").lower().startswith('t')
    smoothing = int(request.GET.get("smooth", "0"))
    wiggle = request.GET.get("wiggle", "False").lower().startswith('t')
    saveprefs = request.GET.get("saveprefs", "False").lower().startswith('t')

    template_only = request.GET.get("template_only", "False").lower().startswith('t')

    if saveprefs:
        fit_view_options = view_options.objects.get(id=1)
        fit_view_options.smoothing = smoothing
        fit_view_options.logscale = logscale
        fit_view_options.save()

    sg = load_sg_from_db_fit(fit.fitid, load_wiggles=wiggle)
    wave_node = sg.leaf_nodes[0]
    obs_wave = wave_node.get_wave()
    wave_node.unfix_value()
    wave_node.prior_predict()
    pred_wave = wave_node.get_wave()
    pred_wave.data.mask = obs_wave.data.mask

    return wave_plus_template_view(wave=None if template_only else obs_wave, template=pred_wave,
                                   logscale=logscale, smoothing=smoothing, request=request,
                                   tmpl_alpha = 0.8 if wiggle else 1)

def phases_from_request(request):
    i=0
    phases = []
    vals = []
    while request.GET.get('p%d_phase' % i, False):
        phases.append(request.GET.get('p%d_phase' % i, False))
        vals.append([
                float(request.GET.get('p%d_1' % i, False)),
                float(request.GET.get('p%d_2' % i, False)),
                float(request.GET.get('p%d_3' % i, False)),
                float(request.GET.get('p%d_4' % i, False)),
                ])
        i += 1
    vals = np.array(vals)
    return phases, vals

def getstr_from_phases(phases, vals, **kwargs):
    entries = {}
    for i in range(len(phases)):
        entries['p%d_phase' % i] = phases[i]
        entries['p%d_1' % i] = vals[i, 0]
        entries['p%d_2' % i] = vals[i, 1]
        entries['p%d_3' % i] = vals[i, 2]
        entries['p%d_4' % i] = vals[i, 3]
    items = entries.items() + kwargs.items()
    getstr = ";".join(['%s=%s' % (k,v) for (k,v) in items])
    return getstr

def custom_template_view(request, fitid):
    fit = get_object_or_404(SigvisaCodaFit, pk=fitid)

    logscale = request.GET.get("logscale", "off").lower().startswith('on')
    smoothing = int(request.GET.get("smooth", "0"))

    phases, vals = phases_from_request(request)
    (phases, vals) = filter_and_sort_template_params(phases, vals, filter_list=Sigvisa().phases)

    return custom_wave_plus_template_view(sta=fit.sta, evid=fit.evid, chan=fit.chan, band=fit.band,
                                   nmid=fit.nmid.nmid, phases=phases, vals=vals, param_type="paired_exp",
                                   logscale=logscale, smoothing=smoothing,
                                   request=request)

def setup_sigvisa_graph(evid, wave, phases, vals):
    event = get_event(evid=evid)
    sg = SigvisaGraph(phases = phases)
    assert(len(sg.toplevel_nodes) == 0)
    assert(len(sg.leaf_nodes) == 0)
    sg.add_event(event)
    sg.add_wave(wave)
    for (phase, val) in zip(phases, vals):
        tm_node = sg.get_template_node(ev=event, wave=wave, phase=phase)
        tm_node.set_value(val)
    for wm in sg.wiggle_nodes:
        wm.prior_predict()
    return sg

def template_debug_view(request, fitid):
    fit = get_object_or_404(SigvisaCodaFit, pk=fitid)
    logscale = request.GET.get("logscale", "off")
    smoothing = int(request.GET.get("smooth", "8"))

    (phases, vals) = phases_from_request(request)
    if request.GET.get("action", "").startswith("delete"):
        badrow = int(request.GET.get("action", None).split('_')[1])
        phases = phases[:badrow] + phases[badrow+1:]
        vals = np.vstack([vals[:badrow, :], vals[badrow+1:]])


    if len(phases) == 0:
        phases, vals = phases_from_fit(fit)

    cursor = Sigvisa().dbconn.cursor()
    wave = load_event_station_chan(fit.evid, str(fit.sta), str(fit.chan), cursor=cursor).filter(str(fit.band) + ";env" + ';hz_%.2f' % fit.hz)
    cursor.close()

    sg = setup_sigvisa_graph(evid=fit.evid, wave=wave, phases=phases, vals=vals)
    wave_node = sg.get_wave_node(wave=wave)
    wave_node.set_noise_model(nmid=fit.nmid.nmid)

    ll = wave_node.log_p()
    param_ll = sg.current_log_p()
    wave_node.unfix_value()
    wave_node.prior_predict()
    nm = wave_node.nm

    data_path = os.path.join(os.getenv("SIGVISA_HOME"), 'logs', 'web_debug')
    ensure_dir_exists(data_path)
    data_hash = hashlib.sha1(repr(vals) + repr(phases) + repr(fitid)).hexdigest()[:8]
    env_fname = os.path.join(data_path, 'template_debug_%s_env.txt' % data_hash)
    np.savetxt(env_fname, wave.data)
    generated_fname = os.path.join(data_path, 'template_debug_%s_generated.txt' % data_hash)
    np.savetxt(generated_fname, wave_node.get_value())

    if request.GET.get("action", "") == "newrow":
        phases.append("P")
        vals = np.vstack([vals, np.array(((vals[0, 0] + 10, 0, 0, 0),))])

    phase_GET_string = getstr_from_phases(phases, vals, logscale=logscale, smooth=smoothing)

    phase_objs = []
    for (i, phase) in enumerate(phases):
        p = {'phase': phase, 'arrival_time': vals[i, 0], 'peak_offset': vals[i, 1],
             'coda_height': vals[i, 2], 'coda_decay': vals[i, 3]}
        phase_objs.append(p)

    return render_to_response('svweb/template_debug.html', {
        'fitid': fitid,
        'fit': fit,
        'phases': phase_objs,
        'logscale': logscale,
        'smoothing': smoothing,
        'phase_GET_string': phase_GET_string,
        'll': ll,
        'param_ll': param_ll,
        'nm': nm,
        'env_fname': env_fname,
        'generated_fname': generated_fname,
    }, context_instance=RequestContext(request))


def template_residual_view(request, fitid):
    fit = get_object_or_404(SigvisaCodaFit, pk=fitid)

    (phases, vals) = phases_from_request(request)
    if len(phases) == 0:
        phases, vals = phases_from_fit(fit)

    cursor = Sigvisa().dbconn.cursor()
    wave = load_event_station_chan(fit.evid, str(fit.sta), str(fit.chan), cursor=cursor).filter(str(fit.band) + ";env" + ';hz_%.2f' % fit.hz)
    cursor.close()

    sg = setup_sigvisa_graph(evid=fit.evid, wave=wave, phases=phases, vals=vals)
    wave_node = sg.get_wave_node(wave=wave)
    wave_node.set_noise_model(nmid=fit.nmid.nmid)
    wave_node.unfix_value()
    wave_node.prior_predict()

    diff = Waveform(data =wave.data - wave_node.get_value(), segment_stats = wave.segment_stats, my_stats=wave.my_stats)

    return view_wave(request, diff)


def rate_fit(request, fitid):
    fit = SigvisaCodaFit.objects.get(fitid=int(fitid))

    next_fitid = int(request.GET.get("next_fitid", None))

    try:
        rating = int(request.POST['approval'])
    except KeyError:
        return HttpResponse("You didn't select a rating.")
    else:
        fit.human_approved = rating
        fit.save()
        return HttpResponseRedirect(reverse('fit_run_detail', args=(next_fitid,)) + "?" + request.GET.urlencode())
    return HttpResponse("Something went wrong.")

def delete_fit(request, fitid):
    try:
        fit = SigvisaCodaFit.objects.get(fitid=int(fitid))
        fit.delete()
        return HttpResponse("Fit %d deleted. <a href=\"javascript:history.go(-1)\">Go back</a>." % int(fitid))
    except Exception as e:
        return HttpResponse("Error deleting fit %d: %s" % (int(fitid), str(e)))


def delete_run(request, runid):
    try:
        run = SigvisaCodaFittingRun.objects.get(runid=int(runid))
        run.delete()
        return HttpResponse("Run %d deleted. <a href=\"javascript:history.go(-1)\">Go back</a>." % int(runid))
    except Exception as e:
        return HttpResponse("Error deleting run %d: %s" % (int(runid), str(e)))


def fit_cost_quality(request, runid):
    run = SigvisaCodaFittingRun.objects.get(runid=int(runid))

    fig = Figure(figsize=(5, 3), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    fig.suptitle("%s iter %d fit quality" % (run.run_name, run.iter))

    axes.set_xlabel("Acost")
    axes.set_ylabel("mb")

    unknown = request.GET.get("unknown", "True").lower().startswith('t')
    good = request.GET.get("good", "True").lower().startswith('t')
    bad = request.GET.get("bad", "True").lower().startswith('t')

    if good:
        good_fits = run.sigvisacodafit_set.filter(human_approved=2)
        g = np.array([(fit.acost, get_event(evid=fit.evid).mb) for fit in good_fits])
        if good_fits.count() > 0:
            axes.scatter(g[:, 0], g[:, 1], c='g', alpha=0.5, s=10, marker='.', edgecolors='none')

    if bad:
        bad_fits = run.sigvisacodafit_set.filter(human_approved=1)
        b = np.array([(fit.acost, get_event(evid=fit.evid).mb) for fit in bad_fits])
        if bad_fits.count() > 0:
            axes.scatter(b[:, 0], b[:, 1], c='r', alpha=0.5, s=10, marker='.', edgecolors='none')

    if unknown:
        unknown_fits = run.sigvisacodafit_set.filter(human_approved=0)
        u = np.array([(fit.acost, get_event(evid=fit.evid).mb) for fit in unknown_fits])
        if unknown_fits.count() > 0:
            axes.scatter(u[:, 0], u[:, 1], c='b', alpha=0.5, s=10, marker='.', edgecolors='none')

    process_plot_args(request, axes)

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response
