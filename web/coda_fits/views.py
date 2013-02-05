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
from database.dataset import *
from database.signal_data import *
from signals.io import *
from sigvisa import *
from noise.noise_model import get_noise_model
from signals.template_models.load_by_name import load_template_model
from source.event import get_event, EventNotFound
from signals.armodel.model import ARModel, ErrorModel
import utils.geog

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datetime import datetime
from pytz import timezone
import hashlib

import plotting.plot as plot
import plotting.histogram as histogram
import textwrap

from coda_fits.models import SigvisaCodaFit, SigvisaCodaFitPhase, SigvisaCodaFittingRun, view_options


def process_plot_args(request, axes):
    xmin = request.GET.get("xmin", "auto")
    xmax = request.GET.get("xmax", "auto")
    ymin = request.GET.get("ymin", "auto")
    ymax = request.GET.get("ymax", "auto")

    if xmin != "auto" and xmax != "auto":
        axes.set_xlim(float(xmin), float(xmax))
    if ymin != "auto" and ymax != "auto":
        axes.set_ylim(float(ymin), float(ymax))

    if xmin == "auto" and xmax == "auto":
        xmin, xmax = axes.get_xlim()
        xlen = float(request.GET.get('len', "-1"))
        if xlen > 0:
            axes.set_xlim(xmin, xmin + xlen)

def error_wave(exception):
    error_text = 'Error plotting waveform: \"%s\"' % str(exception)
    fig = Figure(figsize=(5,3), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    axes.set_xlabel("Time (s)", fontsize=8)
    axes.text(.5, .5, "\n".join(textwrap.wrap(error_text, 60)), horizontalalignment='center', verticalalignment='center', transform = axes.transAxes, fontsize=8)
    canvas=FigureCanvas(fig)
    response=django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response

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


def fit_list_view(request, runid):
    run = SigvisaCodaFittingRun.objects.get(pk=runid)
    fits = SigvisaCodaFit.objects.filter(runid=runid)
    fits_filter = FitsFilterSet(fits, request.GET)

    mean_acost = np.mean([fit.acost for fit in fits_filter.qs])
    mean_time = np.mean([fit.elapsed for fit in fits_filter.qs])
    total_fits = fits_filter.qs.count

    filter_GET_params, filter_args = filterset_GET_string(fits_filter)
    filter_args['runid']=runid

    return render_to_response("coda_fits/fits.html",
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
        'optim_method',
        ]



# detail view for a particular fit
def fit_detail(request, fitid):

    fit = SigvisaCodaFit.objects.get(fitid=fitid)

    # use a FilterSet to determine links to next/previous fits
    fits = SigvisaCodaFit.objects.filter(runid=fit.runid)
    fits_filter = FitsFilterSet(fits, request.GET)
    filter_GET_params, filter_args = filterset_GET_string(fits_filter)

    next = fits_filter.qs.filter(fitid__gt = fitid)
    next = next[0] if next else False
    prev = fits_filter.qs.filter(fitid__lt = fitid)
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
        fit_view_options.sample = False
        fit_view_options.save()


    s = Sigvisa()
    cursor = s.dbconn.cursor()

    # load the waveform so that we can display data about it
    try:
        wave = load_event_station_chan(fit.evid, str(fit.sta), str(fit.chan), cursor=cursor).filter(str(fit.band)+";env")

        wave_stime_str = datetime.fromtimestamp(wave['stime'], timezone('UTC')).strftime(time_format)
        wave_etime_str = datetime.fromtimestamp(wave['etime'], timezone('UTC')).strftime(time_format)
        nm = get_noise_model(waveform=wave)

    except Exception as e:
        wave = Waveform()
        wave_stime_str = str(e)
        wave_etime_str = str(e)
        nm = ARModel([], ErrorModel(0, 1), c=0)

    # load the event so that we can display data about it
    try:
        ev = get_event(evid=fit.evid)

        station_location = s.stations[str(fit.sta)][0:2]
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

    return render_to_response('coda_fits/detail.html', {
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
            'noise_model': nm,
            'fits_filter': fits_filter,
            }, context_instance=RequestContext(request))



@cache_page(60*60*24*365)
def FitImageView(request, fitid):

    fit = get_object_or_404(SigvisaCodaFit, pk=fitid)

    logscale = request.GET.get("logscale", "False").lower().startswith('t')
    smoothing = int(request.GET.get("smooth", "0"))
    sample = request.GET.get("sample", "False").lower().startswith('t')
    saveprefs = request.GET.get("saveprefs", "False").lower().startswith('t')

    if saveprefs:
        fit_view_options = view_options.objects.get(id=1)
        fit_view_options.smoothing = smoothing
        fit_view_options.logscale = logscale
        fit_view_options.save()


    s = Sigvisa()
    cursor = s.dbconn.cursor()
    tm = load_template_model("paired_exp", run_name=None, run_iter=0, model_type="dummy")

    fit_phases = fit.sigvisacodafitphase_set.all()

    try:
        fit_params =np.asfarray([(p.param1, p.param2, p.param3, p.param4) for p in fit_phases])
        phases = tuple([str(p.phase) for p in fit_phases])
        (phases, vals) = filter_and_sort_template_params(phases, fit_params, filter_list=s.phases)


        wave = load_event_station_chan(fit.evid, str(fit.sta), str(fit.chan), cursor=cursor).filter(str(fit.band) + ";env")

        fig = Figure(figsize=(5,3), dpi=144)
        fig.patch.set_facecolor('white')
        axes = fig.add_subplot(111)
        axes.set_xlabel("Time (s)", fontsize=8)
        synth_wave = tm.generate_template_waveform((phases, vals), wave, sample=sample)
        plot.subplot_waveform(wave.filter("smooth_%d" % smoothing) if smoothing > 0 else wave, axes, color='black', linewidth=1.5, logscale=logscale)
        plot.subplot_waveform(synth_wave, axes, color="green", linewidth=3, logscale=logscale, plot_dets=False)
        process_plot_args(request, axes)

    except Exception as e:
        return error_wave(e)

    canvas=FigureCanvas(fig)
    response=django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response

def rate_fit(request, fitid):
    fit = SigvisaCodaFit.objects.get(fitid=int(fitid))

    try:
        rating = int(request.POST['approval'])
    except KeyError:
        return HttpResponse("You didn't select a rating.")
    else:
        fit.human_approved = rating
        fit.save()
        filter_args['pageid']=int(pageid)+1
        return HttpResponseRedirect(reverse('fit_run_detail', kwargs=filter_args))
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

    fig = Figure(figsize=(5,3), dpi=144)
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
            axes.scatter(g[:,0], g[:, 1], c='g', alpha=0.5)

    if bad:
        bad_fits = run.sigvisacodafit_set.filter(human_approved=1)
        b = np.array([(fit.acost, get_event(evid=fit.evid).mb) for fit in bad_fits])
        if bad_fits.count() > 0:
            axes.scatter(b[:,0], b[:, 1], c='r', alpha=0.5)

    if unknown:
        unknown_fits = run.sigvisacodafit_set.filter(human_approved=0)
        u = np.array([(fit.acost, get_event(evid=fit.evid).mb) for fit in unknown_fits])
        if unknown_fits.count() > 0:
            axes.scatter(u[:,0], u[:, 1], c='b', alpha=0.5)

    process_plot_args(request, axes)


    canvas=FigureCanvas(fig)
    response=django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response
