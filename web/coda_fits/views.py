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
from signals.io import *
from sigvisa import *
from signals.template_models.load_by_name import load_template_model
from source.event import Event, EventNotFound
import utils.geog

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from datetime import datetime

import plotting.plot as plot
import textwrap

from coda_fits.models import SigvisaCodaFit, SigvisaCodaFitPhase, SigvisaCodaFittingRun, view_options



class FitListView(django.views.generic.ListView):
    model = SigvisaCodaFit      # shorthand for setting queryset = models.Car.objects.all()
    template_name = 'coda_fits/fits.html'  # optional (the default is app_name/modelNameInLowerCase_list.html; which will look into your templates folder for that path and file)
    context_object_name = "fit_list"    #default is object_list as well as model's_verbose_name_list and/or model's_verbose_name_plural_list, if defined in the model's inner Meta class
    paginate_by = 50  #and that's it !!

    def get_queryset(self):
        a = self.args
        self.run = get_object_or_404(SigvisaCodaFittingRun, pk=self.kwargs['pk'])
        return SigvisaCodaFit.objects.filter(runid=self.run)

def pageid_to_fit(runid, pageid):
    qset = SigvisaCodaFit.objects.filter(runid=runid)
    p = Paginator(list(qset), 1)
    current_fit_page = p.page(pageid)
    fit = current_fit_page[0]
    return fit

# detail view for a particular fit
def fit_detail(request, runid, pageid):

    # get the fit corresponding to the given pageid for this run
    qset = SigvisaCodaFit.objects.filter(runid=runid)
    p = Paginator(list(qset), 1)
    current_fit_page = p.page(pageid)
    fit = current_fit_page[0]

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
        seg = load_event_station(fit.evid, str(fit.sta), cursor=cursor).with_filter("env;"+str(fit.band))
        wave = seg[fit.chan]

        wave_time_str = str(datetime.fromtimestamp(wave['stime']))
    except Exception as e:
        wave = Waveform()
        wave_time_str = str(e)

    # load the event so that we can display data about it
    try:
        ev = Event(evid=fit.evid)

        station_location = s.stations[str(fit.sta)][0:2]
        dist = utils.geog.dist_km((ev.lon, ev.lat), station_location)
        azi = utils.geog.azimuth(station_location, (ev.lon, ev.lat))

        ev_time_str = str(datetime.fromtimestamp(ev.time))
        loc_str = utils.geog.lonlatstr(ev.lon, ev.lat)
    except EventNotFound as e:
        ev = Event()
        dist = None
        azi = None
        loc_str = ""
        ev_time_str = ""

    return render_to_response('coda_fits/detail.html', {
            'fit': fit,
            'fit_view_options': fit_view_options,
            'page_obj': current_fit_page,
            'wave': wave,
            'wave_time_str': wave_time_str,
            'ev': ev,
            'ev_time_str': ev_time_str,
            'loc_str': loc_str,
            'dist': dist,
            'azi': azi,
            }, context_instance=RequestContext(request))

    #
    # SigvisaCodaFit.objects.filter(runid=self.run)


@cache_page(60*60*24*365)
def FitImageView(request, runid, pageid):

    fit = pageid_to_fit(runid, pageid)

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


        seg = load_event_station(fit.evid, str(fit.sta), cursor=cursor).with_filter("env;"+str(fit.band))
        wave = seg[fit.chan]

        fig = plt.figure(figsize=(5,3), dpi=144)
        fig.patch.set_facecolor('white')
        plt.xlabel("Time (s)")
        synth_wave = tm.generate_template_waveform((phases, vals), wave, sample=sample)
        axes = plt.gca()
        plot.subplot_waveform(wave.filter("smooth_%d" % smoothing) if smoothing > 0 else wave, axes, color='black', linewidth=1.5, logscale=logscale)
        plot.subplot_waveform(synth_wave, axes, color="green", linewidth=3, logscale=logscale, plot_dets=False)
        matplotlib.rcParams.update({'font.size': 8})

    except Exception as e:
        error_text = 'Error plotting waveform: \"%s\"' % str(e)
        fig = plt.figure(figsize=(5,3), dpi=144)
        fig.patch.set_facecolor('white')
        axes = plt.gca()
        plt.text(.5, .5, "\n".join(textwrap.wrap(error_text, 60)), horizontalalignment='center', verticalalignment='center', transform = axes.transAxes)
        matplotlib.rcParams.update({'font.size': 8})

    canvas=FigureCanvas(fig)
    response=django.http.HttpResponse(content_type='image/png')
    plt.tight_layout()
    canvas.print_png(response)
    plt.close(fig)
    return response

def rate_fit(request, runid, pageid):
    fit = pageid_to_fit(runid, pageid)
    try:
        rating = int(request.POST['approval'])
    except KeyError:
        return render_to_response('coda_fits/detail.html', {
                'fit': fit,
                'error_message': "You didn't select a rating.",
                }, context_instance=RequestContext(request))
    else:
        fit.human_approved = rating
        fit.save()
        return HttpResponseRedirect(reverse('fit_run_detail', args=(runid, unicode(int(pageid)+1),)))
    return HttpResponse("Something went wrong.")
