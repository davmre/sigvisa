import django
import django.views.generic
from django.shortcuts import render_to_response, get_object_or_404
from django.views.decorators.cache import cache_page
from django.core.cache import cache
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
from sigvisa.signals.common import Waveform
from sigvisa.source.event import Event
from sigvisa.signals.io import fetch_waveform, Segment

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import calendar
from pytz import timezone

import sigvisa.plotting.plot as plot
from sigvisa.plotting.event_heatmap import EventHeatmap
import textwrap
import hashlib

from svweb.models import SigvisaCodaFit, SigvisaCodaFitPhase, SigvisaCodaFittingRun, SigvisaWiggle, SigvisaGridsearchRun, SigvisaGsrunTModel, SigvisaGsrunWave, SigvisaTemplateParamModel

from svweb.plotting_utils import process_plot_args, view_wave, bounds_without_outliers
from sigvisa.signals.common import load_waveform_from_file
from sigvisa.utils.geog import lonlatstr
from sigvisa.infer.gridsearch import propose_origin_times, ev_loc_ll_at_optimal_time


def gridsearch_list_view(request):
    runs = SigvisaGridsearchRun.objects.all()
    # run_filter = ModelsFilterSet(models, request.GET)
    return render_to_response("svweb/gridsearch_runs.html",
                              {'run_list': runs,
                               #                   'model_filter': model_filter,
                               }, context_instance=RequestContext(request))


# detail view for a particular fit

def gridsearch_detail_view(request, gsid):

    # get the fit corresponding to the given pageid for this run
    s = Sigvisa()

#    wiggle = get_object_or_404(SigvisaWiggle, pk=wiggleid)
    gs = get_object_or_404(SigvisaGridsearchRun, pk=gsid)

    nw_str = lonlatstr(gs.lon_nw, gs.lat_nw)
    se_str = lonlatstr(gs.lon_se, gs.lat_se)

    fullname = os.path.join(os.getenv('SIGVISA_HOME'), gs.heatmap_fname)
    hm = EventHeatmap(f=None, calc=False, n=gs.pts_per_side, lonbounds=[gs.lon_nw, gs.lon_se], latbounds=[gs.lat_nw,
                      gs.lat_se], fname=fullname)
    ev = Event(gs.evid)

    dist = hm.set_true_event(ev.lon, ev.lat)
    true_ev_str = lonlatstr(ev.lon, ev.lat)
    maxlon, maxlat, maxll = hm.max()
    max_ev_str = lonlatstr(maxlon, maxlat)

    debug = request.GET.get("debug", "false").lower().startswith('t')
    if debug:
        # for each segment, get the likelihoods at the MAP location and the true location, and their difference
        wave_likelihood_tuples = {}

        segs = get_all_segments_for_gsrun(gs)
        ev_true, tm, wm, phases = get_run_stuff(gs)
        em = EnvelopeModel(template_model=tm, wiggle_model=wm, phases=phases)
        maxll, maxt = ev_loc_ll_at_optimal_time(
            Event(lon=maxlon, lat=maxlat, time=-1, depth=ev.depth, mb=ev.mb, natural_source=ev.natural_source),
            segs.values(),
            em.get_method(gs.likelihood_method),
            tm, phases, return_time=True,
            max_proposals=gs.max_evtime_proposals)

        truell, truet = ev_loc_ll_at_optimal_time(
            Event(lon=ev.lon, lat=ev.lat, time=-1, depth=ev.depth, mb=ev.mb, natural_source=ev.natural_source),
            segs.values(),
            em.get_method(gs.likelihood_method),
            tm, phases, return_time=True,
            max_proposals=gs.max_evtime_proposals)

        ev_max_opt = Event(lon=maxlon, lat=maxlat, time=maxt, depth=ev.depth, mb=ev.mb, natural_source=ev.natural_source)
        ev_true_opt = Event(lon=ev.lon, lat=ev.lat, time=truet, depth=ev.depth, mb=ev.mb, natural_source=ev.natural_source)
        f = em.get_method(gs.likelihood_method)
        overall_maxll = 0
        overall_truell = 0
        for gs_wave in gs.sigvisagsrunwave_set.all():
            seg = segs[gs_wave.gswid]
            wave_maxll, maxparams = f(seg, ev_max_opt)
            wave_truell, trueparams = f(seg, ev_true_opt)
            logodds = wave_truell - wave_maxll
            wave_likelihood_tuples[gs_wave.gswid] = (wave_maxll, wave_truell, logodds)
            overall_maxll += wave_maxll
            overall_truell += wave_truell
        overall_logodds = overall_truell - overall_maxll
        overall_ll = {'maxll': overall_maxll, 'truell': overall_truell, 'logodds': overall_logodds, 'maxev': (
            ev_max_opt.lon, ev_max_opt.lat, '%.1f' % ev_max_opt.time), 'trueev': (ev_true_opt.lon, ev_true_opt.lat, '%.1f' % ev_true_opt.time)}
    else:
        overall_ll = ()
        wave_likelihood_tuples = None

    return render_to_response('svweb/gridsearch_detail.html', {
        'gs': gs,
        'nw_str': nw_str,
        'se_str': se_str,
        'max_ev_str': max_ev_str,
        'true_ev_str': true_ev_str,
        'dist': dist,
        'wave_likelihood_tuples': wave_likelihood_tuples,
        'overall_ll': overall_ll,
    }, context_instance=RequestContext(request))


def delete_gsrun(request, gsid):
    try:
        gs = SigvisaGridsearchRun.objects.get(gsid=int(gsid))
        try:
            os.remove(os.path.join(os.getenv('SIGVISA_HOME'), gs.heatmap_fname))
        except OSError:
            pass
        gs.delete()
        return HttpResponse("Gridsearch run %d deleted. <a href=\"javascript:history.go(-1)\">Go back</a>." % int(gsid))
    except Exception as e:
        return HttpResponse("<html><head></head><body>Error deleting gsrun %d: %s %s</body></html>" % (int(gsid), type(e), str(e)))


@cache_page(60 * 60)
def gs_heatmap_view(request, gsid):
    gs = get_object_or_404(SigvisaGridsearchRun, pk=gsid)

    smooth = request.GET.get("smooth", "true").lower().startswith('t')
    colorbar = request.GET.get("colorbar", "false").lower().startswith('t')
    exp = request.GET.get("exp", "false").lower().startswith('t')
    floor = request.GET.get("floor", "true").lower().startswith('t')

    highlight_lon = float(request.GET.get("lon", "-1000"))
    highlight_lat = float(request.GET.get("lat", "-1000"))

    s = Sigvisa()
    fullname = os.path.join(os.getenv('SIGVISA_HOME'), gs.heatmap_fname)
    hm = EventHeatmap(f=None, calc=False, n=gs.pts_per_side, lonbounds=[gs.lon_nw, gs.lon_se], latbounds=[gs.lat_nw,
                      gs.lat_se], fname=fullname)
    ev = Event(gs.evid)

    sites = [w.sta for w in gs.sigvisagsrunwave_set.all()]
    hm.add_stations(sites)
    hm.set_true_event(ev.lon, ev.lat)

    fig = Figure(figsize=(6, 6), dpi=144)
    fig.patch.set_facecolor('white')

    f_exp = lambda x, x_all: np.exp(x - np.max(x_all))
    f_floor = lambda x, x_all: max(x, scipy.stats.scoreatpercentile(x_all, 10))
    preprocess_list = []
    if exp:
        preprocess_list.append(f_exp)
    if floor:
        preprocess_list.append(f_floor)

    axes = fig.add_subplot(1, 1, 1)
    hm.plot(axes=axes, colorbar=colorbar, nolines=smooth, smooth=smooth, f_preprocess=preprocess_list)

    if highlight_lon > -1000 and highlight_lat > -1000:
        hm.plot_locations(locations=[(highlight_lon, highlight_lat)], marker='o', ms=7, mec='pink', mew=2)

    fig.subplots_adjust(bottom=0.05, top=1, left=0, right=0.9)

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')

    if colorbar:
        fig.tight_layout()
    canvas.print_png(response)
    return response


def get_all_segments_for_gsrun(gs):
    segs = dict()
    waves = dict()
    hz = dict()
    for other_wave in gs.sigvisagsrunwave_set.all():
        gswid = other_wave.gswid
        wave = fetch_waveform(station=str(other_wave.sta), chan=str(other_wave.chan), stime=calendar.timegm(
            other_wave.stime.timetuple()), etime=calendar.timegm(other_wave.etime.timetuple()))
        if gswid not in waves:
            waves[gswid] = []
        waves[gswid].append(wave)
        hz[gswid] = other_wave.hz

    for gswid in waves:
        seg = Segment(waves[gswid])
        segs[gswid] = seg.with_filter("env;hz_%f" % hz[gswid])
    return segs


@cache_page(60 * 60)
def gs_debug_view(request, gswid):

    gs_wave = get_object_or_404(SigvisaGsrunWave, pk=gswid)
    (ev_true, sta, chan, band, stime, etime, wave, tm, wm, phases) = get_wave_stuff(gs_wave)

    lon = float(request.GET.get('lon', ev_true.lon))
    lat = float(request.GET.get('lat', ev_true.lat))
    depth = float(request.GET.get('depth', ev_true.depth))

    # propose times based on this waveform
    times = propose_origin_times(
        Event(lon=lon, lat=lat, depth=depth), [wave, ], tm, phases, max_proposals=gs_wave.gsid.max_evtime_proposals)

    # get the globally optimal time with regard to all waveforms
    em = EnvelopeModel(template_model=tm, wiggle_model=wm, phases=phases, bands=(band,), chans=(chan,))
    segs = get_all_segments_for_gsrun(gs_wave.gsid)
    maxll, maxt = ev_loc_ll_at_optimal_time(
        Event(lon=lon, lat=lat, time=-1, depth=depth, mb=ev_true.mb, natural_source=ev_true.natural_source),
        segs.values(),
        em.get_method(gs_wave.gsid.likelihood_method),
        tm, phases, return_time=True, max_proposals=gs_wave.gsid.max_evtime_proposals)
    times.append(maxt)
    print "maxt is", maxt, "over %d segments" % len(segs)

    errors = [ev_true.time - t for t in times]

    all_params = []
    lls = []
    for t in times:
        ev = Event(lon=lon, lat=lat, time=t, depth=depth, mb=ev_true.mb, natural_source=ev_true.natural_source)
        ev.evid = -1
        (pred_template_params, pred_template) = get_wave_ev_stuff(gs_wave, ev)
        all_params.append(zip(phases, pred_template_params))

        wavell = wm.template_ncost(wave, phases, pred_template_params)
        paramll = tm.log_likelihood((phases, pred_template_params), ev, sta, chan, band)
        ll = wavell + paramll
        lls.append((wavell, paramll, ll))

    proposals = zip(times, errors, all_params, lls)

    global_proposal = proposals[-1]
    proposals = proposals[:-1]

    return render_to_response('svweb/gridsearch_debug.html', {
        'wave': gs_wave,
        'lon': lon,
        'lat': lat,
        'depth': depth,
        'proposals': proposals,
        'ev_true': ev_true,
        'gproposal': global_proposal,
    }, context_instance=RequestContext(request))


def get_run_stuff(gs):
    ev_true = Event(evid=gs.evid)
    all_modelids = []
    for wave in gs.sigvisagsrunwave_set.all():
        wave_modelids = [tpm.modelid.modelid for tpm in wave.sigvisagsruntmodel_set.all()]
        all_modelids += wave_modelids

    shape = SigvisaTemplateParamModel.objects.get(modelid=all_modelids[0]).template_shape
    tm = load_template_model(shape, modelids=all_modelids)
    wm = wiggle_model_by_name(name=gs.wiggle_model_type, tm=tm)
    phases = gs.phases.split(',')
    return ev_true, tm, wm, phases


def get_wave_stuff(gs_wave):

    cache_key = "gs_wave_%d_117" % (gs_wave.gswid, )
    r = cache.get(cache_key)
    if r is None:

        sta = str(gs_wave.sta)
        chan = str(gs_wave.chan)
        band = str(gs_wave.band)
        stime = calendar.timegm(gs_wave.stime.timetuple())
        etime = calendar.timegm(gs_wave.etime.timetuple())

        wave = fetch_waveform(sta, chan, stime, etime).filter(band + ";env;hz_%f" % gs_wave.hz)

        ev_true, tm, wm, phases = get_run_stuff(gs_wave.gsid)

        r = (ev_true, sta, chan, band, stime, etime, wave, tm, wm, phases)

        cache.set(cache_key, r, 20)

    return r


def shash(a):
    s = str(a)
    return hashlib.sha1(s).hexdigest()


def get_wave_ev_stuff(gs_wave, ev):
    cache_key = "gs_wave_%d_ev_%s_117" % (gs_wave.gswid, shash(ev))
    # r = cache.get(cache_key)
    r = None
    if r is None:
        (ev_true, sta, chan, band, stime, etime, wave, tm, wm, phases) = get_wave_stuff(gs_wave)
        pred_template_params = tm.predictTemplate(event=ev, sta=sta, chan=chan, band=band, phases=phases)
        pred_template = tm.generate_template_waveform((phases, pred_template_params), model_waveform=wave)
        r = (pred_template_params, pred_template)
        cache.set(cache_key, r, 20)

    return r


def gs_debug_wave_view(request, gswid):

    gs_wave = get_object_or_404(SigvisaGsrunWave, pk=gswid)

    lon = float(request.GET.get('lon', None))
    lat = float(request.GET.get('lat', None))
    depth = float(request.GET.get('depth', 0))
    mb = float(request.GET.get('mb', None))
    time = float(request.GET.get('time', None))
    logscale = request.GET.get("logscale", "False").lower().startswith('t')

    ev = Event(lon=lon, lat=lat, time=time, depth=depth, mb=mb, natural_source=True)
    ev.evid = -1

    (ev_true, sta, chan, band, stime, etime, wave, tm, wm, phases) = get_wave_stuff(gs_wave)
    (pred_template_params, pred_template) = get_wave_ev_stuff(gs_wave, ev)

    fig = Figure(figsize=(5, 3), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    axes.set_xlabel("Time (s)", fontsize=8)
    plot.subplot_waveform(wave, axes, color="black", linewidth=1.5, logscale=logscale)
    plot.subplot_waveform(pred_template, axes, color="green", linewidth=3, logscale=logscale, plot_dets=False)
    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response
