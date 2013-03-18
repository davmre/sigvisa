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
import os
import pickle
from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *

from sigvisa import *

from sigvisa.models.templates.load_by_name import load_template_model
from sigvisa.signals.common import Waveform
from sigvisa.source.event import Event
from sigvisa.signals.io import fetch_waveform, Segment
from sigvisa.graph.sigvisa_graph import predict_phases

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import calendar
from pytz import timezone

import sigvisa.plotting.plot as plot
from sigvisa.plotting.event_heatmap import EventHeatmap
import textwrap
import hashlib

from svweb.models import SigvisaCodaFit, SigvisaCodaFitPhase, SigvisaCodaFittingRun, SigvisaWiggle, SigvisaGridsearchRun, SigvisaGsrunTModel, SigvisaGsrunWave, SigvisaParamModel

from svweb.plotting_utils import process_plot_args, view_wave, bounds_without_outliers
from svweb.views import wave_plus_template_view
from sigvisa.signals.common import load_waveform_from_file
from sigvisa.utils.geog import lonlatstr, dist_km
from sigvisa.infer.gridsearch import propose_origin_times


def gridsearch_list_view(request):
    runs = SigvisaGridsearchRun.objects.all()
    # run_filter = ModelsFilterSet(models, request.GET)
    return render_to_response("svweb/gridsearch_runs.html",
                              {'run_list': runs,
                               #                   'model_filter': model_filter,
                               }, context_instance=RequestContext(request))


def xy_to_query_ll(request, nw_ll, se_ll, nw_xy=(23,17), se_xy=(389,383)):
    args = request.META['QUERY_STRING'].split(',')
    if len(args) < 2:
        return (None, None)
    else:
        x, y = [float(z) for z in args]

        xscale = float(x - nw_xy[0]) / (se_xy[0] - nw_xy[0])
        yscale = float(y - nw_xy[1]) / (se_xy[1] - nw_xy[1])

        if xscale < 0 or xscale > 1 or yscale > 1 or yscale < 0:
            raise ValueError('query point %d,%d is outside of map range' % (x,y))

        se_lon = se_ll[0]
        se_lat = se_ll[1]
        nw_lon = nw_ll[0]
        nw_lat = nw_ll[1]

        if se_lon < nw_lon:
            se_lon += 360
        if se_lat > nw_lat:
            se_lat -= 180

        new_ll = (nw_lon + xscale * (se_lon - nw_lon),
                  nw_lat - yscale * (nw_lat - se_lat))
        return new_ll

# detail view for a particular fit
def gridsearch_detail_view(request, gsid):

    s = Sigvisa()
    gs = get_object_or_404(SigvisaGridsearchRun, pk=gsid)

    nw_str = lonlatstr(gs.lon_nw, gs.lat_nw)
    se_str = lonlatstr(gs.lon_se, gs.lat_se)

    fullname = os.path.join(os.getenv('SIGVISA_HOME'), gs.heatmap_fname, 'overall.txt')
    hm = EventHeatmap(f=None, calc=False, n=gs.pts_per_side, lonbounds=[gs.lon_nw, gs.lon_se], latbounds=[gs.lat_nw,
                      gs.lat_se], fname=fullname)

    ev = Event(gs.evid)
    with open(os.path.join(os.getenv('SIGVISA_HOME'),
                           gs.heatmap_fname,
                           'graph_%.3f_%.3f.pickle' % (ev.lon, ev.lat)),
              'rb') as f:
        true_sg = pickle.load(f)

    hm.set_true_event(ev.lon, ev.lat)
    true_ev_str = lonlatstr(ev.lon, ev.lat)
    maxlon, maxlat, maxll = hm.max()

    (query_lon, query_lat) = xy_to_query_ll(request, nw_ll=(gs.lon_nw, gs.lat_nw), se_ll=(gs.lon_se, gs.lat_se))
    if not query_lon or not query_lat:
        query_lon, query_lat = maxlon, maxlat
    (query_lon, query_lat) = hm.round_point_to_grid(query_lon, query_lat)
    query_ev_str = lonlatstr(query_lon, query_lat)
    dist = dist_km((query_lon, query_lat), (ev.lon, ev.lat))
    with open(os.path.join(os.getenv('SIGVISA_HOME'),
                           gs.heatmap_fname,
                           'graph_%.3f_%.3f.pickle' % (query_lon, query_lat)),
              'rb') as f:
        query_sg = pickle.load(f)
    query_ev = query_sg.toplevel_nodes[0].get_event()
    assert(np.abs(query_ev.lon - query_lon) < 0.001)

    ws = [w for w in gs.sigvisagsrunwave_set.all()]
    wave_dict = dict([("wave_%s_%s_%s_%.1f" % \
                       (w.sta, w.chan, w.band, calendar.timegm(w.stime.timetuple())), w) \
                      for w in ws])
    true_wave_lls = dict([(key, true_sg.get_wave_node_log_p(true_sg.all_nodes[key])) for key in wave_dict.keys()])
    query_wave_lls = dict([(key, query_sg.get_wave_node_log_p(query_sg.all_nodes[key])) for key in wave_dict.keys()])
    ll_diffs = dict([(key, true_wave_lls[key] - query_wave_lls[key]) for key in wave_dict.keys() ])
    true_ll = true_sg.current_log_p()
    query_ll = query_sg.current_log_p()
    wave_phases = dict([(key, predict_phases(ev=true_sg.toplevel_nodes[0].get_event(), sta=w.sta, phases=true_sg.phases)) for (key,w) in wave_dict.items()])

    query_origin_time_proposals = sorted(propose_origin_times(ev=query_sg.toplevel_nodes[0].get_event(),
                                                       segments=[wn.get_wave() for wn in query_sg.leaf_nodes],
                                                       phases=query_sg.phases,
                                                       max_proposals=gs.max_evtime_proposals))

    return render_to_response('svweb/gridsearch_detail.html', {
        'gs': gs,
        'nw_str': nw_str,
        'se_str': se_str,
        'query_ev_str': query_ev_str,
        'true_ev_str': true_ev_str,
        'dist': dist,
        'wave_dict': wave_dict,
        'true_wave_lls': true_wave_lls,
        'query_wave_lls': query_wave_lls,
        'll_diffs': ll_diffs,
        'query_ev': query_ev,
        'true_ev': ev,
        'true_ll': true_ll,
        'query_ll': query_ll,
        'll_diff': true_ll - query_ll,
        'wave_phases': wave_phases,
        'query_origin_time_proposals': query_origin_time_proposals
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


def gs_heatmap_view(request, gsid):
    gs = get_object_or_404(SigvisaGridsearchRun, pk=gsid)

    query_lon = request.GET.get("query_lon", None)
    query_lat = request.GET.get("query_lat", None)
    wave_label = request.GET.get("wave_label", None)
    smooth = request.GET.get("smooth", "true").lower().startswith('t')
    colorbar = request.GET.get("colorbar", "false").lower().startswith('t')
    exp = request.GET.get("exp", "false").lower().startswith('t')
    floor = request.GET.get("floor", "true").lower().startswith('t')

    s = Sigvisa()
    if wave_label:
        fname = "wave_%s.txt" % wave_label
    else:
        fname = "overall.txt"
    fullname = os.path.join(os.getenv('SIGVISA_HOME'), gs.heatmap_fname, fname)
    if not os.path.exists(fullname):
        raise IOError("heatmap file %s does not exist" % fullname)
    hm = EventHeatmap(f=None, calc=False, n=gs.pts_per_side, lonbounds=[gs.lon_nw, gs.lon_se], latbounds=[gs.lat_nw,
                      gs.lat_se], fname=fullname)
    ev = Event(gs.evid)

    sites = [w.sta for w in gs.sigvisagsrunwave_set.all()]
    hm.add_stations(sites)
    hm.set_true_event(ev.lon, ev.lat)

    fig = Figure(figsize=(5.5, 4), dpi=144)
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

    if query_lon and query_lat:
        hm.plot_locations(locations=[(float(query_lon), float(query_lat))], marker='o', ms=7, mec='pink', mew=2)

    fig.subplots_adjust(bottom=0.0, top=1, left=0, right=0.9)

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')

    if colorbar:
        fig.tight_layout()
    canvas.print_png(response)
    return response

def gs_pickled_env_view(request, gswid):

    gs_wave = get_object_or_404(SigvisaGsrunWave, pk=gswid)

    lon = float(request.GET.get("lon", "-999"))
    lat = float(request.GET.get("lat", "-999"))
    assert( lon > -999 and lat > -999 )

    with open(os.path.join(os.getenv('SIGVISA_HOME'),
                           gs_wave.gsid.heatmap_fname,
                           'graph_%.3f_%.3f.pickle' % (lon, lat)),
              'rb') as f:
        sg = pickle.load(f)
    wave_key = "wave_%s_%s_%s_%.1f" % (gs_wave.sta, gs_wave.chan, gs_wave.band, calendar.timegm(gs_wave.stime.timetuple()))
    wave_node = sg.all_nodes[wave_key]
    obs_env = wave_node.get_wave()
    wave_node.unfix_value()
    wave_node.prior_predict()
    template_env = wave_node.get_wave()
    template_env.data.mask = obs_env.data.mask

    return wave_plus_template_view(wave=obs_env, template=template_env, request=request, logscale=False)

def gs_wave_phase_frame_view(request, gswid):
    gs_wave = get_object_or_404(SigvisaGsrunWave, pk=gswid)
    lon = float(request.GET.get("lon", "-999"))
    lat = float(request.GET.get("lat", "-999"))
    assert( lon > -999 and lat > -999 )

    with open(os.path.join(os.getenv('SIGVISA_HOME'),
                           gs_wave.gsid.heatmap_fname,
                           'graph_%.3f_%.3f.pickle' % (lon, lat)),
              'rb') as f:
        sg = pickle.load(f)

    phase_arrivals = [(tm.phase, tm.get_value()[0]) for tm in sg.template_nodes if list(tm.children)[0].sta == gs_wave.sta and list(tm.children)[0].chan == gs_wave.chan and list(tm.children)[0].band == gs_wave.band]
    phase_arrivals = sorted(phase_arrivals, key = lambda x : x[1])


    return render_to_response('svweb/gridsearch_phase_frame.html', {
        'gswid': gswid,
        'phase_arrivals': phase_arrivals,
        'lon': lon,
        'lat': lat,
    }, context_instance=RequestContext(request))
