import numpy as np

from django.shortcuts import render_to_response, get_object_or_404
from django.template import RequestContext
from django.core.cache import cache
from django.views.decorators.cache import cache_page
import django


from svweb.models import LebOrigin
from svweb.plotting_utils import view_wave

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from sigvisa.plotting.event_heatmap import EventHeatmap
from sigvisa import Sigvisa
from sigvisa.database.dataset import *
import sigvisa.utils.geog as geog
from sigvisa.signals.io import fetch_waveform
from sigvisa.source.event import get_event
from sigvisa.graph.sigvisa_graph import predict_phases_sta
from sigvisa.models.ttime import tt_predict
import hashlib

def event_view(request, evid):

    ss_only = request.GET.get('ss_only', 'false').lower().startswith('t')

    ev = LebOrigin.objects.get(evid=evid)
    s = Sigvisa()
    cursor = s.dbconn.cursor()

    sql_query = "select iarr.sta, iarr.arid, iarr.time, iarr.deltim, iarr.azimuth, iarr.delaz, iarr.slow, iarr.delslo, iarr.snr, iarr.iphase, iarr.amp, iarr.per from leb_arrival iarr, leb_assoc iass, leb_origin ior where iarr.snr > 0 and iarr.arid=iass.arid and iass.orid=ior.orid and ior.evid=%d order by iarr.snr desc" % (
        int(evid),)

    cursor.execute(sql_query)
    rawdets = cursor.fetchall()

    dets = []
    for rd in rawdets:

        try:
            sta = rd[0]

            try:
                a = s.earthmodel.site_info(sta, 0)
                site_ll = tuple(a[0:2])
            except KeyError:
                continue

            site_type = 'ar' if s.earthmodel.site_info(sta, 0)[3] == 1 else 'ss'
            if ss_only and site_type != 'ss':
                continue

            ev_ll = (ev.lon, ev.lat)
            dist = geog.dist_km(site_ll, ev_ll)
            azimuth = geog.azimuth(site_ll, ev_ll)
            phase = rd[DET_PHASE_COL]
            det = {
                'sta': sta,
                'arr': site_type,
                'phase': phase,
                'snr': rd[DET_SNR_COL],
                'dist': dist,
                'time': rd[DET_TIME_COL],
                'amp': rd[DET_AMP_COL],
                'det_azi': rd[DET_AZI_COL],
                'true_azi': azimuth,
                'slo': rd[DET_SLO_COL],
            }
            dets.append(det)
        except:
            raise
            continue


    return render_to_response('svweb/event.html', {
        'ev': ev,
        'dets': dets,
    }, context_instance=RequestContext(request))


def event_context_img_view(request, evid):
    ev = LebOrigin.objects.get(evid=evid)

    s = Sigvisa()

    hm = EventHeatmap(f=None, calc=False, center=(ev.lon, ev.lat), width=100)

    hm.add_stations(s.sitenames)
    hm.set_true_event(ev.lon, ev.lat)

    fig = Figure(figsize=(6, 6), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(1, 1, 1)
    hm.plot(axes=axes, colorbar=False, offmap_station_arrows=False)
    fig.subplots_adjust(bottom=0.05, top=1, left=0, right=0.9)
    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response

def event_wave_view(request, evid):
    sta = str(request.GET.get('sta', None))
    chan = str(request.GET.get('chan', None))
    filter_str = str(request.GET.get('filter_str', None))
    ev = LebOrigin.objects.get(evid=evid)

    event = get_event(evid=ev.evid)
    phases = predict_phases_sta(ev=event, sta=sta, phases="auto")
    phase_atimes = dict([(phase, event.time + tt_predict(event=event, sta=sta, phase=phase)) for phase in phases])

    stime = np.min(phase_atimes.values()) - 10
    etime = np.max(phase_atimes.values()) + 200

    wave = fetch_waveform(station=sta, chan=chan, stime=stime, etime=etime).filter(filter_str)

    return view_wave(request=request, wave=wave, color='black', linewidth=1.0, plot_predictions=phase_atimes)


def events_in_region(left_lon, right_lon, top_lat, bottom_lat, start_time, end_time, min_mb, max_mb, min_depth, max_depth, detected_at_station=None):
    s = Sigvisa()

    # get all events arriving at this station in an arbitrary three-month
    # period (which happens to be March-May 2009)
    cursor = s.dbconn.cursor()
    if left_lon < right_lon:
        lon_cond = "lebo.lon between %f and %f" % (left_lon, right_lon)
    else:
        lon_cond = "( (lon > %f) or (lon < %f) )" % (left_lon, right_lon)
    lat_cond = "lebo.lat between %f and %f" % (bottom_lat, top_lat)

    if not detected_at_station:
        sql_query = "select lebo.lon, lebo.lat, lebo.depth, lebo.time, lebo.mb, lebo.evid  from leb_origin lebo where lebo.time between %f and %f and %s and %s and lebo.mb between %f and %f and lebo.depth between %f and %f" % (start_time, end_time, lon_cond, lat_cond, min_mb, max_mb, min_depth, max_depth)
    else:
        sql_query = "select distinct lebo.lon, lebo.lat, lebo.depth, lebo.time, lebo.mb, lebo.evid from leb_origin lebo, leb_assoc leba, leb_arrival l where lebo.time between %f and %f and %s and %s and lebo.mb between %f and %f and lebo.depth between %f and %f and l.sta='%s' and l.arid=leba.arid and leba.orid=lebo.orid" % (start_time, end_time, lon_cond, lat_cond, min_mb, max_mb, min_depth, max_depth, detected_at_station)
    cache_key = hashlib.sha1(sql_query).hexdigest()
    evs = cache.get(cache_key)
    if not evs:
        cursor.execute(sql_query)
        evs = cursor.fetchall()
        cache.set(cache_key, evs, 60 * 60 * 24 * 365)
    cursor.close()

    return evs

def regional_event_view(request):
    left_lon = float(request.GET.get('left_lon', -180))
    right_lon = float(request.GET.get('right_lon', 180))
    top_lat = float(request.GET.get('top_lat', 90))
    bottom_lat = float(request.GET.get('bottom_lat', -90))
    start_time = float(request.GET.get('start_time', 1238889600))
    end_time = float(request.GET.get('end_time', 1245456000))
    min_mb = float(request.GET.get('min_mb', 2.5))
    max_mb = float(request.GET.get('max_mb', 99))
    min_depth = float(request.GET.get('min_depth', 0))
    max_depth = float(request.GET.get('max_depth', 999))
    detected_at_station = str(request.GET.get('detected_at_station', ''))
    evs = events_in_region(left_lon, right_lon, top_lat, bottom_lat, start_time, end_time, min_mb, max_mb, min_depth, max_depth, detected_at_station=detected_at_station)

    print left_lon, right_lon, top_lat

    return render_to_response('svweb/event_region.html', {
        'evs': evs,
        'left_lon': left_lon,
        'right_lon': right_lon,
        'top_lat': top_lat,
        'bottom_lat': bottom_lat,
        'start_time': start_time,
        'end_time': end_time,
        'min_mb': min_mb,
        'max_mb': max_mb,
        'min_depth': min_depth,
        'max_depth': max_depth,
        'detected_at_station': detected_at_station,
        'n_evs': len(evs),
    }, context_instance=RequestContext(request))


@cache_page(60 * 60)
def regional_event_image_view(request):
    left_lon = float(request.GET.get('left_lon', -180))
    right_lon = float(request.GET.get('right_lon', 180))
    top_lat = float(request.GET.get('top_lat', 90))
    bottom_lat = float(request.GET.get('bottom_lat', -90))
    start_time = float(request.GET.get('start_time', 1238889600))
    end_time = float(request.GET.get('end_time', 1245456000))
    min_mb = float(request.GET.get('min_mb', 2.5))
    max_mb = float(request.GET.get('max_mb', 99))
    min_depth = float(request.GET.get('min_depth', 0))
    max_depth = float(request.GET.get('max_depth', 999))
    detected_at_station = str(request.GET.get('detected_at_station', ''))

    s = Sigvisa()

    hm = EventHeatmap(f=None, calc=False, top_lat=top_lat, bottom_lat=bottom_lat, left_lon=left_lon, right_lon=right_lon)
    hm.add_stations(s.sitenames)
    evs = events_in_region(left_lon, right_lon, top_lat, bottom_lat, start_time, end_time, min_mb, max_mb, min_depth, max_depth, detected_at_station=detected_at_station)
    hm.add_events(evs)

    fig = Figure(dpi=288)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(1, 1, 1)
    hm.plot(axes=axes, event_alpha=0.2, colorbar=False, offmap_station_arrows=False, label_stations=False)
    fig.subplots_adjust(bottom=0.05, top=1, left=0, right=0.9)
    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    canvas.print_png(response, bbox_inches="tight")
    return response
