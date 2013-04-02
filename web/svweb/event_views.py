import numpy as np

from django.shortcuts import render_to_response, get_object_or_404
from django.template import RequestContext
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
from sigvisa.graph.sigvisa_graph import predict_phases
from sigvisa.models.ttime import tt_predict

def event_view(request, evid):

    ss_only = request.GET.get('ss_only', 'true').lower().startswith('t')

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
                site_ll = tuple(s.stations[sta][0:2])
            except KeyError:
                continue

            site_type = 'ar' if s.stations[sta][3] == 1 else 'ss'
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

    hm.add_stations(s.stations.keys())
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
    phases = predict_phases(ev=event, sta=sta, phases="auto")
    phase_atimes = dict([(phase, event.time + tt_predict(event=event, sta=sta, phase=phase)) for phase in phases])

    stime = np.min(phase_atimes.values()) - 10
    etime = np.max(phase_atimes.values()) + 200

    wave = fetch_waveform(station=sta, chan=chan, stime=stime, etime=etime).filter(filter_str)

    return view_wave(request=request, wave=wave, color='black', linewidth=1.0, plot_predictions=phase_atimes)
