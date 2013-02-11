import numpy as np
import sigvisa.utils.geog
from django.shortcuts import render_to_response, get_object_or_404
from django.template import RequestContext
import django

from sigvisa.database.dataset import *
from coda_fits.models import LebOrigin
from sigvisa import Sigvisa

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sigvisa.plotting.event_heatmap import EventHeatmap

def event_view(request, evid):

    ss_only = request.GET.get('ss_only', 'true').lower().startswith('t')

    ev = LebOrigin.objects.get(evid=evid)
    s = Sigvisa()
    cursor = s.dbconn.cursor()

    sql_query = "select iarr.sta, iarr.arid, iarr.time, iarr.deltim, iarr.azimuth, iarr.delaz, iarr.slow, iarr.delslo, iarr.snr, iarr.iphase, iarr.amp, iarr.per from leb_arrival iarr, leb_assoc iass, leb_origin ior where iarr.snr > 0 and iarr.arid=iass.arid and iass.orid=ior.orid and ior.evid=%d order by iarr.snr desc" %  (int(evid),)

    cursor.execute(sql_query)
    rawdets = cursor.fetchall()
    dets = []
    for rd in rawdets:

        try:
            sta = rd[0]
            site_ll = s.stations[sta][0:2]

            site_type = 'ar' if s.stations[sta][3]==1 else 'ss'
            if ss_only and site_type != 'ss':
                continue

            ev_ll = (ev.lon, ev.lat)
            dist = utils.geog.dist_km(site_ll, ev_ll)
            azimuth = utils.geog.azimuth(site_ll, ev_ll)
            phase = rd[DET_PHASE_COL]
            det = {
                'sta': sta, \
                'arr': site_type, \
                'phase': phase, \
                'snr': rd[DET_SNR_COL], \
                'dist': dist, \
                'time': rd[DET_TIME_COL], \
                'amp': rd[DET_AMP_COL], \
                'det_azi': rd[DET_AZI_COL], \
                'true_azi': azimuth, \
                'slo': rd[DET_SLO_COL], \
            }
            dets.append(det)
        except:
            continue
    return render_to_response('coda_fits/event.html', {
        'ev': ev,
        'dets': dets,
    }, context_instance = RequestContext(request))


def event_context_img_view(request, evid):
    ev = LebOrigin.objects.get(evid=evid)

    s = Sigvisa()

    hm = EventHeatmap(f=None, calc=False, center = (ev.lon, ev.lat), width=100)

    hm.add_stations(s.stations.keys())
    hm.set_true_event(ev.lon, ev.lat)

    fig = Figure(figsize=(6,6), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(1,1,1)
    hm.plot(axes=axes, colorbar=False)
    fig.subplots_adjust(bottom=0.05, top=1, left=0, right=0.9)
    canvas=FigureCanvas(fig)
    response=django.http.HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response
