import numpy as np
import sigvisa.utils.geog
from django.shortcuts import render_to_response, get_object_or_404
from django.template import RequestContext
from django.core.cache import cache
import django
import hashlib

from sigvisa.database.dataset import *
from svweb.models import StaticSiteid
from sigvisa import Sigvisa

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sigvisa.plotting.event_heatmap import EventHeatmap


def site_view(request, sta):

    site = StaticSiteid.objects.get(sta=sta)

    return render_to_response('svweb/site.html', {
        'site': site,
    }, context_instance=RequestContext(request))


def site_det_img_view(request, sta):

    site = StaticSiteid.objects.get(sta=sta)

    s = Sigvisa()

    hm = EventHeatmap(f=None, calc=False, center=(site.lon, site.lat), width=100)

    hm.add_stations([sta, ])

    # get all events arriving at this station in an arbitrary three-month
    # period (which happens to be the first three months of 2007)
    cursor = s.dbconn.cursor()
    sql_query = "select lebo.lon, lebo.lat from leb_origin lebo, leb_assoc leba, leb_arrival l where l.sta='%s' and l.time between 1167631200 and 1175403600 and leba.arid=l.arid and lebo.orid=leba.orid" % sta
    cache_key = hashlib.sha1(sql_query).hexdigest()
    evs = cache.get(cache_key)
    if evs is None:
        cursor.execute(sql_query)
        evs = cursor.fetchall()
        cache.set(cache_key, evs, 60 * 60 * 24 * 365)
    hm.add_events(evs)

    fig = Figure(figsize=(6, 6), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(1, 1, 1)
    hm.plot(axes=axes, event_alpha=0.2, colorbar=False)
    fig.subplots_adjust(bottom=0.05, top=1, left=0, right=0.9)
    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response
