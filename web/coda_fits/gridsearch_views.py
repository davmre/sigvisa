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

from sigvisa import *

from signals.template_models.load_by_name import load_template_model
from learn.extract_wiggles import create_wiggled_phase
from signals.waveform_matching.fourier_features import FourierFeatures
from signals.common import Waveform
from source.event import Event

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import calendar
from pytz import timezone

import plotting.plot as plot
from plotting.event_heatmap import EventHeatmap
import textwrap

from coda_fits.models import SigvisaCodaFit, SigvisaCodaFitPhase, SigvisaCodaFittingRun, SigvisaWiggle, SigvisaGridsearchRun, SigvisaGsrunTModel, SigvisaGsrunWave
from coda_fits.views import process_plot_args, error_wave
from signals.common import load_waveform_from_file
from utils.geog import lonlatstr


def gridsearch_list_view(request):
    runs = SigvisaGridsearchRun.objects.all()
    #run_filter = ModelsFilterSet(models, request.GET)
    return render_to_response("coda_fits/gridsearch_runs.html",
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
    hm = EventHeatmap(f=None, calc=False, n=gs.pts_per_side, lonbounds=[gs.lon_nw, gs.lon_se], latbounds=[gs.lat_nw, gs.lat_se], fname=fullname)
    ev = Event(gs.evid)

    dist = hm.set_true_event(ev.lon, ev.lat)
    true_ev_str = lonlatstr(ev.lon, ev.lat)
    maxlon, maxlat, maxll = hm.max()
    max_ev_str = lonlatstr(maxlon, maxlat)
    

    return render_to_response('coda_fits/gridsearch_detail.html', {
        'gs': gs,
        'nw_str': nw_str,
        'se_str': se_str,
        'max_ev_str': max_ev_str,
        'true_ev_str': true_ev_str,
        'dist': dist,
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

    s = Sigvisa()
    fullname = os.path.join(os.getenv('SIGVISA_HOME'), gs.heatmap_fname)
    hm = EventHeatmap(f=None, calc=False, n=gs.pts_per_side, lonbounds=[gs.lon_nw, gs.lon_se], latbounds=[gs.lat_nw, gs.lat_se], fname=fullname)
    ev = Event(gs.evid)

    sites = [w.sta for w in gs.sigvisagsrunwave_set.all()]
    hm.add_stations(sites)
    hm.set_true_event(ev.lon, ev.lat)

    fig = Figure(figsize=(6,6), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)

    fig.subplots_adjust(bottom=0.05, top=1, left=0, right=0.9)

    hm.plot(axes=axes, colorbar=False)

    canvas=FigureCanvas(fig)
    response=django.http.HttpResponse(content_type='image/png')
#    fig.tight_layout()
    canvas.print_png(response)
    return response
