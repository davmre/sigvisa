import django
import django.views.generic

import numpy as np
from database.dataset import *
from database.signal_data import *
from signals.io import *
from sigvisa import *
from plotting.plot_coda_decays import plot_waveform_with_pred

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from coda_fits.models import SigvisaCodaFit


class FitListView(django.views.generic.ListView):
    model = SigvisaCodaFit      # shorthand for setting queryset = models.Car.objects.all()
    template_name = 'coda_fits/fits.html'  # optional (the default is app_name/modelNameInLowerCase_list.html; which will look into your templates folder for that path and file)
    context_object_name = "fit_list"    #default is object_list as well as model's_verbose_name_list and/or model's_verbose_name_plural_list, if defined in the model's inner Meta class
    paginate_by = 50  #and that's it !!

class FitDetailView(django.views.generic.DetailView):
    model = SigvisaCodaFit
    template_name = 'coda_fits/detail.html'
    context_object_name = 'fit'

def FitImageView(request, fitid):

    from signals.template_models.load_by_name import load_template_model

    tm = load_template_model("paired_exp", run_name=None, run_iter=0, model_type="dummy")

    fit = SigvisaCodaFit.objects.get(fitid=fitid)
    s = Sigvisa()

    band_name = s.band_name(low_band = fit.lowband, high_band=fit.highband)
    print "getting band", band_name
    (phases, vals), cost = load_template_params(fit.evid, str(fit.sta), str(fit.chan), band_name, runid=fit.runid.runid)

    seg = load_event_station(fit.evid, str(fit.sta), cursor=s.cursor).with_filter("env;"+band_name)
    wave = seg[fit.chan]

    fig = plot_waveform_with_pred(wave, tm, (phases, vals), logscale=True, title="log scale")
    # plot_waveform_with_pred(pp, wave, tm, (phases, vals), logscale=False, title="linear scale")

    canvas=FigureCanvas(fig)
    response=django.http.HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response
