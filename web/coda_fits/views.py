import django
import django.views.generic

import numpy as np
from database.dataset import *
from database.signal_data import *
from signals.io import *
from sigvisa import *
from signals.template_models.load_by_name import load_template_model

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

import plotting.plot as plot

from coda_fits.models import SigvisaCodaFit, SigvisaCodaFitPhase


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

    logscale = request.GET.get("logscale", "False").lower().startswith('t')
    smoothing = int(request.GET.get("smooth", "0"))
    sample = request.GET.get("sample", "False").lower().startswith('t')

    print smoothing

    s = Sigvisa()
    tm = load_template_model("paired_exp", run_name=None, run_iter=0, model_type="dummy")

    fit = SigvisaCodaFit.objects.get(fitid=fitid)

    fit_phases = fit.sigvisacodafitphase_set.all()

    fit_params =np.asfarray([(p.param1, p.param2, p.param3, p.param4) for p in fit_phases])
    phases = tuple([str(p.phase) for p in fit_phases])
    (phases, vals) = filter_and_sort_template_params(phases, fit_params, filter_list=s.phases)

    seg = load_event_station(fit.evid, str(fit.sta), cursor=s.cursor).with_filter("env;"+str(fit.band))
    wave = seg[fit.chan]

    fig = plt.figure(figsize=(5,3), dpi=144)
    fig.patch.set_facecolor('white')
    plt.xlabel("Time (s)")
    synth_wave = tm.generate_template_waveform((phases, vals), wave, sample=sample)
    axes = plt.gca()
    plot.subplot_waveform(wave.filter("smooth_%d" % smoothing) if smoothing > 0 else wave, axes, color='black', linewidth=1.5, logscale=logscale)
    plot.subplot_waveform(synth_wave, axes, color="green", linewidth=3, logscale=logscale, plot_dets=False)
    matplotlib.rcParams.update({'font.size': 8})
    canvas=FigureCanvas(fig)
    response=django.http.HttpResponse(content_type='image/png')
    canvas.print_png(response)
    plt.close(fig)
    return response
