import django
import django.views.generic
from django.shortcuts import render_to_response, get_object_or_404
from django.views.decorators.cache import cache_page
from django.core.cache import cache
from django.template import RequestContext
from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.core.paginator import Paginator
from django_easyfilters import FilterSet

import numpy as np
import sys
from database.dataset import *
from database.signal_data import *
from sigvisa import *
from signals.template_models.load_by_name import load_template_model
from learn.train_coda_models import load_model


from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datetime import datetime
from pytz import timezone
import hashlib

import plotting.plot as plot
import plotting.histogram as histogram
import textwrap

from coda_fits.models import SigvisaCodaFit, SigvisaCodaFitPhase, SigvisaCodaFittingRun, SigvisaTemplateParamModel
from coda_fits.views import filterset_GET_string, process_plot_args, FitsFilterSet

class ModelsFilterSet(FilterSet):
    fields = [
        'template_shape',
        'param',
        'site',
        'chan',
        'band',
        'phase',
        'model_type',
        ]

def model_list_view(request):
    models = SigvisaTemplateParamModel.objects.all()
    model_filter = ModelsFilterSet(models, request.GET)
    return render_to_response("coda_fits/models.html",
                  {'model_list': model_filter.qs,
                   'model_filter': model_filter,
                   }, context_instance=RequestContext(request))


def get_all_params(fit_qset, phase, template_shape):

    cachekey = hashlib.sha1(str(fit_qset.count) + str(fit_qset.query) + str(phase) + str(template_shape)).hexdigest()
    print cachekey
    print fit_qset.query
    x = cache.get(cachekey)
    if x is None:
        x = []
        i = 0
        for fit in fit_qset:
            p = fit.sigvisacodafitphase_set.filter(phase=phase, template_model=template_shape)
            if len(p) != 1: continue
            else: p = p[0]
            x.append((p.param1, p.param2, p.param3, p.param4, fit.dist, fit.azi, fit.acost, p.amp_transfer, int(fit.human_approved)))
            i += 1
            if i % 100 == 0:
                print i
        x = np.array(x)

        cache.set(cachekey, x, 60*60*24*365)

    return x



def plot_empirical_distance(request, qset, phases, axes, param_idx, max_acost, min_amp, template_shape, **kwargs):

    min_azi = float(request.GET.get("min_azi", "0"))
    max_azi = float(request.GET.get("max_azi", "360"))
    min_depth = float(request.GET.get("min_depth", "0"))
    max_depth = float(request.GET.get("max_depth", "10000"))

    x = {}
    # for each phase, get all (distance, param) pairs
    for phase in phases:
        x[phase] = np.array([(z[4], z[param_idx]) for z in get_all_params(qset, phase, template_shape=template_shape) if z[2] > min_amp and z[6] < max_acost and z[5] > min_azi and z[5] < max_azi])
        print len(x[phase])

    colors = ['b', 'r', 'g', 'y']

    xmin = np.float('-inf')
    xmax = np.float('inf')
    for (i, phase) in enumerate(phases):
        axes.scatter(x[phase][:, 0], x[phase][:, 1], alpha=1 / np.log(x[phase].shape[0]), c=colors[i], s=10, marker='.', edgecolors="none")
        xmin = max(xmin, np.min(x[phase][:, 0]))
        xmax = min(xmax, np.max(x[phase][:, 0]))
    r = xmax-xmin
    axes.set_xlim([xmin-r/20.0, xmax+r/20.0])

def plot_linear_model_distance(request, model_record, axes):

    full_fname = os.path.join(os.getenv("SIGVISA_HOME"), model_record.model_fname)
    model = load_model(full_fname, model_record.model_type)

    distances = np.linspace(0, 20000, 200)
    pred = np.array([ model.predict(np.array(((0, 0, 0, d, 0),))) for d in distances]).flatten()
    axes.plot(distances, pred, 'k-')

    std = np.array([model.std((0, 0, 0, d, 0)) for d in distances])
    var_x = np.concatenate((distances, distances[::-1]))
    var_y = np.concatenate((pred + 2*std, (pred - 2*std)[::-1]))
    axes.fill(var_x, var_y, edgecolor='w', facecolor='#d3d3d3', alpha=0.1)


def plot_gp_model_distance(request, model_record, axes):

    full_fname = os.path.join(os.getenv("SIGVISA_HOME"), model_record.model_fname)
    model = load_model(full_fname, model_record.model_type)

    # TODO : use specified azimuth and depth ranges

    distances = np.linspace(0, 20000, 200)
    pred = np.array([ model.predict(np.array(((0, 0, 0, d, 0),))) for d in distances]).flatten()
    axes.plot(distances, pred, 'k-')

    std = np.array([np.sqrt(model.variance(np.array(((0, 0, 0, d, 0),)))) for d in distances])
    var_x = np.concatenate((distances, distances[::-1]))
    var_y = np.concatenate((pred + 2*std, (pred - 2*std)[::-1]))
    axes.fill(var_x, var_y, edgecolor='w', facecolor='#d3d3d3', alpha=0.1)


def plot_gaussian(request, model_record, axes):

    full_fname = os.path.join(os.getenv("SIGVISA_HOME"), model_record.model_fname)
    model = load_model(full_fname, model_record.model_type)
    xmin = model.mean - 4*model.std
    xmax = model.mean + 4*model.std
    x = np.linspace(xmin, xmax, 200)
    pdf = scipy.stats.norm.pdf(x, loc=model.mean, scale=model.std)
    axes.plot(x, pdf, 'k-')

def plot_fit_param(request, modelid=None, plot_type="histogram", **kwargs):
    fig = Figure(figsize=(5,3), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    param_names = ["arr_time", "peak_offset", "coda_height", "coda_decay", "dist", "azi", "cost", "amp_transfer", "human_approved"]

    d = kwargs

    if modelid is not None:
        model = SigvisaTemplateParamModel.objects.get(modelid=modelid)
        d['runid'] = model.fitting_runid.runid
        d['sta'] = model.site
        d['chan'] = model.chan
        d['band'] = model.band
        d['phases'] = [model.phase,]
        param = model.param
        d['template_shape'] = model.template_shape
        d['min_amp'] = model.min_amp
        d['max_acost'] = model.max_acost
        d['fit_quality'] = "all" if model.require_human_approved!='t' else "2"

        if plot_type == "histogram":
            plot_gaussian(request, model, axes=axes)
        elif plot_type == "distance":
            if model.model_type=="linear_distance":
                plot_linear_model_distance(request, model_record=model, axes=axes)
            elif model.model_type[:2]=="gp":
                plot_gp_model_distance(request, model_record=model, axes=axes)

    else:
        param = request.GET.get("plot_param", "coda_decay")
        d['max_acost'] = float(request.GET.get("max_acost", "inf"))
        d['phases'] = request.GET.get("phases", "P").split(',')
        d['min_amp'] = float(request.GET.get("min_amp", "-inf"))
        d['template_shape'] = request.GET.get("shape", "paired_exp")

    if 'runid' in d:
        d['param_idx'] = param_names.index(param)


        fits = SigvisaCodaFit.objects.filter(runid=int(d['runid']))
        fits_filter = FitsFilterSet(fits, request.GET)


        if plot_type == "histogram":
            plot_empirical_histogram(request=request, qset=fits_filter.qs, axes=axes, **d)
        elif plot_type == "distance":
            plot_empirical_distance(request=request, qset=fits_filter.qs, axes=axes, **d)

    if plot_type == "histogram":
        axes.set_xlabel(param, fontsize=8)
    elif plot_type == "distance":
        axes.set_xlabel("distance (km)", fontsize=8)
        axes.set_ylabel(param, fontsize=8)

        default_bounds = {'coda_decay': [-0.05,0], 'amp_transfer': [-7, 10], 'peak_offset': [0,25]}
        if param in default_bounds:
            axes.set_ylim(default_bounds[param])

    process_plot_args(request, axes)

    canvas=FigureCanvas(fig)
    response=django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response



def plot_empirical_histogram(request, qset, phases, axes, param_idx, max_acost, min_amp, template_shape, **kwargs):

    # get the fit corresponding to the given pageid for this run

    min_azi = float(request.GET.get("min_azi", "0"))
    max_azi = float(request.GET.get("max_azi", "360"))

    pstr = "p[0].param%d" % param_idx

    x = {}
    for phase in phases:
        x[phase] = [z[param_idx] for z in get_all_params(qset, phase, template_shape=template_shape) if z[2] > min_amp and z[6] < max_acost  and z[5] > min_azi and z[5] < max_azi]
        print "got", len(x[phase]), "params"


    colors = ['b', 'r', 'g', 'y']

    for (i, phase) in enumerate(phases):
        histogram.plot_histogram(x[phase], axes=axes, normed=True)



def model_density(request, modelid):
    return plot_fit_param(request, modelid=modelid, plot_type = "histogram")

def model_distance_plot(request, modelid):
    return plot_fit_param(request, modelid=modelid, plot_type = "distance")

def model_heatmap(request, modelid):
    pass

def data_distance_plot(request, **kwargs):
    return plot_fit_param(request, plot_type = "distance", **kwargs)

def data_histogram_plot(request, **kwargs):
    return plot_fit_param(request, plot_type = "histogram", **kwargs)
