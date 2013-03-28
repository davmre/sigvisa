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
from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *
from sigvisa import *
from sigvisa.models.templates.load_by_name import load_template_model
from sigvisa.models.wiggles import load_wiggle_node
from sigvisa.learn.train_param_common import load_model
from sigvisa.learn.train_coda_models import  get_shape_training_data
from sigvisa.learn.train_wiggle_models import  get_wiggle_training_data


from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datetime import datetime
from pytz import timezone
import hashlib

import sigvisa.plotting.plot as plot
import sigvisa.plotting.histogram as histogram
import textwrap

from svweb.models import SigvisaCodaFit, SigvisaCodaFitPhase, SigvisaCodaFittingRun, SigvisaParamModel
from svweb.views import filterset_GET_string, process_plot_args, FitsFilterSet


class ModelsFilterSet(FilterSet):
    fields = [
        'template_shape',
        'fitting_runid',
        'param',
        'site',
        'chan',
        'band',
        'phase',
        'model_type',
    ]


def model_list_view(request):
    models = SigvisaParamModel.objects.all()
    model_filter = ModelsFilterSet(models, request.GET)
    return render_to_response("svweb/models.html",
                              {'model_list': model_filter.qs,
                               'model_filter': model_filter,
                               }, context_instance=RequestContext(request))


def plot_empirical_distance(request, xy_by_phase, axes):

    colors = ['b', 'r', 'g', 'y']

    xmin = np.float('-inf')
    xmax = np.float('inf')
    for (i, phase) in enumerate(sorted(xy_by_phase.keys())):
        X, y = xy_by_phase[phase]
        axes.scatter(X[:, 3], y, alpha=1 / np.log(len(y)), c=colors[i], s=10, marker='.', edgecolors="none")
        xmin = max(xmin, np.min(X[:, 3]))
        xmax = min(xmax, np.max(X[:, 3]))
    r = xmax - xmin
    axes.set_xlim([xmin - r / 20.0, xmax + r / 20.0])


def plot_linear_model_distance(request, model_record, axes):

    full_fname = os.path.join(os.getenv("SIGVISA_HOME"), model_record.model_fname)
    model = load_model(full_fname, model_record.model_type)

    distances = np.linspace(0, 20000, 200)
    pred = np.array([model.predict(np.array(((0, 0, 0, d, 0),))) for d in distances]).flatten()
    axes.plot(distances, pred, 'k-')

    std = np.array([model.std((0, 0, 0, d, 0)) for d in distances])
    var_x = np.concatenate((distances, distances[::-1]))
    var_y = np.concatenate((pred + 2 * std, (pred - 2 * std)[::-1]))
    axes.fill(var_x, var_y, edgecolor='w', facecolor='#d3d3d3', alpha=0.1)


def plot_gp_model_distance(request, model_record, axes):

    full_fname = os.path.join(os.getenv("SIGVISA_HOME"), model_record.model_fname)
    model = load_model(full_fname, model_record.model_type)

    # TODO : use specified azimuth and depth ranges

    distances = np.linspace(0, 20000, 200)
    pred = np.array([model.predict(np.array(((0, 0, 0, d, 0),))) for d in distances]).flatten()
    axes.plot(distances, pred, 'k-')

    std = np.array([np.sqrt(model.variance(np.array(((0, 0, 0, d, 0),)))) for d in distances])
    var_x = np.concatenate((distances, distances[::-1]))
    var_y = np.concatenate((pred + 2 * std, (pred - 2 * std)[::-1]))
    axes.fill(var_x, var_y, edgecolor='w', facecolor='#d3d3d3', alpha=0.1)


def plot_gaussian(request, model_record, axes):

    full_fname = os.path.join(os.getenv("SIGVISA_HOME"), model_record.model_fname)
    model = load_model(full_fname, model_record.model_type)
    xmin = model.mean - 4 * model.std
    xmax = model.mean + 4 * model.std
    x = np.linspace(xmin, xmax, 200)
    pdf = scipy.stats.norm.pdf(x, loc=model.mean, scale=model.std)
    axes.plot(x, pdf, 'k-')


def plot_fit_param(request, modelid=None, runid=None, plot_type="histogram"):
    fig = Figure(figsize=(5, 3), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)

    d = {}

    if modelid is not None:
        model = SigvisaParamModel.objects.get(modelid=modelid)
        runid = model.fitting_runid.runid
        sta = model.site
        chan = model.chan
        band = model.band
        phases = [model.phase, ]
        param = model.param
        template_shape = model.template_shape
        min_amp = model.min_amp
        max_acost = model.max_acost
        basisid = model.wiggle_basisid.basisid if model.wiggle_basisid else None
        require_human_approved = (model.require_human_approved == 't')

        if plot_type == "histogram":
            plot_gaussian(request, model, axes=axes)
        elif plot_type == "distance":
            if model.model_type == "linear_distance":
                plot_linear_model_distance(request, model_record=model, axes=axes)
            elif model.model_type[:2] == "gp":
                plot_gp_model_distance(request, model_record=model, axes=axes)

    else:
        param = request.GET.get("plot_param", "coda_decay")
        sta = request.GET.get("sta", None)
        chan = request.GET.get("chan", None)
        band = request.GET.get("band", None)
        max_acost = float(request.GET.get("max_acost", "200"))
        phases = request.GET.get("phases", "P").split(',')
        min_amp = float(request.GET.get("min_amp", "-10"))
        template_shape = request.GET.get("shape", "paired_exp")
        require_human_approved = str(request.GET.get("human_approved", "0")) == "2"
        basisid=None

        azi_range = request.GET.get("azi", "0i..360i")
        (min_azi, max_azi) = [float(x[:-1]) if x[-1] == 'i' else float(x) for x in azi_range.split('..')]
        d['min_azi'] = min_azi
        d['max_azi'] = max_azi

    if runid is not None:
        run = SigvisaCodaFittingRun.objects.get(runid=runid)

        xy_by_phase = {}
        for phase in phases:
            if basisid:
                wm_node = load_wiggle_node(basisid=basisid)
                X, y, evids = get_wiggle_training_data(run_name=run.run_name, run_iter=run.iter,
                                                site=sta, chan=chan, band=band, phases=phases,
                                                target=param,require_human_approved=require_human_approved,
                                                max_acost=max_acost, min_amp=min_amp,
                                                      wm_node = wm_node, **d)
            else:
                X, y, evids = get_shape_training_data(run_name=run.run_name, run_iter=run.iter,
                                                site=sta, chan=chan, band=band, phases=phases,
                                                target=param,require_human_approved=require_human_approved,
                                                max_acost=max_acost, min_amp=min_amp, **d)
            xy_by_phase[phase] = (X, y)

        if plot_type == "histogram":
            plot_empirical_histogram(request=request, xy_by_phase=xy_by_phase, axes=axes)
        elif plot_type == "distance":
            plot_empirical_distance(request=request, xy_by_phase=xy_by_phase, axes=axes)

    if plot_type == "histogram":
        axes.set_xlabel(param, fontsize=8)
    elif plot_type == "distance":
        axes.set_xlabel("distance (km)", fontsize=8)
        axes.set_ylabel(param, fontsize=8)

        default_bounds = {'coda_decay': [-0.05, 0], 'amp_transfer': [-7, 10], 'peak_offset': [0, 25]}
        if param in default_bounds:
            axes.set_ylim(default_bounds[param])

    process_plot_args(request, axes)

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response


def plot_empirical_histogram(request, xy_by_phase, axes):

    # get the fit corresponding to the given pageid for this run

    for (i, phase) in enumerate(sorted(xy_by_phase.keys())):
        y = xy_by_phase[phase][1]
        histogram.plot_histogram(y, axes=axes, normed=True)


def model_density(request, modelid):
    return plot_fit_param(request, modelid=modelid, plot_type="histogram")


def model_distance_plot(request, modelid):
    return plot_fit_param(request, modelid=modelid, plot_type="distance")


def model_heatmap(request, modelid):
    pass


def data_distance_plot(request, **kwargs):
    return plot_fit_param(request, plot_type="distance", **kwargs)


def data_histogram_plot(request, **kwargs):
    return plot_fit_param(request, plot_type="histogram", **kwargs)
