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
from sigvisa.models.wiggles import load_wiggle_generator
from sigvisa.learn.train_param_common import load_model
from sigvisa.learn.train_coda_models import  get_shape_training_data, get_wiggle_training_data


from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datetime import datetime
from pytz import timezone
import hashlib

import sigvisa.plotting.plot as plot
from sigvisa.plotting.event_heatmap import EventHeatmap
import sigvisa.plotting.histogram as histogram
import textwrap
import sigvisa.utils.geog as geog

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

    colors = ['r', 'g', 'b', 'y']

    xmin = np.float('-inf')
    xmax = np.float('inf')
    for (i, phase) in enumerate(sorted(xy_by_phase.keys())):
        X, y = xy_by_phase[phase]
        alpha = float(request.GET.get('alpha', min(1, 4 / np.log(len(y)))))
        axes.scatter(X[:, 3], y, alpha=alpha, c=colors[i], s=10, marker='.', edgecolors="none")
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


def plot_gp_model_distance(request, model_record, axes, azi=0, depth=0):

    xmax = float(request.GET.get('xmax', 10000))

    kwargs = {}
    parametric_only = request.GET.get('parametric_only', 'f').lower().startswith('t')
    if parametric_only:
        kwargs['parametric_only'] = parametric_only

    full_fname = os.path.join(os.getenv("SIGVISA_HOME"), model_record.model_fname)
    model = load_model(full_fname, model_record.model_type)

    site_loc = Sigvisa().earthmodel.site_info(str(model_record.site), 0)[:2]
    distances = np.linspace(0, xmax, 40)
    pts = [geog.pointRadialDistance(site_loc[0], site_loc[1], azi, d) for d in distances]

    pred = np.array([model.predict(np.array(((pt[0], pt[1], depth, d, azi),)), **kwargs) for (pt, d) in zip(pts, distances)]).flatten()
    axes.plot(distances, pred, 'k-')

    array_inputs = [np.array(((pt[0], pt[1], depth, d, azi),)) for (pt, d) in zip(pts, distances)]
    stds = np.zeros((len(distances),))
    for (i, arr) in enumerate(array_inputs):
        v = model.variance(arr, include_obs=True, **kwargs)
        stds[i] = np.sqrt(v)
    std = np.array(stds)

    var_x = np.concatenate((distances, distances[::-1]))
    var_y = np.concatenate((pred + 2 * std, (pred - 2 * std)[::-1]))
    axes.fill(var_x, var_y, edgecolor='w', facecolor='#d3d3d3', alpha=0.1)

def plot_gp_heatmap(request, model_record, X, y, axes, stddev=False):

    vmin = request.GET.get('vmin', None)
    vmin = float(vmin) if vmin is not None else None
    vmax = request.GET.get('vmax', None)
    vmax = float(vmax) if vmax is not None else None

    full_fname = os.path.join(os.getenv("SIGVISA_HOME"), model_record.model_fname)
    model = load_model(full_fname, model_record.model_type)

    draw_azi = request.GET.get("draw_azi", None)
    if draw_azi:
        azi = float(draw_azi)
        distances = np.linspace(0, 15000, 200)
        site_loc = Sigvisa().earthmodel.site_info(str(model_record.site), 0)
        pts = [geog.pointRadialDistance(site_loc[0], site_loc[1], azi, d) for d in distances]

    ev_locs = X[:, 0:2]
    if stddev:
        print "plotting with stddev"
        f = lambda lon, lat : np.sqrt(model.variance( cond={'lon': lon, 'lat': lat, 'depth': 0 } ))
    else:
        f = lambda lon, lat : model.predict( cond={'lon': lon, 'lat': lat, 'depth': 0 } )
    heatmap_fname = full_fname + ('.std' if stddev else '')  + ".heatmap"
    hm = EventHeatmap(f=f, autobounds=ev_locs, n=25, fname = heatmap_fname)
    hm.add_stations((model_record.site,))
    hm.add_events(locations=ev_locs)
    if draw_azi:
        hm.add_events(pts)
    hm.plot(axes=axes, nolines=True, smooth=True,
            colorbar_format='%.3f',vmin=vmin, vmax=vmax)



    for item in (axes.get_xticklabels() + axes.get_yticklabels()):
        print item
        item.set_fontsize(2)

    axes.set_title(model_record.param)

def plot_data_heatmap(request, sta, param, xy_by_phase, axes):


    X, y = xy_by_phase.values()[0]
    ev_locs = X[:, 0:2]

    ymin = float(request.GET.get('ymin', '-inf'))
    ymax = float(request.GET.get('ymax', 'inf'))
    y = np.array([max(ymin, min(ymax, yy)) for yy in y])

    hm = EventHeatmap(f=None, calc=False, autobounds=ev_locs, autobounds_quantile=0.98)
    hm.add_stations((sta,))
    hm.add_events(locations=ev_locs, yvals=y)
    hm.plot(axes=axes, nolines=True, smooth=True, station_labels=False, meridians=False)

    for item in (axes.get_xticklabels() + axes.get_yticklabels()):
        print item
        item.set_fontsize(2)

    axes.set_title("%s: %s" % (sta, param))


def plot_gaussian(request, model_record, axes):

    full_fname = os.path.join(os.getenv("SIGVISA_HOME"), model_record.model_fname)
    model = load_model(full_fname, model_record.model_type)
    xmin = model.mean - 4 * model.std
    xmax = model.mean + 4 * model.std
    x = np.linspace(xmin, xmax, 200)
    pdf = scipy.stats.norm.pdf(x, loc=model.mean, scale=model.std)
    axes.plot(x, pdf, 'k-')

def plot_adhoc_gaussian(request, xy_by_phase, axes):

    for (i, phase) in enumerate(sorted(xy_by_phase.keys())):
        y = xy_by_phase[phase][1]

        mean = np.mean(y)
        std = np.std(y)

        xmin = mean - 4 * std
        xmax = mean + 4 * std
        x = np.linspace(xmin, xmax, 200)
        pdf = scipy.stats.norm.pdf(x, loc=mean, scale=std)
        axes.plot(x, pdf, 'k-')


def plot_fit_param(request, modelid=None, runid=None, plot_type="histogram"):
    fig = Figure(figsize=(8, 5), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)

    log_transform = bool(request.GET.get("log", False))
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

        gp_plot_azi = float(request.GET.get("azi", 0))
        azi_window = float(request.GET.get("azi_window", -1))
        gp_plot_depth = float(request.GET.get("depth", 0))

        if azi_window > 0:
            d['min_azi'] = gp_plot_azi - azi_window
            d['max_azi'] = gp_plot_azi + azi_window

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
                wg = load_wiggle_generator(basisid=basisid)
                param_num = wg.params().index(param)
                X, y, evids = get_wiggle_training_data(runid=runid,
                                                site=sta, chan=chan, band=band, phases=phases,
                                                target_num=param_num,require_human_approved=require_human_approved,
                                                max_acost=max_acost, min_amp=min_amp,
                                                      wg = wg, **d)
            else:
                X, y, evids = get_shape_training_data(runid=runid,
                                                site=sta, chan=chan, band=band, phases=phases,
                                                target=param,require_human_approved=require_human_approved,
                                                max_acost=max_acost, min_amp=min_amp, **d)
            if log_transform:
                y = np.log(np.abs(y))
            xy_by_phase[phase] = (X, y)

    if modelid is not None:
        if plot_type == "histogram":
            if log_transform:
                plot_adhoc_gaussian(request, xy_by_phase, axes=axes)
            else:
                plot_gaussian(request, model, axes=axes)
        elif plot_type == "distance":
            if model.model_type == "linear_distance":
                plot_linear_model_distance(request, model_record=model, axes=axes)
            else:
                nplots = int(request.GET.get('nplots', 1))
                if nplots == 1:
                    plot_azis = [gp_plot_azi,]
                else:
                    plot_azis = np.linspace(d['min_azi'], d['max_azi'], nplots)
                for azi in plot_azis:
                    plot_gp_model_distance(request, model_record=model, axes=axes, azi=azi, depth=gp_plot_depth)
        elif plot_type == "heatmap": # and model.model_type[:2] == "gp":
                plot_gp_heatmap(request, model_record=model, X=xy_by_phase[model.phase][0], y=xy_by_phase[model.phase][1], axes=axes, stddev=False)
        elif plot_type == "heatmap_std": #and model.model_type[:2] == "gp":
                plot_gp_heatmap(request, model_record=model, X=xy_by_phase[model.phase][0], y=xy_by_phase[model.phase][1], axes=axes, stddev=True)

    if runid is not None:
        if plot_type == "histogram":
            plot_empirical_histogram(request=request, xy_by_phase=xy_by_phase, axes=axes)
        elif plot_type == "distance":
            plot_empirical_distance(request=request, xy_by_phase=xy_by_phase, axes=axes)
        elif plot_type == "heatmap" and modelid is None:
            plot_data_heatmap(request=request, sta=sta, param=param, xy_by_phase=xy_by_phase, axes=axes)

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


    n_bins = int(request.GET.get("bins", -1))
    if n_bins ==-1: n_bins = None

    for (i, phase) in enumerate(sorted(xy_by_phase.keys())):
        y = xy_by_phase[phase][1]
        histogram.plot_histogram(y, axes=axes, n_bins=n_bins, normed=True)


@cache_page(60 * 60 * 24 * 365)
def model_density(request, modelid):
    return plot_fit_param(request, modelid=modelid, plot_type="histogram")

@cache_page(60 * 60 * 24 * 365)
def model_distance_plot(request, modelid):
    return plot_fit_param(request, modelid=modelid, plot_type="distance")

@cache_page(60 * 60 * 24 * 365)
def model_heatmap(request, modelid):
    return plot_fit_param(request, modelid=modelid, plot_type="heatmap")

@cache_page(60 * 60 * 24 * 365)
def model_heatmap_std(request, modelid):
    return plot_fit_param(request, modelid=modelid, plot_type="heatmap_std")


def data_distance_plot(request, **kwargs):
    return plot_fit_param(request, plot_type="distance", **kwargs)


def data_histogram_plot(request, **kwargs):
    return plot_fit_param(request, plot_type="histogram", **kwargs)

def data_heatmap_plot(request, **kwargs):
    return plot_fit_param(request, plot_type="heatmap", **kwargs)
