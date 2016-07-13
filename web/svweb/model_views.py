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
from sigvisa.learn.train_coda_models import  get_training_data


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
        'shrinkage_iter',
    ]


def model_list_view(request):
    models = SigvisaParamModel.objects.all()
    model_filter = ModelsFilterSet(models, request.GET)
    return render_to_response("svweb/models.html",
                              {'model_list': model_filter.qs,
                               'model_filter': model_filter,
                               }, context_instance=RequestContext(request))



def plot_empirical_distance(request, xy_by_phase, alphas_by_phase, axes):

    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1)]

    xmin = np.float('inf')
    xmax = np.float('-inf')
    for (i, phase) in enumerate(sorted(xy_by_phase.keys())):
        X, y = xy_by_phase[phase]
        alphas = alphas_by_phase[phase]
        #alpha = float(request.GET.get('alpha', min(1, 4 / np.log(len(y)))))

        rgba_colors = np.zeros((X.shape[0],4))
        rgba_colors[:, :3] = colors[i]
        rgba_colors[:, 3] = alphas
        axes.scatter(X[:, 3], y, c=rgba_colors, s=10, marker='.', edgecolors="none")
        xmin = min(xmin, np.min(X[:, 3]))
        xmax = max(xmax, np.max(X[:, 3]))
    r = xmax - xmin
    axes.set_xlim([xmin - r / 20.0, xmax + r / 20.0])


def plot_empirical_mb(request, xy_by_phase, axes):

    colors = ['r', 'g', 'b', 'y']

    xmin = np.float('-inf')
    xmax = np.float('inf')
    for (i, phase) in enumerate(sorted(xy_by_phase.keys())):
        X, y = xy_by_phase[phase]
        alpha = float(request.GET.get('alpha', min(1, 4 / np.log(len(y)))))
        axes.scatter(X[:, 4], y, alpha=alpha, c=colors[i], s=10, marker='.', edgecolors="none")
        xmin = max(xmin, np.min(X[:, 4]))
        xmax = min(xmax, np.max(X[:, 4]))
    r = xmax - xmin
    axes.set_xlim([xmin - r / 20.0, xmax + r / 20.0])


def plot_linear_model_distance(request, model_record, axes, mb=3.5):


    nsamples = int(request.GET.get("nsamples", "30"))

    full_fname = os.path.join(os.getenv("SIGVISA_HOME"), model_record.model_fname)
    model = load_model(full_fname, model_record.model_type)

    distances = np.linspace(0, 20000, 200)
    Xs = np.zeros((len(distances), 5))
    Xs[:,3] = distances
    Xs[:,4] = mb


    pred = model.predict(Xs).flatten()
    axes.plot(distances, pred, 'k-')

    std = np.sqrt(model.variance(Xs, include_obs=True))
    var_x = np.concatenate((distances, distances[::-1]))
    var_y = np.concatenate((pred + 2 * std, (pred - 2 * std)[::-1]))
    axes.fill(var_x, var_y, edgecolor='w', facecolor='#d3d3d3', alpha=0.2)


    try:
        for i in range(nsamples):
            ys = model.sample(Xs)
            axes.plot(distances, ys, alpha=0.2)
    except Exception as e:
        print e

def plot_linear_model_mb(request, model_record, axes):

    full_fname = os.path.join(os.getenv("SIGVISA_HOME"), model_record.model_fname)
    model = load_model(full_fname, model_record.model_type)

    mbs = np.linspace(2, 8, 200)
    Xs = np.zeros((len(mbs), 5))
    Xs[:,4] = mbs
    Xs[:,3] = 3500


    pred = model.predict(Xs).flatten()
    axes.plot(mbs, pred, 'k-')

    std = np.sqrt(model.variance(Xs, include_obs=True))
    var_x = np.concatenate((mbs, mbs[::-1]))
    var_y = np.concatenate((pred + 2 * std, (pred - 2 * std)[::-1]))
    axes.fill(var_x, var_y, edgecolor='w', facecolor='#d3d3d3', alpha=0.2)


    try:
        for i in range(30):
            ys = model.sample(Xs)
            axes.plot(mbs, ys, alpha=0.2)
    except Exception as e:
        print e

def plot_gp_model_distance(request, model_record, axes, azi=0, depth=0, mb=3.5):

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

    pred = np.array([model.predict(np.array(((pt[0], pt[1], depth, d, mb),)), **kwargs) for (pt, d) in zip(pts, distances)]).flatten()
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

def plot_localgp_hparams(request, modelid=None, param=None ):

    model_record = SigvisaParamModel.objects.get(modelid=modelid)

    fig = Figure(figsize=(8, 5), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)

    full_fname = os.path.join(os.getenv("SIGVISA_HOME"), model_record.model_fname)
    model = load_model(full_fname, model_record.model_type)
    if param=="noise_var":
        vs = [lgp.noise_var for lgp in model.local_gps]
    elif param=="signal_var":
        vs = [lgp.cov_main.wfn_params[0] for lgp in model.local_gps]
    elif param=="horiz_lscale":
        vs = [lgp.cov_main.dfn_params[0] for lgp in model.local_gps]
    elif param=="depth_lscale":
        vs = [lgp.cov_main.dfn_params[1] for lgp in model.local_gps]
    vmin, vmax = np.min(vs), np.max(vs)
    cluster_centers = model.cluster_centers

    full_fname = os.path.join(os.getenv("SIGVISA_HOME"), model_record.model_fname)
    model = load_model(full_fname, model_record.model_type)

    def f(lon, lat):
        d = {'lon': lon, 'lat': lat, 'depth': 0, 'mb': 4.0 }
        x1 = model.standardize_input_array(d).astype(np.float)
        i = model._x_to_cluster(x1)
        return vs[i]

    heatmap_fname = full_fname + "." +param  + ".heatmap"

    hm = EventHeatmap(f=f, autobounds=cluster_centers, autobounds_quantile=1.0, n=50, fname = heatmap_fname)
    hm.add_stations((model_record.site,))
    hm.add_events(locations=model.X, )
    hm.plot(axes=axes, nolines=True, smooth=True,
            colorbar_format='%.3f',vmin=vmin, vmax=vmax)


    process_plot_args(request, axes)
    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response


def plot_gp_heatmap(request, model_record, X, y, axes, stddev=False):

    ngrid = int(request.GET.get('ngrid', "25"))
    title= request.GET.get('title', model_record.param)

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
        f = lambda lon, lat : np.sqrt(model.variance( cond={'lon': lon, 'lat': lat, 'depth': 0, 'mb': 4.0 } ))
    else:
        f = lambda lon, lat : model.predict( cond={'lon': lon, 'lat': lat, 'depth': 0, 'mb': 4.0 } )
    heatmap_fname = full_fname + ('.std' if stddev else '')  + ".heatmap." + str(ngrid)
    hm = EventHeatmap(f=f, autobounds=ev_locs, n=ngrid, fname = heatmap_fname)
    hm.add_stations((model_record.site,))
    hm.add_events(locations=ev_locs, yvals=y)
    if draw_azi:
        hm.add_events(pts)
    hm.plot(axes=axes, nolines=True, smooth=True,
            colorbar_format='%.3f',vmin=vmin, vmax=vmax)



    for item in (axes.get_xticklabels() + axes.get_yticklabels()):
        print item
        item.set_fontsize(2)

    axes.set_title(title)

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

def plot_laplacian(request, model_record, axes):

    full_fname = os.path.join(os.getenv("SIGVISA_HOME"), model_record.model_fname)
    model = load_model(full_fname, model_record.model_type)
    xmin = model.center - 4 * model.scale
    xmax = model.center + 4 * model.scale
    x = np.linspace(xmin, xmax, 200)
    pdf = scipy.stats.laplace.pdf(x, loc=model.center, scale=model.scale)
    axes.plot(x, pdf, 'r-', linewidth=3)

def plot_beta(request, model_record, axes):

    full_fname = os.path.join(os.getenv("SIGVISA_HOME"), model_record.model_fname)
    model = load_model(full_fname, model_record.model_type)
    x = np.linspace(0, 1, 200)
    pdf = scipy.stats.beta(model.alpha, model.beta).pdf(x)
    axes.plot(x, pdf, 'r-', linewidth=3)

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


def plot_compare_phases(request, runid=None):
    param = request.GET.get("plot_param", "coda_decay")
    sta = request.GET.get("sta", None)
    chan = request.GET.get("chan", None)
    band = request.GET.get("band", None)
    max_acost = float(request.GET.get("max_acost", "inf"))
    phase1 = request.GET.get("phase1", "Pg")
    phase2 = request.GET.get("phase2", "none")
    min_amp = float(request.GET.get("min_amp", "-10"))
    require_human_approved = str(request.GET.get("human_approved", "0")) == "2"
    alpha_scale = float(request.GET.get("alpha_scale", "1.0"))
    ms = int(request.GET.get("ms", "20"))


    x_distance = request.GET.get("x_distance", "f").lower().startswith("t")
    x_mb = request.GET.get("x_mb", "f").lower().startswith("t")
    x_depth = request.GET.get("x_depth", "f").lower().startswith("t")


    try:
        X1, y1, yvars1, evids1 = get_training_data(runid=runid,
                                                   site=sta, chan=chan, band=band, phases=(phase1,),
                                                   target=param,require_human_approved=require_human_approved,
                                                   max_acost=max_acost, min_amp=min_amp)
        evids1 = list(evids1)
    except NoDataException:
        X1, y1, yvars1, evids1 = [], [], [], []

    if phase2 != "none":
        X2, y2, yvars2, evids2 = get_training_data(runid=runid,
                                                   site=sta, chan=chan, band=band, phases=(phase2,),
                                                   target=param,require_human_approved=require_human_approved,
                                                   max_acost=max_acost, min_amp=min_amp)
        evids2 = list(evids2)

    matched = []
    Y_COL, YV_COL, YY_COL, YYV_COL, DIST_COL, MB_COL, DEPTH_COL = np.arange(7)
    for x, y, yvar, evid in zip(X1, y1, yvars1, evids1):
        if phase2 == "none":
            yy, yyvar = 0, 0
        else:
            try:
                idx = evids2.index(evid)
            except:
                continue

            try:
                yy, yyvar = y2[idx], yvars2[idx]
            except:
                import pdb; pdb.set_trace()

        lon, lat, depth, distance, mb = x
        match = (y, yvar, yy, yyvar, distance, mb, depth)
        matched.append(match)

    n = len(matched)
    matched = np.array(matched).reshape((n, 7))


    fig = Figure(figsize=(8, 5), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    
    if x_distance:
        x = matched[:, DIST_COL]
        y = matched[:, Y_COL] - matched[:, YY_COL]
        axes.set_xlabel("distance")
        axes.set_ylabel("%s %s-%s" % (param, phase1, phase2))
    elif x_mb:
        x = matched[:, MB_COL]
        y = matched[:, Y_COL] - matched[:, YY_COL]
        axes.set_xlabel("mb")
        axes.set_ylabel("%s %s-%s" % (param, phase1, phase2))

    elif x_depth:
        x = matched[:, DEPTH_COL]
        y = matched[:, Y_COL] - matched[:, YY_COL]
        axes.set_xlabel("depth")
        axes.set_ylabel("%s %s-%s" % (param, phase1, phase2))

    else:
        x = matched[:, Y_COL]
        y = matched[:, YY_COL]


        axes.set_xlabel("%s %s" % (param, phase1))
        axes.set_ylabel("%s %s" % (param, phase2))

    v = matched[:, YV_COL] + matched[:, YYV_COL]
    alphas = (v + 1e-4)**(-.25 * alpha_scale)
    if len(alphas) > 0:
        alphas /= np.max(alphas)

    rgba_colors = np.zeros((len(matched),4))
    rgba_colors[:,2] = 1.0
    rgba_colors[:, 3] = alphas
    axes.scatter(x, y, color=rgba_colors, s=ms, marker='.', edgecolors="none")

    axes.set_title("%s at %s (%d events)" % (param, sta, len(matched)))

    process_plot_args(request, axes)
    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response


def plot_fit_param(request, modelid=None, runid=None, plot_type="histogram"):
    fig = Figure(figsize=(8, 5), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)

    mb = float(request.GET.get("mb", "3.5"))

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
        max_acost = float(request.GET.get("max_acost", "inf"))
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
        alphas_by_phase = {}
        for phase in phases:
            X, y, yvars, evids = get_training_data(runid=runid,
                                                   site=sta, chan=chan, band=band, phases=(phase,),
                                                   target=param,require_human_approved=require_human_approved,
                                                   max_acost=max_acost, min_amp=min_amp, **d)
            if log_transform:
                y = np.log(np.abs(y))
            xy_by_phase[phase] = (X, y)
            alphas_by_phase[phase] = 1 - np.sqrt(yvars / np.max(yvars))

    if modelid is not None:
        if plot_type == "histogram":
            if log_transform:
                plot_adhoc_gaussian(request, xy_by_phase, axes=axes)
            else:
                if model.model_type == "constant_gaussian":
                    plot_gaussian(request, model, axes=axes)
                elif model.model_type == "constant_laplacian":
                    plot_laplacian(request, model, axes=axes)
                elif model.model_type == "constant_beta":
                    plot_beta(request, model, axes=axes)
                else:
                    raise Exception("don't know how to plot a histogram for %s" % model.model_type)
        elif plot_type == "mb":
            if model.model_type.startswith("param_"):
                plot_linear_model_mb(request, model_record=model, axes=axes)
            else:
                raise Exception("don't know how to plot mb for model %s" % model.model_type)
        elif plot_type == "distance":
            if model.model_type.startswith("param_"):
                plot_linear_model_distance(request, model_record=model, axes=axes, mb=mb)
            else:
                nplots = int(request.GET.get('nplots', 1))
                if nplots == 1:
                    plot_azis = [gp_plot_azi,]
                else:
                    plot_azis = np.linspace(d['min_azi'], d['max_azi'], nplots)
                for azi in plot_azis:
                    plot_gp_model_distance(request, model_record=model, axes=axes, 
                                           azi=azi, depth=gp_plot_depth, mb=mb)
        elif plot_type == "heatmap": # and model.model_type[:2] == "gp":
                plot_gp_heatmap(request, model_record=model, X=xy_by_phase[model.phase][0], y=xy_by_phase[model.phase][1], axes=axes, stddev=False)
        elif plot_type == "heatmap_std": #and model.model_type[:2] == "gp":
                plot_gp_heatmap(request, model_record=model, X=xy_by_phase[model.phase][0], y=xy_by_phase[model.phase][1], axes=axes, stddev=True)

    if runid is not None:
        if plot_type == "histogram":
            plot_empirical_histogram(request=request, xy_by_phase=xy_by_phase, axes=axes)
        elif plot_type == "distance":
            plot_empirical_distance(request=request, xy_by_phase=xy_by_phase, 
                                    alphas_by_phase=alphas_by_phase, axes=axes)
        elif plot_type == "mb":
            plot_empirical_mb(request=request, xy_by_phase=xy_by_phase, axes=axes)
        elif plot_type == "heatmap" and modelid is None:
            plot_data_heatmap(request=request, sta=sta, param=param, xy_by_phase=xy_by_phase, axes=axes)

    if plot_type == "histogram":
        axes.set_xlabel(param, fontsize=8)
    elif plot_type == "distance":
        axes.set_xlabel("distance (km)", fontsize=8)
        axes.set_ylabel(param, fontsize=8)

        default_bounds = {'coda_decay': [-8, 1], 'amp_transfer': [-7, 15], 'peak_offset': [-3, 5]}
        if param in default_bounds:
            axes.set_ylim(default_bounds[param])
    elif plot_type == "mb":
        axes.set_xlabel("mb", fontsize=8)
        axes.set_ylabel(param, fontsize=8)
    elif plot_type == "heatmap":
        pass
    else:
        axes.set_xlabel(plot_type, fontsize=8)
        axes.set_ylabel(param, fontsize=8)

    process_plot_args(request, axes)

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response


def plot_empirical_histogram(request, xy_by_phase, axes):

    # get the fit corresponding to the given pageid for this run


    n_bins = int(request.GET.get("bins", -1))
    draw_stats = str(request.GET.get("draw_stats", 't')).startswith('t')
    if n_bins ==-1: n_bins = None

    for (i, phase) in enumerate(sorted(xy_by_phase.keys())):
        y = xy_by_phase[phase][1]
        histogram.plot_histogram(y, axes=axes, n_bins=n_bins, normed=True, draw_stats=draw_stats)


@cache_page(60 * 60 * 24 * 365)
def model_density(request, modelid):
    return plot_fit_param(request, modelid=modelid, plot_type="histogram")

@cache_page(60 * 60 * 24 * 365)
def model_distance_plot(request, modelid):
    return plot_fit_param(request, modelid=modelid, plot_type="distance")

def model_mb_plot(request, modelid):
    return plot_fit_param(request, modelid=modelid, plot_type="mb")

@cache_page(60 * 60 * 24 * 365)
def model_heatmap(request, modelid):
    return plot_fit_param(request, modelid=modelid, plot_type="heatmap")

@cache_page(60 * 60 * 24 * 365)
def model_heatmap_std(request, modelid):
    return plot_fit_param(request, modelid=modelid, plot_type="heatmap_std")


def data_distance_plot(request, **kwargs):
    return plot_fit_param(request, plot_type="distance", **kwargs)

def data_pairwise_plot(request, runid=None, **kwargs):
    param1 = request.GET.get("param1", None)
    param2 = request.GET.get("param2", None)

    sta = request.GET.get("sta", None)
    chan = request.GET.get("chan", None)
    band = request.GET.get("band", None)
    max_acost = float(request.GET.get("max_acost", "inf"))
    phases = request.GET.get("phases", "P").split(',')
    min_amp = float(request.GET.get("min_amp", "-10"))
    template_shape = request.GET.get("shape", "paired_exp")
    require_human_approved = str(request.GET.get("human_approved", "0")) == "2"

    X, y1, yvars, evids = get_training_data(runid=runid,
                                           site=sta, chan=chan, band=band, phases=phases,
                                           target=param1,require_human_approved=require_human_approved,
                                           max_acost=max_acost, min_amp=min_amp)

    X, y2, yvars, evids = get_training_data(runid=runid,
                                           site=sta, chan=chan, band=band, phases=phases,
                                           target=param2,require_human_approved=require_human_approved,
                                           max_acost=max_acost, min_amp=min_amp)

    alpha = float(request.GET.get('alpha', min(1, 4 / np.log(len(y1)))))


    fig = Figure(figsize=(8, 5), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)

    axes.scatter(y1, y2, alpha=alpha, s=10, marker='.', edgecolors="none")

    axes.set_xlabel(param1)
    axes.set_ylabel(param2)

    process_plot_args(request, axes)

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response

def data_histogram_plot(request, **kwargs):
    return plot_fit_param(request, plot_type="histogram", **kwargs)

def data_heatmap_plot(request, **kwargs):
    return plot_fit_param(request, plot_type="heatmap", **kwargs)
