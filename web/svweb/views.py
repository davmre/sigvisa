import django
import django.views.generic
from django.shortcuts import render_to_response, get_object_or_404
from django.views.decorators.cache import cache_page
from django.core.cache import cache
from django.template import RequestContext
from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse
from django_easyfilters import FilterSet
from django_easyfilters.filters import NumericRangeFilter

import numpy as np
import numpy.ma as ma
import sys
from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *
from sigvisa.signals.io import *
from sigvisa import Sigvisa
from sigvisa.source.event import get_event, EventNotFound
from sigvisa.models.noise.armodel.model import ARModel, ErrorModel
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.graph.load_sigvisa_graph import load_sg_from_db_fit
from sigvisa.models.ttime import tt_predict

import sigvisa.utils.geog as geog

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datetime import datetime
from pytz import timezone
import hashlib

import sigvisa.plotting.plot as plot

from svweb.models import SigvisaCodaFit, SigvisaCodaFitPhase, SigvisaCodaFittingRun, view_options
from svweb.plotting_utils import process_plot_args, view_wave

def filterset_GET_string(filterset):
    """

    Given a django_easyfilters FilterSet object, return a dictionary
    of field->field_value mappings (where the value could be a string
    indicating a range), and a string containing these mappings that
    can be used in a new GET request to recreate the FilterSet.

    """

    def field_names(self):
        field_names = []
        for field in self.fields:
            if isinstance(field, basestring):
                field_names.append(field)
            else:
                field_names.append(field[0])
        return field_names

    def dict_to_GET(d):
        return ';'.join(['%s=%s' % (k, d[k]) for k in d if d[k]])

    field_vals = {}
    for field in field_names(filterset):
        field_vals[field] = filterset.params.get(field, "")
    filter_GET_params = dict_to_GET(field_vals)
    return filter_GET_params, field_vals


def main_view(request):
    return render_to_response("svweb/main.html",
                              {}, context_instance=RequestContext(request))


#@cache_page(60 * 60)
def fit_list_view(request, runid):
    run = SigvisaCodaFittingRun.objects.get(pk=runid)
    fits = SigvisaCodaFit.objects.filter(runid=runid)
    fits_filter = FitsFilterSet(fits, request.GET)

    #mean_acost = np.mean([fit.acost for fit in fits_filter.qs])
    #mean_time = np.mean([fit.elapsed for fit in fits_filter.qs])
    mean_acost = -1
    mean_time = -1
    total_fits = fits_filter.qs.count

    filter_GET_params, filter_args = filterset_GET_string(fits_filter)
    filter_args['runid'] = runid

    return render_to_response("svweb/fits.html",
                              {'fit_list': fits_filter.qs,
                               'fits_filter': fits_filter,
                               'filter_args': filter_args,
                               'filter_GET_params': filter_GET_params,
                               'runid': runid,
                               'mean_acost': mean_acost,
                               'mean_time': mean_time,
                               'total_fits': total_fits,
                               'run': run,
                               }, context_instance=RequestContext(request))


class FitsFilterSet(FilterSet):
    fields = [
        ('evid', {}, NumericRangeFilter),
        'sta',
        'chan',
        'band',
        'human_approved',
        'tmpl_optim_method',
        #'dist',
        'azi',
    ]


# detail view for a particular fit
def fit_detail(request, fitid):

    fit = SigvisaCodaFit.objects.get(fitid=fitid)

    # use a FilterSet to determine links to next/previous fits
    fits = SigvisaCodaFit.objects.filter(runid=fit.runid)
    fits_filter = FitsFilterSet(fits, request.GET)
    filter_GET_params, filter_args = filterset_GET_string(fits_filter)

    next = fits_filter.qs.filter(fitid__gt=fitid)
    next = next[0] if next else False
    prev = fits_filter.qs.filter(fitid__lt=fitid).order_by('-fitid')
    prev = prev[0] if prev else False

    time_format = "%b %d %Y, %H:%M:%S"

    fit_time_str = fit.timestamp.strftime(time_format)

    # get the current display options; initialize to defaults if there are none
    try:
        fit_view_options = view_options.objects.get(id=1)
    except:
        fit_view_options = view_options()
        fit_view_options.id = 1
        fit_view_options.smoothing = fit.smooth
        fit_view_options.logscale = False
        fit_view_options.wiggle = False
        fit_view_options.noise = False
        fit_view_options.save()

    s = Sigvisa()

    try:
        wave = wave_from_fit(fit)

        wave_stime_str = datetime.fromtimestamp(wave['stime'], timezone('UTC')).strftime(time_format)
        wave_etime_str = datetime.fromtimestamp(wave['etime'], timezone('UTC')).strftime(time_format)

    except Exception as e:
        wave = Waveform()
        wave_stime_str = str(e)
        wave_etime_str = str(e)
        nm = ARModel([], ErrorModel(0, 1), c=0)

    # load the event so that we can display data about it
    try:
        ev = get_event(evid=fit.evid)

        station_location = tuple(s.earthmodel.site_info(str(fit.sta), ev.time)[0:2])
        dist = geog.dist_km((ev.lon, ev.lat), station_location)
        azi = geog.azimuth(station_location, (ev.lon, ev.lat))

        ev_time_str = datetime.fromtimestamp(ev.time, timezone('UTC')).strftime(time_format)
        loc_str = geog.lonlatstr(ev.lon, ev.lat)
    except EventNotFound as e:
        ev = Event()
        dist = None
        azi = None
        loc_str = ""
        ev_time_str = ""

    # get other fits of this same arrival
    other_fits = SigvisaCodaFit.objects.filter(sta=fit.sta, evid=fit.evid, chan=fit.chan, band=fit.band)


    return render_to_response('svweb/detail.html', {
        'fit': fit,
        'fit_time_str': fit_time_str,
        'fit_view_options': fit_view_options,
        'filter_GET_params': filter_GET_params,
        'next': next,
        'prev': prev,
        'wave': wave,
        'wave_stime_str': wave_stime_str,
        'wave_etime_str': wave_etime_str,
        'ev': ev,
        'ev_time_str': ev_time_str,
        'loc_str': loc_str,
        'dist': dist,
        'azi': azi,
        'fits_filter': fits_filter,
        'other_fits': other_fits,
    }, context_instance=RequestContext(request))


def wave_plus_template_view(wave, template, logscale=True, smoothing=0, request=None,
                            zoom=1.0, dpi=144, tmpl_alpha=1, tmpl_width=3, 
                            tmpl_color="green", predictions=None):

    fig = Figure(figsize=(zoom*8, 5), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    axes.set_xlabel("Time (s)", fontsize=8)

    if wave is not None:
        nans = np.isnan(wave.data.data)
        wave.data.data[nans] = 0.0
        smoothed = wave.filter("smooth_%d" % smoothing) if smoothing > 0 else wave
        wave.data.data[nans]=np.nan

        plot.subplot_waveform(smoothed, axes, color='black', linewidth=1.5, logscale=logscale)
    if template is not None:
        plot.subplot_waveform(template, axes, color=tmpl_color,
                              linewidth=tmpl_width, alpha = tmpl_alpha,
                              logscale=logscale, plot_dets=False)

    if request is not None:
        process_plot_args(request, axes)

    if predictions is not None:
        plot.plot_pred_atimes(predictions, template, axes=axes, color="purple", alpha=1.0)


    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response

def custom_wave_plus_template_view(sta, st, et, chan, band, smooth, hz, fit_params, nmid, tmshape, env, **kwargs):

    cursor = Sigvisa().dbconn.cursor()
    wave = fetch_waveform(str(sta), str(chan), float(st), float(et), cursor=cursor).filter(str(band) + "%s;smooth_%d;hz_%d" % (";env" if env else "", smooth, hz))
    cursor.close()

    sg = setup_sigvisa_graph(evid=evid, tmshape=tmshape, wave=wave, fit_params=fit_params)
    wave_node = sg.get_wave_node(wave=wave)
    wave_node.set_noise_model(nmid=nmid)
    wave_node.unfix_value()
    if wave_node.is_env:
        wave_node.parent_predict()
        synth_wave = wave_node.get_wave()
        synth_wave.data.mask = wave.data.mask
    else:
        synth_wave = wave_node.assem_env() + wave_node.nm_env.c
        synth_wave = ma.masked_array(synth_wave, wave.data.mask)


    return wave_plus_template_view(wave=wave, template=synth_wave, **kwargs)

def tmshape_from_fit(fit):
    tmshape = "paired_exp"
    fit_phases = fit.sigvisacodafitphase_set.all()
    for p in fit_phases:
        tmshape = str(p.template_model)
        break
    return tmshape

def phases_from_fit(fit):
    fit_phases = fit.sigvisacodafitphase_set.all()
    fit_params = dict()
    ua_counter = 0
    for p in fit_phases:
        phasename = p.phase
        if phasename=="UA":
            ua_counter += 1
            phasename += str(ua_counter)
        fit_params[phasename] = {'arrival_time': p.arrival_time,
                                 'peak_offset': p.peak_offset,
                                 'coda_height': p.coda_height,
                                 'peak_decay': p.peak_decay,
                                 'mult_wiggle_std': p.mult_wiggle_std,
                                 'coda_decay': p.coda_decay}
    return fit_params

def wave_from_fit(fit):
    cursor = Sigvisa().dbconn.cursor()

    try:
        stime = float(fit.stime)
        etime = float(fit.etime)
    except:
        import calendar
        stime = calendar.timegm(fit.stime.timetuple())
        etime = calendar.timegm(fit.etime.timetuple())

    wave = fetch_waveform(str(fit.sta), str(fit.chan), stime, etime, cursor=cursor).filter(str(fit.band) + (";env" if fit.env=="t" else "") + (';smooth_%d' % fit.smooth) + ';hz_%.2f' % fit.hz)
    cursor.close()
    return wave


#@cache_page(60 * 60)
def FitImageView(request, fitid):

    fit = get_object_or_404(SigvisaCodaFit, pk=fitid)

    logscale = request.GET.get("logscale", "False").lower().startswith('t')
    smoothing = int(request.GET.get("smooth", "0"))
    wiggle = request.GET.get("wiggle", "False").lower().startswith('t')
    components = request.GET.get("components", None)
    noise = request.GET.get("noise", "False").lower().startswith('t')
    saveprefs = request.GET.get("saveprefs", "False").lower().startswith('t')
    only_eid = int(request.GET.get("eid", -9999))
    only_phase = request.GET.get("phase", None)
    pred_phases = request.GET.get("pred_phases", "none")
    template_only = request.GET.get("template_only", "False").lower().startswith('t')
    zoom = float(request.GET.get("zoom", "1.0"))

    if saveprefs:
        fit_view_options = view_options.objects.get(id=1)
        fit_view_options.smoothing = smoothing
        fit_view_options.logscale = logscale
        fit_view_options.save()

    sg = load_sg_from_db_fit(fit.fitid)
    wave_node = list(sg.station_waves.values()[0])[0]
    obs_wave = wave_node.get_wave()
    wave_node.unfix_value()
    wave_node._parent_values()

    if noise:
        means = wave_node.tssm.component_means(wave_node.get_value().data)
        pred_wave = Waveform(means[0], segment_stats=wave_node.mw.segment_stats.copy(), my_stats=wave_node.mw.my_stats.copy())
        pred_wave.data.mask = obs_wave.data.mask
    elif components is not None:
        components = components.split(",")
        means = wave_node.tssm.component_means(wave_node.get_value().data)
        pred_wave = np.zeros((wave_node.npts,))

        for i, (eid, phase, env, start_idx, npts, component_type) in enumerate(wave_node.tssm_components):
            if phase is None: continue
            if only_eid > -9999 and eid != only_eid: continue
            if only_phase is not None and phase != only_phase: continue
            if component_type not in components: continue

            from sigvisa.utils.array import array_overlap
            a, b = array_overlap(pred_wave, means[i]*env[:len(means[i])], start_idx)
            a += b
        pred_wave += wave_node.nm.c
        pred_wave = Waveform(pred_wave, segment_stats=wave_node.mw.segment_stats.copy(), my_stats=wave_node.mw.my_stats.copy())
        pred_wave.data.mask = obs_wave.data.mask
    elif not wave_node.is_env:
        pred_wave = wave_node.assem_env() + wave_node.nm_env.c
        pred_wave = Waveform(pred_wave, segment_stats=wave_node.mw.segment_stats.copy(), my_stats=wave_node.mw.my_stats.copy())
        pred_wave.data.mask = obs_wave.data.mask
    else:
        for fphase in fit.sigvisacodafitphase_set.all():
            for i, (eid, phase, env, start_idx, npts, component_type) in enumerate(wave_node.tssm_components):
                if phase != fphase.phase: continue
                if component_type != "wavelet": continue
                cssm = wave_node.tssm.get_component(i)

                if wiggle:
                    messages = read_messages(fphase.message_fname, fit.runid.runid)
                    pmeans, pvars = messages[fphase.wiggle_family +"_posterior"]
                    print "pmeans", pmeans
                    cssm.set_coef_prior(pmeans, pvars)
                    pm, pv = cssm.get_coef_prior();
                    print "pm", pm
                    #wave_node.tssm.ssms[i].coef_means[:] = pmeans
                else:
                    pm, pv = cssm.get_coef_prior();
                    n_coefs = len(pm)
                    fakemeans = np.zeros((n_coefs,))
                    fakevars = np.ones((n_coefs,))
                    cssm.set_coef_prior(fakemeans, fakevars)

                v = wave_node.tssm.mean_obs(wave_node.npts)

        wave_node.parent_predict()
        pred_wave = wave_node.get_wave()
        pred_wave.data.mask = obs_wave.data.mask

    pred_phase_atimes = None
    phases = None
    eid = sg.evnodes.keys()[0]
    ev = sg.get_event(eid)
    sta = sg.station_waves.keys()[0]
    if pred_phases=="all":
        # pred atimes for all phases that exist
        phases = ("P", "PKP", "S", "PKPbc", "PcP", "pP", "Lg", "PKPab", "ScP", "PKKPbc", "Pg", "Rg", "Sn", "Pn")
    elif pred_phases=="model":
        # pred atimes for the phases this sg knows about
        phases = sg.phases
    if phases is not None:
        pred_phase_atimes = {}
        for phase in phases:
            try:
                tt = tt_predict(ev, sta, phase=phase)
            except:
                tt = -1
            if tt > 0:
                pred_phase_atimes[phase] = ev.time + tt

    
    return wave_plus_template_view(wave=None if template_only else obs_wave, template=pred_wave,
                                   logscale=logscale, smoothing=smoothing, request=request,
                                   tmpl_alpha = 0.8 if wiggle else 1, 
                                   tmpl_color="red" if noise else "green", 
                                   predictions=pred_phase_atimes,
                                   zoom=zoom)


def phases_from_request(request):
    fit_params = {}
    for (p, v) in request.GET.items():
        try:
            phase, param = str(p).split('__')
        except:
            continue
        phase = str(phase)
        param = str(param)
        if phase not in fit_params:
            fit_params[phase] = dict()
        if param =="newname":
            fit_params[phase][param] = v
        else:
            fit_params[phase][param] = float(v)
    fit_params_newnames = dict()
    for (phase, vals) in fit_params.items():
        try:
            newname = vals['newname']
            del vals['newname']
            fit_params_newnames[newname] = vals
        except KeyError:
            fit_params_newnames[phase] = vals
    print fit_params_newnames
    return fit_params_newnames

def getstr_from_phases(fit_params, **kwargs):
    entries = {}
    for (phase, vals) in fit_params.items():
        for (p, v) in vals.items():
            entries['%s__%s' % (phase, p)] = v
    items = entries.items() + kwargs.items()
    getstr = ";".join(['%s=%s' % (k,v) for (k,v) in items])
    print getstr
    return getstr

def custom_template_view(request, fitid):
    fit = get_object_or_404(SigvisaCodaFit, pk=fitid)

    logscale = request.GET.get("logscale", "off").lower().startswith('on')
    smoothing = int(request.GET.get("smooth", "0"))

    fit_params = phases_from_request(request)
    """
    filtered_fit_params = dict()
    for (phase, v) in fit_params.items():
        if phase in Sigvisa().phases:
            filtered_fit_params[phase] = v
    fit_params = filtered_fit_params
    """

    tmshape = tmshape_from_fit(fit)

    return custom_wave_plus_template_view(sta=fit.sta, st=fit.stime, et=fit.etime,
                                          chan=fit.chan, band=fit.band,
                                          smooth=fit.smooth, hz=fit.hz,
                                   nmid=fit.nmid.nmid, fit_params=fit_params, tmshape=tmshape,
                                   logscale=logscale, smoothing=smoothing,
                                   request=request)

def setup_sigvisa_graph(fit_params, tmshape, evid, wave):
    event = get_event(evid=evid)
    sg = SigvisaGraph(phases = fit_params.keys(), template_shape=tmshape)
    assert(len(sg.toplevel_nodes) == 0)
    assert(len(sg.leaf_nodes) == 0)
    wave_node = sg.add_wave(wav)

    sg.add_event(event)

    for (phase, vals) in fit_params.items():
        print "setting", phase, vals
        if phase.startswith("UA"):
            sg.create_unassociated_template(wave_node, atime=vals['arrival_time'], initial_vals=vals)
        else:
            sg.set_template(eid=event.eid, sta=wave['sta'],
                            phase=phase, band=wave['band'],
                            chan=wave['chan'], values= vals)
    for wm in sg.wiggle_nodes:
        wm.parent_predict()
    return sg

def template_debug_view(request, fitid):
    fit = get_object_or_404(SigvisaCodaFit, pk=fitid)
    logscale = request.GET.get("logscale", "off")
    smoothing = int(request.GET.get("smooth", "0"))

    fit_params = phases_from_request(request)
    print fit_params
    if request.GET.get("action", "").startswith("delete"):
        badphase = str(request.GET.get("action", None).split('_')[1])
        del fit_params[badphase]

    if len(fit_params) == 0:
        fit_params = phases_from_fit(fit)

    wave = wave_from_fit(fit)

    fit_phases = fit.sigvisacodafitphase_set.all()
    tmshape=tmshape_from_fit(fit)

    sg = setup_sigvisa_graph(evid=fit.evid, tmshape=tmshape, wave=wave, fit_params=fit_params)
    wave_node = sg.get_wave_node(wave=wave)
    wave_node.set_noise_model(nmid=fit.nmid.nmid)

    ll = wave_node.log_p()
    param_ll = sg.current_log_p()
    wave_node.unfix_value()
    wave_node.parent_predict()
    nm = wave_node.nm

    data_path = os.path.join(os.getenv("SIGVISA_HOME"), 'logs', 'web_debug')
    ensure_dir_exists(data_path)
    data_hash = hashlib.sha1(repr(fit_params) + repr(fitid)).hexdigest()[:8]
    env_fname = os.path.join(data_path, 'template_debug_%s_env.txt' % data_hash)
    np.savetxt(env_fname, wave.data)
    generated_fname = os.path.join(data_path, 'template_debug_%s_generated.txt' % data_hash)
    np.savetxt(generated_fname, wave_node.get_value())

    if request.GET.get("action", "") == "newrow":
        (k,v) = fit_params.items()[0]
        fit_params[k+"copy"] = v.copy()

    phase_GET_string = getstr_from_phases(fit_params, logscale=logscale, smooth=smoothing)

    phase_objs = []
    for (phase, vals) in fit_params.items():
        p = dict(vals.items() + [('phase', phase),] )
        phase_objs.append(p)

    return render_to_response('svweb/template_debug.html', {
        'fitid': fitid,
        'fit': fit,
        'phases': phase_objs,
        'logscale': logscale,
        'smoothing': smoothing,
        'phase_GET_string': phase_GET_string,
        'll': ll,
        'param_ll': param_ll,
        'nm': nm,
        'env_fname': env_fname,
        'generated_fname': generated_fname,
    }, context_instance=RequestContext(request))


def template_residual_view(request, fitid):
    fit = get_object_or_404(SigvisaCodaFit, pk=fitid)

    fit_params = phases_from_request(request)
    if len(fit_params) == 0:
        fit_params = phases_from_fit(fit)

    wave = wave_from_fit(fit)

    sg = setup_sigvisa_graph(evid=fit.evid, tmshape=tmshape_from_fit(fit), wave=wave, fit_params=fit_params)
    wave_node = sg.get_wave_node(wave=wave)
    wave_node.set_noise_model(nmid=fit.nmid.nmid)
    wave_node.unfix_value()
    wave_node.parent_predict()

    diff = Waveform(data =wave.data - wave_node.get_value(), segment_stats = wave.segment_stats, my_stats=wave.my_stats)

    return view_wave(request, diff)


def rate_fit(request, fitid):
    fit = SigvisaCodaFit.objects.get(fitid=int(fitid))

    next_fitid = int(request.GET.get("next_fitid", None))

    try:
        rating = int(request.POST['approval'])
    except KeyError:
        return HttpResponse("You didn't select a rating.")
    else:
        fit.human_approved = rating
        fit.save()
        return HttpResponseRedirect(reverse('fit_run_detail', args=(next_fitid,)) + "?" + request.GET.urlencode())
    return HttpResponse("Something went wrong.")

def delete_fit(request, fitid):
    try:
        fit = SigvisaCodaFit.objects.get(fitid=int(fitid))
        fit.delete()
        return HttpResponse("Fit %d deleted. <a href=\"javascript:history.go(-1)\">Go back</a>." % int(fitid))
    except Exception as e:
        return HttpResponse("Error deleting fit %d: %s" % (int(fitid), str(e)))


def delete_run(request, runid):
    try:
        run = SigvisaCodaFittingRun.objects.get(runid=int(runid))
        run.delete()
        return HttpResponse("Run %d deleted. <a href=\"javascript:history.go(-1)\">Go back</a>." % int(runid))
    except Exception as e:
        return HttpResponse("Error deleting run %d: %s" % (int(runid), str(e)))


def fit_cost_quality(request, runid):
    run = SigvisaCodaFittingRun.objects.get(runid=int(runid))

    fig = Figure(figsize=(5, 3), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    fig.suptitle("%s iter %d fit quality" % (run.run_name, run.iter))

    axes.set_xlabel("Acost")
    axes.set_ylabel("mb")

    unknown = request.GET.get("unknown", "True").lower().startswith('t')
    good = request.GET.get("good", "True").lower().startswith('t')
    bad = request.GET.get("bad", "True").lower().startswith('t')

    if good:
        good_fits = run.sigvisacodafit_set.filter(human_approved=2)
        g = np.array([(fit.acost, get_event(evid=fit.evid).mb) for fit in good_fits])
        if good_fits.count() > 0:
            axes.scatter(g[:, 0], g[:, 1], c='g', alpha=0.5, s=10, marker='.', edgecolors='none')

    if bad:
        bad_fits = run.sigvisacodafit_set.filter(human_approved=1)
        b = np.array([(fit.acost, get_event(evid=fit.evid).mb) for fit in bad_fits])
        if bad_fits.count() > 0:
            axes.scatter(b[:, 0], b[:, 1], c='r', alpha=0.5, s=10, marker='.', edgecolors='none')

    if unknown:
        unknown_fits = run.sigvisacodafit_set.filter(human_approved=0)
        u = np.array([(fit.acost, get_event(evid=fit.evid).mb) for fit in unknown_fits])
        if unknown_fits.count() > 0:
            axes.scatter(u[:, 0], u[:, 1], c='b', alpha=0.5, s=10, marker='.', edgecolors='none')

    process_plot_args(request, axes)

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response

def suspicious_fit_view(request, runid):
    runid = int(runid)
    s = Sigvisa()
    suspicion_cutoff = 4.0

    #suspicious_fitids = s.sql("select distinct f.fitid from sigvisa_coda_fit f, sigvisa_coda_fit_phase fp, sigvisa_ttr_consistency tc where f.fitid=fp.fitid and fp.fpid=tc.fpid and f.runid=%d and abs(tc.tt_residual - tc.ttr_neighbor_median) > %f" % (runid, suspicion_cutoff))

    suspicious_fitids = s.sql("select distinct f.fitid from sigvisa_coda_fit f where f.human_approved='3'")

    fit_info = []
    for (fitid,) in suspicious_fitids:
        r = s.sql("select fp.phase, tc.tt_residual, tc.ttr_neighbor_median, tc.ttr_neighbor_stddev, tc.neighbor_fitids from sigvisa_coda_fit_phase fp, sigvisa_ttr_consistency tc where fp.fitid=%d and fp.fpid=tc.fpid" % (fitid))


        phases = []
        zs = []
        for (phase, ttr, median, stddev, neighbor_str) in r:
            z = np.abs(ttr-median)/stddev
            phases.append((phase, ttr, median, stddev, np.abs(ttr-median), z))
            zs.append(z)

        neighbors = eval(neighbor_str)

        fit_info.append((fitid, phases, neighbors))


    return render_to_response("svweb/suspicious_fits.html",
                              {'fit_info': fit_info,
                               'runid': runid,
                               'discr_threshold': suspicion_cutoff
                               }, context_instance=RequestContext(request))
