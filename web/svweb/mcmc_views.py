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
import numpy.ma as ma
import sys
import os
import time
import cPickle as pickle
from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *
from sigvisa.utils.geog import azimuth_gap
from sigvisa.infer.analyze_mcmc import load_trace, trace_stats
from sigvisa.models.ttime import tt_predict
from sigvisa.plotting.plot import plot_with_fit_shapes, plot_pred_atimes, subplot_waveform
from sigvisa.plotting.event_heatmap import EventHeatmap
from sigvisa.plotting.heatmap import event_bounds, find_center

from sigvisa.signals.io import Waveform
from sigvisa import *

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from stat import S_ISREG, ST_CTIME, ST_MODE





def mcmc_list_view(request, sort_by_time=False):

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")

    mcmc_run_dirs = os.listdir(mcmc_log_dir)

    mcmc_runs = []
    for rundir in mcmc_run_dirs:
        mcmcrun = dict()
        mcmcrun['dir'] = rundir
        mcmcrun['machine_time'] = os.path.getctime(os.path.join(mcmc_log_dir, rundir))
        mcmcrun['time'] = str(time.ctime(mcmcrun['machine_time']))

        steps = [int(d[5:]) for d in os.listdir(os.path.join(mcmc_log_dir, rundir)) if d.startswith("step_")]
        if len(steps) == 0:
            mcmcrun['steps'] = 0
        else:
            mcmcrun['steps'] = np.max(steps)

        cmd = ""
        try:
            with open(os.path.join(mcmc_log_dir, rundir, 'cmd.txt'), 'r') as f:
                cmd = f.read()
        except IOError:
            pass
        mcmcrun['cmd'] = cmd
        mcmc_runs.append(mcmcrun)

    if sort_by_time:
        mcmc_runs = sorted(mcmc_runs, key = lambda  r : r['machine_time'], reverse=True)
    else:
        mcmc_runs = sorted(mcmc_runs, key = lambda  r : r['dir'], reverse=True)

    print "loaded %d mcmc runs" % (len(mcmc_run_dirs),)

    return render_to_response("svweb/mcmc_runs.html",
                              {'run_list': mcmc_runs,
                               }, context_instance=RequestContext(request))


def graphs_by_step(ev_dir, start_step=0, end_step=9999999):
    sgs = dict()
    for d in os.listdir(ev_dir):
        if not d.startswith("step_"): continue
        n_step = int(d[5:])

        if n_step < start_step or n_step > end_step:
            continue

        with open(os.path.join(ev_dir, d, 'pickle.sg'), 'rb') as f:
            sgs[n_step] = pickle.load(f)
    return sgs

def graph_for_step(ev_dir, step):
    with open(os.path.join(ev_dir, "step_%06d" % step, 'pickle.sg'), 'rb') as f:
        sg = pickle.load(f)
    return sg

def final_mcmc_state(ev_dir):
    max_step = np.max([int(d[5:]) for d in os.listdir(ev_dir) if d.startswith('step')])
    sg = graph_for_step(ev_dir, max_step)
    return sg, max_step

def mcmc_run_detail(request, dirname):
    s = Sigvisa()
    burnin = int(request.GET.get('burnin', '-1'))
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    cmd = "not found"
    try:
        with open(os.path.join(mcmc_log_dir, dirname, 'cmd.txt'), 'r') as f:
            cmd = f.read()
    except IOError:
        pass

    relative_run_dir = os.path.join("logs", "mcmc", dirname)
    analyze_cmd = "python infer/analyze_mcmc.py %s 10 %s/events.pkl t" % (relative_run_dir, relative_run_dir)


    try:
        with open(os.path.join(mcmc_run_dir, 'events.pkl'), 'rb') as f:
            true_evs = pickle.load(f)
    except Exception as e:
        print e
        true_evs = []

    sg, max_step = final_mcmc_state(mcmc_run_dir)


    stas = sg.station_waves.keys()
    try:
        gp_hparams = sg.jointgp_hparam_prior.keys()
    except AttributeError:
        gp_hparams = None

    wns = [n.label for n in np.concatenate(sg.station_waves.values())]

    eids = sg.evnodes.keys()
    evs = []
    site_names = sg.site_elements.keys()
    site_info = np.array([s.earthmodel.site_info(sta, 0) for sta in site_names])

    if true_evs is not None:
        true_ev_strs = [str(ev) for ev in true_evs]
    else:
        true_ev_strs = []

    if burnin < 0:
        burnin = 100 if max_step > 150 else 10
    for eid in eids:
        ev_trace_file = os.path.join(mcmc_run_dir, 'ev_%05d.txt' % eid)
        trace, _, _ = load_trace(ev_trace_file, burnin=burnin)

        llon, rlon, blat, tlat = event_bounds(trace)

        results, txt = trace_stats(trace, true_evs)
        ev = sg.get_event(eid)
        evdict = {'eid': eid,
                  'evstr': str(ev),
                  'azgap': 0.0, #azimuth_gap(ev.lon, ev.lat, site_info),
                  'dist_mean': results['dist_mean'] if "dist_mean" in results else "",
                  'lon_std_km': results['lon_std_km'] if "lon_std_km" in results else "",
                  'lat_std_km': results['lat_std_km'] if "lat_std_km" in results else "",
                  'top_lat': tlat,
                  'bottom_lat': blat,
                  'left_lon': llon,
                  'right_lon': rlon,
                  'example_node': sg.extended_evnodes[eid][4].label,
        }
        evs.append(evdict)

    return render_to_response("svweb/mcmc_run_detail.html",
                              {'wns': wns,
                               'dirname': dirname,
                               'full_dirname': mcmc_run_dir,
                               'cmd': cmd,
                               'analyze_cmd': analyze_cmd,
                               'max_step': max_step,
                               'evs': evs,
                               'true_ev_strs': true_ev_strs,
                               'stas': stas,
                               'gp_hparams': gp_hparams,
                               }, context_instance=RequestContext(request))

def rundir_eids(mcmc_run_dir):
    eids = []
    ev_re = re.compile(r'ev_(\d+).txt')
    for fname in os.listdir(mcmc_run_dir):
        m = ev_re.match(fname)
        if m is not None:
            eid = int(m.group(1))
            eids.append(eid)
    return eids

def mcmc_hparam_posterior(request, dirname, sta, hparam):


    def plot_dist(ax, samples):
        import seaborn as sns
        sns.distplot(samples, ax=ax)

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    burnin = int(request.GET.get('burnin', '-1'))


    sg, max_step = final_mcmc_state(mcmc_run_dir)
    wn = sg.station_waves[sta][0]
    srate = float(wn.srate)

    # analyze the wavelet basis to lay out the visualization
    (starray, etarray, idarray, M, N), target_std = wn.wavelet_basis
    ids_by_length = {idarray[i]: etarray[i]-starray[i] for i in range(len(starray))}

    # longest basis elements first
    ids_sorted = sorted(ids_by_length.keys(), key = lambda k : -ids_by_length[k])

    st_range = np.max(starray) - np.min(starray)
    # time resolution of the shortest element
    res = np.min(np.diff(starray[idarray==ids_sorted[-1]]))
    nbins = int(st_range/res)+1


    # load hparam samples from file
    ffname = os.path.join(mcmc_run_dir, "gp_hparams", "%s_%s" % (sta, hparam))
    a = np.loadtxt(ffname)
    if burnin < 0:
        burnin = 100 if a.shape[0] > 150 else 10
    assert(a.shape[1]==len(starray))

    true_vals = None
    try:
        # for each sta, for each hparam, for each param, we'll have a true hparam
        with open(os.path.join(mcmc_run_dir, "gp_hparams", "true.pkl"), 'rb') as f:
            true_dict = pickle.load(f)
        true_vals = true_dict[sta][hparam]
    except IOError:
        pass

    f = Figure((16, 8))
    f.patch.set_facecolor('white')
    gs = gridspec.GridSpec(len(ids_sorted), nbins)

    firstax=dict()
    for i in range(a.shape[1]):
        st, et, pid = starray[i], etarray[i], idarray[i]
        id_idx = ids_sorted.index(pid)

        st_idx = (st-np.min(starray))/res

        my_id_sts = starray[idarray==idarray[i]]
        width = (my_id_sts[1] - my_id_sts[0])/res


        shared_ax = firstax[pid] if pid in firstax else None
        ax = f.add_subplot(gs[id_idx,st_idx:st_idx+width],
                           sharey=shared_ax,
                           sharex=shared_ax)
        #ax.patch.set_facecolor('white')

        samples = a[burnin:, i]
        plot_dist(ax, samples)
        ax.set_xticks(np.linspace(0, np.max(a), 3))

        if true_vals is not None:
            ax.axvline(true_vals[i], lw=1, color="green")

        if pid not in firstax:
            firstax[pid] = ax
        else:
            ax.get_yaxis().set_visible(False)
        t_start = max(0, starray[i]/srate)
        t_end = etarray[i]/srate
        ax.set_title("%d" % i)
        ax.annotate('%.1fs:%.1fs\nmean %.1f\nstd %.1f' % (t_start, t_end, np.mean(samples), np.std(samples)), (0.3, 0.85), xycoords='axes fraction', size=10)




    canvas = FigureCanvas(f)
    response = django.http.HttpResponse(content_type='image/png')
    f.tight_layout()
    canvas.print_png(response)
    return response


#@cache_page(60 * 60 * 60)
def mcmc_event_posterior(request, dirname):
    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    left_lon = float(request.GET.get('left_lon', '-180'))
    right_lon = float(request.GET.get('right_lon', '180'))
    top_lat = float(request.GET.get('top_lat', '90'))
    bottom_lat = float(request.GET.get('bottom_lat', '-90'))
    burnin = int(request.GET.get('burnin', '100'))
    plot_true = request.GET.get('plot_true', 't').lower().startswith('t')
    plot_train = request.GET.get('plot_train', 't').lower().startswith('t')
    plot_mean = request.GET.get('plot_mean', 't').lower().startswith('t')
    stds = float(request.GET.get('stds', '-1'))


    horiz_deg = right_lon-left_lon
    vert_deg = top_lat-bottom_lat
    aspect_ratio = horiz_deg / vert_deg

    proj = "cyl"
    if top_lat==90 and bottom_lat==-90 and left_lon==-180 and right_lon==180:
        proj="robin"

    sg, max_step = final_mcmc_state(mcmc_run_dir)

    sites = sg.site_elements.keys()

    f = Figure((8*aspect_ratio, 8))
    f.patch.set_facecolor('white')
    ax = f.add_subplot(111)

    hm = EventHeatmap(f=None, calc=False, left_lon=left_lon, right_lon=right_lon, top_lat=top_lat, bottom_lat=bottom_lat)
    hm.add_stations(sites)
    hm.init_bmap(axes=ax, nofillcontinents=True, projection=proj, resolution="c")

    hm.plot(axes=ax, nolines=True, smooth=True,
            colorbar_format='%.3f')


    eids = rundir_eids(mcmc_run_dir)
    import seaborn as sns
    shape_colors = sns.color_palette("hls", len(eids))
    eid_patches = []
    eid_labels = []
    for eid_i, eid in enumerate(sorted(eids)):
        ev_trace_file = os.path.join(mcmc_run_dir, 'ev_%05d.txt' % eid)
        trace, min_step, max_step = load_trace(ev_trace_file, burnin=burnin)
        n = trace.shape[0]
        print eid, trace.shape
        if len(trace.shape)==2:
            if stds <= 0:
                scplot = hm.plot_locations(trace[:, 0:2], marker=".", ms=8, mfc=shape_colors[eid_i-1], mew=0, mec="none", alpha=1.0/np.log(n+1))
            eid_patches.append(mpatches.Patch(color=shape_colors[eid_i-1]))
            eid_labels.append('%d' % eid)

            if plot_mean:
                clon, clat = find_center(trace[:, 0:2])
                loc = np.array(((clon, clat), ))
                hm.plot_locations(loc,  labels=None, marker="+", ms=16, mfc="none", mec=shape_colors[eid_i-1], mew=2, alpha=1)

            if stds > 0:
                 m = np.mean(trace[:, 0:2], axis=0)
                 centered_trace = trace[:, 0:2] - m
                 cov = np.dot(centered_trace.T, centered_trace)/float(centered_trace.shape[0])
                 hm.plot_covs([m,], [cov,], stds=stds, colors=[shape_colors[eid_i-1],], alpha=0.2)

    f.legend(handles=eid_patches, labels=eid_labels)

    if plot_true:
        try:
            with open(os.path.join(mcmc_run_dir, 'events.pkl'), 'rb') as evfile:
                true_evs = pickle.load(evfile)
        except Exception as e:
            print e
            true_evs = []
        if true_evs is None:
            true_evs = []
        for ev in true_evs:
            loc = np.array(((ev.lon, ev.lat), ))
            hm.plot_locations(loc,  labels=None, marker="*", ms=16, mfc="none", mec="#44FF44", mew=2, alpha=1)

    if plot_train:
        try:
            with open(os.path.join(mcmc_run_dir, 'train_events.pkl'), 'rb') as evfile:
                train_evs = pickle.load(evfile)
        except Exception as e:
            print e
            train_evs = []
        if train_evs is None:
            train_evs = []
        for ev in train_evs:
            loc = np.array(((ev.lon, ev.lat), ))
            hm.plot_locations(loc,  labels=None, marker="*", ms=8, mfc="none", mec="#448844", mew=2, alpha=1)

        try:
            with open(os.path.join(mcmc_run_dir, 'obs_events.pkl'), 'rb') as evfile:
                obs_evs = pickle.load(evfile)
        except Exception as e:
            print e
            obs_evs = []
        if obs_evs is None:
            obs_evs = []
        for ev in obs_evs:
            loc = np.array(((ev.lon, ev.lat), ))
            hm.plot_locations(loc,  labels=None, marker="*", ms=8, mfc="none", mec="#887722", mew=2, alpha=1)




    canvas = FigureCanvas(f)
    response = django.http.HttpResponse(content_type='image/png')
    f.tight_layout()
    canvas.print_png(response)
    return response


def mcmc_param_posterior(request, dirname, node_label):
    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    sgs = graphs_by_step(mcmc_run_dir)

    vals = [float(sg.all_nodes[node_label].get_value()) for sg in sgs.values()]



    return HttpResponse("key: %s\nsamples: %d\n\nmean: %.3f\nstd: %.3f\nmin: %.3f\nmax: %.3f" % (node_label, len(vals),  np.mean(vals), np.std(vals), np.min(vals), np.max(vals)), content_type="text/plain")


def mcmc_arrivals(request, dirname, wn_label, step):

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    sg = graph_for_step(mcmc_run_dir, int(step))

    wn = sg.all_nodes[wn_label]

    response = ""

    for (eid, phase) in sorted(wn.arrivals()):
        v, tg = wn.get_template_params_for_arrival(eid=eid, phase=phase)
        response += "eid %d, phase %s:\n" % (eid, phase)
        for (key, val) in v.items():
            response += " %s: %s\n" % (key, val)
        response += "\n"

    return HttpResponse(response, content_type="text/plain")


def mcmc_signal_posterior_page(request, dirname, wn_label):
    pass

def mcmc_signal_posterior_wave(request, dirname, wn_label, key1):
    zoom = float(request.GET.get("zoom", '1'))
    vzoom = float(request.GET.get("vzoom", '1'))

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    sg, max_step = final_mcmc_state(mcmc_run_dir)
    wn = sg.all_nodes[wn_label]

    arrival_info = wn.signal_component_means()

    if key1=="signal" or key1=="noise":
        d = arrival_info[key1]
    else:
        eid, phase = key1.split("_")
        eid = int(eid)
        key2 = request.GET.get("component", 'combined')
        d = arrival_info[(eid, phase)][key2]

    len_s = float(len(d))/wn.srate
    f = Figure((10.0 * zoom, 5*vzoom))
    f.patch.set_facecolor('white')
    axes = f.add_subplot(111)
    t = np.linspace(0, len_s, len(d))
    axes.plot(t, d)

    canvas = FigureCanvas(f)
    response = django.http.HttpResponse(content_type='image/png')
    f.tight_layout()
    canvas.print_png(response)
    return response


def mcmc_wave_posterior(request, dirname, wn_label):

    zoom = float(request.GET.get("zoom", '1'))
    vzoom = float(request.GET.get("vzoom", '1'))
    pred_signal = request.GET.get("pred_signal", 'false').lower().startswith('t')
    pred_signal_var = request.GET.get("pred_signal_var", 'false').lower().startswith('t')
    plot_predictions = request.GET.get("plot_predictions", 'true').lower().startswith('t')
    plot_dets = request.GET.get("plot_dets", 'leb')
    plot_template_arrivals = request.GET.get("plot_templates", 'true').lower().startswith('t')
    step = request.GET.get("step", 'all')

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    if step=="all":
        sgs = graphs_by_step(mcmc_run_dir)
    else:
        sgs = {int(step): graph_for_step(mcmc_run_dir, int(step))}

    last_step = np.max(sgs.keys())
    last_sg = sgs[last_step]

    ev_alpha = 0.8 / len(sgs)
    ua_alpha = 0.4/len(sgs)

    nevents = last_sg.next_eid-1

    wn = last_sg.all_nodes[wn_label]
    real_wave = wn.get_wave()
    real_wave.data = ma.masked_array(real_wave.data, copy=True)
    len_mins = (wn.et - wn.st) / 60.0

    f = Figure((10*zoom, 5*vzoom))
    f.patch.set_facecolor('white')
    axes = f.add_subplot(111)
    subplot_waveform(wn.get_wave(), axes, color='black', linewidth=1.5, plot_dets=None)

    import matplotlib.cm as cm
    shape_colors = dict([(eid, cm.get_cmap('jet')(np.random.rand()*.5)) for (eid, phase) in wn.arrivals()])
    steps = sgs.keys()
    alpha = 1.0/len(steps)
    for step in steps:
        wn = sgs[step].all_nodes[wn_label]

        if pred_signal:
            wn._parent_values()
            pred_signal = wn.tssm.mean_obs(wn.npts)
            w = Waveform(pred_signal, srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
            subplot_waveform(w, axes, color='green', linewidth=2.5)
            if pred_signal_var:
                signal_var = wn.tssm.obs_var(wn.npts)
                w1 = Waveform(pred_signal+2*np.sqrt(signal_var), srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
                subplot_waveform(w1, axes, color='red', linewidth=1.0)
                w2 = Waveform(pred_signal-2*np.sqrt(signal_var), srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
                subplot_waveform(w2, axes, color='red', linewidth=1.0)
        else:
            shape_colors = plot_with_fit_shapes(fname=None, wn=wn,title=wn_label, axes=axes, plot_dets=plot_dets, shape_colors=shape_colors, plot_wave=False, alpha=alpha)

        if plot_predictions:
            predictions = []
            for (eid, phase) in wn.arrivals():
                if eid < 0: continue
                event = sgs[step].get_event(eid)
                predictions.append([phase+"_%d" % eid, event.time+tt_predict(event, wn.sta, phase)])
            plot_pred_atimes(dict(predictions), wn.get_wave(), axes=axes, color="purple", alpha=alpha, draw_text=False)

    if plot_predictions:
        plot_pred_atimes(dict(predictions), wn.get_wave(), axes=axes, color="purple", alpha=1.0, draw_bars=False)

    if plot_template_arrivals:
        atimes = dict([("%d_%s" % (eid, phase), wn.get_template_params_for_arrival(eid=eid, phase=phase)[0]['arrival_time']) for (eid, phase) in wn.arrivals()])
        colors = dict([("%d_%s" % (eid, phase), shape_colors[eid]) for (eid, phase) in wn.arrivals()])
        plot_pred_atimes(dict(atimes), wn.get_wave(), axes=axes, color=colors, alpha=1.0, bottom_rel=-0.1, top_rel=0.0)

    canvas = FigureCanvas(f)
    response = django.http.HttpResponse(content_type='image/png')
    f.tight_layout()
    canvas.print_png(response)
    return response


def mcmcrun_analyze(request, dirname):

    from sigvisa.infer.analyze_mcmc import analyze_run

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    burnin=int(request.GET.get('burnin', '100'))
    plot_template_posteriors=request.GET.get('plot_templates', 'False').lower().startswith('t')

    try:
        with open(os.path.join(mcmc_run_dir, "events.pkl"), 'rb') as f:
            evs = pickle.load(f)
    except IOError:
        evs = []

    analyze_run(run_dir=mcmc_run_dir, burnin=burnin, true_evs=evs, plot_template_posteriors=plot_template_posteriors)
    return HttpResponse("Ran analysis with %d steps of burnin on MCMC run %s. Check server logs for details. <a href=\"javascript:history.go(-1)\">Go back</a>." % (burnin, dirname))

import mimetypes

def mcmcrun_browsedir(request, dirname, path):

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    current_path = os.path.join(mcmc_run_dir, path)

    if os.path.isdir(current_path):

        fnames = sorted(os.listdir(current_path))

        files = []
        for fname in fnames:
            d = os.path.isdir(os.path.join(current_path, fname))
            t = mimetypes.guess_type(fname)[0] or 'application/octetstream'
            files.append((fname,d,t))


        return render_to_response("svweb/filesystem_directory.html",
                                  {'dlist': [f for (f, d, t) in files if d],
                                   'flist': [{'name':f, 'type':t} for (f, d, t) in files if not d],
                                   'path': path, 'dirname': dirname
                               }, context_instance=RequestContext(request))

    else:
        mimetype=mimetypes.guess_type(path)[0]
        return HttpResponse(open(current_path).read(), mimetype=mimetype)
