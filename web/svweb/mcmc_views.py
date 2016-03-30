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
from sigvisa.results.compare import find_matching
from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *
from sigvisa.utils.geog import azimuth_gap, dist_km
from sigvisa.infer.analyze_mcmc import load_trace, trace_stats, match_true_ev
from sigvisa.infer.template_xc import get_arrival_signal, fastxc
from sigvisa.infer.correlations.ar_correlation_model import ar_advantage, iid_advantage
from sigvisa.models.ttime import tt_predict
from sigvisa.plotting.plot import plot_with_fit_shapes, plot_pred_atimes, subplot_waveform
from sigvisa.plotting.event_heatmap import EventHeatmap
from sigvisa.plotting.heatmap import event_bounds, find_center
from sigvisa.utils.array import time_to_index, index_to_time
from sigvisa.signals.io import Waveform
from sigvisa import *

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from stat import S_ISREG, ST_CTIME, ST_MODE





def mcmc_list_view(request, sort_by_time=True):

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

def graph_from_file(run_dir, fname):
    with open(os.path.join(run_dir, fname), 'rb') as f:
        sg = pickle.load(f)
    return sg

def mcmc_lp_posterior(request, dirname):
    s = Sigvisa()

    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    lps = np.loadtxt(os.path.join(mcmc_run_dir, "lp.txt"))
    times = np.loadtxt(os.path.join(mcmc_run_dir, "times.txt"))

    n = min(len(lps), len(times))
    if n > 1:
        n -= 1 # avoid partially written final lines
    lps = lps[:n]
    times = times[:n]

    f = Figure()
    f.patch.set_facecolor('white')
    ax = f.add_subplot(111)
    ax.plot(times, lps)
    ax.set_xlabel("elapsed time")
    ax.set_ylabel("log density")


    n = len(lps)
    recent_lps = lps[n/2:]
    lpmin, lpmax = np.min(recent_lps), np.max(recent_lps)
    lprange = lpmax-lpmin
    print n, recent_lps, lpmin, lpmax, lprange
    ax.set_ylim((lpmin-lprange/5.0, lpmax+lprange/5.0))

    canvas = FigureCanvas(f)
    response = django.http.HttpResponse(content_type='image/png')
    f.tight_layout()
    canvas.print_png(response)
    return response

def mcmc_ev_detail(request, dirname, eid_str):
    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    pred_signal = request.GET.get('pred_signal', 'false').startswith("t")
    if pred_signal:
        extra_wave_args=";pred_signal=true;pred_signal_var=true"
    else:
        extra_wave_args=";pred_env=True"


    step = int(request.GET.get('step', '-1'))
    sgfile = str(request.GET.get('sgfile', ''))
    if sgfile:
        sg = graph_from_file(mcmc_run_dir, sgfile)
    else:
        if step < 0:
            max_step = np.max([int(d[5:]) for d in os.listdir(mcmc_run_dir) if d.startswith('step')])
            step = max_step
        sg = graph_for_step(mcmc_run_dir, step)

    if eid_str[:4]=="true":
        with open(os.path.join(mcmc_run_dir, 'events.pkl'), 'rb') as f:
            true_evs = pickle.load(f)
        idx = int(eid_str[4:])
        ev = true_evs[idx]
        eid = 9999
        sg.add_event(ev, eid=eid)
    else:
        eid = int(eid_str)
        ev = sg.get_event(eid)
    ev_str = str(ev)

    r = []
    for sta, wns in sorted(sg.station_waves.items()):
        for wn in wns:
            arrs = [(eid2, phase) for (eid2, phase) in wn.arrivals() if eid2==eid]
            if len(arrs) == 0: continue
            tms = [wn.get_template_params_for_arrival(eid, phase)[0] for (eid, phase) in arrs]
            atimes = [tm["arrival_time"] for tm in tms]
            min_atime = np.min(atimes)
            max_atime = np.max(atimes)
            r.append((wn.label, min_atime-10, max_atime + 60, repr(tms)))

    proposalpath = "ev_%05d/proposal.png" % eid
    sgfilestr = "sgfile=%s" % sgfile if sgfile else ""
    return render_to_response("svweb/mcmc_ev_detail.html",
                              {'r': r,
                               'dirname': dirname,
                               'full_dirname': mcmc_run_dir,
                               'eid': eid,
                               'ev_str': ev_str,
                               'step': step,
                               'sgfilestr': sgfilestr,
                               'proposalpath': proposalpath,
                               'extra_wave_args': extra_wave_args,
                               }, context_instance=RequestContext(request))


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

    true_evs = sorted(true_evs, key = lambda ev : ev.time)

    sgfile = str(request.GET.get('sgfile', ''))
    if sgfile:
        sg = graph_from_file(mcmc_run_dir, sgfile)
        max_step = 0
    else:
        sg, max_step = final_mcmc_state(mcmc_run_dir)


    stas = sg.station_waves.keys()
    gp_hparams = []
    try:
        if len(sg._jointgp_hparam_nodes) > 0:
            gp_hparams = ["wiggles", "template"]
    except AttributeError:
        pass

    #except AttributeError:
    #    gp_hparams = None

    wns = []
    for sta in sorted(sg.station_waves.keys()):
        wns.append((sta, [n.label for n in sg.station_waves[sta]]))

    eids = sg.evnodes.keys()
    evs = []
    site_names = sg.site_elements.keys()
    site_info = np.array([s.earthmodel.site_info(sta, 0) for sta in site_names])

    if true_evs is not None:
        true_ev_strs = [("true%d" % idx,str(ev)) for idx,ev in enumerate(true_evs)]
    else:
        true_ev_strs = []

    if burnin < 0:
        burnin = 100 if max_step > 150 else 10 if max_step > 10 else 0

    X = []

    eids = sorted(eids)
    inferred_evs = [sg.get_event(eid) for eid in eids]
    trueX = [(ev.lon, ev.lat, ev.depth, ev.time, ev.mb) for ev in true_evs]
    inferredX = [(ev.lon, ev.lat, ev.depth, ev.time, ev.mb) for ev in inferred_evs]
    indices = find_matching(trueX, inferredX, max_delta_deg=5.0, max_delta_time=100.0)

    phases_used = sg.phases

    for i, eid in enumerate(eids):
        ev_trace_file = os.path.join(mcmc_run_dir, 'ev_%05d.txt' % eid)
        trace, _, _ = load_trace(ev_trace_file, burnin=burnin)

        llon, rlon, blat, tlat = event_bounds(trace)
        X.append([llon, tlat])
        X.append([llon, blat])
        X.append([rlon, tlat])
        X.append([rlon, blat])

        results, txt = trace_stats(trace, true_evs)
        ev = sg.get_event(eid)
        try:
            enode = sg.extended_evnodes[eid][4].label
        except:
            enode = None

        matches = [idx for (idx, j) in indices if j==i]
        matched = ""
        dist = ""
        if len(matches) > 0:
            matched = matches[0] + 1
            true_ev = true_evs[matched-1]
            dist = dist_km((ev.lon, ev.lat), (true_ev.lon, true_ev.lat))

        evdict = {'eid': eid,
                  'evstr': str(ev),
                  'azgap': 0.0, #azimuth_gap(ev.lon, ev.lat, site_info),
                  'matched': matched,
                  'dist': dist,
                  'lon_std_km': results['lon_std_km'] if "lon_std_km" in results else "",
                  'lat_std_km': results['lat_std_km'] if "lat_std_km" in results else "",
                  'top_lat': tlat,
                  'bottom_lat': blat,
                  'left_lon': llon,
                  'right_lon': rlon,
                  'example_node': enode,
        }
        evs.append(evdict)

    X = np.array(X, dtype=np.float)
    if len(X) > 0:
        if len(trueX) > 0:
            X = np.vstack((X, np.asarray(trueX)[:, :2]))
    else:
        if len(trueX) > 0:
            X = np.asarray(trueX)[:, :2]

    if sg.inference_region is not None:
        r = sg.inference_region
        left_bound, right_bound = r.left_lon, r.right_lon
        bottom_bound, top_bound = r.bottom_lat, r.top_lat
    else:
        try:
            left_bound, right_bound, bottom_bound, top_bound = event_bounds(X, quantile=0.9999)
        except:
            left_bound, right_bound, bottom_bound, top_bound= -180, 180, -90, 90

    bounds = dict()
    bounds["top"] = top_bound + 0.2
    bounds["bottom"] = bottom_bound - 0.2
    bounds["left"] = left_bound - 0.2
    bounds["right"] = right_bound + 0.2

    jointgps = (sg.wiggle_model_type=="gp_joint")
    sgfilestr = "sgfile=%s" %sgfile if sgfile else ""
    return render_to_response("svweb/mcmc_run_detail.html",
                              {'wns': wns,
                               'phases_used': phases_used,
                               'dirname': dirname,
                               'full_dirname': mcmc_run_dir,
                               'cmd': cmd,
                               'analyze_cmd': analyze_cmd,
                               'max_step': max_step,
                               'evs': evs,
                               'true_ev_strs': true_ev_strs,
                               'stas': stas,
                               'gp_hparams': gp_hparams,
                               'bounds': bounds,
                               'jointgps': jointgps,
                               'sgfilestr': sgfilestr
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

def conditional_signal_posterior(request, dirname, sta, phase):
    s = Sigvisa()

    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    sgfile = str(request.GET.get('sgfile', ''))
    if sgfile:
        sg = graph_from_file(mcmc_run_dir, sgfile)
        max_step = 0
    else:
        sg, max_step = final_mcmc_state(mcmc_run_dir)

    from sigvisa.models.wiggles.wavelets import implicit_to_explicit


    (starray, etarray, idarray, M, levels, N)= sg.station_waves.values()[0][0].wavelet_basis
    prototypes = [np.asarray(m).flatten() for m in M]
    basis = implicit_to_explicit(starray, etarray, idarray, prototypes, levels, N)
    n_basis = np.sum(levels[:-1])
    print n_basis

    eids = sg.evnodes.keys()

    eid_wiggle_posteriors = dict()

    wiggle_gpmodels = dict([(k, v) for (k, v) in sg._joint_gpmodels[sta].items() if k[0].startswith("db")])

    holdout_evidence = dict()
    for eid in eids:
        try:
            holdout_evidence[eid] = np.sum([jgp.holdout_evidence(eid) for jgp, wnodes in wiggle_gpmodels.values()])
        except KeyError:
            continue

    n = len(eids)
    f = Figure((16, 4*n))
    f.patch.set_facecolor('white')
    gs = gridspec.GridSpec(n, 2)

    j = 0
    for wn in sg.station_waves[sta]:
        wn._parent_values()

        tssm = wn.transient_ssm()
        npts = wn.npts
        base_mean = tssm.mean_obs(npts)
        base_var = tssm.obs_var(npts)

        w = wn.get_value()
        lp1, marginals, step_ells = wn.tssm.all_filtered_cssm_coef_marginals(w)
        srate = wn.srate
        stime = wn.st
        timevals = np.arange(stime, stime + npts / srate, 1.0 / srate)[0:npts]

        posterior_means, posterior_vars = zip(*marginals)
        posterior_means, posterior_vars = np.concatenate(posterior_means), np.concatenate(posterior_vars)
        
        for i, (eid, pphase, _, sidx, cnpts, ctype) in enumerate(wn.tssm_components):
            if ctype != "wavelet": continue
            if pphase != phase: continue
    
            cond_means, cond_vars = zip(*[jgp.posterior(eid) for jgp in wn.wavelet_param_models[phase]])
            cond_means, cond_vars = np.asarray(cond_means, dtype=np.float64), np.asarray(cond_vars, dtype=np.float64)
            cssm = wn.arrival_ssms[(eid, phase)]
            cssm.set_coef_prior(cond_means, cond_vars)

            pred_mean = tssm.mean_obs(npts)
            pred_var = tssm.obs_var(npts)
            lp2 = tssm.run_filter(w)


            posterior_means[n_basis:] = 0.0
            posterior_vars[n_basis:] = 1.0
            cssm.set_coef_prior(posterior_means, posterior_vars)
            post_mean = tssm.mean_obs(npts)
            post_var = tssm.obs_var(npts)
            lp3 = tssm.run_filter(w)


            cstime = stime + sidx/srate - 5.0
            cetime= cstime + cnpts / srate + 5.0

            w_local = w[sidx : sidx + cnpts]
            ymin  = np.min(w_local) - .5
            ymax  = np.max(w_local) + .5

            ax = f.add_subplot(gs[j, 0])
            ax.plot(timevals, w)
            ax.plot(timevals, pred_mean, lw=2)
            #ax.plot(timevals, pred_mean+np.sqrt(pred_var))

            ax.fill_between(timevals, pred_mean+2*np.sqrt(pred_var), 
                            pred_mean-2*np.sqrt(pred_var), facecolor="green", alpha=0.2)

            ax.set_xlim([cstime, cetime])
            ax.set_ylim([ymin, ymax])

            ax.set_title("lp %f conditional" % (lp2))
            ax = f.add_subplot(gs[j, 1])
            #plot_wavelet_dist_samples(ax, wn.srate, basis, posterior_means, posterior_vars, c="blue")
            #plot_wavelet_dist_samples(ax, wn.srate, basis, cond_means, cond_vars, c="green")
            ax.plot(timevals, w)
            ax.plot(timevals, base_mean, lw=2)
            ax.fill_between(timevals, base_mean+2*np.sqrt(base_var), 
                            base_mean-2*np.sqrt(base_var), facecolor="green", alpha=0.2)

            ax.set_xlim([cstime, cetime])
            ax.set_ylim([ymin, ymax])            
            ax.set_title("lp %f prior" % (lp1))

            """
            ax = f.add_subplot(gs[j, 2])
            #plot_wavelet_dist_samples(ax, wn.srate, basis, posterior_means, posterior_vars, c="blue")
            #plot_wavelet_dist_samples(ax, wn.srate, basis, cond_means, cond_vars, c="green")
            ax.plot(timevals, w)
            ax.plot(timevals, post_mean, lw=2)
            ax.fill_between(timevals, post_mean+2*np.sqrt(post_var), 
                            post_mean-2*np.sqrt(post_var), facecolor="green", alpha=0.2)

            ax.set_xlim([cstime, cetime])            
            ax.set_title("lp %f filtered" % (lp3))
            ax.set_ylim([ymin, ymax])
            """

            j += 1


    canvas = FigureCanvas(f)
    response = django.http.HttpResponse(content_type='image/png')
    f.tight_layout()
    canvas.print_png(response)
    return response

def conditional_wiggle_posterior(request, dirname, sta, phase):
    s = Sigvisa()

    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    sgfile = str(request.GET.get('sgfile', ''))
    if sgfile:
        sg = graph_from_file(mcmc_run_dir, sgfile)
        max_step = 0
    else:
        sg, max_step = final_mcmc_state(mcmc_run_dir)

    from sigvisa.models.wiggles.wavelets import implicit_to_explicit


    (starray, etarray, idarray, M, levels, N)= sg.station_waves.values()[0][0].wavelet_basis
    prototypes = [np.asarray(m).flatten() for m in M]
    basis = implicit_to_explicit(starray, etarray, idarray, prototypes, levels, N)

    def plot_wavelet_dist_samples(ax, srate, basis, wmeans, wvars, c="blue"):
        wmeans = np.asarray(wmeans).flatten()
        wvars = np.asarray(wvars, dtype=float).flatten()

        n = basis.shape[1]
        x = np.linspace(0, n/float(srate), n)
        for n in range(30):
            ws = np.random.randn(basis.shape[0])*np.sqrt(wvars)+wmeans
            w = np.dot(basis.T, ws) + 1
            ax.plot(x, w, c=c, linestyle="-", alpha=0.2, lw=3)


    def plot_wavelet_dist(ax, srate, basis, wmeans, wvars, c="blue"):
        wmeans = np.asarray(wmeans).flatten()
        wvars = np.asarray(wvars, dtype=float).flatten()

        w = np.dot(basis.T, wmeans) + 1
        wv = np.asarray(np.diag(np.dot(basis.T, np.dot(np.diag(wvars), basis))), dtype=float)
        n = basis.shape[1]
        x = np.linspace(0, n/float(srate), n)
        ax.plot(x, w, c=c, linestyle="-", lw=2)
        ax.fill_between(x, w+2*np.sqrt(wv), w-2*np.sqrt(wv), facecolor=c, alpha=0.2)



    eids = sg.evnodes.keys()

    eid_wiggle_posteriors = dict()

    wiggle_gpmodels = dict([(k, v) for (k, v) in sg._joint_gpmodels[sta].items() if k[0].startswith("db")])

    holdout_evidence = dict()
    for eid in eids:
        try:
            holdout_evidence[eid] = np.sum([jgp.holdout_evidence(eid) for jgp, wnodes in wiggle_gpmodels.values()])
        except KeyError:
            continue

    n = len(eids)
    f = Figure((12, 4*n))
    f.patch.set_facecolor('white')
    gs = gridspec.GridSpec(n, 2)

    j = 0
    for wn in sg.station_waves[sta]:
        wn.pass_jointgp_messages()
        try:
            ell, all_prior_means, all_prior_vars, all_posterior_means, all_posterior_vars = wn._coef_message_cache
            idx = 0
            for i, (eid, component_phase, scale, sidx, npts, component_type) in enumerate(wn.tssm_components):
                if component_phase == phase:
                    break
                elif component_type=="wavelet":
                    idx += basis.shape[0]

            idx_end = idx +  basis.shape[0]
            prior_means = all_prior_means[idx:idx_end]
            prior_vars = all_prior_vars[idx:idx_end]
            posterior_means = all_posterior_means[idx:idx_end]
            posterior_vars = all_posterior_vars[idx:idx_end]

                
        except:
            import pdb; pdb.set_trace()

        for i, (eid, pphase, _, _, _, ctype) in enumerate(wn.tssm_components):
            if ctype != "wavelet": continue
            if pphase != phase: continue
            cond_means, cond_vars = zip(*[jgp.posterior(eid) for jgp in wn.wavelet_param_models[phase]])

            ax = f.add_subplot(gs[j, 0])
            plot_wavelet_dist(ax, wn.srate, basis, posterior_means, posterior_vars, c="blue")
            plot_wavelet_dist(ax, wn.srate, basis, cond_means, cond_vars, c="green")
            ax.set_title("%s - %d - evidence %f" % (sta, eid, holdout_evidence[eid]))
            ax = f.add_subplot(gs[j, 1])
            plot_wavelet_dist_samples(ax, wn.srate, basis, posterior_means, posterior_vars, c="blue")
            plot_wavelet_dist_samples(ax, wn.srate, basis, cond_means, cond_vars, c="green")
            ax.set_title("%s - %d - evidence %f" % (sta, eid, holdout_evidence[eid]))
            j += 1

    canvas = FigureCanvas(f)
    response = django.http.HttpResponse(content_type='image/png')
    f.tight_layout()
    canvas.print_png(response)
    return response


def mcmc_alignment_posterior(request, dirname, sta, phase):
    import seaborn as sns

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    burnin = int(request.GET.get('burnin', '50'))
    xmin = float(request.GET.get('xmin', '-5'))
    xmax = float(request.GET.get('xmax', '5'))
    ymax = float(request.GET.get('ymax', '-1'))
    temp = float(request.GET.get('temp', '10'))
    nplots = int(request.GET.get('nplots', '20'))
    titles = request.GET.get('titles', 'f').startswith('t')
    plotxc = request.GET.get('plotxc', 'f').startswith('t')
    plot_true_alignment = request.GET.get('plot_true_alignment', 'f').startswith('t')

    sgfile = str(request.GET.get('sgfile', ''))
    if sgfile:
        sg = graph_from_file(mcmc_run_dir, sgfile)
        max_step = 0
    else:
        sg, max_step = final_mcmc_state(mcmc_run_dir)

    eid_request = request.GET.get('eids', 'None')
    if eid_request == "None":
        eids = sg.evnodes.keys()
    else:
        eids = [int(eid) for eid in eid_request.split(',')]

    true_atimes = dict()
    try:
        with open(os.path.join(mcmc_run_dir, "sw.pkl"), "rb") as f:
            sw = pickle.load(f)
        residuals = sw.tm_params[sta]['tt_residual']
        for eid in eids:
            true_ev = sw.all_evs[eid-1]
            true_atimes[eid] = true_ev.time + tt_predict(true_ev, sta, phase) \
                               + residuals[eid-1]
    except IOError:
        pass

    atimes = dict()
    for eid in eids:

        eid_dir = os.path.join(mcmc_run_dir, "ev_%05d" % eid)
        for fname in os.listdir(eid_dir):
            if sta not in fname: continue
            if not fname.endswith("_%s" % phase): continue

            tmpl = np.loadtxt(os.path.join(eid_dir, fname))
            atimes[eid] = tmpl[burnin:, 1]
            atimes[eid] = atimes[eid]
            break

    f = Figure((3*len(eids), 1.5*len(eids)))
    f.patch.set_facecolor('white')
    gs = gridspec.GridSpec(len(atimes)*2+1, len(atimes))

    sorted_eids = sorted(atimes.keys())
    shared_ax = None
    shared_ax_xc = None
    for i, eid1 in enumerate(sorted_eids):
        ev1 = sg.get_event(eid1)
        pred_t1 = ev1.time + tt_predict(ev1, sta, phase=phase)

        ax = f.add_subplot(gs[2*i+1:2*i+3,i],
                           sharex=shared_ax)
        if shared_ax is None:
            shared_ax = ax

        mean_atime = np.mean(atimes[eid1])
        try:
            sns.distplot(atimes[eid1]-mean_atime, ax=ax)
        except ZeroDivisionError:
            pass

        ax.set_xticks(np.linspace(xmin, xmax, 5))
        ax.set_xlim([xmin, xmax])

        ax.axvline(pred_t1-mean_atime, lw=1, color="green")

        if eid1 in true_atimes:
            ax.axvline(true_atimes[eid1]-mean_atime, lw=1, color="red")

        if ymax > 0:
            ax.set_ylim([0, ymax])


        if titles:
            ax.set_title("atimes %d" % eid1)

        try:
            wn = [wn for wn in sg.station_waves[sta] if (eid1, phase) in wn.arrivals() and mean_atime > wn.st][0]
            s1, _, _ = get_arrival_signal(sg, eid1, phase, wn, 2, 10, atime = mean_atime)
        except Exception as e:
            print e
            s1 = None

        for j, eid2 in enumerate(sorted_eids[i+1:]):
            ev2 = sg.get_event(eid2)

            ax = f.add_subplot(gs[2*i+1:2*i+3,i+j+1],
                               sharex=shared_ax)

            pred_t2 = ev2.time + tt_predict(ev2, sta, phase=phase)
            pred_diff = pred_t1-pred_t2

            n = min(len(atimes[eid1]), len(atimes[eid2]))
            reltimes = atimes[eid1][:n] - atimes[eid2][:n]
            mean_reltime = np.mean(reltimes)
            reltimes -= mean_reltime
            try:
                sns.distplot(reltimes, ax=ax)
            except ZeroDivisionError:
                pass

            ax.axvline(pred_diff-mean_reltime, lw=1, color="green")

            if eid1 in true_atimes and eid2 in true_atimes:
                true_reltime = true_atimes[eid1] - true_atimes[eid2]
                ax.axvline(true_reltime-mean_reltime, lw=1, color="red")

            if titles:
                ax.set_title("atime diff %d %d" % (eid1, eid2))

            if s1 is None:
                print "no signal for", eid1
                continue

            # compute xcorr relative to mean relative time
            try:
                wn = [wn for wn in sg.station_waves[sta] if (eid2, phase) in wn.arrivals() and np.mean(atimes[eid2]) > wn.st][0]
                s2, _, _ = get_arrival_signal(sg, eid2, phase, wn, 2-xmin, 10+xmax, atime=np.mean(atimes[eid2]))
            except IndexError as e:
                print e
                continue


            if plotxc:
                ax2 = f.add_subplot(gs[2*(i+j+1)+1:2*(i+j+1)+3, i],
                                    sharex=shared_ax,
                                    sharey=shared_ax_xc)


                xc = fastxc(s1, s2)
                xcdist = np.exp(temp*xc)
                xcdist /= np.sum(xcdist)
                x = np.linspace(xmin, xmax, len(xcdist))
                ax2.plot(x, xc)

                if titles:
                    ax2.set_title("xcorr: %d %d" % (eid1, eid2))
            else:
                ax2 = f.add_subplot(gs[2*(i+j+1)+1:2*(i+j+1)+3, i],
                                    sharex=shared_ax_xc,
                                    sharey=None)
                x1 = np.linspace(-2+xmin, 10+xmax, len(s2))
                visible = [(x1> -2) * (x1 <10) ]
                ax2.plot(x1, s2/np.max(s2[visible]))


                idxs = np.linspace(0, len(reltimes)-1, nplots)
                idxs = np.array([int(idx) for idx in idxs])

                #ats = atimes[eid2][idxs]
                #mean_at2 = np.mean(atimes[eid2])

                # truths
                # mean_reltime = mean(atime1 - atime2) = mean(atime1) - mean(atime2)
                # we load s1 around mean_atime1
                # we load s2 around mean_atime2
                # so reltime=0 corresponds to plotting s1 at
                # mean_atime1 and s2 at mean_atime2, which requires no offset
                if plot_true_alignment:
                    offset = true_reltime-mean_reltime
                    x2 = np.linspace(-2+offset, 10+offset, len(s1))
                    visible = [(x2> -2) * (x2 <10) ]
                    ax2.plot(x2, s1/np.max(s1[visible]), alpha = 1.0, c="red")
                else:
                    for offset in reltimes[idxs]:
                        x2 = np.linspace(-2+offset, 10+offset, len(s1))
                        visible = [(x2> -2) * (x2 <10) ]
                        ax2.plot(x2, s1/np.max(s1[visible]), alpha = 1.0/nplots, c="green")

                ax2.set_xlim(-2, 10)
                ax2.set_ylim(0, 1)
                if titles:
                    ax2.set_title("waves: %d %d" % (eid1, eid2))


            if shared_ax_xc is None:
                shared_ax_xc = ax2


    f.suptitle("atime alignments at %s, phase %s" % (sta, phase))

    canvas = FigureCanvas(f)
    response = django.http.HttpResponse(content_type='image/png')
    f.tight_layout()
    canvas.print_png(response)
    return response

def mcmc_phase_stack(request, dirname, sta, phase, base_eid):

    radius_km = float(request.GET.get('radius_km', '15.0'))
    signal_len = float(request.GET.get('signal_len', '20.0'))
    xc_type = request.GET.get('xc_type', 'xc')

    buffer_s = 8.0

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    base_eid = int(base_eid)
    sta = str(sta)
    phase = str(phase)

    sgfile = str(request.GET.get('sgfile', ''))
    if sgfile:
        sg = graph_from_file(mcmc_run_dir, sgfile)
        max_step = 0
    else:
        sg, max_step = final_mcmc_state(mcmc_run_dir)

    # get the signal for this eid, with some leeway
    base_ev = sg.get_event(base_eid)
    base_wn = sg.get_arrival_wn(sta, base_eid, phase, band=None, chan=None)
    atime = base_wn.get_template_params_for_arrival(base_eid, phase)[0]["arrival_time"]
    sidx = time_to_index(atime-buffer_s, base_wn.st, base_wn.srate)
    base_stime = index_to_time(sidx, base_wn.st, base_wn.srate)
    eidx = sidx + int((signal_len + 2*buffer_s) * base_wn.srate)
    default_idx = time_to_index(atime, base_stime, base_wn.srate)
    base_signal = base_wn.get_value()[sidx:eidx].copy()
    base_default_window = base_signal[default_idx :default_idx + int(signal_len * base_wn.srate)]
    base_signal /= np.linalg.norm(base_default_window)

    # get all other events within the radius
    nearby_eids = []
    eid_distances = []
    nearby_eid_wns = []
    for eid in sg.evnodes.keys():
        ev = sg.get_event(eid)
        dist = dist_km((ev.lon, ev.lat), (base_ev.lon, base_ev.lat))
        if dist < radius_km:
            try:
                wn = sg.get_arrival_wn(sta, eid, phase, band=None, chan=None)
            except:
                continue

            nearby_eids.append(eid)
            eid_distances.append(dist)
            nearby_eid_wns.append(wn)
            
    # for each event, extract signal for the appropriate phase
    eid_signals = []
    eid_alignments = []
    eid_ttrs = []
    for eid, wn in zip(nearby_eids, nearby_eid_wns):
        ev = sg.get_event(eid)
        pred_atime = ev.time + tt_predict(ev, sta, phase)        
        atime = wn.get_template_params_for_arrival(eid, phase)[0]["arrival_time"]
        eid_ttrs.append(atime - pred_atime)

        sidx = time_to_index(atime, wn.st, wn.srate)
        eidx = sidx + int(signal_len * wn.srate)
        eid_signal = wn.get_value()[sidx:eidx].copy()
        eid_signal /= np.linalg.norm(eid_signal)
        eid_signals.append(eid_signal)

        # also then do a correlation to get the best offset
        if xc_type=="xc":
            xc = fastxc(eid_signal, base_signal)
        elif xc_type=="ar":
            xc = ar_advantage(base_signal, eid_signal, wn.nm)
            xc /= np.sum(xc)
        elif xc_type=="iid":
            xc = iid_advantage(base_signal, eid_signal)
            xc /= np.sum(xc)

        align_s = np.argmax(xc) / wn.srate - buffer_s
        align_idx = np.argmax(xc) + sidx

        #print "eid", eid, "atime", atime, "xc_atime", atime + align_s
        eid_alignments.append((np.max(xc), xc[default_idx], align_s))


    # plot the signals overlayed at their current offsets
    # also plot at the preferred offsets
    f = Figure((14, 2*(len(nearby_eids)+1)))
    f.patch.set_facecolor('white')
    gs = gridspec.GridSpec(len(nearby_eids)+1, 2)

    ax1 = f.add_subplot(gs[0, 0])
    ax2 = f.add_subplot(gs[0, 1])
    xs_base = np.linspace(-buffer_s, signal_len + buffer_s, len(base_signal))
    xs_default = np.linspace(0, signal_len, len(eid_signals[0]))
    ax1.plot(xs_base, base_signal, linewidth=2, color="black")
    ax2.plot(xs_base, base_signal, linewidth=2, color="black")

    best_xcs = sorted(np.arange(len(nearby_eids)), key = lambda i : -eid_alignments[i][0])

    for i, eid_idx in enumerate(best_xcs):
        eid = nearby_eids[eid_idx]
        signal = eid_signals[eid_idx]
        (xc_peak, default_xc, alignment) = eid_alignments[eid_idx]
        dist = eid_distances[eid_idx]
        ttr = eid_ttrs[eid_idx]

        xs_aligned = np.linspace(alignment, alignment + signal_len, len(signal))
        ax1.plot(xs_default, signal)
        ax2.plot(xs_aligned, signal)

        my_ax1 = f.add_subplot(gs[i+1, 0])
        my_ax2 = f.add_subplot(gs[i+1, 1])
        my_ax1.plot(xs_base, base_signal, linewidth=0.5, color="black")
        my_ax2.plot(xs_base, base_signal, linewidth=0.5, color="black")

        my_ax1.set_title("eid %d dist %.2fkm" % (eid, dist))
        my_ax2.set_title("xc peak %.2f at ttr=%.2fs vs %.2f at ttr=%.2fs" % (xc_peak, ttr+alignment, default_xc, ttr))

        my_ax1.plot(xs_default, signal, linewidth=1.3, alpha=0.7)
        my_ax2.plot(xs_aligned, signal, linewidth=1.3, alpha=0.7)
        my_ax1.set_xlim([-buffer_s, signal_len+buffer_s])
        my_ax2.set_xlim([-buffer_s, signal_len+buffer_s])

    canvas = FigureCanvas(f)
    response = django.http.HttpResponse(content_type='image/png')
    f.tight_layout()
    canvas.print_png(response)
    return response

def mcmc_compare_gps_doublets(request, dirname, sta):
    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    sta = str(sta)

    xc_type = request.GET.get('xc_type', 'xc')
    zoom = float(request.GET.get('zoom', '1.0'))
    step = int(request.GET.get('step', '-1'))

    sgfile = str(request.GET.get('sgfile', ''))
    if sgfile:
        sg = graph_from_file(mcmc_run_dir, sgfile)
    elif step < 0:
        sg, step = final_mcmc_state(mcmc_run_dir)
    else:
        sg = graph_for_step(mcmc_run_dir, step)

    pred_signals = {}
    pred_atime_idxs = {}
    extracted_signals = {}
    neighbor_events = {}
    actual_signals = {}
    actual_signal_offset = {}
    nms = {}
    for eid in sg.evnodes.keys():
        try:
            wn = sg.get_arrival_wn(sta, eid, "Lg", band=None, chan=None)
        except:
            continue

        (start_idxs, end_idxs, identities, basis_prototypes, levels, N) = wn.wavelet_basis

        # find the closest neighbor to this event
        ev = sg.get_event(eid)
        nearest_eid = None
        nearest_dist = np.inf
        for eid2 in sg.evnodes.keys():
            if eid == eid2: continue
            ev2 = sg.get_event(eid2)
            dist = dist_km((ev.lon, ev.lat), (ev2.lon, ev2.lat))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_eid = eid2
        neighbor_events[eid] = (nearest_eid, nearest_dist)
        neighbor_eid = nearest_eid

        # for each phase, extract the GP predicted signal and the signal from the nearest neighbor
        pred_signals[eid] = {}
        extracted_signals[eid] = {}
        pred_atime_idxs[eid] = {}
        pred_atimes = {}
        nms[eid] = wn.nm
        earliest_start_idx = wn.npts
        latest_end_idx = 0
        

        for phase in sg.ev_arriving_phases(eid, sta=sta):

            # extract phase signal for neighbor wn
            try:
                neighbor_wn = sg.get_arrival_wn(sta, neighbor_eid, phase, band=wn.band, chan=wn.chan)
            except Exception as e:
                print "no neighbor for %s" % phase, e
                continue
            neighbor_idx = neighbor_wn.arrival_start_idx(neighbor_eid, phase)
            neighbor_end_idx = neighbor_idx + N
            # truncate extracted signal to omit other arriving phases
            for (eeid, pphase) in neighbor_wn.arrivals():
                other_idx = neighbor_wn.arrival_start_idx(eeid, pphase)
                if other_idx > neighbor_idx and other_idx < neighbor_end_idx:
                    neighbor_end_idx = other_idx
            extracted_signal = neighbor_wn.get_value()[neighbor_idx:neighbor_end_idx]
            extracted_signals[eid][phase] = extracted_signal

            # get leave-one-out GP prediction
            cond_means, cond_vars = zip(*[jgp.posterior(eid) for jgp in wn.wavelet_param_models[phase]])
            cond_means, cond_vars = np.asarray(cond_means, dtype=np.float64), np.asarray(cond_vars, dtype=np.float64)
            cssm = wn.arrival_ssms[(eid, phase)]
            cssm.set_coef_prior(cond_means, cond_vars)
            pred_mean = wn.tssm.mean_obs(wn.npts)
            start_idx = wn.arrival_start_idx(eid, phase)
            end_idx = start_idx + (neighbor_end_idx-neighbor_idx)
            pred_signal = pred_mean[start_idx:end_idx]
            pred_signals[eid][phase] = pred_signal
            print "pred", eid, phase

            earliest_start_idx = min(start_idx, earliest_start_idx)
            latest_end_idx = max(end_idx, latest_end_idx)

            pred_atimes[phase] = ev.time + tt_predict(ev, sta, phase)
            

        buffer_idx = int(5.0 * wn.srate)
        actual_signals[eid] = wn.get_value()[earliest_start_idx - buffer_idx: latest_end_idx + buffer_idx]
        for phase, atime in pred_atimes.items():
            pred_idx = int((atime - wn.st) * wn.srate)
            pred_atime_idxs[eid][phase] = pred_idx - (earliest_start_idx - buffer_idx)

    f = Figure((14*zoom, 2*zoom*len(actual_signals)))
    f.patch.set_facecolor('white')
    gs = gridspec.GridSpec(len(actual_signals), 2)

    phase_colors = {"Pn": "blue", "Pg": "green", "Sn": "red", "Lg": "purple", "P": "brown"}

    def do_xc(short_s, long_s, long_nm=None):
        if xc_type=="xc":
            return fastxc(short_s, long_s)
        elif xc_type=="ar":
            return ar_advantage(long_s, short_s, long_nm)
        elif xc_type=="iid":
            norm_longs = long_s / np.std(long_s)
            return iid_advantage(norm_longs, short_s)


    total_pred_xc = 0.0
    total_extracted_xc = 0.0

    for i, (eid, s) in enumerate(actual_signals.items()):

        ax1 = f.add_subplot(gs[i, 0])
        ax2 = f.add_subplot(gs[i, 1])
        ax1.plot(s, linewidth=0.5, color="black")
        ax2.plot(s, linewidth=0.5, color="black")
        pred_title = "eid %d pred xcs" % eid
        extracted_title = "eid %d extracted xcs" % eid
        s1 = s.copy()
        s2 = s.copy()

        total_ev_pred_xc = 0.0
        total_ev_extracted_xc = 0.0

        for phase in sorted(extracted_signals[eid].keys()):
            color = phase_colors[phase]

            pred_signal = pred_signals[eid][phase]
            print len(pred_signal), len(s)

            pred_idx = pred_atime_idxs[eid][phase]
            slack_idx = 100 # 10s times srate=10.0
            sidx_xc = max(pred_idx - slack_idx, 0)
            xc_s1 = s1[sidx_xc : pred_idx + slack_idx + len(pred_signal)]
            xc_s2 = s2[sidx_xc : pred_idx + slack_idx + len(pred_signal)]

            xc_pred = do_xc(pred_signal, xc_s1, nms[eid])
            sidx_pred = np.argmax(xc_pred) + sidx_xc
            idxs_pred = np.arange(sidx_pred, sidx_pred + len(pred_signal))
            s1[idxs_pred] = 0
            ax1.plot(idxs_pred, pred_signal, color=color)
            pred_title += " %s %.2f" % (phase, np.max(xc_pred))
            total_ev_pred_xc += np.max(xc_pred)

            extracted_signal = extracted_signals[eid][phase]
            xc_extracted = do_xc(extracted_signal, xc_s2, nms[eid])
            sidx_extracted = np.argmax(xc_extracted) + sidx_xc
            idxs_extracted = np.arange(sidx_extracted, sidx_extracted + len(extracted_signal))
            s2[idxs_extracted] = 0
            target_norm=np.linalg.norm(s[idxs_extracted])
            scaling = target_norm / np.linalg.norm(extracted_signal)
            ax2.plot(idxs_extracted, extracted_signal * scaling, color=color)
            extracted_title += " %s %.2f" % (phase, np.max(xc_extracted))
            total_ev_extracted_xc += np.max(xc_extracted)

        pred_title += " total %.2f" % total_ev_pred_xc
        extracted_title += " total %.2f" % total_ev_extracted_xc
        ax1.set_title(pred_title, fontsize=10)
        ax2.set_title(extracted_title, fontsize=10)
        total_pred_xc += total_ev_pred_xc
        total_extracted_xc += total_ev_extracted_xc

    f.suptitle("total pred %.2f extracted %.2f" % (total_pred_xc, total_extracted_xc))

    canvas = FigureCanvas(f)
    response = django.http.HttpResponse(content_type='image/png')
    f.tight_layout()
    canvas.print_png(response)
    return response

def mcmc_compare_gps_doublets_wavelets(request, dirname, sta, phase):
    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    zoom = float(request.GET.get('zoom', '1.0'))
    step = int(request.GET.get('step', '-1'))

    sta = str(sta)
    phase = str(phase)

    sgfile = str(request.GET.get('sgfile', ''))
    if sgfile:
        sg = graph_from_file(mcmc_run_dir, sgfile)
    elif step < 0:
        sg, step = final_mcmc_state(mcmc_run_dir)
    else:
        sg = graph_for_step(mcmc_run_dir, step)

    aligned_wavelets = {}
    pred_wavelets = {}
    neighbor_wavelets = {}
    for eid in sg.evnodes.keys():
        try:
            wn = sg.get_arrival_wn(sta, eid, "Lg", band=None, chan=None)
        except:
            continue

        (start_idxs, end_idxs, identities, basis_prototypes, levels, N) = wn.wavelet_basis

        # find the closest neighbor to this event
        ev = sg.get_event(eid)
        nearest_eid = None
        nearest_dist = np.inf
        for eid2 in sg.evnodes.keys():
            if eid == eid2: continue
            ev2 = sg.get_event(eid2)
            dist = dist_km((ev.lon, ev.lat), (ev2.lon, ev2.lat))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_eid = eid2

        neighbor_eid = nearest_eid
        try:
            neighbor_wn = sg.get_arrival_wn(sta, neighbor_eid, phase, band=wn.band, chan=wn.chan)
        except Exception as e:
            print "no neighbor for %s" % phase, e
            continue

        def extract_wavelet_posterior(wn, eid, phase):
                    # extract phase signal for neighbor wn
            d = wn.get_value().data
            ell, marginals, step_ells = wn.tssm.all_filtered_cssm_coef_marginals(d)            
            for i, (eid, pphase, scale, sidx, npts, component_type) in enumerate(wn.tssm_components):
                if component_type != "wavelet": continue
                if pphase!=phase: continue
                wmeans, wvars = marginals[i]
                break
            return wmeans, wvars

        aligned_wavelets[eid], _ = extract_wavelet_posterior(wn, neighbor_eid, phase)
        neighbor_wavelets[eid], _ = extract_wavelet_posterior(neighbor_wn, neighbor_eid, phase)

        cond_means, cond_vars = zip(*[jgp.posterior(eid) for jgp in wn.wavelet_param_models[phase]])
        cond_means, cond_vars = np.asarray(cond_means, dtype=np.float64), np.asarray(cond_vars, dtype=np.float64)
        pred_wavelets[eid] = cond_means
        print "predicted", eid

    f = Figure((14*zoom, 2*zoom*len(aligned_wavelets)))
    f.patch.set_facecolor('white')
    gs = gridspec.GridSpec(len(aligned_wavelets), 2)

    for i, (eid, w) in enumerate(aligned_wavelets.items()):

        w_norm = w / np.linalg.norm(w)

        ax1 = f.add_subplot(gs[i, 0])
        ax2 = f.add_subplot(gs[i, 1])
        ax1.plot(w_norm, linewidth=0.5, color="black")
        ax2.plot(w_norm, linewidth=0.5, color="black")
        pred_title = "eid %d pred xcs" % eid
        extracted_title = "eid %d extracted xcs" % eid

        pred_w_norm = pred_wavelets[eid] / np.linalg.norm(pred_wavelets[eid])
        neighbor_w_norm = neighbor_wavelets[eid] / np.linalg.norm(neighbor_wavelets[eid])
        ax1.plot(pred_w_norm, color="blue", alpha=0.5, linewidth=2)
        ax2.plot(neighbor_w_norm, color="green", alpha=0.5, linewidth=2)

        pred_corr = np.dot(w_norm, pred_w_norm) 
        neighbor_corr = np.dot(w_norm, neighbor_w_norm) 

        ax1.set_title("eid %d pred corr %.2f" % (eid, pred_corr))
        ax2.set_title("eid %d neighbor corr %.2f" % (eid, neighbor_corr))


    canvas = FigureCanvas(f)
    response = django.http.HttpResponse(content_type='image/png')
    f.tight_layout()
    canvas.print_png(response)
    return response


def safe_loadtxt(fname):
    # np.loadtxt breaks if the last line is only
    # partially written (e.g. when inference is
    # still running and buffers have been flushed
    # in a way that doesn't align with line
    # boundaries). This method runs np.loadtxt
    # ignoring the last line of a file.
    with open(fname, 'r') as f:
        lines = f.readlines()
    import StringIO

    inp = StringIO.StringIO("\n".join(lines[:-1]))
    return np.loadtxt(inp)

def mcmc_hparam_posterior(request, dirname, sta, target):

    def plot_dist(ax, samples):
        import seaborn as sns
        sns.distplot(samples, ax=ax)

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    burnin = int(request.GET.get('burnin', '-1'))

    sgfile = str(request.GET.get('sgfile', ''))
    if sgfile:
        sg = graph_from_file(mcmc_run_dir, sgfile)
        max_step = 0
    else:
        sg, max_step = final_mcmc_state(mcmc_run_dir)

    true_vals = None
    # load hparam samples from file
    #import pdb; pdb.set_trace()
    #assert(a.shape[1]==len(starray))
    
    if target == "wiggles":
        targets = ["level%d" %i for i in range(10)]
    elif target == "template":
        targets = ("tt_residual", "amp_transfer", "coda_decay", "peak_decay", "peak_offset", "mult_wiggle_std")
    else:
        raise Exception("unrecognized target %s" % target)
    
    hkeys = []
    keys = None
    for hparam_key, hparam_nodes in sg._jointgp_hparam_nodes.items():
        for target in targets:
            if sta in hparam_key and target in hparam_key:
                hkeys.append(hparam_key)
                if keys is None:
                    keys = sorted(hparam_nodes.keys())
                else:
                    assert(keys == sorted(hparam_nodes.keys()))

    f = Figure((16, 1.2*len(hkeys)))
    f.patch.set_facecolor('white')
    gs = gridspec.GridSpec(len(hkeys), len(keys))
    
    axes = [None,] * len(keys)
    for i, hkey in enumerate(sorted(hkeys)):
        ffname = os.path.join(mcmc_run_dir, "gp_hparams", hkey)
        a = safe_loadtxt(ffname)
        if burnin < 0:
            burnin = 100 if a.shape[0] > 150 else 10 if a.shape[0] > 10 else 0

        for j, k in enumerate(keys):
            ax = f.add_subplot(gs[i,j:j+1], sharex=axes[j])
            if axes[j] is None:
                axes[j] = ax
            if k=="noise_var" and "level" in hkey:
                ax.set_xlim([0, 1])
            #ax.patch.set_facecolor('white')

            samples = a[burnin:, j]
            if np.std(samples) > 0:
                plot_dist(ax, samples)
            #ax.set_xticks(np.linspace(0, np.max(a), 3))

            if true_vals is not None:
                ax.axvline(true_vals[i], lw=1, color="green")

            hpphase = hkey.split(";")[-2]
            hpclass = hkey.split(";")[-1]
            ax.set_title("%s %s %s" % (hpphase, hpclass, k))
            ax.annotate('mean %.1f\nstd %.1f' % (np.mean(samples), np.std(samples)), (0.1, 0.1), xycoords='axes fraction', size=10)

    canvas = FigureCanvas(f)
    response = django.http.HttpResponse(content_type='image/png')
    f.tight_layout()
    canvas.print_png(response)
    return response


def mcmc_event_proposals(request, dirname):

    zoom = float(request.GET.get('zoom', '1.0'))
    proposal_type = request.GET.get('proposal_type', 'hough')

    def extract_proposal(line):
        parts = line.split(",")
        loc = parts[1]
        m = re.match(r" loc ([0-9\.]+) ([EW]) ([0-9\.]+) ([NS])", loc)
        lon, ew, lat, ns = m.groups()
        if ew=="E":
            lon = float(lon)
        else:
            lon = -float(lon)
        if ns == "N":
            lat = float(lat)
        else:
            lat = -float(lat)

        depth = float(parts[2][6:-2])
        t = float(parts[3][5:])
        mb = float(parts[4][3:])
        return (lon, lat, depth, mb, t)

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    proposedX = []
    if proposal_type=="hough":
        with open(os.path.join(mcmc_run_dir, "hough_proposals.txt"), 'r') as f:
            for line in f:
                if line.startswith("proposed ev: "):
                    proposedX.append(extract_proposal(line))
    elif proposal_type=="correlation":
        with open(os.path.join(mcmc_run_dir, "correlation_proposals.txt"), 'r') as f:
            for line in f:
                if line.startswith("proposed ev: "):
                    proposedX.append(extract_proposal(line))
    else:
        raise Exception("unknown proposal type %s" % proposal_type)
    proposedX = np.asarray(proposedX).reshape((-1, 5))
    

    try:
        with open(os.path.join(mcmc_run_dir, 'events.pkl'), 'rb') as evfile:
            true_evs = pickle.load(evfile)
    except Exception as e:
        print e
        true_evs = []
    trueX = np.asarray([(ev.lon, ev.lat, ev.depth, ev.mb, ev.time) for ev in true_evs]).reshape((-1, 5))

    sgfile = str(request.GET.get('sgfile', ''))
    if sgfile:
        sg = graph_from_file(mcmc_run_dir, sgfile)
        max_step = 0
    else:
        sg, max_step = final_mcmc_state(mcmc_run_dir)

    inferred_evs = [sg.get_event(eid) for eid in sg.evnodes.keys() if eid not in sg.fixed_events]
    inferredX = np.asarray([(ev.lon, ev.lat, ev.depth, ev.mb, ev.time) for ev in inferred_evs]).reshape((-1, 5))

    allX = np.vstack((proposedX, trueX, inferredX))


    f = Figure((12*zoom, 12*zoom))
    f.patch.set_facecolor('white')
    gs = gridspec.GridSpec(5, 1)
    ax = f.add_subplot(gs[:4, 0])

    if sg.inference_region is not None:
        region = sg.inference_region
        hm = EventHeatmap(f=None, calc=False, 
                          left_lon=region.left_lon,
                          right_lon=region.right_lon,
                          top_lat=region.top_lat,
                          bottom_lat=region.bottom_lat)
    else:
        hm = EventHeatmap(f=None, calc=False, autobounds = allX, autobounds_quantile=1.0)

    hm.add_stations(sg.station_waves.keys())
    hm.init_bmap(axes=ax, nofillcontinents=True, projection="cyl", resolution="c")

    hm.plot(axes=ax, nolines=True, smooth=True,
            colorbar_format='%.3f')

    hm.plot_locations(trueX,  labels=None, marker="*", ms=16, mfc="none", mew=2, alpha=1)
    hm.plot_locations(inferredX,  labels=None, marker="+", ms=16, mew=2, mec="blue", alpha=1.0)
    hm.plot_locations(proposedX,  labels=None, marker=".", ms=12, mfc="red", mew=0, mec="none", alpha=0.5)


    ax = f.add_subplot(gs[4, 0])
    ax.scatter(proposedX[:, 4], proposedX[:, 1], marker=".", color="red", alpha=0.5)
    ax.scatter(inferredX[:, 4], inferredX[:, 1], marker="+", color="blue", s=32)
    ax.scatter(trueX[:, 4], trueX[:, 1], marker="*", s=32)
    ax.set_ylabel("lat")
    ax.set_xlabel("time")

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

    sgfile = str(request.GET.get('sgfile', ''))
    if sgfile:
        sg = graph_from_file(mcmc_run_dir, sgfile)
        max_step = 0
    else:
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

    try:
        with open(os.path.join(mcmc_run_dir, 'events.pkl'), 'rb') as evfile:
            true_evs = pickle.load(evfile)
    except Exception as e:
        print e
        true_evs = []

    for eid_i, eid in enumerate(sorted(eids)):
        ev_trace_file = os.path.join(mcmc_run_dir, 'ev_%05d.txt' % eid)
        trace, min_step, max_step = load_trace(ev_trace_file, burnin=burnin)

        if len(trace.shape) != 2 or trace.shape[0] < 2:
            continue

        if plot_true:
            true_ev = match_true_ev(trace, true_evs)
            if true_ev is not None:
                true_loc = np.array(((true_ev.lon, true_ev.lat), ))
                hm.plot_locations(true_loc,  labels=None, marker="*", ms=16, mfc="none", mec=shape_colors[eid_i-1], mew=2, alpha=1)
                true_evs.remove(true_ev)
                print len(true_evs)

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
                if true_ev is not None:
                    hm.drawline(loc[0], true_loc[0], color=shape_colors[eid_i-1])

            if stds > 0:
                 m = np.mean(trace[:, 0:2], axis=0)
                 centered_trace = trace[:, 0:2] - m
                 cov = np.dot(centered_trace.T, centered_trace)/float(centered_trace.shape[0])
                 hm.plot_covs([m,], [cov,], stds=stds, colors=[shape_colors[eid_i-1],], alpha=0.2)

    f.legend(handles=eid_patches, labels=eid_labels)


    # plot any true events unmatched to inferred events
    if plot_true:
        for true_ev in true_evs:
            loc = np.array(((true_ev.lon, true_ev.lat), ))
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

    sgfile = str(request.GET.get('sgfile', ''))
    if sgfile:
        sg = graph_from_file(mcmc_run_dir, sgfile)
    else:
        sg = graph_for_step(mcmc_run_dir, int(step))

    wn = sg.all_nodes[wn_label]

    response = ""

    for (eid, phase) in sorted(wn.arrivals()):
        v, tg = wn.get_template_params_for_arrival(eid=eid, phase=phase)

        tmnodes = sg.get_template_nodes(eid, wn.sta, phase, wn.band, wn.chan)

        response += "eid %d, phase %s:\n" % (eid, phase)
        for (key, val) in v.items():

            k, node = tmnodes[key]
            pv = node._parent_values()

            model_txt = ""
            if node.deterministic():
                parent_key = node.default_parent_key()
                parent = node.parents[parent_key]
                pv = parent._parent_values()

                if parent.modeled_as_joint():
                    model = parent.joint_conditional_dist()
                    lp = model.log_p(parent.get_value())
                else:
                    model = parent.model


                try:
                    pmean = model.predict(cond=pv)
                except:
                    pmean = np.nan
                try:
                    pstd = np.sqrt(model.variance(cond=pv))
                except:
                    pstd = 0.0

                try:
                    modelid = parent.modelid
                except:
                    modelid = -1

                try:
                    lp = model.log_p(parent.get_value(), cond=pv)
                except:
                    lp = np.nan

                model_txt = "determined by parent with value %.2f, mean %.2f std %.2f, lp %.2f under model %s" % (parent.get_value(), pmean, pstd, lp, modelid)
            else:
                try:
                    lp = node.log_p()
                except:
                    lp = np.nan


                if node.modeled_as_joint():
                    model = node.joint_conditional_dist()
                    lp = model.log_p(node.get_value(), cond=pv)
                else:
                    model = node.model
    
                try:
                    pmean = model.predict(cond=pv)
                except:
                    pmean = np.nan
                try:
                    pstd = np.sqrt(model.variance(cond=pv))
                except:
                    pstd = 0.0
                try:
                    modelid = model.modelid
                except:
                    modelid = -1

                model_txt = "mean %.2f std %.2f, lp %.2f under model %d" % (pmean, pstd, lp, modelid)

            try:
                response += " %s: %.3f  %s\n" % (key, val, model_txt)
            except:
                response += " %s: %s\n  %s" % (key, val, model_txt)
        response += "\n"


    response += "\nnoise model params %s mean %.2f step std %.2f stationary std %.2f\n" % (wn.nm.params, wn.nm.c, wn.nm.em.std, np.sqrt(wn.nm.marginal_variance()))
    

    return HttpResponse(response, content_type="text/plain")


def mcmc_signal_posterior_page(request, dirname, wn_label):
    pass

def mcmc_signal_posterior_wave(request, dirname, wn_label, key1):
    zoom = float(request.GET.get("zoom", '1'))
    vzoom = float(request.GET.get("vzoom", '1'))

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    sgfile = str(request.GET.get('sgfile', ''))
    if sgfile:
        sg = graph_from_file(mcmc_run_dir, sgfile)
        max_step = 0
    else:
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

def mcmc_wave_gpvis(request, dirname, wn_label):
    zoom = float(request.GET.get("zoom", '1'))
    vzoom = float(request.GET.get("vzoom", '1'))
    samples = int(request.GET.get("samples", '0'))
    step = request.GET.get("step", '-1')

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    sgfile = str(request.GET.get('sgfile', ''))
    if sgfile:
        sg = graph_from_file(mcmc_run_dir, sgfile)
    else:
        sg = graph_for_step(mcmc_run_dir, int(step))

    wn = sg.all_nodes[wn_label]

    f = Figure((10*zoom, 5*vzoom))
    f.patch.set_facecolor('white')
    ax = f.add_subplot(111)

    wn._parent_values()
    wn._set_cssm_priors_from_model()
    tssm = wn.transient_ssm()

    m1 = tssm.mean_obs(1700)
    s1 = tssm.obs_var(1700)

    ax.plot(wn.get_value(), color='black')
    ax.plot(m1, lw=2, color='green')
    ax.fill_between(np.arange(wn.npts), m1+2*np.sqrt(s1),  m1-2*np.sqrt(s1), facecolor="green", alpha=0.2)

    for i in range(samples):
        z = tssm.prior_sample(1700, i)
        ax.plot(z, lw=1, color='blue', alpha=0.4)


    canvas = FigureCanvas(f)
    response = django.http.HttpResponse(content_type='image/png')
    f.tight_layout()
    canvas.print_png(response)
    return response

def mcmc_wave_posterior(request, dirname, wn_label):

    zoom = float(request.GET.get("zoom", '1'))
    vzoom = float(request.GET.get("vzoom", '1'))
    plot_pred_env = request.GET.get("pred_env", 'true').lower().startswith('t')
    plot_pred_signal = request.GET.get("pred_signal", 'false').lower().startswith('t')
    pred_signal_var = request.GET.get("pred_signal_var", 'false').lower().startswith('t')
    plot_predictions = request.GET.get("plot_predictions", 'true').lower().startswith('t')
    plot_posterior = request.GET.get("plot_posterior", 'true').lower().startswith('t')

    plot_dets = request.GET.get("plot_dets", 'leb')
    plot_template_arrivals = request.GET.get("plot_templates", 'true').lower().startswith('t')
    model_lw = float(request.GET.get("model_lw", '2'))
    signal_lw = float(request.GET.get("signal_lw", '1.5'))
    step = request.GET.get("step", 'all')
    stime = float(request.GET.get("stime", '-1'))
    etime = float(request.GET.get("etime", '-1'))
    

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    sgfile = str(request.GET.get('sgfile', ''))
    if sgfile:
        sg = graph_from_file(mcmc_run_dir, sgfile)
        sgs = {0: sg}
    elif step=="all":
        sgs = graphs_by_step(mcmc_run_dir)
    else:
        sgs = {int(step): graph_for_step(mcmc_run_dir, int(step))}

    last_step = np.max(sgs.keys())
    last_sg = sgs[last_step]

    ev_alpha = 0.8 / len(sgs)
    ua_alpha = 0.4/len(sgs)

    nevents = last_sg.next_eid-1

    wn = last_sg.all_nodes[wn_label]
    len_mins = (wn.et - wn.st) / 60.0

    f = Figure((10*zoom, 5*vzoom))
    f.patch.set_facecolor('white')
    axes = f.add_subplot(111)
    subplot_waveform(wn.get_wave(), axes, color='black', linewidth=signal_lw, plot_dets=None)

    import matplotlib.cm as cm
    shape_colors = dict([(eid, cm.get_cmap('jet')(np.random.rand()*.5)) for (eid, phase) in wn.arrivals()])
    steps = sgs.keys()
    alpha = 1.0/len(steps)
    for step in steps:
        wn = sgs[step].all_nodes[wn_label]

        try:
            wn.tssm
        except AttributeError:
            wn.tssm = wn.transient_ssm()

        if plot_pred_env:
            wn._parent_values()
            pred_env = wn.assem_env() + wn.nm_env.c
            w = Waveform(pred_env, srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
            subplot_waveform(w, axes, color='green', linewidth=2.5)
            
        if plot_pred_signal:
            pv = wn._parent_values()
            wn._set_cssm_priors_from_model(parent_values=pv)
            pred_signal = wn.tssm.mean_obs(wn.npts)
            w = Waveform(pred_signal, srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
            subplot_waveform(w, axes, color='green', linewidth=2.5, alpha=0.8)
            if pred_signal_var:
                signal_var = wn.tssm.obs_var(wn.npts)

                bottom = pred_signal-2*np.sqrt(signal_var)
                top = pred_signal+2*np.sqrt(signal_var)
                w1 = Waveform(bottom, srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
                subplot_waveform(w1, axes, color='green', linewidth=1.0, fill_y2=top, alpha=0.1)
                #w2 = Waveform(pred_signal-2*np.sqrt(signal_var), srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
                #subplot_waveform(w2, axes, color='red', linewidth=1.0)
        elif plot_posterior:
            shape_colors = plot_with_fit_shapes(fname=None, wn=wn,title=wn_label, axes=axes, plot_dets=plot_dets, shape_colors=shape_colors, plot_wave=False, alpha=alpha, model_lw=model_lw, zorder=5)

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
        plot_pred_atimes(dict(atimes), wn.get_wave(), axes=axes, color=colors, alpha=1.0, bottom_rel=0.0, top_rel=0.1)

    if stime > 0 and etime > 0:
        sidx = max(int((stime - wn.st) * wn.srate), 0)
        eidx = max(sidx, min(int((etime - wn.st) * wn.srate), wn.npts))
        d = wn.get_value()[sidx:eidx]
        try:
            dmax, dmin = np.max(d), np.min(d)
        except:
            dmax, dmin = np.max(wn.get_value()), np.min(wn.get_value())

        try:
            pd = pred_env[sidx:eidx]
            dmax = max(np.max(pd), dmax)
            dmin = min(np.min(pd), dmin)
        except:
            pass

        drange = dmax - dmin
        slack = drange / 20.0
        axes.set_xlim((stime, etime))
        axes.set_ylim((dmin-slack, dmax+slack))

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

    elif current_path.endswith(".pkl") or current_path.endswith(".sg"):
        url = reverse('mcmcrun_detail', kwargs={"dirname": dirname})
        return HttpResponseRedirect(url + "?sgfile=%s" % path)
    else:
        mimetype=mimetypes.guess_type(path)[0]
        return HttpResponse(open(current_path).read(), content_type=mimetype)

def mcmc_ev_trace_pairwise_plot(request, dirname, eid, param1, param2):


    cols = {"step": 0, "lon": 1, "lat": 2, "depth": 3, "time": 4, "mb": 5, "natural_source": 6}

    burnin = int(request.GET.get('burnin', '0'))
    eid = int(eid)
    
    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    eidfile = os.path.join(mcmc_run_dir, "ev_%05d.txt" % eid)
    trace = np.loadtxt(eidfile)

    col1 = cols[param1]
    col2 = cols[param2]

    steps = trace[burnin:, 0]
    steps -= np.min(steps)
    nsteps = len(steps)
    alphas = (steps + 1.0) / float(nsteps)
    
    x = trace[burnin:, col1]
    y = trace[burnin:, col2]

    f = Figure((5, 5))
    f.patch.set_facecolor('white')
    ax = f.add_subplot(111)
    for (xp, yp, alpha) in zip(x, y, alphas):
        ax.scatter(xp, yp, alpha=alpha)
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    
    canvas = FigureCanvas(f)
    response = django.http.HttpResponse(content_type='image/png')
    f.tight_layout()
    canvas.print_png(response)
    return response
