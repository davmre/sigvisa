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
from sigvisa.infer.analyze_mcmc import load_trace, trace_stats, match_true_ev
from sigvisa.infer.template_xc import get_arrival_signal, fastxc
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

def mcmc_lp_posterior(request, dirname):
    s = Sigvisa()

    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    lps = np.loadtxt(os.path.join(mcmc_run_dir, "lp.txt"))
    f = Figure()
    f.patch.set_facecolor('white')
    ax = f.add_subplot(111)
    ax.plot(lps)
    ax.set_xlabel("step")
    ax.set_ylabel("log density")

    canvas = FigureCanvas(f)
    response = django.http.HttpResponse(content_type='image/png')
    f.tight_layout()
    canvas.print_png(response)
    return response

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
    gp_hparams = []
    if len(sg._jointgp_hparam_nodes) > 0:
        gp_hparams = ["wiggles", "template"]

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
        true_ev_strs = [str(ev) for ev in true_evs]
    else:
        true_ev_strs = []

    if burnin < 0:
        burnin = 100 if max_step > 150 else 10 if max_step > 10 else 0

    X = []
    for eid in eids:
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
                  'example_node': enode,
        }
        evs.append(evdict)

    X = np.array(X, dtype=np.float)
    try:
        left_bound, right_bound, bottom_bound, top_bound = event_bounds(X, quantile=0.99)
    except:
        left_bound, right_bound, bottom_bound, top_bound= -180, 180, -90, 90
    bounds = dict()
    bounds["top"] = top_bound + 0.2
    bounds["bottom"] = bottom_bound - 0.2
    bounds["left"] = left_bound - 0.2
    bounds["right"] = right_bound + 0.2

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
                               'bounds': bounds,
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
    gs = gridspec.GridSpec(n, 3)

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

            ax = f.add_subplot(gs[j, 0])
            ax.plot(timevals, w)
            ax.plot(timevals, pred_mean, lw=2)
            #ax.plot(timevals, pred_mean+np.sqrt(pred_var))

            ax.fill_between(timevals, pred_mean+2*np.sqrt(pred_var), 
                            pred_mean-2*np.sqrt(pred_var), facecolor="green", alpha=0.2)

            ax.set_xlim([cstime, cetime])
            ax.set_ylim([np.min(w)-.5, np.max(w)+.5])

            ax.set_title("lp %f" % (lp2))
            ax = f.add_subplot(gs[j, 1])
            #plot_wavelet_dist_samples(ax, wn.srate, basis, posterior_means, posterior_vars, c="blue")
            #plot_wavelet_dist_samples(ax, wn.srate, basis, cond_means, cond_vars, c="green")
            ax.plot(timevals, w)
            ax.plot(timevals, base_mean, lw=2)
            ax.fill_between(timevals, base_mean+2*np.sqrt(base_var), 
                            base_mean-2*np.sqrt(base_var), facecolor="green", alpha=0.2)

            ax.set_xlim([cstime, cetime])
            ax.set_ylim([np.min(w)-.5, np.max(w)+.5])            
            ax.set_title("lp %f" % (lp1))


            ax = f.add_subplot(gs[j, 2])
            #plot_wavelet_dist_samples(ax, wn.srate, basis, posterior_means, posterior_vars, c="blue")
            #plot_wavelet_dist_samples(ax, wn.srate, basis, cond_means, cond_vars, c="green")
            ax.plot(timevals, w)
            ax.plot(timevals, post_mean, lw=2)
            ax.fill_between(timevals, post_mean+2*np.sqrt(post_var), 
                            post_mean-2*np.sqrt(post_var), facecolor="green", alpha=0.2)

            ax.set_xlim([cstime, cetime])            
            ax.set_title("lp %f" % (lp3))
            ax.set_ylim([np.min(w)-.5, np.max(w)+.5])
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
            ell, prior_means, prior_vars, posterior_means, posterior_vars = wn._coef_message_cache
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

    f = Figure((16, 8))
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

            hpclass = hkey.split(";")[-1]
            ax.set_title("%s %s" % (hpclass, k))
            ax.annotate('mean %.1f\nstd %.1f' % (np.mean(samples), np.std(samples)), (0.1, 0.1), xycoords='axes fraction', size=10)

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

def mcmc_wave_gpvis(request, dirname, wn_label):
    zoom = float(request.GET.get("zoom", '1'))
    vzoom = float(request.GET.get("vzoom", '1'))
    samples = int(request.GET.get("samples", '0'))
    step = request.GET.get("step", '-1')

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

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
    plot_pred_signal = request.GET.get("pred_signal", 'false').lower().startswith('t')
    pred_signal_var = request.GET.get("pred_signal_var", 'false').lower().startswith('t')
    plot_predictions = request.GET.get("plot_predictions", 'true').lower().startswith('t')
    plot_dets = request.GET.get("plot_dets", 'leb')
    plot_template_arrivals = request.GET.get("plot_templates", 'true').lower().startswith('t')
    model_lw = float(request.GET.get("model_lw", '2'))
    signal_lw = float(request.GET.get("signal_lw", '1.5'))
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


        if plot_pred_signal:
            wn._parent_values()
            pred_env = wn.assem_env() + wn.nm_env.c
            #pred_signal = wn.tssm.mean_obs(wn.npts)
            w = Waveform(pred_env, srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
            subplot_waveform(w, axes, color='green', linewidth=2.5)
            if pred_signal_var:
                signal_var = wn.tssm.obs_var(wn.npts)
                w1 = Waveform(pred_signal+2*np.sqrt(signal_var), srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
                subplot_waveform(w1, axes, color='red', linewidth=1.0)
                w2 = Waveform(pred_signal-2*np.sqrt(signal_var), srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
                subplot_waveform(w2, axes, color='red', linewidth=1.0)
        else:
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
        return HttpResponse(open(current_path).read(), content_type=mimetype)
