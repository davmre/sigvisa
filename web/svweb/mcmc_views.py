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
from sigvisa.models.ttime import tt_predict
from sigvisa.plotting.plot import plot_with_fit_shapes, plot_pred_atimes
from sigvisa import *

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas








def mcmc_list_view(request):

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")

    mcmc_run_dirs = os.listdir(mcmc_log_dir)

    mcmc_runs = []
    for rundir in sorted(mcmc_run_dirs):
        mcmcrun = dict()
        mcmcrun['dir'] = rundir
        mcmcrun['time'] = str(time.ctime(os.path.getmtime(os.path.join(mcmc_log_dir, rundir))))

        cmd = ""
        try:
            with open(os.path.join(mcmc_log_dir, rundir, 'cmd.txt'), 'r') as f:
                cmd = f.read()
        except IOError:
            pass
        mcmcrun['cmd'] = cmd
        mcmc_runs.append(mcmcrun)

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

def final_mcmc_state(ev_dir):
    max_step = np.max([int(d[5:]) for d in os.listdir(ev_dir) if d.startswith('step')])
    with open(os.path.join(ev_dir, "step_%06d" % max_step, 'pickle.sg'), 'rb') as f:
        sg = pickle.load(f)
    return sg

def mcmc_run_detail(request, dirname):
    s = Sigvisa()
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


    sg = final_mcmc_state(mcmc_run_dir)
    wns = [n.label for n in np.concatenate(sg.station_waves.values())]

    return render_to_response("svweb/mcmc_run_detail.html",
                              {'wns': wns,
                               'dirname': dirname,
                               'full_dirname': mcmc_run_dir,
                               'cmd': cmd,
                               'analyze_cmd': analyze_cmd,
                               }, context_instance=RequestContext(request))


def mcmc_wave_posterior(request, dirname, wn_label):

    zoom = float(request.GET.get("zoom", '1'))
    plot_predictions = request.GET.get("plot_predictions", 'true').lower().startswith('t')
    plot_dets = request.GET.get("plot_dets", 'leb')

    s = Sigvisa()
    mcmc_log_dir = os.path.join(s.homedir, "logs", "mcmc")
    mcmc_run_dir = os.path.join(mcmc_log_dir, dirname)

    sgs = graphs_by_step(mcmc_run_dir)

    last_step = np.max(sgs.keys())
    last_sg = sgs[last_step]

    ev_alpha = 0.8 / len(sgs)
    ua_alpha = 0.4/len(sgs)

    nevents = last_sg.next_eid-1

    import seaborn as sns
    ev_colors = sns.color_palette("hls", nevents)

    wn = last_sg.all_nodes[wn_label]

    real_wave = wn.get_wave()
    real_wave.data = ma.masked_array(real_wave.data, copy=True)


    len_mins = (wn.et - wn.st) / 60.0


    f = Figure((len_mins * zoom, 5))
    plot_with_fit_shapes(fname=None, wn=wn,title=wn_label, fig=f, plot_dets=plot_dets)

    ax = f.gca()

    if plot_predictions:
        predictions = []
        for (eid, phase) in wn.arrivals():
            if eid < 0: continue
            event = last_sg.get_event(eid)
            predictions.append([phase+"_%d" % eid, event.time+tt_predict(event, wn.sta, phase)])
        plot_pred_atimes(dict(predictions), real_wave, axes=ax, color="purple")


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
