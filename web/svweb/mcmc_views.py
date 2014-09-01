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
import os
import time
import cPickle as pickle
from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *

from sigvisa import *

from sigvisa.source.event import Event
from sigvisa.signals.io import fetch_waveform, Segment
from sigvisa.graph.sigvisa_graph import predict_phases

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from sigvisa.plotting.event_heatmap import EventHeatmap
import hashlib

from svweb.models import SigvisaCodaFit, SigvisaCodaFitPhase, SigvisaCodaFittingRun, SigvisaWiggle, SigvisaGridsearchRun, SigvisaGsrunModel, SigvisaGsrunWave, SigvisaParamModel

from sigvisa.plotting.plot import bounds_without_outliers
from svweb.plotting_utils import process_plot_args, view_wave
from svweb.views import wave_plus_template_view
from sigvisa.signals.common import load_waveform_from_file
from sigvisa.utils.geog import lonlatstr, dist_km
from sigvisa.infer.gridsearch import propose_origin_times



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

def mcmcrun_detail(request, dirname, path):

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
