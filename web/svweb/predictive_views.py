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
import os
import cPickle as pickle

from sigvisa.plotting.plot import plot_with_fit_shapes, plot_pred_atimes, subplot_waveform
from sigvisa.signals.io import Waveform
from sigvisa import Sigvisa
from sigvisa.source.event import Event
from sigvisa.infer.template_xc import fastxc
from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, TimeRangeRunSpec, do_coarse_to_fine, initialize_from, do_inference, initialize_sg


from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec



def set_atimes_from_alignment(sg, wn):

    for (eid, phase) in wn.arrivals():
        if phase=="UA": 
            continue
    
        (_, _, _, _, _, N) = wn.wavelet_basis
        v, tg = wn.get_template_params_for_arrival(eid, phase)
        logenv = tg.abstract_logenv_raw(v, srate=wn.srate, fixedlen=N)

        cssm = wn.arrival_ssms[(eid, phase)]
        wiggle = cssm.mean_obs(N)
        pred_signal = np.exp(logenv)*wiggle

        if np.max(np.abs(pred_signal)) < 1e-4:
            print "skipping alignment for phase %s because no wiggles predicted" % phase
            continue

        pred_atime_idx = int((v["arrival_time"] - wn.st) * wn.srate)
        window_pre_idx = int(15.0 * wn.srate)
        window_start_idx = pred_atime_idx - window_pre_idx
        window_end_idx = pred_atime_idx + int(15.0*wn.srate) + N
        xc_window = wn.get_value()[ window_start_idx : window_end_idx ]
        
        print "pred signal len", N, "window len", len(xc_window)
        r = fastxc(pred_signal, xc_window)
        align_offset = np.argmax(r) - window_pre_idx
        align_time = v["arrival_time"] + align_offset / wn.srate

        print "phase", phase, "predicted atime", v["arrival_time"], "aligned atime", align_time

        # TODO: SET ARRIVAL TIME
        tnodes = sg.get_template_nodes(eid, wn.sta, phase, wn.band, wn.chan)
        k_atime, n_atime = tnodes["arrival_time"]
        n_atime.set_value(align_time, key=k_atime)

def PredictedSignalView(request, runid):
    # plot the posterior wiggle
    # (mean plus samples?)
    # also show the wiggle coefs

    runid=int(runid)

    sta = request.GET.get("sta", "")

    hz = float(request.GET.get("hz", "10.0"))

    wiggle_family = request.GET.get("wiggle_family", "db4_2.0_3_20.0")

    lon = float(request.GET.get("lon", "0.0"))
    lat = float(request.GET.get("lat", "0.0"))
    depth = float(request.GET.get("depth", "0.0"))
    time = float(request.GET.get("time", "0.0"))
    mb = float(request.GET.get("mb", "4.0"))
    evid = int(request.GET.get("evid", "-1"))

    zoom = float(request.GET.get("zoom", '1'))

    align = request.GET.get("align", 't').startswith('t')

    if evid > 0:
        ev = Event(evid=evid)
    else:
        ev = Event(lon=lon, lat=lat, depth=depth, time=time, mb=mb)

    rs = EventRunSpec(evs=[ev,], stas=[sta,], initialize_events=True)
    ms = ModelSpec(template_model_type="gpparam", wiggle_family=wiggle_family, 
                   min_mb=1.0,
                   phases=["P", "S", "Lg", "pP", "Pg"],
                   wiggle_model_type="gplocal+lld+none", 
                   raw_signals=True, 
                   max_hz=hz,
                   #dummy_fallback=True,
                   runids=(runid,))
    sg = rs.build_sg(ms)
    initialize_sg(sg, ms , rs)

    for n in sg.extended_evnodes[1]:
        if n in sg.evnodes[1].values(): continue
        if n.deterministic(): continue
        n.parent_predict()



    wn = sg.station_waves[sta][0]

    wn._parent_values()
    lp = wn.log_p()
    #wn._set_cssm_priors_from_model()

    if align:
        set_atimes_from_alignment(sg, wn)

    fig = Figure(figsize=(10*zoom, 5*zoom), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    axes.set_xlabel("Time (s)", fontsize=8)

    subplot_waveform(wn.get_wave(), axes, color='black', linewidth=1.0, plot_dets=None)


    pred_signal = wn.tssm.mean_obs(wn.npts)
    w = Waveform(pred_signal, srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
    subplot_waveform(w, axes, color='green', linewidth=2.5)

    signal_var = wn.tssm.obs_var(wn.npts)
    w1 = Waveform(pred_signal+2*np.sqrt(signal_var), srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
    subplot_waveform(w1, axes, color='red', linewidth=1.0)
    w2 = Waveform(pred_signal-2*np.sqrt(signal_var), srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
    subplot_waveform(w2, axes, color='red', linewidth=1.0)


    atimes = dict([("%d_%s" % (eid, phase), wn.get_template_params_for_arrival(eid=eid, phase=phase)[0]['arrival_time']) for (eid, phase) in wn.arrivals()])
    plot_pred_atimes(dict(atimes), wn.get_wave(), axes=axes, alpha=1.0, bottom_rel=0.0, top_rel=0.1)


    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response

def SignalLibraryView(request):
    # plot the posterior wiggle
    # (mean plus samples?)
    # also show the wiggle coefs

    modelid = int(request.GET.get("modelid", ""))
    idx = int(request.GET.get("idx", ""))

    zoom = float(request.GET.get("zoom", '1'))

    s = Sigvisa()
    fname = os.path.join(s.homedir, "db_cache", "history_%d.pkl" % modelid)
    with open(fname, 'rb') as f:
        library = pickle.load(f)

    x, entry = library[idx]
    n_wns = len(entry)

    fig = Figure(figsize=(10*zoom, 5*zoom), dpi=144)
    fig.patch.set_facecolor('white')
    for i, k in enumerate(entry.keys()):
        axes = fig.add_subplot(n_wns,1,i+1)
        s = entry[k]
        xs = np.linspace(0, 20, len(s))
        axes.plot(xs, s)
        axes.set_title("%s %s" % (x, k))

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response

