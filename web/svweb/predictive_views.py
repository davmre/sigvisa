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

from sigvisa.treegp.gp import sort_morton
from sigvisa.plotting.plot import plot_with_fit_shapes, plot_pred_atimes, subplot_waveform
from sigvisa.signals.io import Waveform
from sigvisa import Sigvisa
from sigvisa.learn.train_wiggle_models import wiggle_params_by_level
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.graph.graph_utils import predict_phases_sta, parse_key
from sigvisa.models.ttime import tt_predict

from sigvisa.source.event import Event, get_event
from sigvisa.infer.template_xc import fastxc
from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, TimeRangeRunSpec, do_coarse_to_fine, initialize_from, do_inference, initialize_sg

from sigvisa.learn.train_param_common import load_modelid, load_modelid_evids

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec



def set_atimes_from_alignment(sg, wn, max_align=15.0):

    for (eid, phase) in wn.arrivals():
        if phase=="UA": 
            continue
    
        (_, _, _, _, _, N) = wn.wavelet_basis
        v, tg = wn.get_template_params_for_arrival(eid, phase)
        logenv = tg.abstract_logenv_raw(v, srate=wn.srate, fixedlen=N)

        cssm = wn.arrival_ssms[(eid, phase)]
        wiggle = cssm.mean_obs(N)


        pred_signal = np.exp(logenv)*wiggle
        #pred_signal = wiggle

        if np.max(np.abs(pred_signal)) < 1e-4:
            print "skipping alignment for phase %s because no wiggles predicted" % phase
            continue

        pred_atime_idx = int((v["arrival_time"] - wn.st) * wn.srate)
        window_pre_idx = int(max_align * wn.srate)
        window_start_idx = pred_atime_idx - window_pre_idx
        window_end_idx = pred_atime_idx + int(max_align*wn.srate) + N
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

    wn._parent_values()


def evid_fits(request, runid, evid, sta):
    runid=int(runid)
    evid = int(evid)
    s = Sigvisa()
    query = "select fitid from sigvisa_coda_fit where runid=%d and evid=%d and sta='%s'" % (runid, evid, sta)
    fitids = s.sql(query)
    fitid = fitids[0][0]
    return HttpResponseRedirect(reverse('fit_run_detail', args=(fitid,)))
    
def local_hparams(request, runid, sta):
    with open("cached_hparams_%s_%s.txt" % (runid, sta), "r") as f:
        hp = f.read()

    return HttpResponse(hp, content_type="text/plain")

    

def PredictedSignalMap(request, runid, sta):

    s = Sigvisa()
    
    evids = np.loadtxt(os.path.join(s.homedir, "notebooks", "thesis", "train_evids.txt"), dtype=int)

    markerCode = ""
    for i, evid in enumerate(evids):
        ev = get_event(evid=evid)
        #lon, lat, depth, dist, mb = x
        evid = evids[i]
        popuptxt = "loc %.2f %.2f<br>depth %.1fkm mb %.1f"% (ev.lon, ev.lat, ev.depth, ev.mb)
        evid_url = reverse('evid_fits', args=(runid, sta, evid))
        aligned_url = reverse('aligned_signal', args=(runid,)) + "?evid=%d;sta=%s" % (evid, sta)
        popuptxt += "<br>evid <a href=\"%s\">%d</a>" % (evid_url, ev.evid)
        popuptxt += "<br><a href=\"%s\">aligned signal</a>" % (aligned_url)
        popuptxt += "<br><a href=\"\#\" onclick=\"snapToEvent(marker_{id}, {depth})\">snap to location</a>".format(id=i, depth=ev.depth)

        evstr =  """
        var marker_{id} = L.circleMarker([{lat}, {lon}]);
        marker_{id}.bindPopup('{popup}');
        evmarkers.addLayer(marker_{id})
        """.format(id=i, lat=ev.lat, lon=ev.lon, popup=popuptxt)
        markerCode += evstr

    return render_to_response("svweb/gpsignal_visualize.html",
                              {'chan': "auto",
                               'sta': sta,
                               'band': "freq_0.8_4.5",
                               'phases': "P,Pn,Pg,Sn,Lg",
                               'depth': 0.0,
                               'zoom': 4.0,
                               'markerCode': markerCode,
                               'runid': runid,
                               }, context_instance=RequestContext(request))



def get_hparams(sg, wn, ev):
    
    x = np.array(((ev.lon, ev.lat, ev.depth, 0.0, 0.0),))

    def extract_hparams(model):

        idx = model._x_to_cluster(x)[0]
        lgp = model.local_gps[idx]
        svar = lgp.cov_main.wfn_params[0]
        lscale_horiz, lscale_depth = lgp.cov_main.dfn_params
        nvar = lgp.noise_var

        return np.sqrt(svar), np.sqrt(nvar), lscale_horiz, lscale_depth
        

    hparam_str = "gp hparams for %s\n\n" % (wn.label)
    phases = set()
    for n in sg.extended_evnodes[1]:
        try:
            (eid, phase, sta, chan, band, param) = parse_key(n.label)
            sstd, nstd, lscale_horiz, lscale_depth = extract_hparams(n.model)
        except Exception as e:
            continue

        phases.add(phase)
        hparam_str +=  "%s %s signal %.1f noise %.1f horiz %.4f depth %.4f\n" % (phase, param, sstd, nstd, lscale_horiz, lscale_depth)


    hparam_str += "\n"
    params_by_level = wiggle_params_by_level(wn.srate, sg.wiggle_family)
    for phase in phases:
        for params in params_by_level:
            param_nums = [int(p.split("_")[-1]) for p in params]
            minparam, maxparam = np.min(param_nums), np.max(param_nums)
            minmodel = wn.wavelet_param_models[phase][minparam]

            sstd, nstd, lscale_horiz, lscale_depth = extract_hparams(minmodel)
            label = "%s wavelets %d-%d: " % (phase, minparam, maxparam)
            hparam_str += label + "signal %.1f noise %.1f horiz %.4f depth %.4f\n" % (sstd, nstd, lscale_horiz, lscale_depth)

    return hparam_str

def PredictedSignalView(request, runid):

    runid=int(runid)

    sta = request.GET.get("sta", "")

    hz = float(request.GET.get("hz", "10.0"))

    wiggle_family = request.GET.get("wiggle_family", "db4_2.0_3_20.0")

    lon = float(request.GET.get("lon", "0.0"))
    lat = float(request.GET.get("lat", "0.0"))
    depth = float(request.GET.get("depth", "0.0"))
    time = float(request.GET.get("time", "1167609600"))
    mb = float(request.GET.get("mb", "4.0"))
    evid = int(request.GET.get("evid", "-1"))
    phases = str(request.GET.get("phases", "Pn,Pg,Lg,Sn,P")).split(",")

    zoom = float(request.GET.get("zoom", '1'))


    align = request.GET.get("align", 't').startswith('t')
    max_align = float(request.GET.get("max_align", '15.0'))

    sg = SigvisaGraph(template_model_type="gpparam", 
                      template_shape="lin_polyexp",
                      wiggle_family=wiggle_family,
                      min_mb=1.0,
                      phases=phases,
                      wiggle_model_type="gplocal+lld+none",
                      raw_signals=True,
                      base_srate=hz,
                      #dummy_fallback=True,
                      runids=(runid,))

    if evid > 0:
        ev = Event(evid=evid)
    else:
        ev = Event(lon=lon, lat=lat, depth=depth, time=time, mb=mb)

    s = Sigvisa()
    arriving_phases = predict_phases_sta(ev, sta, phases)
    atimes = [ev.time + tt_predict(ev, sta, phase) for phase in arriving_phases]
    stime = np.min(atimes) - 5.0
    etime = np.max(atimes) + 30.0
    npts = int((etime - stime) * hz)
    d = np.zeros((npts,))
    chan=s.default_vertical_channel[sta]
    band="freq_0.8_4.5"
    wave = Waveform(d, sta=sta, srate=hz, stime=stime, etime=etime, 
                    filter_str=band, chan=chan)
    wn = sg.add_wave(wave)

    sg.add_event(ev)

    for n in sg.extended_evnodes[1]:
        if n in sg.evnodes[1].values(): continue
        if n.deterministic(): continue
        n.parent_predict()

    wn._set_cssm_priors_from_model()

    try:
        hparams = get_hparams(sg, wn, ev)
    except Exception as e:
        hparams = "hparams failed with exception %s" % (str(e))

    with open("cached_hparams_%d_%s.txt" % (runid, sta), "w") as f:
        f.write(hparams)

    fig = Figure(figsize=(10*zoom, 5), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    axes.set_xlabel("Time (s)", fontsize=8)

    
    pred_signal = wn.tssm.mean_obs(wn.npts)
    w = Waveform(pred_signal, srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
    subplot_waveform(w, axes, color='green', linewidth=1.2, alpha=0.7)

    signal_var = wn.tssm.obs_var(wn.npts)
    w2 = Waveform(pred_signal-2*np.sqrt(signal_var), srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
    subplot_waveform(w2, axes, color='green', linewidth=1.0, fill_y2=pred_signal+2*np.sqrt(signal_var), alpha=0.2)



    atimes = dict([("%d_%s" % (eid, phase), wn.get_template_params_for_arrival(eid=eid, phase=phase)[0]['arrival_time']) for (eid, phase) in wn.arrivals()])
    plot_pred_atimes(dict(atimes), wn.get_wave(), axes=axes, alpha=1.0, bottom_rel=0.0, top_rel=0.1)

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response

def AlignedSignalView(request, runid):
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
    time = float(request.GET.get("time", "1167609600"))
    mb = float(request.GET.get("mb", "4.0"))
    evid = int(request.GET.get("evid", "-1"))
    phases = str(request.GET.get("phases", "Pn,Pg,Lg,Sn,P")).split(",")

    zoom = float(request.GET.get("zoom", '10'))

    align = request.GET.get("align", 't').startswith('t')
    max_align = float(request.GET.get("max_align", '15.0'))

    if evid > 0:
        ev = Event(evid=evid)
    else:
        ev = Event(lon=lon, lat=lat, depth=depth, time=time, mb=mb)

    rs = EventRunSpec(evs=[ev,], stas=[sta,], initialize_events=True)
    ms = ModelSpec(template_model_type="gpparam", wiggle_family=wiggle_family, 
                   min_mb=1.0,
                   phases=phases,
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
        set_atimes_from_alignment(sg, wn, max_align=max_align)

    fig = Figure(figsize=(10*zoom, 5), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)
    axes.set_xlabel("Time (s)", fontsize=8)

    

    subplot_waveform(wn.get_wave(), axes, color='black', linewidth=1.0, plot_dets=None)
    pred_signal = wn.tssm.mean_obs(wn.npts)
    w = Waveform(pred_signal, srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
    subplot_waveform(w, axes, color='green', linewidth=1.2, alpha=0.7)

    signal_var = wn.tssm.obs_var(wn.npts)
    w2 = Waveform(pred_signal-2*np.sqrt(signal_var), srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
    subplot_waveform(w2, axes, color='green', linewidth=1.0, fill_y2=pred_signal+2*np.sqrt(signal_var), alpha=0.2)


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

def ClusterFits(request, runid, sta, cluster):
    runid = int(runid)
    cluster = int(cluster)

    all_phases = ["Pn", "Pg", "Sn", "Lg"]
    
    s = Sigvisa()
    cluster_evids = np.loadtxt(os.path.join(s.homedir, 
                                            "train_clusters", 
                                            "cluster_%03d" % cluster), 
                               dtype=int)
    


    fitids = []
    all_data = {}
    X = []
    for evid in cluster_evids:
        query = "select fitid from sigvisa_coda_fit where runid=%d and evid=%d and sta='%s'" % (runid, evid, sta)
        try:
            fitid = s.sql(query)[0][0]
        except:
            continue

        ev = get_event(evid)

        query = "select phase, fpid, arrival_time from sigvisa_coda_fit_phase where fitid=%d" % (fitid)
        r = s.sql(query)
        phases = {}
        for phase, fpid, atime in r:
            pred_atime = tt_predict(ev, sta, phase) + ev.time
            ttr = atime - pred_atime
            phases[phase] = (phase, fpid, ttr)

        column_list = []
        for phase in all_phases:
            if phase in phases:
                column_list.append(phases[phase])
            else:
                column_list.append((phase, 0, 0))

        evstr = str(ev)
        X.append((ev.lon, ev.lat))
        fitids.append(fitid)
        all_data[fitid] = (evid, evstr, fitid, column_list)
        
    X = np.asarray(X)
    fitids = np.asarray(fitids)
    X, fitids = sort_morton(X, fitids)
    sorted_data=[]
    for fitid in fitids:
        sorted_data.append(all_data[fitid])

    return render_to_response("svweb/cluster_wiggles.html",
                              {'sta': sta,
                               'runid': runid,
                               'cluster': cluster,
                               'all_phases': all_phases,
                               'fitids': sorted_data,
                               }, context_instance=RequestContext(request))
