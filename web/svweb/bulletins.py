import numpy as np

import mimetypes

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

from matplotlib.figure import Figure

import cPickle as pickle

from sigvisa.utils.geog import dist_km
from sigvisa.utils.fileutils import mkdir_p
from sigvisa.source.event import Event, get_event
from sigvisa import *
from sigvisa.plotting.plot import savefig, subplot_waveform, plot_pred_atimes
from sigvisa.plotting.event_heatmap import EventHeatmap

from sigvisa.signals.common import Waveform

from sigvisa.infer.coarse_to_fine_init import EventRunSpec, ModelSpec, initialize_sg

from predictive_views import set_atimes_from_alignment

import os

bulletin_cache_basedir = "/home/dmoore/python/sigvisa/web/rating_cache/orids/"

def orid_ev(orid):
    s = Sigvisa()
    r = s.sql("select lon, lat, depth, time, mb from sigvisa_origin where orid=%d" % orid)
    lon, lat, depth, time, mb = r[0]
    ev = Event(lon=lon, lat=lat, depth=depth, time=time, mb=mb)
    return ev

def bulletin_ev_signal_page(request, orid):

    orid = int(orid)

    mkdir_p(os.path.join(bulletin_cache_basedir, '%d' % orid))

    rater = request.GET.get('rater', 'dave')
    score_threshold = float(request.GET.get('score_threshold', '125'))

    stas = "ANMO,ELK,ILAR,KDAK,NEW,NVAR,PDAR,PFO,TXAR,ULM,YBH,YKA".split(",")
    phases = "P,Pn,Pg,Lg,Sn".split(',')

    orids = np.loadtxt("/home/dmoore/python/sigvisa/filtered_orids", dtype=np.int)
    idx = np.searchsorted(orids, orid)
    #assert(orids[idx] == orid)
    # temp hack
    next_orid = orids[(idx+1) % len(orids)]
    prev_orid = orids[(idx-1) % len(orids)]
    filter_GET_params = ";rater=%s;score_threshold=%.2f" % (rater, score_threshold)

    ev = orid_ev(orid)
    ev_str = str(ev)

    s = Sigvisa()
    tmp = []
    for sta in stas:
        slon, slat = s.earthmodel.site_info(sta, 0)[:2]
        tmp.append((sta,  dist_km((slon, slat), (ev.lon, ev.lat))))
    sta_info = sorted(tmp, key = lambda d : d[1])

    nearby_query = "select orid from sigvisa_origin where lon between %f and %f and lat between %f and %f and matched_isc_orid > 0" % (ev.lon -2 , ev.lon + 2, ev.lat-2, ev.lat+2)
    nearby_orids = [r[0] for r in s.sql(nearby_query)]
    nearby_evs = [orid_ev(norid) for norid in nearby_orids]
    nearby_ev_dists = [dist_km((ev.lon, ev.lat), (nev.lon, nev.lat)) for nev in nearby_evs]
    nearby_evstrs = [",".join(str(nev).split(",")[1:5]) for nev in nearby_evs]
    nearby_ev_info = sorted(zip(nearby_ev_dists, nearby_orids, nearby_evstrs))[:4]

    ratings = s.sql("select rating from sigvisa_origin_rating where orid=%d and rater='%s'" % (orid, rater))
    if len(ratings) > 0:
        assert (len(ratings) == 1)
        rated = ratings[0][0]
    else:
        rated = '0'

    bulletin_ev_loc_helper(orid, ev, stas)
    sg = bulletin_sg(orid, stas, phases)
    for sta in stas:
        try:
            bulletin_ev_signal_vis_helper(orid, sg, sta)
        except Exception as e:
            print "WARNING: error at %s: %s" % (sta, e)
            continue
        bulletin_ev_signal_corr_helper(orid, sg, sta)

    return render_to_response("svweb/bulletin_signals.html",
                          {'sta_info': sta_info,
                           'orid': orid,
                           'rated': rated,
                           'ev_str': ev_str,
                           'nearby_ev_info': nearby_ev_info,
                           'next_orid': next_orid,
                           'prev_orid': prev_orid,
                           'filter_GET_params': filter_GET_params
                           }, context_instance=RequestContext(request))


def bulletin_sg(orid, stas, phases, runid=25, band="freq_0.8_4.5", hz=10):
    orid = int(orid)

    def build_sg():
        ev = orid_ev(orid)

        rs = EventRunSpec(evs=[ev,], stas=stas, 
                          initialize_events=True,
                          pre_s=50, post_s=100,
                          disable_conflict_checking=True)
        ms1 = ModelSpec(template_model_type="gpparam",
                wiggle_family="db4_2.0_3_20.0",
                wiggle_model_type="gplocal+lld+none",
                uatemplate_rate=1e-3,
                max_hz=hz,
                phases=phases,
                bands=(band,),
                runids=(runid,),
                min_mb=1.0,
                skip_levels=0,
                dummy_fallback=True,
                raw_signals=True,
                hack_param_constraint=True,
                vert_only=True)
        sg = rs.build_sg(ms1)
        initialize_sg(sg, ms1, rs)

        for n in sg.extended_evnodes[1]:
            if n in sg.evnodes[1].values(): continue
            if n.deterministic(): continue
            n.parent_predict()

        return sg

    fname = os.path.join(bulletin_cache_basedir, "%d" % orid, "pickle.sg")
    try:
        with open(fname, 'rb') as f:
            sg = pickle.load(f)
    except IOError:
        sg = build_sg()
        with open(fname, 'wb') as f:
            pickle.dump(sg, f)

    return sg

def bulletin_ev_rate(request, orid):
    orid = int(orid)

    rating = int(request.POST['approval'])
    next_orid = int(request.GET.get("next_orid", None))

    rater = request.GET.get('rater', 'dave')

    try:
        rating = int(request.POST['approval'])
    except KeyError:
        return HttpResponse("You didn't select a rating.")
    else:
        s = Sigvisa()
        q = "insert into sigvisa_origin_rating (orid, rater, rating) values (%d, '%s', %d)" % (orid, rater, rating)
        s.sql(q)
        print q
        s.dbconn.commit()
        return HttpResponseRedirect(reverse('bulletin_ev_signal_page', args=(next_orid,)) + "?" + request.GET.urlencode())
    return HttpResponse("Something went wrong.")


def bulletin_ev_loc_helper(orid, ev, stas):
    fname = os.path.join(bulletin_cache_basedir, "%d" % orid, "loc.png")
    if os.path.exists(fname):
        return

    s = Sigvisa()
    hm = EventHeatmap(f=None, left_lon=-126, right_lon=-100, top_lat=49, bottom_lat=33, calc=True, n=0)

    f = Figure(figsize=(13, 8))
    ax = f.add_subplot(111)
    hm.init_bmap(axes=ax, nofillcontinents=True, projection="cyl")
    hm.plot_earth(y_fontsize=14, x_fontsize=14)

    train_evids = np.loadtxt("/home/dmoore/python/sigvisa/notebooks/thesis/train_evids.txt")
    train_evs = [get_event(evid=int(evid)) for evid in train_evids]
    train_locs = [(tev.lon, tev.lat) for tev in train_evs]

    normed_locations = np.array([hm.normalize_lonlat(*location[:2]) for location in train_locs ])
    hm.plot_locations(normed_locations, marker=".", ms=6, mec="none", mew=0,
                          alpha=0.3, color="red")

    sta_locs = np.asarray([s.earthmodel.site_info(sta, 0)[:2] for sta in stas])
    hm.plot_locations(sta_locs, labels=stas, marker="^", ms=10, mfc="none", mec="blue", mew=3, alpha=1,
                      offmap_arrows=True, arrow_color="blue", 
                      label_x_off=12, label_y_off=0, label_pts=14,
                      edge_x_off=-60, edge_y_off=-20)

    inferred_locs = [(ev.lon, ev.lat)]
    normed_locations = np.array([hm.normalize_lonlat(*location[:2]) for location in inferred_locs ])
    hm.plot_locations(normed_locations, marker="*", ms=24, mec="purple", mew=0,
                      alpha=0.95, color="purple")

    savefig(fname, f, bbox_inches='tight')



def bulletin_ev_loc(request, orid):
    orid = int(orid)
    fname = os.path.join(bulletin_cache_basedir, "%d" % orid, "loc.png")
    mimetype=mimetypes.guess_type(fname)[0]
    return HttpResponse(open(fname).read(), content_type=mimetype)


def bulletin_ev_signal_vis_helper(orid, sg, sta):
    

    fname = os.path.join(bulletin_cache_basedir, "%d" % orid, "%s_vis.png" % sta)
    if os.path.exists(fname):
        return

    try:
        wn = sg.station_waves[sta][0]
    except IndexError:
        s = Sigvisa()
        sta = s.get_default_sta(sta)
        wn = sg.station_waves[sta][0]

    f = Figure((20, 5))
    f.patch.set_facecolor('white')
    ax = f.add_subplot(111)
    atime_args = {"color": "purple",
                  "top_rel": 0.95,
                  "bottom_rel": 0.05}
    wn.plot(ax=ax, model_lw=None, unass_lw=None, ev_lw=None, atime_args=atime_args)
    savefig(fname, f)

def bulletin_ev_signal_vis(request, orid, sta):
    orid = int(orid)
    fname = os.path.join(bulletin_cache_basedir, "%d" % orid, "%s_vis.png" % sta)
    mimetype=mimetypes.guess_type(fname)[0]
    return HttpResponse(open(fname).read(), content_type=mimetype)

def bulletin_ev_signal_corr_helper(orid, sg, sta):
    

    fname = os.path.join(bulletin_cache_basedir, "%d" % orid, "%s_corr.png" % sta)
    if os.path.exists(fname):
        return

    try:
        wn = sg.station_waves[sta][0]
    except IndexError:
        s = Sigvisa()
        sta = s.get_default_sta(sta)
        wn = sg.station_waves[sta][0]

    wn._parent_values()
    lp = wn.log_p()

    pred_atimes = dict([("%d_%s" % (eid, phase), wn.get_template_params_for_arrival(eid=eid, phase=phase)[0]['arrival_time']) for (eid, phase) in wn.arrivals()])

    set_atimes_from_alignment(sg, wn, max_align=15.0)

    f = Figure((140, 5))
    f.patch.set_facecolor('white')
    axes = f.add_subplot(111)

    subplot_waveform(wn.get_wave(), axes, color='black', linewidth=1.0, plot_dets=None)
    pred_signal = wn.tssm.mean_obs(wn.npts)
    w = Waveform(pred_signal, srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
    subplot_waveform(w, axes, color='red', linewidth=1.2, alpha=0.9)

    signal_var = wn.tssm.obs_var(wn.npts)
    w2 = Waveform(pred_signal-2*np.sqrt(signal_var), srate=wn.srate, stime=wn.st, sta=wn.sta, band=wn.band, chan=wn.chan)
    subplot_waveform(w2, axes, color='red', linewidth=1.0, fill_y2=pred_signal+2*np.sqrt(signal_var), alpha=0.15)


    atimes = dict([("%d_%s" % (eid, phase), wn.get_template_params_for_arrival(eid=eid, phase=phase)[0]['arrival_time']) for (eid, phase) in wn.arrivals()])
    plot_pred_atimes(dict(atimes), wn.get_wave(), axes=axes, alpha=1.0, bottom_rel=0.0, top_rel=0.1)

    plot_pred_atimes(dict(pred_atimes), wn.get_wave(), axes=axes, color="purple", alpha=1.0, 
                     bottom_rel=0.05, top_rel=0.95)


    savefig(fname, f)


def bulletin_ev_signal_corr(request, orid, sta):
    orid = int(orid)
    fname = os.path.join(bulletin_cache_basedir, "%d" % orid, "%s_corr.png" % sta)
    mimetype=mimetypes.guess_type(fname)[0]
    return HttpResponse(open(fname).read(), content_type=mimetype)

def bulletin_ev_signal_sac(request, orid, sta):
    orid = int(orid)
    fname = os.path.join(bulletin_cache_basedir, "%d" % orid, "%s.sac" % sta)
    mimetype=mimetypes.guess_type(fname)[0]
    return HttpResponse(open(fname).read(), content_type=mimetype)

