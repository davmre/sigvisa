import numpy as np
from sigvisa import Sigvisa
import os
import sys

from sigvisa.signals.io import fetch_waveform
from sigvisa.source.event import get_event

from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, TimeRangeRunSpec, do_coarse_to_fine, initialize_from, do_inference, initialize_sg
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.treegp.gp import GPCov
from sigvisa.graph.region import Region
from sigvisa.signals.common import Waveform


from sigvisa.models.ttime import tt_predict
from sigvisa.infer.template_xc import fastxc


from sigvisa.infer.template_xc import fastxc

from sigvisa.explore.correlate_validation_events import load_arrivals

from sigvisa.infer.correlations.historical_signal_library import get_historical_signals
from sigvisa.infer.correlations.ar_correlation_model import estimate_ar, ar_advantage, iid_advantage
from sigvisa.infer.correlations.weighted_event_posterior import build_ttr_model_array, hack_ev_time_posterior_with_weight

import cPickle as pickle


def build_sg(sta, phases=None):
    # create an sg with appropriate models
    region_lon = (-126, -100)
    region_lat = (32, 49)
    region_stime = 1167609600.0
    region_etime = region_stime + 1000.0

    min_mb = 4.0
    uatemplate_rate=1e-4
    hz = 10.0
    runid=14
    if phases is None:
        phases = "P,Pg,S,Lg".split(",")

    region = Region(lons=region_lon, lats=region_lat,
                    times=(region_stime, region_etime),
                    rate_bulletin="isc",
                    rate_train_start=1167609600,
                    rate_train_end=1199145600)

    sg = SigvisaGraph(template_model_type="gpparam",
                    wiggle_model_type="gplocal+lld+none",
                    wiggle_family="db4_2.0_3_20.0",
                      template_shape="lin_polyexp",
                    uatemplate_rate=uatemplate_rate,
                    phases=phases,
                    runids=(runid,),
                    inference_region=region,
                    dummy_fallback=False,
                    hack_param_constraint=True,
                    min_mb=min_mb,
                    raw_signals=True)
    
    s = Sigvisa()
    chans = s.sql("select distinct chan from sigvisa_param_model where site='%s'  and fitting_runid=%d" % (sta, runid))
    chan = chans[0][0]
    
    null_data = np.zeros((800,))
    null_wave = Waveform(null_data, stime=region_stime + 1000.0, srate=10.0, sta=sta, chan=chan, 
                         filter_str="freq_0.8_4.5;hz_10", band="freq_0.8_4.5")
    sg.add_wave(null_wave)
    
    return sg

def proposal_xc(validation_signal, proposals, ar_nm=None, use_xc=False):

    xs = np.array([x for (x, sta_lls) in proposals])
    gpsignals = [signals.values()[0] for (x, signals) in proposals]
    
    if use_xc:
        xcs = np.array([np.max(fastxc(s, validation_signal)) for s in gpsignals])
    elif ar_nm is None:
        xcs = np.array([np.max(iid_advantage(validation_signal, s)) for s in gpsignals])
    else:
        xcs = np.array([np.max(ar_advantage(validation_signal, s, ar_nm)) for s in gpsignals])
    
    p = sorted(np.arange(len(xcs)), key = lambda i : -xcs[i])
    sorted_xcs = xcs[p]
    sorted_xs = xs[p]
    sorted_gpsignals = [gpsignals[j] for j in p]
    
    return sorted_xcs, sorted_xs, sorted_gpsignals

def locate_event(signal, training_evs, training_signals, use_xc=False, ar_nm=None):
    xcs = []
    for ev, s in zip(training_evs, training_signals):

        if use_xc:
            xc = fastxc(s, signal)
        elif ar_nm is None:
            xc = iid_advantage(signal, s)
        else:
            xc = ar_advantage(signal, s, ar_nm)
        xcs.append(np.max(xc))
    xcs = np.array(xcs)
    p = sorted(np.arange(len(xcs)), key = lambda i : -xcs[i])
    sorted_xcs = xcs[p]
    sorted_evs = [training_evs[j] for j in p]
    sorted_signals = [training_signals[j] for j in p]

    return sorted_evs, sorted_xcs, sorted_signals

from sigvisa.models.noise.nm_node import load_noise_model_prior
from sigvisa.models.noise.armodel.model import ARModel, ErrorModel

def get_prior_nm(wn, runid):
    waveform = wn.get_wave()
    prior_mean_dist, prior_var_dist, prior_param_dist = \
      load_noise_model_prior(sta=waveform["sta"], chan=waveform["chan"],
                             band=waveform["band"], hz=waveform["srate"],
                             runids=(runid,),
                             env = wn.nm_node.is_env)
        
    em = ErrorModel(mean=0.0, std=np.sqrt(prior_var_dist.predict()))
    prior_nm = ARModel(params=prior_param_dist.predict(), em=em,
                        c=prior_mean_dist.predict(), sf=waveform["srate"])
    return prior_nm



def compute_validation_training_alignments(sta, phase):
    fname = "alignments_%s_%s.pkl" % (sta, phase)
    try:
        with open(fname, 'rb') as f:
            r = pickle.load(f)
        print "skipping existing file", fname
        return
    except IOError as e:
        pass


    sg = build_sg(sta, phases=[phase,])
    proposals = get_historical_signals(sg, phase)
    training_evs, training_signals = load_arrivals(sta, phase, 14, label="training")
    validation_evs, validation_signals = load_arrivals(sta, phase, 14, label="validation")

    wn = sg.station_waves[sta][0]
    prior_nm = get_prior_nm(wn, 14)

    results = dict()

    xs = np.array([x for (x, sta_lls) in proposals])
    gpsignals = [signals.values()[0] for (x, signals) in proposals]

    results["xs"] = xs
    results["gpsignals"] = gpsignals

    for vev, vs in zip(validation_evs, validation_signals):

        xcs = np.array([fastxc(s, vs) for s in gpsignals])
        iid_advantages = np.array([iid_advantage(vs, s) for s in gpsignals])
        ar_advantages = np.array([ar_advantage(vs, s, prior_nm) for s in gpsignals])
    
        rdict = {}
        rdict["ev"] = vev
        rdict["signal"] = vs
        rdict["xcs"] = xcs
        rdict["iid_advantages"] = iid_advantages
        rdict["ar_advantages"] =  ar_advantages

        results[vev.evid] = rdict

        print sta, phase, vev


    fname = "alignments_%s_%s.pkl" % (sta, phase)
    with open(fname, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":   

    stas = sys.argv[1].split(",")

    for sta in stas:
        try:
            compute_validation_training_alignments(sta, "P")
        except Exception as e:
            print e
            pass

        try:
            compute_validation_training_alignments(sta, "Pg")
        except Exception as e:
            print e
            pass

        try:
            compute_validation_training_alignments(sta, "S")
        except Exception as e:
            print e
            pass

        try:
            compute_validation_training_alignments(sta, "Lg")
        except Exception as e:
            print e
            pass


