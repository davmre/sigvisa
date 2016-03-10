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

from sigvisa.utils.geog import dist_km
from sigvisa.infer.template_xc import fastxc

from sigvisa.explore.correlate_validation_events import load_arrivals

from sigvisa.infer.correlations.historical_signal_library import get_historical_signals
from sigvisa.infer.correlations.ar_correlation_model import estimate_ar, ar_advantage, iid_advantage
from sigvisa.infer.correlations.event_proposal import get_atime_posteriors, reweight_uniform_top, reweight_uniform_all
from sigvisa.infer.correlations.weighted_event_posterior import build_ttr_model_array, hack_ev_time_posterior_with_weight

import cPickle as pickle



def is_proposable(sg, ev, proposal_type):

    stas, phase, temper = proposal_type

    atime_lls = get_atime_posteriors(sg, phase, ar_advantage=False)

    global_srate = 1.0
    global_stime = sg.inference_region.stime
    N = int((sg.inference_region.etime - sg.inference_region.stime)*global_srate)

    proposal_otime_likelihoods = [hack_ev_time_posterior_with_weight(sg, x, sta_lls, stas=stas,
                                                                     N=N,
                                                                     global_stime = global_stime,
                                                                     global_srate = global_srate,
                                                                     temper=temper)
                                for (x, sta_lls) in atime_lls]

    proposal_weights = []
    proposal_otime_posteriors = []
    for ll in proposal_otime_likelihoods:
        C = np.max(ll)
        posterior = np.exp(ll-C)
        Z = np.sum(posterior)
        posterior /= Z
        logZ = np.log(Z) + C
        proposal_weights.append(logZ)
        proposal_otime_posteriors.append(posterior)

    proposal_weights = reweight_uniform_all(reweight_uniform_top(proposal_weights, n=15), uniform_prob=0.05)

    for i in range(len(atime_lls)):

        if proposal_weights[i] < 0.02: continue

        x, _ = atime_lls[i]
        lon, lat, depth = x[0,0], x[0,1], x[0,2]

        otime_dist = proposal_otime_posteriors[i]
        mle_time = np.argmax(otime_dist) / global_srate + global_stime

        if dist_km((lon, lat), (ev.lon, ev.lat)) < 30.0:
            if np.abs(mle_time - ev.time) > 50.0:
                print "WARNING proposed event %s is close to ev %s, but proposed time %.1f does not match" % (str(x), str(ev), mle_time)
            else:
                print "successful proposal for", ev
                return True

    print "unsuccessful proposal"
    return False


        
def build_hourlong_sg(stas, ev):

    runid = 14
    hz=10.0
    uatemplate_rate=4e-4
    bands=["freq_0.8_4.5"]
    phases=["P", "S", "Lg", "Pg"]
    hack_constraint = True
    raw_signals = True

    region_lon = (-126, -100)
    region_lat = (32, 49)

    region_stime = ev.time - 400
    region_etime = ev.time + 400



    runids=(runid,)

    rs = TimeRangeRunSpec(sites=stas, start_time=region_stime, end_time=region_etime)

    region_stime = rs.start_time
    region_etime = rs.end_time
    region = Region(lons=region_lon, lats=region_lat, 
                    times=(region_stime, region_etime),
                    rate_bulletin="isc", 
                    min_mb=2.0,
                    rate_train_start=1167609600,
                    rate_train_end=1199145600)


    ms1 = ModelSpec(template_model_type="gpparam",
                    wiggle_family="db4_2.0_3_20.0",
                    wiggle_model_type="gplocal+lld+none",
                    uatemplate_rate=uatemplate_rate,
                    max_hz=hz,
                    phases=phases,
                    bands=bands,
                    runids=runids,
                    inference_region=region,
                    dummy_fallback=True,
                    raw_signals=raw_signals,
                    hack_param_constraint=hack_constraint,
                    vert_only=True)

    sg = rs.build_sg(ms1)
    return sg

def build_proposals(stas, phases):

    tempers = (0.1, 1.0, 10.0)

    proposals = []

    for temper in tempers:
        for phase in phases:

            proposals.append((stas, phase, temper))

            for sta in stas:
                proposals.append(((sta,), phase, temper))

            sta_pairs = set()
            while len(sta_pairs) < 5:
                sta1 = np.random.choice(stas)
                sta2 = np.random.choice(stas)
                candidate_pair = tuple(sorted((sta1, sta2)))
                sta_pairs.add(candidate_pair)

            for sta_pair in sta_pairs:
                proposals.append((sta_pair, phase, temper))                

            sta_triples = set()
            while len(sta_triples) < 5:
                sta1 = np.random.choice(stas)
                sta2 = np.random.choice(stas)
                sta3 = np.random.choice(stas)
                candidate_triple = tuple(sorted((sta1, sta2, sta3)))
                sta_triples.add(candidate_triple)

            for sta_triple in sta_triples:
                proposals.append((sta_triple, phase, temper))                

    return proposals


def experiment():
    stas = tuple("ANMO,ELK,ILAR,KDAK,NEW,NVAR,PDAR,PFO,TXAR,ULM,YBH,YKA".split(","))
    phases= ["P", "Lg", "S", "Pg"]

    s = Sigvisa()
    validation_evid_file = os.path.join(s.homedir, "notebooks", "thesis", "validation_evids.txt")
    validation_evids = np.loadtxt(validation_evid_file)

    proposable_events = dict()
    proposals = build_proposals(stas, phases)
    for proposal_type in proposals:
        proposable_events[proposal_type] = 0

    outfile = open("multi_station.txt", "w")

    for evid in validation_evids:

        ev = get_event(evid)
        sg = build_hourlong_sg(stas, ev)
        for proposal_type in proposals:
            try:
                print "evid %d testing proposal %s" % (evid, str(proposal_type))
                if is_proposable(sg, ev, proposal_type):
                    proposable_events[proposal_type] += 1
            except Exception as e:
                print "Exception", e

        outfile.write("evid %d status %s\n" % (evid, proposable_events))
        outfile.flush()
        print "evid", evid, "status", proposable_events

    outfile.close()

if __name__ == "__main__":   

    experiment()
