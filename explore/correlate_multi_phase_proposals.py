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
from sigvisa.infer.correlations.event_proposal import compute_proposal_distribution, reweight_uniform_top, reweight_uniform_all


import cPickle as pickle



def is_proposable(sg, ev, proposal_type):

    stas, phases, temper = proposal_type

    xs, weights, posteriors = compute_proposal_distribution(sg, stas, phases, temper=temper)

    proposal_weights = reweight_uniform_all(reweight_uniform_top(weights, n=15), uniform_prob=0.05)

    for i in range(len(xs)):

        if weights[i] < 0.02: continue

        x = xs[i]
        lon, lat, depth = x[0,0], x[0,1], x[0,2]

        otime_dist = posteriors[i]
        mle_time = np.argmax(otime_dist) / 1.0  + sg.inference_region.stime

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

def build_proposals():

    tempers = (1.0, 10.0)

    proposals = []
    phase_sets = [("Lg"), ("P", "Lg"), ("P"), ("P", "Pg", "S", "Lg")]
    sta_sets = [("ULM",), ("NEW",), ("ULM", "NEW")]

    for temper in tempers:
        for phases in phase_sets:
            for stas in sta_sets:
                proposals.append((stas, phases, temper))

    return proposals


def experiment():
    #stas = tuple("ANMO,ELK,ILAR,KDAK,NEW,NVAR,PDAR,PFO,TXAR,ULM,YBH,YKA".split(","))
    stas = ("ULM", "NEW")
    phases= ["P", "Lg", "S", "Pg"]

    s = Sigvisa()
    validation_evid_file = os.path.join(s.homedir, "notebooks", "thesis", "validation_evids.txt")
    validation_evids = np.loadtxt(validation_evid_file)

    proposable_events = dict()
    proposals = build_proposals()
    for proposal_type in proposals:
        proposable_events[proposal_type] = []

    for evid in validation_evids:

        ev = get_event(evid)
        sg = build_hourlong_sg(stas, ev)
        for proposal_type in proposals:
            try:
                print "evid %d testing proposal %s" % (evid, str(proposal_type))
                if is_proposable(sg, ev, proposal_type):
                    proposable_events[proposal_type].append(evid)
            except Exception as e:
                print "Exception", e

        with open("multi_phase.txt", "w") as f:
            f.write("%s\n" % repr(proposable_events))
            for (k, evids) in proposable_events.items():
                f.write("%s: %d\n" % (str(k), len(evids)))




if __name__ == "__main__":   

    experiment()
