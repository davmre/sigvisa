import numpy as np

from sigvisa import Sigvisa
from sigvisa.source.event import get_event, Event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, SyntheticRunSpec, do_coarse_to_fine, initialize_from, do_inference
from sigvisa.infer.swap_rpc.sg_server import run_parallel_coarse_to_fine
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.treegp.gp import GPCov
from sigvisa.synthetic.doublets import SampledWorld
from sigvisa.graph.region import Region

import os, sys, traceback
import cPickle as pickle
from optparse import OptionParser

# excludign ELK cause the noise model lookup fails for some reason
#stas = ["PDAR","NVAR","ANMO"]
stas = "NEW,PDAR,NVAR,ANMO,TXAR,PFO,YKA,ULM,ILAR".split(",")


region_lon = (-126, -100)
region_lat = (32, 49)
region_stime = 1239040000
region_etime = region_stime + 7200


def main(seed=1,  resume_from="", server_only=False, client_only=None, ctf_only=False):


    uatemplate_rate=1e-4
    hz = 10.0
    runid=3

    phases=["P", "S"]

    n_events = 2

    region = Region(lons=region_lon, lats=region_lat, times=(region_stime, region_etime))

    sw = SampledWorld(seed=seed)
    sw.sample_sg(runid=3, wiggle_model_type="dummy", wiggle_family="iid", sites=stas, phases=phases, tmtype="param", uatemplate_rate=uatemplate_rate, sample_uatemplates=True, n_events=n_events, min_mb=3.5, force_mb=None, len_s=region_etime-region_stime, tt_buffer_s=1000, hz=hz, dumpsg=False, dummy_fallback=True, stime=region_stime, evs=None, region=region, raw_signals=True)

    rs = SyntheticRunSpec(sw=sw, runid=runid, raw_signals=True)

    ms_coarse = ModelSpec(template_model_type="param",
                          wiggle_family="iid",
                          uatemplate_rate=uatemplate_rate,
                          max_hz=0.5,
                          phases=phases,
                          dummy_fallback=True,
                          hack_param_constraint = True,
                          hack_coarse_tts = 20,
                          inference_region=region,
                          min_mb=3.5,
                          raw_signals=False,
                          vert_only=True)

    ms_coarse.add_inference_round(enable_event_moves=True, enable_event_openworld=True, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'], prior_births_only=True, steps=100000)


    mss = []

    tt_coarseness = (200, 200, 100, 90, 50, 50)
    signal_coarseness = (10, 8, 7, 4, 4, 2)
    uatemplate_rates = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    hzs = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
    for ttc, sc, hz, utr in zip(tt_coarseness, signal_coarseness, hzs, uatemplate_rates):
        ms_coarse = ModelSpec(template_model_type="param",
                              wiggle_family="iid",
                              uatemplate_rate=utr,
                              max_hz=hz,
                              phases=phases,
                              dummy_fallback=True,
                              hack_param_constraint = True,
                              hack_coarse_tts = ttc,
                              hack_coarse_signal = sc,
                              inference_region=region,
                              min_mb=3.5,
                              raw_signals=False,
                              vert_only=True)
        mss.append(ms_coarse)


    mss[0].add_inference_round(enable_event_moves=True, enable_event_openworld=True, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'], prior_births_only=True, steps=10000000)
    for i in range(1, len(mss)):
        mss[i].add_inference_round(enable_event_moves=True, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'], prior_births_only=True, steps=10000000)

    specs = dict([("coarse_%d" % i, (mss[i], rs))  for i in range(len(mss))])
    names = sorted(specs.keys())


    #sg = rs.build_sg(ms_coarse)
    #do_inference(sg, ms_coarse, rs, dump_interval=10, print_interval=10, model_switch_lp_threshold=None)

    
    run_parallel_coarse_to_fine(names, specs, 
                                server_only=server_only, 
                                client_only=client_only,
                                min_swap_s = 1.0, 
                                max_swap_s = 1000.0,
                                allowable_wait_s = 1.0)


if __name__ == "__main__":
    try:

        parser = OptionParser()
        parser.add_option("--seed", dest="seed", default=1, type=int)
        parser.add_option("--resume_from", dest="resume_from", default="", type=str)
        parser.add_option("--server_only", dest="server_only", default=False, action="store_true")
        parser.add_option("--client_only", dest="client_only", default=None, type="str")
        parser.add_option("--ctf_only", dest="ctf_only", default=False, action="store_true")

        (options, args) = parser.parse_args()
        main(seed=options.seed, 
             ctf_only=options.ctf_only,
             resume_from=options.resume_from,
             client_only=options.client_only,
             server_only=options.server_only)

    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
