import numpy as np

from sigvisa import Sigvisa
from sigvisa.source.event import get_event, Event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, SyntheticRunSpec, do_coarse_to_fine, initialize_from, do_inference
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.treegp.gp import GPCov
from sigvisa.synthetic.doublets import SampledWorld
from sigvisa.graph.region import Region

import os, sys, traceback
import cPickle as pickle
from optparse import OptionParser

# excludign ELK cause the noise model lookup fails for some reason
stas = "NEW,PDAR,NVAR,ANMO,TXAR,PFO,YKA,ULM,ILAR".split(",")

region_lon = (-126, -100)
region_lat = (32, 49)
region_stime = 1239040000
region_etime = region_stime + 7200

def main(seed=1, n_events=2, resume_from=""):

    evs = None
    if n_events==-1:
        ev = Event(lon=-105.427, lat=43.731, depth=0.0, time=1239041017.07, mb=4.0, natural_source=False)
        evs = [ev,]

    uatemplate_rate=1e-4
    hz = 2.0
    runid=3
    phases=["P", "S", "Lg", "PcP", "ScP", "pP", "Pg"]

    region = Region(lons=region_lon, lats=region_lat, times=(region_stime, region_etime))

    sw = SampledWorld(seed=seed)
    sw.sample_sg(runid=3, wiggle_model_type="dummy", wiggle_family="iid", sites=stas, phases=phases, tmtype="param", uatemplate_rate=uatemplate_rate, sample_uatemplates=True, n_events=n_events, min_mb=3.5, force_mb=None, len_s=region_etime-region_stime, tt_buffer_s=1000, hz=hz, dumpsg=False, dummy_fallback=True, stime=region_stime, evs=evs, region=region)

    rs = SyntheticRunSpec(sw=sw, runid=runid)



    ms1 = ModelSpec(template_model_type="param",
                    wiggle_family="iid",
                    uatemplate_rate=uatemplate_rate,
                    max_hz=hz,
                    phases=phases,
                    inference_region=region,
                    dummy_fallback=True,
                    hack_param_constraint=True,
                    min_mb=3.5,
                    vert_only=True)

    if len(resume_from) > 0:
        with open(resume_from, 'rb') as f:
            sg = pickle.load(f)
    else:
        sg = rs.build_sg(ms1)
        ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=200)

    #k = sg.evnodes.keys()
    #for eid in k:
    #    if eid > 100:
    #        sg.remove_event(eid)

    #evtrue = Event(lon=-109.07, lat=41.68, depth=3.8, time=1239045221.9, mb=3.0)
    #from sigvisa.infer.event_mcmc import ev_phasejump_helper
    #ev_phasejump_helper(sg, eid=5, new_ev=evtrue, 
    #                    params_changed=["lon", "lat", "depth", "time", "mb"], 
    #                    adaptive_decouple=True,
    #                    birth_type="mh", fix_result=None)


    ms1.add_inference_round(enable_event_moves=True, enable_event_openworld=True, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc',], steps=1000)
    do_inference(sg, ms1, rs, dump_interval=10, print_interval=10, model_switch_lp_threshold=None)


if __name__ == "__main__":
    try:

        parser = OptionParser()
        parser.add_option("--seed", dest="seed", default=1, type=int)
        parser.add_option("--n_events", dest="n_events", default=2, type=int)
        parser.add_option("--resume_from", dest="resume_from", default="", type=str)
        (options, args) = parser.parse_args()
        main(seed=options.seed, n_events=options.n_events, resume_from=options.resume_from)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
