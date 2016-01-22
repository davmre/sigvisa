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
stas = "NEW,PDAR,NVAR,ANMO,TXAR,PFO,YKA,ULM,ILAR".split(",")

region_lon = (-126, -100)
region_lat = (32, 49)
region_stime = 1239040000
region_etime = region_stime + 7200

def main(seed=1, n_events=2, resume_from="", server_only=False, client_only=None):

    evs = None
    if n_events==-1:
        ev = Event(lon=-105.427, lat=43.731, depth=0.0, time=1239041017.07, mb=4.0, natural_source=False)
        evs = [ev,]

    uatemplate_rate=1e-4
    hz = 20.0
    runid=3
    phases=["P", "S", "Lg", "PcP", "ScP", "pP", "Pg"]

    region = Region(lons=region_lon, lats=region_lat, times=(region_stime, region_etime))

    sw = SampledWorld(seed=seed)
    sw.sample_sg(runid=3, wiggle_model_type="dummy", wiggle_family="iid", sites=stas, phases=phases, tmtype="param", uatemplate_rate=uatemplate_rate, sample_uatemplates=True, n_events=n_events, min_mb=3.5, force_mb=None, len_s=region_etime-region_stime, tt_buffer_s=1000, hz=hz, dumpsg=False, dummy_fallback=True, stime=region_stime, evs=evs, region=region)

    rs = SyntheticRunSpec(sw=sw, runid=runid)


    ms_coarse = ModelSpec(template_model_type="param",
                          wiggle_family="iid",
                          uatemplate_rate=uatemplate_rate,
                          max_hz=1.0,
                          phases=phases,
                          inference_region=region,
                          dummy_fallback=True,
                          hack_param_constraint=True,
                          min_mb=3.5,
                          vert_only=True)

    ms_fine = ModelSpec(template_model_type="param",
                        wiggle_family="iid",
                        uatemplate_rate=uatemplate_rate,
                        max_hz=hz,
                        phases=phases,
                        inference_region=region,
                        dummy_fallback=True,
                        hack_param_constraint=True,
                        min_mb=3.5,
                        vert_only=True)


    ms_coarse.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=1000)
    ms_fine.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=200)


    names = ("coarse", "fine")
    specs = {"coarse": (ms_coarse, rs),
             "fine": (ms_fine, rs)}

    #sg = rs.build_sg(ms_fine)
    #do_inference(sg, ms_fine, rs, dump_interval=10, print_interval=10, model_switch_lp_threshold=None)


    run_parallel_coarse_to_fine(names, specs, server_only=server_only, client_only=client_only)


if __name__ == "__main__":
    try:

        parser = OptionParser()
        parser.add_option("--seed", dest="seed", default=1, type=int)
        parser.add_option("--n_events", dest="n_events", default=2, type=int)
        parser.add_option("--resume_from", dest="resume_from", default="", type=str)
        parser.add_option("--server_only", dest="server_only", default=False, action="store_true")
        parser.add_option("--client_only", dest="client_only", default=None, type="str")

        (options, args) = parser.parse_args()
        main(seed=options.seed, n_events=options.n_events, 
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
