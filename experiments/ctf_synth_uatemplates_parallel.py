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
stas = ["PDAR",]


def main(seed=1,  resume_from="", server_only=False, client_only=None, ctf_only=False):


    uatemplate_rate=1e-1
    hz = 20.0
    runid=3
    phases=["P",]

    sw = SampledWorld(seed=seed)
    sw.sample_sg(runid=3, wiggle_model_type="dummy", wiggle_family="iid", sites=stas, phases=phases, tmtype="param", uatemplate_rate=uatemplate_rate, sample_uatemplates=True, n_events=0, min_mb=3.5, force_mb=None, len_s=400, tt_buffer_s=1000, hz=hz, dumpsg=False, dummy_fallback=True, stime=1239040000, evs=None, raw_signals=True)

    rs = SyntheticRunSpec(sw=sw, runid=runid, raw_signals=True)


    ms_coarse = ModelSpec(template_model_type="param",
                          wiggle_family="iid",
                          uatemplate_rate=uatemplate_rate,
                          max_hz=1.0,
                          phases=phases,
                          dummy_fallback=True,
                          hack_param_constraint=True,
                          min_mb=3.5,
                          raw_signals=False,
                          vert_only=True)

    """
    ms_medium = ModelSpec(template_model_type="param",
                          wiggle_family="iid",
                          uatemplate_rate=uatemplate_rate,
                          max_hz=5.0,
                          phases=phases,
                          dummy_fallback=True,
                          hack_param_constraint=True,
                          min_mb=3.5,
                          vert_only=True)
    """

    ms_fine = ModelSpec(template_model_type="param",
                        wiggle_family="iid",
                        uatemplate_rate=uatemplate_rate,
                        max_hz=hz,
                        phases=phases,
                        dummy_fallback=True,
                        hack_param_constraint=True,
                        min_mb=3.5,
                        vert_only=True,
                        raw_signals=True)


    ms_coarse.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=100000)
    #ms_medium.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=not ctf_only, enable_template_moves=True, disable_moves=['atime_xc'], steps=100000)
    ms_fine.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=not ctf_only, enable_template_moves=True, disable_moves=['atime_xc'], steps=40000)


    names = ("coarse", "fine")
    specs = {"coarse": (ms_coarse, rs),
             #"medium": (ms_medium, rs),
             "fine": (ms_fine, rs)}

    #names = ("fine1", "fine2", "fine3")
    #specs = {"fine1": (ms_fine, rs),
    #         "fine2": (ms_fine, rs),
    #         "fine3": (ms_fine, rs)}

    #sg = rs.build_sg(ms_fine)
    #do_inference(sg, ms_fine, rs, dump_interval=10, print_interval=10, model_switch_lp_threshold=None)

    
    run_parallel_coarse_to_fine(names, specs, 
                                server_only=server_only, 
                                client_only=client_only,
                                min_swap_s = 1.0, 
                                max_swap_s = 20.0)


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
