import numpy as np

from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, TimeRangeRunSpec, do_coarse_to_fine, initialize_from, do_inference, initialize_sg
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.treegp.gp import GPCov
from sigvisa.graph.region import Region

import os, sys, traceback
import cPickle as pickle


#stas = "ANMO,ELK,ILAR,KDAK,NEW,NVAR,PDAR,PFO,TXAR,ULM,YBH,YKA".split(",")
stas = "PDAR".split(",")

region_lon = (-126, -100)
region_lat = (32, 49)


def main():

    evid = 877774
    wiggle_family="db4_2.0_3_20.0"
    uatemplate_rate = 4e-4
    phases = ["P", "Lg"]


    rs = EventRunSpec(evids=[evid,], 
                      stas=stas, 
                      runids=(8,6), 
                      pre_s = 50,
                      post_s = 300,
                      force_event_wn_matching=False,
                      initialize_events=False)

    region_stime = 1174078000
    region_etime = region_stime + 3600
    region = Region(lons=region_lon, lats=region_lat, times=(region_stime, region_etime))
    
    ms1 = ModelSpec(template_model_type="param",
                    wiggle_family="iid",
                    uatemplate_rate=uatemplate_rate,
                    max_hz=2.0,
                    phases=phases,
                    inference_region=region,
                    dummy_fallback=True,
                    vert_only=True)
    ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'], steps=50)
    #ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=50)
    #ms1.add_inference_round(enable_event_moves=True, enable_event_openworld=True, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=10)

    ms2 = ModelSpec(template_model_type="param",
                    wiggle_family=wiggle_family,
                    wiggle_model_type="gp_lld",
                    uatemplate_rate=uatemplate_rate,
                    max_hz=10.0,
                    phases=phases,
                    inference_region=region,
                    dummy_fallback=True,
                    raw_signals=True,
                    vert_only=True)
    #ms2.add_inference_round(enable_event_moves=False, enable_event_openworld=False, 
    #                        enable_template_openworld=True, enable_template_moves=True, steps=20)
    ms2.add_inference_round(enable_event_moves=True, enable_event_openworld=True, 
                            enable_template_openworld=True, enable_template_moves=True, steps=500)

    ms = [ms1, ms2]
    do_coarse_to_fine(ms, rs, max_steps_intermediate=50, model_switch_lp_threshold=10)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
