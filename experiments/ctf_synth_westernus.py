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

# excludign ELK cause the noise model lookup fails for some reason
stas = "NEW,PDAR,NVAR,ANMO,TXAR,PFO,YKA,ULM,ILAR".split(",")

region_lon = (-126, -100)
region_lat = (32, 49)
region_stime = 1239040000
region_etime = region_stime + 7200

def main():

    n_events = 1
    uatemplate_rate=1e-4
    seed=202
    hz = 2.0
    runid=3
    phases=["P", "S", "Lg", "PcP", "ScP", "pP", "Pg"]

    ev = Event(lon=-105.427, lat=43.731, depth=0.0, time=1239041017.07, mb=4.0, natural_source=False)

    sw = SampledWorld(seed=seed)
    sw.sample_sg(runid=3, wiggle_model_type="dummy", wiggle_family="iid", sites=stas, phases=phases, tmtype="param", uatemplate_rate=uatemplate_rate, sample_uatemplates=True, n_events=n_events, min_mb=3.0, force_mb=3.7, len_s=region_etime-region_stime, tt_buffer_s=1000, hz=hz, dumpsg=False, dummy_fallback=False, evs = [ev,], stime=region_stime)

    rs = SyntheticRunSpec(sw=sw, runid=runid)

    region = Region(lons=region_lon, lats=region_lat, times=(region_stime, region_etime))

    ms1 = ModelSpec(template_model_type="param",
                    wiggle_family="iid",
                    uatemplate_rate=uatemplate_rate,
                    max_hz=hz,
                    phases=phases,
                    inference_region=region,
                    dummy_fallback=True,
                    hack_param_constraint=True,
                    vert_only=True)

    if len(sys.argv) > 1:
        with open(sys.argv[1], 'rb') as f:
            sg = pickle.load(f)
    else:
        sg = rs.build_sg(ms1)
        ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=200)


    ms1.add_inference_round(enable_event_moves=True, enable_event_openworld=True, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc',], steps=1000)
    do_inference(sg, ms1, rs, dump_interval=10, print_interval=10, model_switch_lp_threshold=None)


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
