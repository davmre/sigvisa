import numpy as np

from sigvisa import Sigvisa
from sigvisa.source.event import get_event, Event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, SyntheticRunSpec, do_coarse_to_fine, initialize_from, do_inference, initialize_sg
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.treegp.gp import GPCov
from sigvisa.synthetic.doublets import SampledWorld, build_joint_sg
from sigvisa.graph.region import Region

import os, sys, traceback
import cPickle as pickle
from optparse import OptionParser


#stas = "NEW,PDAR,NVAR,ANMO,TXAR,PFO,YKA,ULM,ILAR".split(",")
#stas = "ANMO,ELK,ILAR,KDAK,NEW,NVAR,PDAR,PFO,TXAR,ULM,YBH,YKA".split(",")
#stas=["TXAR",]

region_lon = (-126, -100)
region_lat = (32, 49)
region_stime = 1239040000.0
region_etime = region_stime + 3600.0

def main(stas, seed=1, n_events=5, resume_from="", use_hough=False, init_true=False, hack_event_rate=True, min_mb=2.5):


    uatemplate_rate=1e-4
    hz = 10.0
    runids=(24,18)
    phases=["P", "Pn", "Pg", "Sn", "Lg"]

    tmtype = "param" # "gpparam"

    # correlation proposal caching assumes that data from the same
    # time windows are constant, which is not true as we vary
    # synthetic random seeds, so this is a hack to avoid cache
    # problems. 
    offset = seed*100 + n_events

    region = Region(lons=region_lon, lats=region_lat, 
                    times=(region_stime, region_etime),
                    rate_bulletin="isc", 
                    rate_train_start=1167609600 + offset,
                    rate_train_end=1199145600 + offset)

    sw = SampledWorld(seed=seed)
    sw.sample_sg(runids=runids, wiggle_model_type="gplocal+lld+none", wiggle_family="db4_2.0_3_20.0", sites=stas, phases=phases, uatemplate_rate=uatemplate_rate, sample_uatemplates=True, n_events=n_events, min_mb=min_mb, force_mb=None, len_s=region_etime-region_stime, tt_buffer_s=1000, hz=hz, dumpsg=False, dummy_fallback=True, stime=region_stime, evs=None, region=region, raw_signals=True, tmtype=tmtype)

    rs = SyntheticRunSpec(sw=sw)

    ms1 = ModelSpec(template_model_type=tmtype,
                    wiggle_model_type="gplocal+lld+none",
                    wiggle_family="db4_2.0_3_20.0",
                    uatemplate_rate=uatemplate_rate,
                    max_hz=hz,
                    phases=phases,
                    runids=runids,
                    inference_region=region,
                    dummy_fallback=True,
                    hack_param_constraint=True,
                    min_mb=min(min_mb, 3.0),
                    raw_signals=True,
                    vert_only=True)

    if len(resume_from) > 0:
        with open(resume_from, 'rb') as f:
            sg = pickle.load(f)
    elif init_true:
        sg = rs.build_sg(ms1)
        initialize_sg(sg, ms1, rs)
        ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, enable_phase_openworld=False, disable_moves=['atime_xc'], steps=100)
        ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=200)
    else:
        sg = rs.build_sg(ms1)
        ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=30)

    if hack_event_rate:
        sg.event_rate = n_events / float(region_etime - region_stime)

    #sg.correlation_proposal_stas=stas

    ms1.add_inference_round(enable_event_moves=True, enable_event_openworld=True, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc',], steps=1000, propose_correlation=True, propose_hough = use_hough)


    do_inference(sg, ms1, rs, dump_interval_s=10, print_interval_s=10, model_switch_lp_threshold=None)


if __name__ == "__main__":
    try:

        parser = OptionParser()
        parser.add_option("--seed", dest="seed", default=1, type=int)
        parser.add_option("--min_mb", dest="min_mb", default=2.5, type=float)
        parser.add_option("--n_events", dest="n_events", default=2, type=int)
        parser.add_option("--resume_from", dest="resume_from", default="", type=str)
        parser.add_option("--use_hough", dest="use_hough", default=False, action="store_true")
        parser.add_option("--init_true", dest="init_true", default=False, action="store_true")
        parser.add_option("--stas", dest="stas", default=None, type=str)


        (options, args) = parser.parse_args()

        if options.stas is None:
            stas = "ANMO,ELK,ILAR,KDAK,NEW,NVAR,PDAR,PFO,TXAR,ULM,YBH,YKA".split(",")
        else:
            stas = options.stas.split(",")

        main(stas=stas, seed=options.seed, n_events=options.n_events, 
             resume_from=options.resume_from, use_hough=options.use_hough,
             init_true=options.init_true, min_mb=options.min_mb)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
