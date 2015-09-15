import numpy as np

from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, TimeRangeRunSpec, do_coarse_to_fine, initialize_from, do_inference, initialize_sg
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.treegp.gp import GPCov

import os, sys, traceback
import cPickle as pickle
from optparse import OptionParser


stas = "ASAR,FITZ,ILAR,MKAR,WRA,YKA,KURK,SONM,BVAR,CTA,CMAR,ZALV,AKTO,INK,AAK,AKBB,ARCES,CPUP,DZM,FINES,JKA,KBZ,KSRS,LPAZ,NOA,NVAR,PETK,PLCA,PMG,STKA,TORD,URZ,USRK,VNDA".split(",")
#stas = "ASAR,MKAR,FITZ".split(",")

#stas="MKAR,ASAR,WRA,PETK,FINES,FITZ,YKA,VNDA,JKA,HFS,MJAR".split(",")

def generate_leb_truth(hour=0.0, len_hours=2.0, runid=37, hz=2.0, enable_uatemplates=False, uatemplate_rate=4e-4, raw_signals=False, bands=["freq_0.8_4.5"]):

    rs = TimeRangeRunSpec(sites=stas, runids=(runid,), dataset="training",
                          hour=hour, len_hours=len_hours,
                          initialize_events="leb")

    ms1 = ModelSpec(template_model_type="param",
                    wiggle_family="iid",
                    uatemplate_rate=uatemplate_rate,
                    max_hz=hz,
                    raw_signals=raw_signals,
                    phases="leb",
                    bands=bands,
                    dummy_fallback=True,
                    vert_only=True)
    ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=enable_uatemplates, enable_template_moves=True, disable_moves=['atime_xc'], steps=500)

    sg = rs.build_sg(ms1)
    initialize_sg(sg, ms1, rs)
    do_inference(sg, ms1, rs, dump_interval=10, print_interval=10, model_switch_lp_threshold=None)




def main(hour=0.0, len_hours=2.0, runid=37, hz=2.0, tmpl_steps=500, ev_steps=1000, resume_from=None, deserialize=None, uatemplate_rate=4e-4, raw_signals=False, bands=["freq_0.8_4.5"]):

    

    # python infer/run_mcmc.py --dataset=training --hour=0 --len_hours=2.0 --sites=MKAR,ASAR,WRA,PETK,FINES,FITZ,YKA,VNDA,JKA,HFS,MJAR --runid=26 --phases=P,S --skip=10 --hz=2.0 --nm=ar --uatemplate_rate=1e-4 --steps=1000 --wiggle_family=iid --initialize_leb=yes --dummy_fallback

    rs = TimeRangeRunSpec(sites=stas, runids=(runid,), dataset="training", hour=hour, len_hours=len_hours)

    ms1 = ModelSpec(template_model_type="param",
                    wiggle_family="iid",
                    uatemplate_rate=uatemplate_rate,
                    max_hz=hz,
                    phases=["P",],
                    bands=bands,
                    #bands=["freq_1.0_2.0","freq_2.0_3.0"],
                    dummy_fallback=True,
                    raw_signals=raw_signals,
                    vert_only=True)

    if resume_from is not None:
        with open(resume_from, 'rb') as f:
            sg = pickle.load(f)
        sg.phases=["P",]
        sg.uatemplate_rate = uatemplate_rate
    else:
        sg = rs.build_sg(ms1)


        if deserialize is not None:
            sg.deserialize_from_tgz(deserialize)
        else:
            ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=tmpl_steps)

    ms1.add_inference_round(enable_event_moves=True, enable_event_openworld=True, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=ev_steps)

    do_inference(sg, ms1, rs, dump_interval=10, print_interval=10, model_switch_lp_threshold=None)


if __name__ == "__main__":

    parser = OptionParser()


    parser.add_option("--raw", dest="raw", default=False, action="store_true",
                      help="use raw signals instead of envelopes")
    parser.add_option("--hz", dest="hz", default=2.0, type=float,
                      help="downsample signals to this rate")
    parser.add_option("--bands", dest="bands", default="freq_0.8_4.5", type=str,
                      help="comma-separated frequency bands")
    parser.add_option("--leb", dest="leb", default=False, action="store_true",
                      help="fix events to LEB bulletin")
    parser.add_option("--leb_uatemplates", dest="leb_uatemplates", default=False, action="store_true",
                      help="enable uatemplate births in LEB inference")
    parser.add_option("--hour", dest="hour", default=0.0, type=float,
                      help="start time, relative to training dataset")
    parser.add_option("--len_hours", dest="len_hours", default=2.0, type=float,
                      help="length of signal to use")
    parser.add_option("--tmpl_steps", dest="tmpl_steps", default=500, type=int,
                      help="steps to take in tmpl-only inference")
    parser.add_option("--uatemplate_rate", dest="uatemplate_rate", default=4e-4, type=float,
                      help="")
    parser.add_option("--ev_steps", dest="ev_steps", default=1000, type=int,
                      help="steps to take in full inference ")
    parser.add_option("--resume_from", dest="resume_from", default=None, type=str,
                      help="saved sg state to initialize inference")
    parser.add_option("--deserialize", dest="deserialize", default=None, type=str,
                      help="serialized (tgz) sg state to initialize inference")
    parser.add_option("--runid", dest="runid", default=1, type=int,
                      help="runid for models to load")

    (options, args) = parser.parse_args()

    bands = options.bands.split(",")

    if options.leb:
        generate_leb_truth(hour=options.hour, len_hours=options.len_hours, runid=options.runid, enable_uatemplates=options.leb_uatemplates, uatemplate_rate=options.uatemplate_rate, raw_signals=options.raw, hz=options.hz, bands=bands)
    else:
        main(hour=options.hour, len_hours=options.len_hours, resume_from=options.resume_from, runid=options.runid, tmpl_steps=options.tmpl_steps, ev_steps=options.ev_steps, deserialize=options.deserialize, uatemplate_rate=options.uatemplate_rate, raw_signals=options.raw, hz=options.hz, bands=bands)
