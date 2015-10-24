import numpy as np

from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, TimeRangeRunSpec, do_coarse_to_fine, initialize_from, do_inference, initialize_sg
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.treegp.gp import GPCov

import os, sys, traceback
import cPickle as pickle



# RES,
stas="YKA,USRK,NOA,KSRS,MKAR,JNU,WRA,FINES,FRB,NVAR".split(",")
stas="AKASG,ASAR,FINES,ILAR,JHJ,KSRS,MJAR,NOA,NVAR,PDAR,SONM,TXAR,USRK,WRA,YKA,ZALV,JNU,LPAZ,MKAR,VRAC,CMAR,FITZ".split(",")
stime = 1160357728
etime = stime + 2500
runid=26 # 37
hz=2.0
uatemplate_rate=1e-6
phases = ["P", "S"] #["P",]

def generate_leb_truth():

    rs = TimeRangeRunSpec(sites=stas, runids=(runid,), dataset="training",
                          start_time=stime-10,end_time=etime,
                          initialize_events="leb")

    ms1 = ModelSpec(template_model_type="param",
                    wiggle_family="iid",
                    uatemplate_rate=uatemplate_rate,
                    max_hz=hz,
                    phases="leb",
                    dummy_fallback=True,
                    vert_only=True,
                    bands=["freq_2.0_3.0",])
    ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'], steps=500)

    sg = rs.build_sg(ms1)
    initialize_sg(sg, ms1, rs)
    do_inference(sg, ms1, rs, dump_interval=10, print_interval=10, model_switch_lp_threshold=None)


def main():


    # python infer/run_mcmc.py --dataset=training --hour=0 --len_hours=2.0 --sites=MKAR,ASAR,WRA,PETK,FINES,FITZ,YKA,VNDA,JKA,HFS,MJAR --runid=26 --phases=P,S --skip=10 --hz=2.0 --nm=ar --uatemplate_rate=1e-4 --steps=1000 --wiggle_family=iid --initialize_leb=yes --dummy_fallback


    rs = TimeRangeRunSpec(sites=stas, runids=(runid,), start_time=stime, end_time=etime,)

    ms1 = ModelSpec(template_model_type="param",
                    wiggle_family="iid",
                    uatemplate_rate=uatemplate_rate,
                    max_hz=hz,
                    phases=phases,
                    dummy_fallback=True,
                    bands=["freq_2.0_3.0",],
                    seed=1,
                    vert_only=True)

    if len(sys.argv) > 1:
        with open(sys.argv[1], 'rb') as f:
            sg = pickle.load(f)
        sg.uatemplate_rate = uatemplate_rate
    else:
        sg = rs.build_sg(ms1)
        ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=500)

    ms1.add_inference_round(enable_event_moves=True, enable_event_openworld=True, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=1000)

    do_inference(sg, ms1, rs, dump_interval=10, print_interval=10, model_switch_lp_threshold=None)



if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1]=="leb":
            generate_leb_truth()
        else:
            main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
