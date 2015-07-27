import numpy as np

from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, SyntheticRunSpec, do_coarse_to_fine, initialize_from, do_inference
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.treegp.gp import GPCov
from sigvisa.synthetic.doublets import SampledWorld

import os, sys, traceback
import cPickle as pickle

#stas = ['ASAR', 'KURK', 'MKAR', 'SONM', 'BVAR', 'FITZ', 'CTA', 'CMAR', 'WRA', 'ZALV', 'MJAR', 'AKTO', 'INK']

#evids = [5334501, 5334991, 5334726, 5335144, 5349684, 5335822, 5348178, 5334971, 5349536, 5335079, 5335116, 5335138, 5350499, 5336237, 5335425, 5335424, 5349441, 5336640, 5335577, 5350077, 5336889, 5335760, 5336967, 5337111, 5336015, 5337461, 5351821, 5351657, 5336724, 5351713, 5338302, 5338318, 5338388]

stas="ASAR,FITZ,ILAR,MKAR,WRA,YKA,KURK,SONM,BVAR,CTA,CMAR,ZALV,AKTO,INK,AAK,AKBB,ARCES,CPUP,DZM,FINES,JKA,KBZ,KSRS,LPAZ,NOA,NVAR,PETK,PLCA,PMG,STKA,TORD,URZ,USRK,VNDA".split(",")



# GOAL: emulate the command
# python experiments/^Cent_localization.py --runid=37 --seed=201 --skip=2 --steps=5000 --sites=ASAR,FITZ,ILAR,MKAR,WRA,YKA,KURK,SONM,BVAR,CTA,CMAR,ZALV,AKTO,INK,AAK,AKBB,ARCES,CPUP,DZM,FINES,JKA,KBZ,KSRS,LPAZ,NOA,NVAR,PETK,PLCA,PMG,STKA,TORD,URZ,USRK,VNDA --uatemplate_rate=1e-10 --n_events=3 --nm_type=ar --template_move_type=rw --template_openworld --openworld  --run_label=spinup --wiggle_family=iid


def main():

    n_events = 8
    uatemplate_rate=1e-6
    seed=202
    hz = 2.0
    runid=37
    phases=["P", "S"]

    sw = SampledWorld(seed=seed)
    sw.sample_sg(runid=37, wiggle_model_type="dummy", wiggle_family="iid", sites=stas, phases=phases, tmtype="param", uatemplate_rate=uatemplate_rate, sample_uatemplates=False, n_events=n_events, min_mb=4.0, force_mb=None, len_s=2000, tt_buffer_s=500, hz=hz, dumpsg=False, dummy_fallback=True)

    rs = SyntheticRunSpec(sw=sw, runid=runid)

    ms1 = ModelSpec(template_model_type="param",
                    wiggle_family="iid",
                    uatemplate_rate=uatemplate_rate,
                    max_hz=hz,
                    phases=phases,
                    dummy_fallback=True,
                    vert_only=True)

    if len(sys.argv) > 1:
        with open(sys.argv[1], 'rb') as f:
            sg = pickle.load(f)
    else:
        sg = rs.build_sg(ms1)
        ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=500)


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
