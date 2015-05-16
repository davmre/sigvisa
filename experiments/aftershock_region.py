import numpy as np

from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, do_coarse_to_fine, initialize_from, do_inference
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.treegp.gp import GPCov

import os, sys, traceback
import cPickle as pickle

stas = ['ASAR', 'KURK', 'MKAR', 'SONM', 'BVAR', 'FITZ', 'CTA', 'CMAR', 'WRA', 'ZALV', 'MJAR', 'AKTO', 'INK']

evids = [5334501, 5334991, 5334726, 5335144, 5349684, 5335822, 5348178, 5334971, 5349536, 5335079, 5335116, 5335138, 5350499, 5336237, 5335425, 5335424, 5349441, 5336640, 5335577, 5350077, 5336889, 5335760, 5336967, 5337111, 5336015, 5337461, 5351821, 5351657, 5336724, 5351713, 5338302, 5338318, 5338388]


def sigvisa_locate_ctf():

    rs = EventRunSpec(evids=evids, stas=stas, runids=(37,), disable_conflict_checking=True)

    ms1 = ModelSpec(template_model_type="param", wiggle_family="iid", max_hz=5.0)
    ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'])
    ms1.add_inference_round(enable_event_moves=True, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'])

    ms2 = ModelSpec(template_model_type="gp_joint", wiggle_family="iid", wiggle_model_type="dummy", inference_preset="closedworld")

    ms3 = ModelSpec(template_model_type="gp_joint", wiggle_family="db4uvars_2.0_3_30.0", wiggle_model_type="dummy", inference_preset="closedworld")


    ms4 = ModelSpec(template_model_type="gp_joint", wiggle_family="db4uvars_2.0_3_30.0", wiggle_model_type="gp_joint", inference_preset="closedworld")

    ms = [ms1, ms2, ms3, ms4]
    do_coarse_to_fine(ms, rs, max_steps_intermediate=100)


def continue_from():
    with open("/home/dmoore/python/sigvisa/logs/mcmc/01935/step_000019/pickle.sg", 'rb') as f:
        sg_old = pickle.load(f)

    rs = EventRunSpec(evids=evids, stas=stas, runids=(37,), disable_conflict_checking=True)
    ms3 = ModelSpec(template_model_type="gp_joint", wiggle_family="db4uvars_2.0_3_30.0", wiggle_model_type="dummy", inference_preset="closedworld")

    sg = rs.build_sg(ms3)
    initialize_from(sg, ms3, sg_old, None)


    #print sg_old.current_log_p(verbose=True)
    print sg.current_log_p()
    print sg.current_log_p()

    with open("/home/dmoore/python/sigvisa/logs/mcmc/01936/step_000000/pickle.sg", 'wb') as f:
        pickle.dump(sg, f, 2)



    #sg_old.current_log_p_breakdown()
    #sg.current_log_p_breakdown()

    #do_inference(sg, ms3, rs, max_steps=1000, model_switch_lp_threshold=-1000)


def main():
    if sys.argv[1]=="ctf":
        sigvisa_locate_ctf()
    if sys.argv[1]=="continue":
        continue_from()

"""
def relative_atimes():

    waves = load_waves(filter_str="freq_0.8_4.5")

    rel_atimes = dict()
    for sta in stas:
        for evid1 in evids:
            w1 = waves[evid1]['sta']
            patch1 =
            for evid2 in evids:
                if evid2 <= evid1: continue

                w2 = waves[evid2]['sta']

def xcorr_valid(a,b):
    a = (a - np.mean(a)) / (np.std(a) * np.sqrt(len(a)))
    b = (b - np.mean(b)) / (np.std(b) * np.sqrt(len(a)))

    xc = np.correlate(a, b, 'valid')
    xcmax = np.max(xc)
    offset = np.argmax(xc)
    return xcmax, offset
"""

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
