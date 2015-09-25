import numpy as np

from sigvisa import Sigvisa
from sigvisa.source.event import get_event, Event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, do_coarse_to_fine, initialize_from, do_inference, initialize_sg
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.treegp.gp import GPCov
from sigvisa.utils.geog import dist_km
from sigvisa.utils.isc import load_isc
import os, sys, traceback
import cPickle as pickle

#stas = ['ASAR', 'KURK', 'MKAR', 'SONM', 'BVAR', 'FITZ', 'CTA', 'CMAR', 'WRA', 'ZALV', 'MJAR', 'AKTO', 'INK']
stas=["MKAR",]

# 5334939, 
evids = [5335822, 5334501, 5334991, 5334726, 5335144, 5349684, 5348178, 5334971, 5349536, 5335079, 5335116, 5335138, 5350499, 5336237, 5335425, 5335424, 5349441, 5336640, 5335577, 5350077, 5336889, 5335760, 5336967, 5337111, 5336015, 5337461, 5351821, 5351657, 5336724, 5351713, 5338302, 5338318, 5338388]



def profile():
    isc_evs = [load_isc(evid) for evid in evids]
    
    main_ev = isc_evs[0]
    isc_evs = [ev for ev in isc_evs if ev is not None and dist_km((main_ev.lon, main_ev.lat), (ev.lon, ev.lat)) < 15]

    rs = EventRunSpec(evs=isc_evs, stas=stas, runids=(1,), disable_conflict_checking=True)


    ms3 = ModelSpec(template_model_type="gp_joint", wiggle_family="db4_2.0_3_30.0", wiggle_model_type="dummy", raw_signals=True, max_hz=10.0, skip_levels=2)
    ms3.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True)

    #ms = [ms3,]
    sg = rs.build_sg(ms3)
    initialize_sg(sg, ms3, rs)
    do_inference(sg, ms3, rs, max_steps=1000, model_switch_lp_threshold=-1000,  dump_interval=5)

def sigvisa_locate_ctf():

    seed=5

    isc_evs = [load_isc(evid) for evid in evids]
    main_ev = isc_evs[0]
    #main_ev = get_event(evid=5335822)
    isc_evs = [ev for ev in isc_evs if ev is not None and dist_km((main_ev.lon, main_ev.lat), (ev.lon, ev.lat)) < 15]

    rs = EventRunSpec(evs=isc_evs, stas=stas, runids=(1,), disable_conflict_checking=True)

    ms1 = ModelSpec(template_model_type="param", wiggle_family="iid", max_hz=10.0, raw_signals=True, seed=seed)
    ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'])
    #ms1.add_inference_round(enable_event_moves=True, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'])

    ms2 = ModelSpec(template_model_type="gp_joint", wiggle_family="iid", wiggle_model_type="dummy", raw_signals=True, max_hz=10.0, seed=seed)
    ms2.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'])

    ms3 = ModelSpec(template_model_type="gp_joint", wiggle_family="db4_2.0_3_20.0", wiggle_model_type="dummy", raw_signals=True, max_hz=10.0, seed=seed)
    ms3.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True)


    ms4 = ModelSpec(template_model_type="gp_joint", wiggle_family="db4_2.0_3_20.0", wiggle_model_type="gp_joint", raw_signals=True, max_hz=10.0, seed=seed)
    ms4.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True)

    ms = [ms1, ms2, ms4]
    do_coarse_to_fine(ms, rs, max_steps_intermediate=20, dump_interval=5)


def continue_from(old_sgfile):
    with open(old_sgfile, 'rb') as f:
        sg_old = pickle.load(f)

    isc_evs = [load_isc(evid) for evid in evids]
    main_ev = isc_evs[0]
    #main_ev = get_event(evid=5335822)
    isc_evs = [ev for ev in isc_evs if ev is not None and dist_km((main_ev.lon, main_ev.lat), (ev.lon, ev.lat)) < 15]

    rs = EventRunSpec(evs=isc_evs, stas=stas, runids=(1,), disable_conflict_checking=True)
    #ms3 = ModelSpec(template_model_type="gp_joint", wiggle_family="db4_2.0_3_15.0", wiggle_model_type="dummy", raw_signals=True, max_hz=10.0)
    #ms3.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True)

    ms4 = ModelSpec(template_model_type="gp_joint", wiggle_family="db4_2.0_3_20.0", wiggle_model_type="gp_joint", raw_signals=True, max_hz=10.0, phases=["P", ], jointgp_param_run_init=1)
    ms4.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True)


    sg = rs.build_sg(ms4)
    initialize_from(sg, ms4, sg_old, None)


    for k, nodes in sg._jointgp_hparam_nodes.items():
        if not k.startswith("level_var") and "level" in k:
            nodes['horiz_lscale'].set_value(40.0)
            nodes['depth_lscale'].set_value(20.0)
            nodes['noise_var'].set_value(0.95)

    #print sg_old.current_log_p(verbose=True)
    print sg.current_log_p()
    print sg.current_log_p()

    #with open("/home/dmoore/python/sigvisa/logs/mcmc/01936/step_000000/pickle.sg", 'wb') as f:
    #    pickle.dump(sg, f, 2)



    #sg_old.current_log_p_breakdown()
    #sg.current_log_p_breakdown()

    do_inference(sg, ms4, rs, max_steps=1000, model_switch_lp_threshold=-1000, dump_interval=5)


def main():
    if sys.argv[1]=="ctf":
        sigvisa_locate_ctf()
    if sys.argv[1]=="continue":
        continue_from(sys.argv[2])
    if sys.argv[1]=="profile":
        profile()

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
