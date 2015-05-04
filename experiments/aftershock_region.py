import numpy as np

from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.signals.io import fetch_waveform, load_event_station_chan, MissingWaveform
from sigvisa.infer.run_mcmc import run_open_world_MH
from sigvisa.infer.mcmc_logger import MCMCLogger
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.treegp.gp import GPCov

import os, sys, traceback
import cPickle as pickle

stas = ['ASAR', 'KURK', 'MKAR', 'SONM', 'BVAR', 'FITZ', 'CTA', 'CMAR', 'WRA', 'ZALV', 'MJAR', 'AKTO', 'INK']

evids = [5334501, 5334991, 5334726, 5335144, 5349684, 5335822, 5348178, 5334971, 5349536, 5335079, 5335116, 5335138, 5350499, 5336237, 5335425, 5335424, 5349441, 5336640, 5335577, 5350077, 5336889, 5335760, 5336967, 5337111, 5336015, 5337461, 5351821, 5351657, 5336724, 5351713, 5338302, 5338318, 5338388]

def load_evs():
    evs = dict()
    for evid in evids:
        evs[evid] = get_event(evid=evid)
    return evs

def load_waves(filter_str="freq_0.8_4.5;env;hz_5.0"):

    s = Sigvisa()

    cache_fname = os.path.join(s.homedir, 'scratch/aftershock_waves_%s.pkl' % filter_str)
    try:
        with open(cache_fname, 'rb') as f:
            waves = pickle.load(f)

    except IOError:


        cursor = s.dbconn.cursor()
        waves = dict()
        for evid in evids:
            waves[evid]=dict()
            for sta in stas:
                try:
                    waves[evid][sta]=load_event_station_chan(evid, sta, chan="auto", cursor=cursor, pre_s=10, post_s=120, exclude_other_evs=True, phases=['P']).filter(filter_str)
                except MissingWaveform as e:
                    print e
                    continue
        cursor.close()

        with open(cache_fname, 'wb') as f:
            pickle.dump(waves, f)


    return waves


def build_sg(evs, waves, **kwargs):

    sg = SigvisaGraph(template_shape="lin_polyexp",
                      phases=["P",], nm_type = "ar", **kwargs)

    for i in waves.keys():
        for sta in waves[i].keys():
            wn = sg.add_wave(waves[i][sta], disable_conflict_checking=True)

    for ev in evs.values():
        sg.add_event(ev)

    return sg

def sigvisa_locate_basic():

    evs = load_evs()
    waves = load_waves()

    tm_type_str="tt_residual:constant_laplacian,peak_offset:param_linear_mb,amp_transfer:param_sin1,coda_decay:param_linear_distmb,peak_decay:param_linear_distmb"
    tm_types = {}
    for p in tm_type_str.split(','):
        (param, model_type) = p.strip().split(':')
        tm_types[param] = model_type

    sg = build_sg(evs, waves, template_model_type=tm_types, wiggle_model_type="dummy", wiggle_family="iid", runids=(28,), force_event_wn_matching=True)

    logger = MCMCLogger(write_template_vals=False, dump_interval=10)

    with open(os.path.join(logger.run_dir, "events.pkl"), "wb") as f:
        pickle.dump(evs.values(), f)

    run_open_world_MH(sg, steps=1000,
                      enable_template_moves=True,
                      enable_event_moves=True,
                      logger=logger,
                      enable_event_openworld=False,
                      enable_template_openworld=False)

def sigvisa_locate_joint(jointtts=False, jointwiggles=False):

    evs = load_evs()
    waves = load_waves()

    tm_type_str="tt_residual:constant_laplacian,peak_offset:param_linear_mb,amp_transfer:param_sin1,coda_decay:param_linear_distmb,peak_decay:param_linear_distmb"
    tm_types = {}
    for p in tm_type_str.split(','):
        (param, model_type) = p.strip().split(':')
        tm_types[param] = model_type
    wiggle_model_type="dummy"
    wiggle_family="iid"

    gpc_wiggle = 0.1, GPCov(wfn_str="compact2", wfn_params=np.array((0.7,)), dfn_str="lld", dfn_params=np.array((80.0, 10.0)))
    gpc_tt = 0.3, GPCov(wfn_str="se", wfn_params=np.array((6.0,)), dfn_str="lld", dfn_params=np.array((100.0, 10.0)))
    jgpp = {}
    if jointtts:
        tm_types['tt_residual'] = "gp_joint"
        jgpp['tt_residual'] = gpc_tt
    if jointwiggles:
        jgpp['wiggle'] = gpc_wiggle
        wiggle_model_type="gp_joint"
        wiggle_family="db4uvars_2.0_3_10_0.4"

    sg = build_sg(evs, waves, template_model_type=tm_types, wiggle_model_type=wiggle_model_type, wiggle_family=wiggle_family, force_event_wn_matching=True,  runids=(28,))

    logger = MCMCLogger(write_template_vals=True, dump_interval=10, write_gp_hparams=True)

    with open(os.path.join(logger.run_dir, "events.pkl"), "wb") as f:
        pickle.dump(evs.values(), f)

    run_open_world_MH(sg, steps=1000,
                      enable_template_moves=True,
                      enable_event_moves=True,
                      logger=logger,
                      enable_event_openworld=False,
                      enable_template_openworld=False)





def main():
    if sys.argv[1]=="basic":
        sigvisa_locate_basic()
    if sys.argv[1]=="krigedtts":
        sigvisa_locate_joint(jointtts=True)
    if sys.argv[1]=="wavematch":
        sigvisa_locate_joint(jointtts=False, jointwiggles=True)
    if sys.argv[1]=="everything":
        sigvisa_locate_joint(jointtts=True, jointwiggles=True)

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
