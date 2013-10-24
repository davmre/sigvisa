import numpy as np
import sys
import os
import traceback
import pickle
import copy

from sigvisa import Sigvisa

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.infer.run_mcmc import run_open_world_MH
import sigvisa.infer.run_mcmc
from sigvisa.infer.template_mcmc import birth_move, death_move, split_move, merge_move
from sigvisa.plotting.plot import plot_with_fit
from sigvisa.signals.common import Waveform
from sigvisa.utils.fileutils import clear_directory, mkdir_p

BASE_DIR = os.path.join(os.getenv("SIGVISA_HOME"), "tests", "mcmc", "one_station_templates")

def sample_template(seed=None, wiggle_family="dummy"):
    mkdir_p(BASE_DIR)

    sg = SigvisaGraph(template_model_type="dummy", template_shape="paired_exp",
                      wiggle_model_type="dummy", wiggle_family=wiggle_family,
                      phases="leb", nm_type = "ar", wiggle_len_s = 60.0)

    wave = Waveform(data = np.zeros(500), srate=5.0, stime=1239915900.0, sta="FIA3", chan="SHZ", filter_str="freq_2.0_3.0;env;hz_5.0")
    wn = sg.add_wave(wave)
    sg.uatemplate_rate = .01

    if seed is not None:
        np.random.seed(seed)

    tg = sg.template_generator('UA')
    tg.hack_force_mean = np.log(wn.nm.c * 5)
    templates = sg.prior_sample_uatemplates(wn, wiggles=True)

    print "sampled", len(templates), 'templates'
    wave = wn.get_wave()

    #sg.debug_dump(dump_path=os.path.join(BASE_DIR, 'sampled_seed%d' % seed))
    f = open(os.path.join(BASE_DIR, 'sampled_seed%d' % seed), 'wb')
    pickle.dump(wave, f)
    f.close()
    #np.savetxt(os.path.join(BASE_DIR, 'sampled_wave_seed%d' % seed))

    return wave, templates


def main(seed=3):

    wiggles=False
    if wiggles:
        wiggle_family = "fourier_0.8"
    else:
        wiggle_family = "dummy"


    print os.path.join(BASE_DIR, 'sampled_seed%d' % seed)
    try:
        f = open(os.path.join(BASE_DIR, 'sampled_seed%d' % seed), 'rb')
    except IOError:
        sample_template(seed=seed, wiggle_family = wiggle_family)
        f = open(os.path.join(BASE_DIR, 'sampled_seed%d' % seed), 'rb')
    wave = pickle.load(f)
    f.close()

    sg = SigvisaGraph(template_model_type="dummy", template_shape="paired_exp",
                      wiggle_model_type="dummy", wiggle_family=wiggle_family,
                      phases="leb", nm_type = "ar", wiggle_len_s = 60.0)
    wn = sg.add_wave(wave)
    sg.uatemplate_rate=.01


    tg = sg.template_generator('UA')
    tg.hack_force_mean = np.log(wn.nm.c * 5)

    run_seed = 2
    np.random.seed(run_seed)
    sigvisa.infer.run_mcmc.global_stds['arrival_time'] = 1.0
    run_open_world_MH(sg, skip=100, steps=500,
                      run_dir = os.path.join(BASE_DIR, 'mcmcrun_seed%d_run%d' % (seed, run_seed)),
                      enable_template_openworld=True,
                      enable_template_moves=True,
                      enable_event_moves=False,
                      enable_event_openworld=False,
                      dumpsteps=True)

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
