import numpy as np
import sys
import os
import traceback
import pickle
import copy

from sigvisa import Sigvisa

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.infer.run_mcmc import run_open_world_MH
from sigvisa.infer.mcmc_logger import MCMCLogger
from sigvisa.infer.template_mcmc import birth_move, death_move #, split_move, merge_move
from sigvisa.plotting.plot import plot_with_fit
from sigvisa.signals.common import Waveform
from sigvisa.utils.fileutils import clear_directory, mkdir_p

BASE_DIR = os.path.join(os.getenv("SIGVISA_HOME"), "tests", "mcmc", "one_station_templates")

def sample_template(seed=None, wiggle_family="dummy", nm_type="ar", srate=1.0, sta="FIA3", chan="SHZ", hardcoded=False, len_s=1000, n_templates=None, return_graph=False, rate=1e-3):
    mkdir_p(BASE_DIR)

    sg = SigvisaGraph(template_model_type="dummy", template_shape="lin_polyexp",
                      wiggle_model_type="dummy", wiggle_family=wiggle_family,
                      phases="leb", nm_type = nm_type, wiggle_len_s = 60.0)

    wave = Waveform(data = np.zeros(int(1000*srate)), srate=srate, stime=1239915900.0, sta=sta, chan=chan, filter_str="freq_2.0_3.0;env;hz_%.1f" % srate)
    wn = sg.add_wave(wave)
    sg.uatemplate_rate = rate

    if seed is not None:
        np.random.seed(seed)

    tg = sg.template_generator('UA')
    tg.hack_force_mean = np.log(wn.nm.c * 10)

    if hardcoded:
        templates = add_hardcoded_templates(sg, wn)
    else:
        templates = sg.prior_sample_uatemplates(wn, n_templates=n_templates, wiggles=True)

    #print "sampled", len(templates), 'templates'
    wave = wn.get_wave()

    #sg.debug_dump(dump_path=os.path.join(BASE_DIR, 'sampled_seed%d' % seed))
    #f = open(os.path.join(BASE_DIR, 'sampled_seed%d' % seed), 'wb')
    #pickle.dump(wave, f)
    #f.close()
    #np.savetxt(os.path.join(BASE_DIR, 'sampled_wave_seed%d' % seed))

    if return_graph:
        return wave, templates, sg, wn
    else:
        return wave, templates


def add_hardcoded_templates(sg, wn):
    st = wn.st
    templates = []

    atime = st + 100
    v1 = {'arrival_time': atime,
          'peak_offset': 1.0,
          'peak_decay': -4.0,
          'coda_decay': -4.0,
          'coda_height': 3.0,}

    tnodes = sg.create_unassociated_template(wn, atime, initial_vals=v1)
    templates.append(tnodes)

    sg._topo_sort()
    wn.unfix_value()
    wn.parent_sample()
    wn.fix_value()

    return templates


def main(seed=3):

    wiggles=False
    if wiggles:
        wiggle_family = "fourier_0.8"
    else:
        wiggle_family = "dummy"


    nm_type="l1"

    #print os.path.join(BASE_DIR, 'sampled_seed%d' % seed)
    #try:
    #    f = open(os.path.join(BASE_DIR, 'sampled_seed%d' % seed), 'rb')
    #except IOError:
    print "sampling templates"
    wave, templates = sample_template(seed=seed, wiggle_family = wiggle_family, nm_type=nm_type, hardcoded=True)
    print "done"
    #f = open(os.path.join(BASE_DIR, 'sampled_seed%d' % seed), 'rb')
    #wave = pickle.load(f)
    #f.close()



    sg = SigvisaGraph(template_model_type="dummy", template_shape="lin_polyexp",
                      wiggle_model_type="dummy", wiggle_family=wiggle_family,
                      phases="leb", nm_type = nm_type, wiggle_len_s = 60.0)
    wn = sg.add_wave(wave)
    sg.uatemplate_rate=1e-4


    tg = sg.template_generator('UA')
    tg.hack_force_mean = np.log(wn.nm.c * 5)

    run_seed = 2
    np.random.seed(run_seed)
    logger = MCMCLogger(run_dir=None, write_template_vals=True, dump_interval=50)
    run_open_world_MH(sg, steps=5000,
                      enable_template_openworld=True,
                      enable_template_moves=True,
                      enable_event_moves=False,
                      logger=logger,
                      enable_event_openworld=False)

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
