import numpy as np
import sys
import os
import traceback
import cPickle as pickle
import copy
import hashlib

from sigvisa import Sigvisa
from optparse import OptionParser

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.infer.run_mcmc import run_open_world_MH
from sigvisa.infer.mcmc_logger import MCMCLogger
from sigvisa.infer.template_mcmc import birth_move, death_move, split_move, merge_move
from sigvisa.plotting.plot import plot_with_fit
from sigvisa.signals.common import Waveform
from sigvisa.utils.fileutils import clear_directory, mkdir_p
from sigvisa.utils.geog import dist_km
from sigvisa.synthetic import sample_event, wave_dirname

BASE_DIR = os.path.join(os.getenv("SIGVISA_HOME"), "experiments", "event_localization")



def perturb_ev(ev, perturb_amt=0.0):
    ev.lon += 5.0 * perturb_amt * np.random.randn()
    ev.lon = (ev.lon + 180) % 360 - 180
    ev.lat += 5.0 * perturb_amt * np.random.randn()
    ev.lat = min(90.0, ev.lat)
    ev.lat = max(-90.0, ev.lat)
    ev.depth += 100.0 * perturb_amt * np.random.randn()
    ev.depth = max(0.0, ev.depth)
    ev.depth = min(ev.depth, 700.0)
    ev.time += 20.0 * perturb_amt * np.random.randn()
    ev.mb += perturb_amt * np.random.randn()

def load_graph(sg, wave_dir, max_distance=None):

    with open(os.path.join(wave_dir, "events.pkl"), 'rb') as f:
        evs = pickle.load(f)

    sta_waves = os.listdir(wave_dir)
    i = 0
    j = 0
    s = Sigvisa()
    for sw in sta_waves:
        if not sw.startswith('wave_'): continue
        j += 1
        with open(os.path.join(wave_dir, sw), 'rb') as f:
            wave = pickle.load(f)

            # optionally skip sites that are not close to any of the events we sampled
            if max_distance is not None:
                sta = wave['sta']
                try:
                    slon, slat = s.earthmodel.site_info(sta, 0)[0:2]
                except KeyError:
                    continue
                best_dist = 99999
                for ev in evs:
                    best_dist = min(best_dist, dist_km((slon, slat), (ev.lon, ev.lat)))
                if best_dist > max_distance:
                    continue

            sg.add_wave(wave)
            i += 1
    print "loaded %d of %d stations" % (i, j)
    return evs


def setup_graph(seed, perturb_amt, tmtype, runid, phases, init_events, max_distance, uatemplate_rate, sample_uatemplates, n_events, min_mb, force_mb, sites=None, tmshape="lin_polyexp", nm_type="ar", wiggle_model_type="dummy", wiggle_family="dummy"):

    np.random.seed(seed)
    sg = SigvisaGraph(template_model_type=tmtype, template_shape=tmshape,
                      wiggle_model_type=wiggle_model_type, wiggle_family=wiggle_family,
                      nm_type = nm_type, phases=phases, runids=(runid,))

    sg.uatemplate_rate = uatemplate_rate

    wave_dir = wave_dirname(seed=seed, runid=runid, wiggle_model_type=wiggle_model_type, wiggle_family=wiggle_family, sites=sites, phases=phases, tmtype=md5hash(tmtype), uatemplate_rate=uatemplate_rate, sample_uatemplates=sample_uatemplates, n_events=n_events, min_mb=min_mb, force_mb=force_mb, nm_type=nm_type, base_dir=BASE_DIR)
    try:
        evs = load_graph(sg, wave_dir, max_distance=max_distance)
    except (IOError, OSError) as e:
        sample_event(runid, seed, wiggle_model_type, wiggle_family, sites, phases, tmtype, uatemplate_rate, sample_uatemplates, n_events, min_mb, force_mb, nm_type=nm_type, wave_dir=wave_dir)
        evs = load_graph(sg, wave_dir, max_distance=max_distance)

    if init_events:
        for ev in evs:
            perturb_ev(ev, perturb_amt)
            evnodes = sg.add_event(ev)

    return sg, evs, wave_dir

def main():

    parser = OptionParser()

    parser.add_option("--seed", dest="seed", default=4, type="int",
                      help="random seed for sampling world (1000)")
    parser.add_option("--runid", dest="runid", default=19, type="int")
    parser.add_option("--max_distance", dest="max_distance", default=None, type="float")
    parser.add_option("--perturb_amt", dest="perturb_amt", default=0.0, type="float")
    parser.add_option("--openworld", dest="openworld", default=False, action="store_true")
    parser.add_option("--template_openworld", dest="template_openworld", default=False, action="store_true")
    parser.add_option("--init_openworld", dest="init_openworld", default=False, action="store_true")
    parser.add_option("--sample_uatemplates", dest="sample_uatemplates", default=False, action="store_true", help="include unassociated templates in the synthetic world (default is to only sample templates from events)")
    parser.add_option("--uatemplate_rate", dest="uatemplate_rate", default=0.001, type="float")
    parser.add_option("--n_events", dest="n_events", default=None, type="int", help="force the specified number of events (default is to sample from the prior)")
    parser.add_option("--min_mb", dest="min_mb", default=4.5, type="float", help="sample event magnitudes from an exponential(10) distribution with the given origin. (4.5)")
    parser.add_option("--force_mb", dest="force_mb", default=None, type="float", help="force event magnitude to the given value.")
    parser.add_option("--load_saved_state", dest="load_saved_state", default=None, type="str", help="initialize inference at the state stored in a pickled SigvisaGraph file")
    parser.add_option("--phases", dest="phases", default='P', type="str")
    parser.add_option("--sites", dest="sites", default=None, type="str")
    parser.add_option("--steps", dest="steps", default=20000, type="int",
                      help="MCMC steps to take (20000)")
    parser.add_option("--wiggle_model_type", dest="wiggle_model_type", default="dummy", type="str",
                      help="model to use for wiggle coef priors (dummy)")
    parser.add_option("--wiggle_family", dest="wiggle_family", default="dummy", type="str",
                      help="wiggle parameterization to use (dummy)")
    parser.add_option("--nm_type", dest="nm_type", default="ar", type="str",
                      help="type of noise model to use (ar)")
    parser.add_option("--template_move_type", dest="template_move_type", default="hamiltonian", type="str", help="options are 'hamiltonian' (default), 'rw', or 'both'")
    parser.add_option("--run_label", dest="run_label", default="", type="str", help="any label to describe additional properties of this run")
    parser.add_option("--skip", dest="skip", default=50, type="int",
                      help="how often to print/save MCMC state, in steps (50)")


    (options, args) = parser.parse_args()

    tmtype = {'tt_residual': 'constant_laplacian',
              'peak_offset': 'param_linear_mb',
              'amp_transfer': 'param_sin1',
              'coda_decay': 'param_linear_distmb',
              'peak_decay': 'param_linear_distmb',
              'mult_wiggle_std': "constant_beta"}

    phases = options.phases.split(',')
    sites = options.sites.split(',')

    logger = MCMCLogger(run_dir=None, write_template_vals=True, dump_interval=options.skip)
    run_dir = logger.run_dir
    mkdir_p(run_dir)

    if options.load_saved_state is not None:
        import cPickle as pickle
        with open(options.load_saved_state, 'rb') as f:
            sg = pickle.load(f)
        if options.perturb_amt > 0:
            for eid in sg.evnodes.keys():
                ev = sg.get_event(eid)
                pev = copy.copy(ev)
                perturb_ev(pev, options.perturb_amt)
                print "perturbed", ev, "\nto", pev
                sg.set_event(eid, pev)
        eventpkl = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(options.load_saved_state)), "events.pkl")))
        print "wave dir", eventpkl
        os.symlink(eventpkl, os.path.join(run_dir, 'events.pkl'))

        sg.uatemplate_rate = options.uatemplate_rate

    else:
        init_events = options.init_openworld or (not options.openworld)
        sg, evs, wave_dir = setup_graph(options.seed, options.perturb_amt, tmtype, options.runid, options.phases, init_events, max_distance = options.max_distance, uatemplate_rate=options.uatemplate_rate, sample_uatemplates=options.sample_uatemplates, n_events=options.n_events, min_mb=options.min_mb, force_mb=options.force_mb, sites=sites, nm_type=options.nm_type, wiggle_model_type=options.wiggle_model_type, wiggle_family=options.wiggle_family)

        print "got %d evs" % len(evs)
        for ev in evs:
            print ev

        os.symlink(os.path.join(wave_dir, 'events.pkl'), os.path.join(run_dir, 'events.pkl'))

    print "now running inference"

    np.random.seed(1)
    run_open_world_MH(sg, steps=options.steps,
                      logger=logger,
                      enable_template_openworld=options.template_openworld,
                      enable_template_moves=True,
                      enable_event_moves=True,
                      enable_event_openworld=options.openworld,
                      template_move_type=options.template_move_type)

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
