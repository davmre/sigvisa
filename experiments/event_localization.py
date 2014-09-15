import numpy as np
import sys
import os
import traceback
import pickle
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

BASE_DIR = os.path.join(os.getenv("SIGVISA_HOME"), "experiments", "event_localization")

def hash(x, n=8):
    return hashlib.md5(repr(x)).hexdigest()[:n]

def sample_event(wave_dir, runid, seed, wiggles, sites, phases, tmtype, uatemplate_rate, sample_uatemplates, n_events, min_mb, force_mb, stime=1238889600.0, len_s=1000, tmshape="lin_polyexp"):
    mkdir_p(wave_dir)

    if wiggles:
        wiggle_family = "fourier_0.8"
    else:
        wiggle_family = "dummy"
    sg = SigvisaGraph(template_model_type=tmtype, template_shape=tmshape,
                      wiggle_model_type="dummy", wiggle_family=wiggle_family,
                      nm_type = "ar", phases=phases, runid=runid)

    s = Sigvisa()


    wns = dict()
    for site in sites:
        try:
            sta = s.get_default_sta(site)
            chan = s.canonical_channel_name[s.default_vertical_channel[sta]]
            wave = Waveform(data = np.zeros(2000), srate=1.0, stime=stime, sta=sta, chan=chan, filter_str="freq_2.0_3.0;env;smooth_15;hz_1.0")
            wns[sta] = sg.add_wave(wave)
        except Exception as e:
            print e
            continue

    if seed is not None:
        np.random.seed(seed)
        s.sigmodel.srand(seed)


    evs = sg.prior_sample_events(stime=stime, etime=stime+len_s, n_events=n_events, min_mb=min_mb, force_mb=force_mb)
    print "sampled", len(evs), "evs"

    sg.uatemplate_rate = uatemplate_rate
    if sample_uatemplates:
        for wn in wns.values():
            sg.prior_sample_uatemplates(wn)

    #sg.debug_dump(dump_path=os.path.join(BASE_DIR, 'sampled_seed%d' % seed))

    for (sta, wn) in wns.items():
        wn.unfix_value()
        wn.parent_sample()

        f = open(os.path.join(wave_dir, "wave_%s" % sta), 'wb')
        pickle.dump(wn.get_wave(), f)
        f.close()
    #np.savetxt(os.path.join(BASE_DIR, 'sampled_wave_seed%d' % seed))

    f = open(os.path.join(wave_dir, "events.txt"), 'w')
    for (eid, d) in sg.evnodes.items():
        lon = d['loc'].get_local_value('lon')
        lat = d['loc'].get_local_value('lat')
        depth = d['loc'].get_local_value('depth')
        mb = d['mb'].get_local_value('mb')
        time = d['time'].get_local_value('time')
        source = d['natural_source'].get_local_value('natural_source')
        f.write("%d\t%f\t%f\t%f\t%f\t%f\t%d\n" % (eid, lon, lat, depth, time, mb, source))
    f.close()

    with open(os.path.join(wave_dir, "events.pkl"), 'wb') as f:
        pickle.dump(evs, f)

    sg.debug_dump(dump_path=os.path.join(wave_dir, 'init_dump'))

    return evs


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

def wave_dirname(**kwargs):
    return os.path.join(BASE_DIR, hash("sampled_" + '_'.join([':'.join((str(k),str(v))) for (k,v) in kwargs.items() if v ])))

def setup_graph(seed, perturb_amt, tmtype, runid, phases, init_events, max_distance, uatemplate_rate, sample_uatemplates, sample_single, n_events, min_mb, force_mb, wiggles=False, sites=None, tmshape="lin_polyexp"):

    sg = SigvisaGraph(template_model_type=tmtype, template_shape=tmshape,
                      wiggle_model_type="dummy", wiggle_family='dummy',
                      nm_type = "ar", phases=phases, runid=runid)

    sg.uatemplate_rate = uatemplate_rate

    wave_dir = wave_dirname(seed=seed, runid=runid, wiggles=wiggles, sites=sites, phases=phases, tmtype=hash(tmtype), uatemplate_rate=uatemplate_rate, sample_uatemplates=sample_uatemplates, n_events=n_events, min_mb=min_mb, force_mb=force_mb)
    try:
        evs = load_graph(sg, wave_dir, max_distance=max_distance)
    except (IOError, OSError) as e:
        sample_event(wave_dir, runid, seed, wiggles, sites, phases, tmtype, uatemplate_rate, sample_uatemplates, n_events, min_mb, force_mb)
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
    parser.add_option("--sample_single", dest="sample_single", default=False, action="store_true")
    parser.add_option("--perturb_amt", dest="perturb_amt", default=0.0, type="float")
    parser.add_option("--openworld", dest="openworld", default=False, action="store_true")
    parser.add_option("--init_openworld", dest="init_openworld", default=False, action="store_true")
    parser.add_option("--sample_uatemplates", dest="sample_uatemplates", default=False, action="store_true", help="include unassociated templates in the synthetic world (default is to only sample templates from events)")
    parser.add_option("--uatemplate_rate", dest="uatemplate_rate", default=0.001, type="float")
    parser.add_option("--n_events", dest="n_events", default=None, type="int", help="force the specified number of events (default is to sample from the prior)")
    parser.add_option("--min_mb", dest="min_mb", default=4.5, type="float", help="sample event magnitudes from an exponential(10) distribution with the given origin. (4.5)")
    parser.add_option("--force_mb", dest="force_mb", default=None, type="float", help="force event magnitude to the given value.")
    parser.add_option("--phases", dest="phases", default='P', type="str")
    parser.add_option("--sites", dest="sites", default=None, type="str")
    parser.add_option("--steps", dest="steps", default=20000, type="int",
                      help="MCMC steps to take (20000)")

    (options, args) = parser.parse_args()

    tmtype = {'tt_residual': 'constant_laplacian',
              'peak_offset': 'param_linear_mb',
              'amp_transfer': 'param_sin1',
              'coda_decay': 'param_linear_distmb',
              'peak_decay': 'param_linear_distmb',}

    phases = options.phases.split(',')
    sites = options.sites.split(',')

    init_events = options.init_openworld or (not options.openworld)
    sg, evs, wave_dir = setup_graph(options.seed, options.perturb_amt, tmtype, options.runid, options.phases, init_events, max_distance = options.max_distance, uatemplate_rate=options.uatemplate_rate, sample_uatemplates=options.sample_uatemplates, sample_single=options.sample_single, n_events=options.n_events, min_mb=options.min_mb, force_mb=options.force_mb, sites=sites)

    print "got %d evs" % len(evs)
    for ev in evs:
        print ev
    print "now running inference"

    hash_options = hash(str(options))
    openworld_str = '_open' if options.openworld else ''
    #run_dir = os.path.join(BASE_DIR,
    #                       'mcmcrun_seed%d_%s' % (options.seed, hash_options))
    #mkdir_p(run_dir)
    #with open(os.path.join(run_dir, 'gold'), 'w') as f:
    #    f.write(wave_dir + '\n')

    logger = MCMCLogger(run_dir=None, write_template_vals=True, dump_interval=50)
    run_dir = logger.run_dir
    with open(os.path.join(run_dir, 'cmd.txt'), 'w') as f:
        f.write(" ".join(sys.argv))


    mkdir_p(run_dir)
    os.symlink(os.path.join(wave_dir, 'events.pkl'), os.path.join(run_dir, 'events.pkl'))
    np.random.seed(1)
    run_open_world_MH(sg, steps=options.steps,
                      logger=logger,
                      enable_template_openworld=options.openworld,
                      enable_template_moves=True,
                      enable_event_moves=True,
                      enable_event_openworld=options.openworld)

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
