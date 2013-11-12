import numpy as np
import sys
import os
import traceback
import pickle
import copy

from sigvisa import Sigvisa
from optparse import OptionParser

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.infer.run_mcmc import run_open_world_MH
from sigvisa.infer.template_mcmc import birth_move, death_move, split_move, merge_move
from sigvisa.plotting.plot import plot_with_fit
from sigvisa.signals.common import Waveform
from sigvisa.utils.fileutils import clear_directory, mkdir_p
from sigvisa.utils.geog import dist_km

BASE_DIR = os.path.join(os.getenv("SIGVISA_HOME"), "tests", "mcmc", "event_localization")

def sample_event(wave_dir, runid, seed=None, wiggles = False, sites=10, phases='auto', tmtype='dummy', uatemplate_rate=None, sample_uatemplates=False, sample_single=False):
    mkdir_p(wave_dir)

    if wiggles:
        wiggle_family = "fourier_0.8"
    else:
        wiggle_family = "dummy"
    sg = SigvisaGraph(template_model_type=tmtype, template_shape="paired_exp",
                      wiggle_model_type="dummy", wiggle_family=wiggle_family,
                      nm_type = "ar", phases=phases, runid=runid)

    s = Sigvisa()

    cursor = s.dbconn.cursor()
    #site_perm = np.random.permutation(len(s.siteid_minus1_to_name))
    #sites = np.array(s.siteid_minus1_to_name)[site_perm[:sites]]
    #sites = ['ILAR','WRA','FINES','YKA','JNU','ASAR','NVAR','AKASG','STKA']
    f = open('trained_stations.txt')
    sites = f.readline().split(',')
    f.close()




    wns = dict()
    for site in sites:
        try:
            sta = s.get_default_sta(site)
            chan = s.canonical_channel_name[s.default_vertical_channel[sta]]
            wave = Waveform(data = np.zeros(10000), srate=5.0, stime=1239915900.0, sta=sta, chan=chan, filter_str="freq_2.0_3.0;env;hz_5.0")
            wns[sta] = sg.add_wave(wave)
        except Exception as e:
            print e
            continue

    if seed is not None:
        np.random.seed(seed)
        s.sigmodel.srand(seed)

    if sample_single:
        ev = sg.prior_sample_event(min_mb=4.5, stime=1239915400.0, etime=1239916400.0)
        sg.add_event(ev)
        evs = [ev,]
    else:
        evs = sg.prior_sample_events(min_mb=4.5, stime=1239915400.0, etime=1239916400.0)

        print "sampled", len(evs), "evs"

    if sample_uatemplates:
        sg.uatemplate_rate = uatemplate_rate
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

def setup_graph(seed, perturb_amt, tmtype, runid, phases, init_events, max_distance, uatemplate_rate, sample_uatemplates, sample_single):
    sg = SigvisaGraph(template_model_type=tmtype, template_shape="paired_exp",
                      wiggle_model_type="dummy", wiggle_family='dummy',
                      nm_type = "ar", phases=phases, runid=runid)

    sg.uatemplate_rate = uatemplate_rate

    wave_dir = os.path.join(BASE_DIR, 'sampled_seed%d' % seed)
    if sample_uatemplates:
        wave_dir += "_rate%.4f" % uatemplate_rate
    if sample_single:
        wave_dir += "_single"
    try:
        evs = load_graph(sg, wave_dir, max_distance=max_distance)
    except (IOError, OSError) as e:
        sample_event(wave_dir=wave_dir, runid=runid, seed=seed, tmtype=tmtype, phases=phases, uatemplate_rate=uatemplate_rate, sample_uatemplates=sample_uatemplates, sample_single=sample_single)
        evs = load_graph(sg, wave_dir, max_distance=max_distance)

    if init_events:
        for ev in evs:
            perturb_ev(ev, perturb_amt)
            evnodes = sg.add_event(ev)

    return sg, evs

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
    parser.add_option("--sample_uatemplates", dest="sample_uatemplates", default=False, action="store_true")
    parser.add_option("--uatemplate_rate", dest="uatemplate_rate", default=0.001, type="float")
    parser.add_option("--phases", dest="phases", default='P', type="str")
    parser.add_option("--steps", dest="steps", default=20000, type="int",
                      help="MCMC steps to take (20000)")

    (options, args) = parser.parse_args()

    tmtype = {'tt_residual': 'constant_gaussian',
              'peak_offset': 'constant_gaussian',
              'amp_transfer': 'param_dist5',
              'coda_decay': 'param_dist5',}

    phases = options.phases.split(',')

    init_events = options.init_openworld or (not options.openworld)
    sg, evs = setup_graph(options.seed, options.perturb_amt, tmtype, options.runid, options.phases, init_events, max_distance = options.max_distance, uatemplate_rate=options.uatemplate_rate, sample_uatemplates=options.sample_uatemplates, sample_single=options.sample_single)

    print "got %d evs" % len(evs)
    for ev in evs:
        print ev
    print "now running inference"

    openworld_str = '_open' if options.openworld else ''
    run_dir = os.path.join(BASE_DIR,
                           'mcmcrun_seed%d_dist%.1f_perturb%.2f%s_%srate%.4f%s' % (options.seed, options.max_distance, options.perturb_amt, openworld_str, "sample" if options.sample_uatemplates else "", options.uatemplate_rate, 'single' if options.sample_single else 'multiple'))
    np.random.seed(1)
    run_open_world_MH(sg, skip=5000, steps=options.steps,
                      run_dir = run_dir,
                      enable_template_openworld=options.openworld,
                      enable_template_moves=True,
                      enable_event_moves=False,
                      enable_event_openworld=False,
#                      enable_event_openworld=options.openworld,
                      dumpsteps=False)

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
