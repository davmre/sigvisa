import numpy as np
import sys
import os
import traceback
import pickle
import copy

from sigvisa import Sigvisa

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.infer.run_mcmc import run_open_world_MH
from sigvisa.infer.template_mcmc import birth_move, death_move, split_move, merge_move
from sigvisa.plotting.plot import plot_with_fit
from sigvisa.signals.common import Waveform
from sigvisa.utils.fileutils import clear_directory, mkdir_p

BASE_DIR = os.path.join(os.getenv("SIGVISA_HOME"), "tests", "mcmc", "event_localization")

def sample_event(runid, seed=None, wiggles = False, sites=10, phases='auto', tmtype='dummy'):
    clear_directory(BASE_DIR)
    mkdir_p(os.path.join(BASE_DIR, 'sampled_seed%d' % seed))

    if wiggles:
        wiggle_family = "fourier_0.8"
    else:
        wiggle_family = "dummy"
    sg = SigvisaGraph(template_model_type=tmtype, template_shape="paired_exp",
                      wiggle_model_type="dummy", wiggle_family=wiggle_family,
                      nm_type = "ar", phases=phases, runid=runid)

    if seed is not None:
        np.random.seed(seed)

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

    evs = sg.prior_sample_events(min_mb=4.5, stime=1239915400.0, etime=1239916400.0)

    #sg.debug_dump(dump_path=os.path.join(BASE_DIR, 'sampled_seed%d' % seed))

    for (sta, wn) in wns.items():
        wn.unfix_value()
        wn.parent_sample()

        f = open(os.path.join(BASE_DIR, 'sampled_seed%d' % seed, "wave_%s" % sta), 'wb')
        pickle.dump(wn.get_wave(), f)
        f.close()
    #np.savetxt(os.path.join(BASE_DIR, 'sampled_wave_seed%d' % seed))

    f = open(os.path.join(BASE_DIR, 'sampled_seed%d' % seed, "events.txt"), 'w')
    for (eid, d) in sg.evnodes.items():
        lon = d['loc'].get_local_value('lon')
        lat = d['loc'].get_local_value('lat')
        depth = d['loc'].get_local_value('depth')
        mb = d['mb'].get_local_value('mb')
        time = d['time'].get_local_value('time')
        source = d['natural_source'].get_local_value('natural_source')
        f.write("%d\t%f\t%f\t%f\t%f\t%f\t%d\n" % (eid, lon, lat, depth, time, mb, source))
    f.close()

    with open(os.path.join(BASE_DIR, 'sampled_seed%d' % seed, "events.pkl"), 'wb') as f:
        pickle.dump(evs, f)

    sg.debug_dump(dump_path=os.path.join(BASE_DIR, 'sampled_seed%d' % seed, 'init_dump'))

    return evs

def main(seed=1):

    runid = 19
    tmtype = {'tt_residual': 'constant_gaussian',
              'peak_offset': 'constant_gaussian',
              'amp_transfer': 'param_dist5',
              'coda_decay': 'param_dist5',}
    #phases = ['Pn','Pg', 'Sn', 'Lg']
    phases=['P',]

    def load_graph(sg, wave_dir):
        sta_waves = os.listdir(wave_dir)
        for sw in sta_waves:
            if not sw.startswith('wave_'): continue
            with open(os.path.join(wave_dir, sw), 'rb') as f:
                wave = pickle.load(f)
                sg.add_wave(wave)
        with open(os.path.join(wave_dir, "events.pkl"), 'rb') as f:
            evs = pickle.load(f)
        return evs

    #tmtype = "gplocal+lld+dist0"
    sg = SigvisaGraph(template_model_type=tmtype, template_shape="paired_exp",
                      wiggle_model_type="dummy", wiggle_family='dummy',
                      nm_type = "ar", phases=phases, runid=runid)

    wave_dir = os.path.join(BASE_DIR, 'sampled_seed%d' % seed)
    try:
        evs = load_graph(sg, wave_dir)
    except (IOError, OSError) as e:
        sample_event(runid=runid, seed=seed, tmtype=tmtype, phases=phases)
        evs = load_graph(sg, wave_dir)

    print "got %d evs" % len(evs)
    for ev in evs:
        print ev
    print "now running inference"


    for ev in evs:
        sg.add_event(ev)

    np.random.seed(1)
    run_open_world_MH(sg, skip=100, steps=5000,
                      run_dir = os.path.join(BASE_DIR, 'mcmcrun'),
                      enable_template_openworld=False,
                      enable_template_moves=True,
                      enable_event_moves=True,
                      enable_event_openworld=False,
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
