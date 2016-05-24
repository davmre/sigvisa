import numpy as np

import os
import hashlib

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.utils.fileutils import clear_directory, mkdir_p
from sigvisa.utils.geog import dist_km

import cPickle as pickle

BASE_DIR = os.path.join(os.getenv("SIGVISA_HOME"), "synthetic", "worlds")

def md5hash(x, n=8):
    return hashlib.md5(repr(x)).hexdigest()[:n]

def wave_dirname(base_dir=None, **kwargs):
    base_dir = BASE_DIR if base_dir is None else base_dir
    return os.path.join(BASE_DIR, md5hash("sampled_" + '_'.join([':'.join((str(k),str(v))) for (k,v) in kwargs.items() if v ])))

def sample_event(runids, seed, wiggle_model_type, wiggle_family, sites, phases, tmtype, uatemplate_rate, sample_uatemplates, n_events, min_mb, force_mb, stime=1238889600.0, len_s=1000, tt_buffer_s=500, tmshape="lin_polyexp", band="freq_0.8_4.5", hz=5.0, dummy_fallback=False, return_all=False, wave_dir=None, dumpsg=True, evs=None, region=None, raw_signals=False):

    if wave_dir is None:
        wave_dir = wave_dirname(seed=seed, runids=runids, wiggle_model_type=wiggle_model_type, wiggle_family=wiggle_family, sites=sites, phases=phases, tmtype=md5hash(tmtype), uatemplate_rate=uatemplate_rate, sample_uatemplates=sample_uatemplates, n_events=n_events, min_mb=min_mb, force_mb=force_mb, nm_type="ar", raw_signals=raw_signals)
    mkdir_p(wave_dir)

    sg = SigvisaGraph(template_model_type=tmtype, template_shape=tmshape,
                      wiggle_model_type=wiggle_model_type, wiggle_family=wiggle_family,
                      phases=phases, runids=runids, dummy_fallback=dummy_fallback, 
                      inference_region=region, min_mb=min_mb, raw_signals=raw_signals)

    s = Sigvisa()

    wns = dict()
    for site in sites:
        try:
            sta = s.get_default_sta(site)
            chan = s.canonical_channel_name[s.default_vertical_channel[sta]]
            wave = Waveform(data = np.zeros((int(len_s*hz),)), srate=hz, stime=stime, sta=sta, chan=chan, filter_str="%s;%shz_%.1f" % (band, "env;" if not raw_signals else "",  hz))
            wns[sta] = sg.add_wave(wave)
        except Exception as e:
            print "wave sampling exception:", e
            continue

    if seed is not None:
        np.random.seed(seed)
        s.sigmodel.srand(seed)

    if evs is None:
        evs = sg.prior_sample_events(stime=stime, etime=stime+len_s-tt_buffer_s, n_events=n_events,  force_mb=force_mb)

    else:
        for ev in evs:
            sg.add_event(ev, sample_templates=True)
    print "sampled", len(evs), "evs"

    sg.uatemplate_rate = uatemplate_rate
    if sample_uatemplates:
        for wn in wns.values():
            sg.prior_sample_uatemplates(wn)

    #sg.debug_dump(dump_path=os.path.join(BASE_DIR, 'sampled_seed%d' % seed))

    #
    
    waves = []
    for (sta, wn) in wns.items():

        nm_node = wn.nm_node
        nm_node.parent_sample()

        wn.unfix_value()
        wn.parent_sample()

        f = open(os.path.join(wave_dir, "wave_%s" % sta), 'wb')
        wave = wn.get_wave()
        waves.append(wave)
        pickle.dump(wave, f)
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

    if dumpsg:
        sg.debug_dump(dump_path=os.path.join(wave_dir, 'init_dump'))


    if return_all:
        return evs, waves, sg
    else:
        return evs
