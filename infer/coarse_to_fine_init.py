import time
import numpy as np
import os

import cPickle as pickle
import hashlib

from sigvisa.database.dataset import read_timerange, read_events, EV_MB_COL, EV_EVID_COL
from sigvisa.signals.io import load_event_station_chan, load_segments, fetch_waveform, MissingWaveform
from sigvisa.infer.run_mcmc import run_open_world_MH
from sigvisa.infer.mcmc_logger import MCMCLogger
from sigvisa import Sigvisa
from sigvisa.source.event import get_event, Event
from sigvisa.models.ttime import tt_predict
from sigvisa.graph.sigvisa_graph import SigvisaGraph, MAX_TRAVEL_TIME

class RunSpec(object):
    def build_sg(self, modelspec):
        kwargs = modelspec.sg_params.copy()
        if modelspec.signal_params['raw_signals']:
            kwargs["raw_signals"] = True

        sg = SigvisaGraph(runids=self.runids, **kwargs)
        waves = self.get_waves(modelspec)
        for (wave, wave_env) in waves:
            sg.add_wave(wave, wave_env=wave_env)

        try:
            sg.seed = self.seed
        except AttributeError:
            sg.seed = 0
        return sg

class SyntheticRunSpec(RunSpec):

    def __init__(self, sw, runid, init_noise=False):
        self.sw = sw
        self.runids = (runid,)
        self.init_noise = init_noise

    def get_waves(self, modelspec):
        w = self.sw.waves
        waves = [(w[evid][sta], None) for evid in w.keys() for sta in w[evid].keys()]
        return waves

    def get_init_events(self):
        if self.init_noise:
            evs = []
            for ev in self.sw.all_evs:
                ev_init = Event(lon=ev.lon+1*np.random.randn(),
                                lat=ev.lat-1*np.random.randn(),
                                depth=ev.depth,
                                mb=ev.mb+0.4 *np.random.randn(),
                                time=ev.time+20.0*np.random.randn())
                evs.append(ev_init)
            return evs
        else:
            return self.sw.all_evs

class EventRunSpec(RunSpec):

    def __init__(self, sites=None, stas=None, evids=None, evs=None, runids=None, initialize_events=True,
                 pre_s=10, post_s=120, force_event_wn_matching=True, disable_conflict_checking=False):

        self.sites = sites
        self.stas = stas
        self.runids = runids
        self.evids = evids
        self.evs = evs
        self.pre_s = pre_s
        self.post_s = post_s
        self.initialize_events=initialize_events
        self.force_event_wn_matching = force_event_wn_matching
        self.disable_conflict_checking = disable_conflict_checking

    def get_waves(self, modelspec):
        s = Sigvisa()
        cursor = s.dbconn.cursor()

        stas = self.stas
        if stas is None:
            stas = s.sites_to_stas(self.sites, refsta_only=not modelspec.sg_params['arrays_joint'])


        k = hashlib.md5(repr(self.__dict__) + repr(modelspec.sg_params) + repr(modelspec.signal_params)).hexdigest()[:10]
        cache_fname = os.path.join(s.homedir, 'scratch/evwaves_%s.pkl' % k)
        try:
            with open(cache_fname, 'rb') as f:
                waves = pickle.load(f)
        except IOError:

            if self.evids is not None:
                waves = self._get_waves_from_evids(modelspec, stas, cursor)
            else:
                waves = self._get_waves_from_evs(modelspec, stas, cursor)
            cursor.close()
            with open(cache_fname, 'wb') as f:
                pickle.dump(waves, f)

        return waves

    def _get_waves_from_evs(self, modelspec, stas, cursor):
        waves = []
        for sta in stas:
            ptimes = np.array([tt_predict(ev, sta, "P") + ev.time for ev in self.evs])
            for i, ev in enumerate(self.evs):
                ptime = ptimes[i]
                stime = ptime - 10.0
                etime = ptime + 150.0
                try:
                    next_ptime = np.min(ptimes[ptimes > ptime])
                    etime = min(etime, next_ptime - 25)
                except ValueError:
                    pass

                try:
                    wave = fetch_waveform(sta, chan="auto", stime=stime, etime=etime, cursor=cursor)
                    bands = modelspec.signal_params['bands']
                    hz = modelspec.signal_params['max_hz']
                    assert(len(bands)==1)

                    wave_env = wave.filter("%s;env;hz_%.1f" % (bands[0], hz))
                    if modelspec.signal_params['raw_signals']:
                        wave = wave.filter("%s;hz_%.1f" % (bands[0], hz))
                        waves.append((wave, wave_env))
                    else:
                        waves.append((wave_env, None))
                except MissingWaveform as e:
                    print e
                    continue
        return waves

    def _get_waves_from_evids(self, modelspec, stas, cursor):
        waves = []
        for evid in self.evids:
            for sta in stas:
                try:
                    wave=load_event_station_chan(evid, sta, chan="auto", cursor=cursor,
                                                 pre_s=self.pre_s,
                                                 post_s=self.post_s,
                                                 exclude_other_evs=True,
                                                 phases=modelspec.sg_params['phases'])
                    bands = modelspec.signal_params['bands']
                    hz = modelspec.signal_params['max_hz']
                    assert(len(bands)==1)

                    wave_env = wave.filter("%s;env;hz_%.1f" % (bands[0], hz))
                    if modelspec.signal_params['raw_signals']:
                        wave = wave.filter("%s;hz_%.1f" % (bands[0], hz))
                        waves.append((wave, wave_env))
                    else:
                        waves.append((wave_env, None))
                except MissingWaveform as e:
                    print e
                    continue
        return waves

    def get_init_events(self):
        if self.initialize_events:
            if self.evids is not None:
                evs = [get_event(evid=evid) for evid in self.evids]
            else:
                evs = self.evs
        else:
            evs = []
        return evs

    def build_sg(self, modelspec):
        kwargs = modelspec.sg_params.copy()
        kwargs['force_event_wn_matching'] = self.force_event_wn_matching
        if modelspec.signal_params['raw_signals']:
            kwargs["raw_signals"] = True

        sg = SigvisaGraph(runids=self.runids, **kwargs)
        waves = self.get_waves(modelspec)
        for (wave, wave_env) in waves:
            sg.add_wave(wave, disable_conflict_checking=self.disable_conflict_checking, wave_env=wave_env)
        return sg


class TimeRangeRunSpec(RunSpec):
    def __init__(self, sites, dataset="training", hour=0.0, len_hours=2.0, start_time=None, end_time=None, runids=None, initialize_events=None):

        self.sites = sites
        self.runids = runids

        if start_time is not None:
            self.start_time = start_time
            self.end_time = end_time
        else:
            s = Sigvisa()
            cursor = s.dbconn.cursor()
            #print "loading signals from dataset %s" % options.dataset
            (stime, etime) = read_timerange(cursor, dataset, hours=None, skip=0)
            cursor.close()

            self.start_time = stime + hour * 3600
            self.end_time = self.start_time + len_hours*3600.0

        self.initialize_events=initialize_events

    def get_waves(self, modelspec):
        s = Sigvisa()
        cursor = s.dbconn.cursor()

        stas = s.sites_to_stas(self.sites, refsta_only=not modelspec.sg_params['arrays_joint'])

        etime = self.end_time + MAX_TRAVEL_TIME

        segments = load_segments(cursor, stas, self.start_time, etime, chans = modelspec.signal_params['chans'])

        waves = []
        for seg in segments:
            for band in modelspec.signal_params['bands']:
                filtered_seg = seg.with_filter(band)
                if modelspec.signal_params['smooth'] is not None:
                    filtered_seg = filtered_seg.with_filter("smooth_%d" % modelspec.signal_params['smooth'])
                filtered_seg = filtered_seg.with_filter("hz_%.3f" % modelspec.signal_params['max_hz'])


                filtered_seg_env = filtered_seg.with_filter("env")


                for chan in filtered_seg.get_chans():
                    wave_env = filtered_seg_env[chan]
                    if modelspec.signal_params['raw_signals']:
                        wave = filtered_seg[chan]
                        waves.append((wave, wave_env))
                    else:
                        waves.append((wave_env, None))

        return waves


    def get_init_events(self):
        if self.initialize_events is None:
            evs = []
        elif self.initialize_events == "leb":
            s = Sigvisa()
            cursor = s.dbconn.cursor()

            events, orid2num = read_events(cursor, self.start_time, self.end_time, 'leb')
            events = [evarr for evarr in events if evarr[EV_MB_COL] > 2]
            evs = []
            eid = 1
            for evarr in events:
                ev = get_event(evid=evarr[EV_EVID_COL])
                ev.eid = eid
                eid += 1
                evs.append(ev)

            cursor.close()
        elif isinstance(self.initialize_events, list) or \
             isinstance(self.initialize_events, tuple):
            evs = [get_event(evid=evid) for evid in self.initialize_events]
        else:
            raise Exception("unrecognized event initialization %s" % self.initialize_events)
        return evs

class ModelSpec(object):

    def __init__(self, ms_label=None, vert_only=True, inference_preset=None, seed=0, **kwargs):
        sg_params = {
            'template_shape': "lin_polyexp",
            'template_model_type': "param",
            'wiggle_family': "iid",
            'wiggle_model_type': "dummy",
            'skip_levels': 1,
            'dummy_fallback': False,
            'phases': ["P",],
            'arrays_joint': False,
            'uatemplate_rate': 1e-6,
            'min_mb': 2.5,
            'jointgp_hparam_prior': None,
            'jointgp_param_run_init': None,
            'hack_param_constraint': False,
            'inference_region': None,
        }
        signal_params = {
            'max_hz': 5.0,
            'raw_signals': False,
            'bands': ["freq_0.8_4.5"],
            'chans': None, # need to set this
            'smooth': None
        }

        if vert_only:
            signal_params['chans'] = ['auto'] # 'vertical'
        for k in kwargs.keys():
            if k in sg_params:
                sg_params[k] = kwargs[k]
            elif k in signal_params:
                signal_params[k] = kwargs[k]
            else:
                raise KeyError("unrecognized param %s" % k)

        self.sg_params = sg_params
        self.signal_params = signal_params
        self.ms_label = ms_label
        self.seed = seed

        self.inference_rounds = []
        if inference_preset == "closedtmpl":
            self.add_inference_round(enable_template_openworld=False,
                                     enable_event_openworld=False, 
                                     enable_event_moves=False,
                                     disable_moves=["atime_xc"])
        if inference_preset == "closedworld":
            self.add_inference_round(enable_template_openworld=False,
                                     enable_event_openworld=False, )
        elif inference_preset == "closedworld_noatime":
            self.add_inference_round(enable_template_openworld=False,
                                     enable_event_openworld=False, disable_moves=['atime_xc'])
        elif inference_preset == "openworld":
            self.add_inference_round(enable_template_openworld=True,
                                     enable_event_openworld=True)


    def add_inference_round(self, **kwargs):
        inference_params = {
            'enable_template_moves': True,
            'enable_template_openworld': True,
            'enable_event_moves': True,
            'enable_event_openworld': True,
            'enable_hparam_moves': True,
            'steps': -1,
        }
        for k in kwargs.keys():
            inference_params[k] = kwargs[k]
        self.inference_rounds.append(inference_params)


    def update_args(self, kwargs):
        pass



def initialize_sg(sg, modelspec, runspec):
    evs = runspec.get_init_events()
    for ev in evs:
        sg.add_event(ev)

def initialize_from(sg_new, ms_new, sg_old, ms_old):
    """
    We will eventually need special cases for certain types of
    differences. But this is a start.
    """

    # copy events from old SG to new
    eids_old = sg_old.evnodes.keys()
    for eid in eids_old:
        ev = sg_old.get_event(eid)
        sg_new.add_event(ev, eid=eid)

    # copy param values
    for k in sg_old.all_nodes.keys():
        try:
            n1 = sg_new.all_nodes[k]
            n2 = sg_old.all_nodes[k]
            n1.set_value(n2.get_value())
        except:
            continue

def do_inference(sg, modelspec, runspec, max_steps=None, model_switch_lp_threshold=500, dump_interval=50, print_interval=10):

    # save 'true' events if they are known
    # build logger
    # save modelspec to the inference directory
    # save sw to the inference directory
    # run inference with appropriate params
    # and hooks to monitor convergence?

    logger = MCMCLogger( write_template_vals=True, dump_interval=dump_interval, print_interval=print_interval, write_gp_hparams=True)
    logger.dump(sg)

    sg.seed = modelspec.seed

    try:
        sw = runspec.sw
    except:
        sw = None
    if sw is not None:
        with open(os.path.join(logger.run_dir, "events.pkl"), "wb") as f:
            pickle.dump(sw.all_evs, f)

        with open(os.path.join(logger.run_dir, "sw.pkl"), "wb") as f:
            pickle.dump(sw, f)

    np.random.seed(modelspec.seed)

    def lp_change_threshold(logger):
        if model_switch_lp_threshold is None: return False
        lps = logger.lps
        if len(lps) < 20:
            return False
        m1 = np.mean( lps[-10:])
        m2 = np.mean( lps[-20:])
        diff = m1-m2
        return diff < model_switch_lp_threshold

    step = 0


    for inference_params in modelspec.inference_rounds:

        if max_steps is not None:
            inference_params["steps"] = max_steps

        if "fix_outside_templates" in inference_params:
            if inference_params["fix_outside_templates"]:
                sg.fix_outside_region(fix_templates=True)
            del inference_params["fix_outside_templates"]

        run_open_world_MH(sg, logger=logger,
                          stop_condition=lp_change_threshold,
                          start_step=step,
                          **inference_params)
        step = logger.last_step


def do_coarse_to_fine(modelspecs, runspec,
                      model_switch_lp_threshold=1000,
                      max_steps_intermediate=100,
                      max_steps_final=5000, **kwargs):

    sg_old, ms_old = None, None

    sgs = [runspec.build_sg(modelspec) for modelspec in modelspecs]

    for (sg, modelspec) in zip(sgs[:-1], modelspecs[:-1]):
        if sg_old is None:
            initialize_sg(sg, modelspec, runspec)
        else:
            initialize_from(sg, modelspec, sg_old, ms_old)

        do_inference(sg, modelspec, runspec,
                     model_switch_lp_threshold=model_switch_lp_threshold,
                     max_steps = max_steps_intermediate, **kwargs)
        sg_old, ms_old = sg, modelspec

    sg, modelspec = sgs[-1], modelspecs[-1]
    initialize_from(sg, modelspec, sg_old, ms_old)
    do_inference(sg, modelspec, runspec,
                 model_switch_lp_threshold=None,
                 max_steps = max_steps_final, **kwargs)


def synth_location_seq():

    ms1 = ModelSpec(template_model_type="param", wiggle_family="iid")
    ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'])
    ms1.add_inference_round(enable_event_moves=True, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'])

    ms2 = ModelSpec(template_model_type="gp_joint", wiggle_family="iid", wiggle_model_type="dummy", inference_preset="closedworld")

    ms3 = ModelSpec(template_model_type="gp_joint", wiggle_family="db4uvars_2.0_3_10.0", wiggle_model_type="gp_joint", inference_preset="closedworld")

    return [ms1, ms2, ms3]
