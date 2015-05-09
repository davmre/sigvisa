import time
import numpy as np
import os

import cPickle as pickle

from sigvisa.database.dataset import read_timerange, read_events, EV_MB_COL, EV_EVID_COL
from sigvisa.signals.io import load_event_station_chan, load_segments, fetch_waveform
from sigvisa.infer.run_mcmc import run_open_world_MH
from sigvisa.infer.mcmc_logger import MCMCLogger
from sigvisa import Sigvisa
from sigvisa.source.event import get_event

from sigvisa.graph.sigvisa_graph import SigvisaGraph

class SyntheticRunSpec(object):

    def __init__(self, sw, runid):
        self.sw = sw
        self.runid = runid

    def get_waves(self, modelspec):
        w = self.sw.waves
        waves = [w[evid][sta] for evid in w.keys() for sta in w[evid].keys()]
        return waves

    def get_init_events(self):
        return self.sw.all_evs

class RunSpec(object):
    def __init__(sites, dataset="training", hour=0.0, len_hours=2.0, start_time=None, end_time=None, runid=None, initialize_events=None):

        self.sites = sites
        self.runid = runid

        if start_time is not None:
            self.start_time = start_time
            self.end_time = end_time
        else:
            #print "loading signals from dataset %s" % options.dataset
            (stime, etime) = read_timerange(cursor, dataset, hours=None, skip=0)
            self.start_time = stime + hour * 3600
            self.end_time = self.start_time + len_hours*3600.0

        self.initialize_events=initialize_events

    def get_waves(modelspec):
        s = Sigvisa()
        cursor = s.dbconn.cursor()

        stas = s.sites_to_stas(sites, refsta_only=not modelspec.sg_params['arrays_joint'])

        segments = load_segments(cursor, stas, self.start_time, self.end_time, chans = modelspac.signal_params['chans'])

        waves = []
        for seg in segments:
            for band in modelspec.signal_params['bands']:
                filtered_seg = seg.with_filter(band)
                if modelspec.signal_params['smooth'] is not None:
                    filtered_seg = filtered_seg.with_filter("smooth_%d" % modelspec.signal_params['smooth'])
                filtered_seg = filtered_seg.with_filter("hz_%.3f" % modelspec.signal_params['max_hz'])

                for chan in filtered_seg.get_chans():
                    wave = filtered_seg[chan]
                    waves.append(wave)

        return waves


    def get_init_events(self):
        if self.initialize_events is None:
            evs = []
        elif self.initialize_events == "leb":
            s = Sigvisa()
            cursor = s.dbconn.cursor()
            evs = get_leb_events(sg, cursor)
            cursor.close()
        elif isinstance(self.initialize_events, list) or \
             isinstance(self.initialize_events, tuple):
            evs = [get_event(evid=evid) for evid in self.initialize_events]
        else:
            raise Exception("unrecognized event initialization %s" % self.initialize_events)

class ModelSpec(object):

    def __init__(self, ms_label=None, vert_only=True, inference_preset=None, seed=0, **kwargs):
        sg_params = {
            'template_shape': "lin_polyexp",
            'template_model_type': "param",
            'wiggle_family': "iid",
            'wiggle_model_type': "dummy",
            'dummy_fallback': False,
            'nm_type': "ar",
            'phases': ["P",],
            'arrays_joint': False,
            'uatemplate_rate': 1e-6,
            'jointgp_hparam_prior': None,
        }
        signal_params = {
            'max_hz': 5.0,
            'bands': ["freq_0.8_4.5",],
            'chans': None, # need to set this
            'smooth': None
        }

        if vert_only:
            chans = ['vertical']
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
        if inference_preset == "closedworld":
            self.add_inference_round(enable_template_openworld=False,
                                     enable_event_openworld=False)
        elif inference_preset == "openworld":
            self.add_inference_round(enable_template_openworld=True,
                                     enable_event_openworld=True)


    def add_inference_round(self, **kwargs):
        inference_params = {
            'enable_template_moves': True,
            'enable_template_openworld': True,
            'enable_event_moves': True,
            'enable_event_openworld': True,
        }
        for k in kwargs.keys():
            inference_params[k] = kwargs[k]
        self.inference_rounds.append(inference_params)

"""
How can I specify a run?

list of sites, time bounds *for all sites*, model runid,

OR list of generated signals with times. okay so ultimately there's going to be a set of waves. either I provide it explicitly or specify it implicitly. But the particular set of waves will depend on things like whether we're doing arrayjoint. Different models will have different sets of waves.

max signal hz, uatemplate rate, wiggle family, wiggle model type, initialization...

I should be able to apply the *same* list

"""

def build_sg(modelspec, runspec):
    sg = SigvisaGraph(runids=(runspec.runid,), **modelspec.sg_params)
    waves = runspec.get_waves(modelspec)
    for wave in waves:
        sg.add_wave(wave)
    return sg

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

def do_inference(sg, modelspec, runspec, max_steps=None, model_switch_lp_threshold=500):

    # save 'true' events if they are known
    # build logger
    # save modelspec to the inference directory
    # save sw to the inference directory
    # run inference with appropriate params
    # and hooks to monitor convergence?

    logger = MCMCLogger( write_template_vals=True, dump_interval=50, print_interval=10, write_gp_hparams=True)


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
        run_open_world_MH(sg, logger=logger,
                          stop_condition=lp_change_threshold,
                          steps=max_steps,
                          start_step=step,
                          **inference_params)
        step = logger.last_step

def do_coarse_to_fine(modelspecs, runspec,
                      model_switch_lp_threshold=1000,
                      max_steps_intermediate=100,
                      max_steps_final=5000):

    sg_old, ms_old = None, None

    sgs = [build_sg(modelspec, runspec) for modelspec in modelspecs]

    for (sg, modelspec) in zip(sgs[:-1], modelspecs[:-1]):
        if sg_old is None:
            initialize_sg(sg, modelspec, runspec)
        else:
            initialize_from(sg, modelspec, sg_old, ms_old)

        do_inference(sg, modelspec, runspec,
                     model_switch_lp_threshold=model_switch_lp_threshold,
                     max_steps = max_steps_intermediate)
        sg_old, ms_old = sg, modelspec

    sg, modelspec = sgs[-1], modelspecs[-1]
    initialize_from(sg, modelspec, sg_old, ms_old)
    do_inference(sg, modelspec, runspec,
                 model_switch_lp_threshold=None,
                 max_steps = max_steps_final)


def synth_location_seq():

    ms1 = ModelSpec(template_model_type="param", wiggle_family="iid")
    ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'])
    ms1.add_inference_round(enable_event_moves=True, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'])

    ms2 = ModelSpec(template_model_type="gp_joint", wiggle_family="iid", wiggle_model_type="dummy", inference_preset="closedworld")

    ms3 = ModelSpec(template_model_type="gp_joint", wiggle_family="db4uvars_2.0_3_10.0", wiggle_model_type="gp_joint", inference_preset="closedworld")

    return [ms1, ms2, ms3]
