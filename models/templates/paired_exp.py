import numpy as np
import sys, os
from sigvisa import *
import sigvisa_c

import models.noise.noise_model as noise_model
from source.event import get_event
from signals.common import *
from models.templates.template_model import TemplateModel

ARR_TIME_PARAM, PEAK_OFFSET_PARAM, CODA_HEIGHT_PARAM, CODA_DECAY_PARAM, NUM_PARAMS = range(4+1)

def set_dummy_wiggles(sta):
    s = Sigvisa()
    siteid = s.name_to_siteid_minus1[sta]+1
    for chan in s.chans:
        c = sigvisa_c.canonical_channel_num(chan)
        for band in s.bands:
            b = sigvisa_c.canonical_band_num(band)
            for p in s.phases:
                pid = s.phaseids[p]
                sigmodel.set_wiggle_process(siteid, b, c, pid, 1, 0.05, np.array([.8,-.2]))


class PairedExpTemplateModel(TemplateModel):

#    target_fns = {"decay": lambda r : r[FIT_CODA_DECAY], "onset": lambda r : r[FIT_PEAK_DELAY], "amp": lambda r: r[FIT_CODA_HEIGHT] - r[FIT_MB], "amp_transfer": lambda r : r[FIT_CODA_HEIGHT] - SourceSpectrumModel().source_logamp(r[FIT_MB], int(r[FIT_PHASEID]), bandid=int(r[FIT_BANDID]))}

    def params(self):
        return ("arrival_time", "peak_offset", "coda_height", "coda_decay")

    def model_name(self):
        return "paired_exp"


    def abstract_logenv_raw(self, vals, min_logenv = -7, idx_offset = 0, srate=40):
        arr_time, peak_offset, coda_height, coda_decay = vals
        assert(idx_offset >= 0 and idx_offset < 1)

        if np.isnan(vals).any() or coda_decay > 0:
            #                print "WARNING: invalid parameters", vals
            return np.empty((0,))

        if coda_decay > -0.001:
            l =1200*srate
        else:
            l = int(max(0, min(1200, peak_offset + (min_logenv - coda_height) / coda_decay) * srate))
        d = np.empty((l,))

        peak_idx = max(0, peak_offset*srate)
        if peak_idx != 0:
            onset_slope = np.exp(coda_height) / peak_idx
        else:
            onset_slope = 0

        try:
            intro_len = min(len(d), int(idx_offset+peak_idx)+1)
            if intro_len > 0 and onset_slope > 0:
                intro_env = (np.arange(intro_len) - idx_offset) * onset_slope + np.exp(min_logenv)

                # avoid taking log(0)
                intro_env[0] = 0.1
                d[0:intro_len] = np.log(intro_env)

                # say peak_idx is 9.9 and idx_offset is 0.2, so we have intro_len=11
                # then at t=1, we have onset[0.8]
                # up to t=10, where we have onset[9.8], which is what we want.
                # now imagine peak_idx is 10 and idx_offset is 0; again we have intro_len=11
                # now at t=10 we have onset[10], which is fine.

            # now for case a, at t=11 we are 10.8 into the signal, so we want decay[0.9]
            # for case b, at t=11 we are 11 into the signal, so we want decay[1]
            # in general at t=intro_len we are intro_len - idx_offset into the signal, so we want decay[intro_len - idx_offset - peak_idx]
            initial_decay = intro_len - idx_offset - peak_idx
            d[intro_len:] = (np.arange(len(d)-intro_len) + initial_decay)/srate * coda_decay + coda_height
            if len(d) > 0:
                d[0] = -999
        except Exception as e:
            print e
            raise
        return d


    def waveform_log_likelihood(self, wave, template_params):
        s = self.sigvisa
        phases, vals = template_params
        phaseids = [s.phaseids[phase] for phase in phases]
        noise_model.set_noise_process(wave)
        tr = wave.as_obspy_trace()
        tr.data = tr.data.filled(np.float('nan'))
        ll = s.sigmodel.trace_log_likelihood(tr, phaseids, vals);
        return ll

    def low_bounds(self, phases, default_atimes=None):
        bounds = np.ones((len(phases), len(self.params()))) * -np.inf
        bounds[:, CODA_HEIGHT_PARAM] = -7
        bounds[:, PEAK_OFFSET_PARAM] = 0
        bounds[:, CODA_DECAY_PARAM] = -.2
        if default_atimes is not None:
            bounds[:, ARR_TIME_PARAM] = default_atimes - 15
        return bounds

    def high_bounds(self, phases, default_atimes=None):
        bounds = np.ones((len(phases), len(self.params()))) * np.inf
        bounds[:, PEAK_OFFSET_PARAM] = 25
        bounds[:, CODA_DECAY_PARAM] = 0
        bounds[:, CODA_HEIGHT_PARAM] = 10
        if default_atimes is not None:
            bounds[:, ARR_TIME_PARAM] = default_atimes + 15
        return bounds

    def heuristic_starting_params(self, wave, detected_phases_only=True):
        s = Sigvisa()

        ev = get_event(wave['evid'])
        if detected_phases_only:
            arrivals = wave['event_arrivals']
            arrival_phases = [s.phasenames[pid_m1] for pid_m1 in arrivals[:, DET_PHASE_COL]]
            all_phases = arrival_phases
        else:
            all_phases = s.arriving_phases(ev, wave['sta'])

        start_params = np.zeros((len(all_phases), 4))
        for (i, phase) in enumerate(all_phases):
            start_params[i, ARR_TIME_PARAM] = ev.time + s.sigmodel.mean_travel_time(ev.lon, ev.lat, ev.depth, wave['siteid']-1, s.phaseids[phase]-1)
            start_params[i, PEAK_OFFSET_PARAM] = 1
#            arrival_idx = int((start_params[i, ARR_TIME_PARAM] - wave['stime']) * wave['srate'])
            start_params[i, CODA_HEIGHT_PARAM] = np.log(np.max(wave.data)) +.2 #np.log(np.max(wave.data[arrival_idx: arrival_idx + wave['srate']*5]))+.5
            start_params[i, CODA_DECAY_PARAM] = -0.001
        return (all_phases, start_params)
