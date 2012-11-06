import numpy as np
import sys, os
from sigvisa import *
import sigvisa_c

import signals.noise_model as noise_model
from signals.common import *
from signals.template_model import TemplateModel
from signals.template_models.heuristic_coda_fits import *

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
        return ("arrival_time", "onset_period", "amp_transfer", "decay")

    def model_name(self):
        return "paired_exp"

    def generate_template_waveform(self, template_params, model_waveform, logscale=False, sample=False):
        s = self.sigvisa

        srate = model_waveform['srate']
        st = model_waveform['stime']
        et = model_waveform['etime']
        siteid = model_waveform['siteid']
        c = sigvisa_c.canonical_channel_num(model_waveform['chan'])
        b = sigvisa_c.canonical_band_num(model_waveform['band'])

        noise_model.set_noise_process(model_waveform)
        phases, vals = template_params
        phaseids = [s.phaseids[phase] for phase in phases]

        if not sample:
            env = s.sigmodel.generate_trace(st, et, int(siteid), int(b), int(c), srate, phaseids, vals)
        else:
            env = s.sigmodel.sample_trace(st, et, int(siteid), int(b), int(c), srate, phaseids, vals)

        data = np.log(env) if logscale else env
        wave = Waveform(data = data, segment_stats=model_waveform.segment_stats.copy(), my_stats=model_waveform.my_stats.copy())

        try:
            del wave.segment_stats['evid']
            del wave.segment_stats['event_arrivals']
        except KeyError:
            pass
        return wave

    def waveform_log_likelihood(self, wave, template_params):
        s = self.sigvisa
        phases, vals = template_params
        phaseids = [s.phaseids[phase] for phase in phases]
        noise_model.set_noise_process(wave)
        tr = wave.as_obspy_trace()
        tr.data = tr.data.filled(np.float('nan'))
        ll = s.sigmodel.trace_log_likelihood(tr, phaseids, vals);
        return ll

    def low_bounds(self, phases):
        bounds = np.ones((len(phases), len(self.params()))) * -np.inf
        bounds[:, PE_PEAK_OFFSET_PARAM] = 0
        bounds[:, PE_CODA_HEIGHT_PARAM] = 0
        bounds[:, PE_CODA_DECAY_PARAM] = -.2
        return bounds

    def high_bounds(self, phases):
        bounds = np.ones((len(phases), len(self.params()))) * np.inf
        bounds[:, PE_PEAK_OFFSET_PARAM] = 15
        bounds[:, PE_CODA_DECAY_PARAM] = 0
        return bounds

    def heuristic_starting_params(self, wave):
        smoothed = wave.filter('smooth').as_obspy_trace()
        noise_model = get_noise_model(wave)
        smoothed.stats.noise_floor = noise_model.c
        template_params = find_starting_params(smoothed)
        return template_params


