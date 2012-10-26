import numpy as np
import sys, os
from sigvisa import *

from signals.template_model import TemplateModel

from signals.template_models.heuristic_coda_fits import *

class PairedExpTemplateModel(TemplateModel):

#    target_fns = {"decay": lambda r : r[FIT_CODA_DECAY], "onset": lambda r : r[FIT_PEAK_DELAY], "amp": lambda r: r[FIT_CODA_HEIGHT] - r[FIT_MB], "amp_transfer": lambda r : r[FIT_CODA_HEIGHT] - SourceSpectrumModel().source_logamp(r[FIT_MB], int(r[FIT_PHASEID]), bandid=int(r[FIT_BANDID]))}

    def params(self):
        return ("arrival_time", "onset_period", "amp_transfer", "decay")

    def model_name(self):
        return "paired_exp"

    def valid_params(self, params):
        phases, vals = params

        for i, phase in enumerate(phases):
            if np.isnan(vals[i, PEAK_HEIGHT_PARAM]) or np.isnan(vals[i, CODA_HEIGHT_PARAM]):
                return False
            if vals[i, CODA_HEIGHT_PARAM] > vals[i, PEAK_HEIGHT_PARAM] + 1:
                return False
            if vals[i, CODA_DECAY_PARAM] >= 0:
                return False
            if vals[i, PEAK_DECAY_PARAM] < 0:
                return False
        return True

    def generate_template_waveform(self, template_params, model_waveform, logscale=False, sample=False):
        s = self.sigvisa

        srate = trace['srate']
        st = trace['stime']
        et = trace['etime']
        siteid = trace['siteid']
        c = sigvisa_c.canonical_channel_num(trace['chan'])
        b = sigvisa_c.canonical_band_num(trace['band'])

        phases, vals = template_params
        phaseids = [s.phaseids[phase] for phase in phases]

        if not sample:
            env = s.sigmodel.generate_wave(st, et, int(siteid), int(b), int(c), srate, phaseids, vals)
        else:
            env = s.sigmodel.sample_wave(st, et, int(siteid), int(b), int(c), srate, phaseids, vals)

        data = np.log(env) if logscale else env
        wave = Waveform(data = data, segment_stats=model_waveform.segment_stats.copy(), my_stats=model_waveform.my_stats.copy())

        return wave

    def waveform_cost(self, wave, template_params):
        if not self.valid_params(template_params):
            return float('inf')

        s = self.sigvisa
        phases, vals = template_params
        phaseids = [s.phaseids[phase] for phase in phases]

        c = -1 *s.sigmodel.trace_likelihood(wave.as_obspy_trace(), vals);
        return c

    def heuristic_starting_params(self, wave):
        smoothed = wave.filter('smooth').as_obspy_trace()
        noise_model = get_noise_model(wave)
        smoothed.stats.noise_floor = noise_model.c
        vals = find_starting_params(smoothed)
        return (arrs['phase_arrivals'], vals)


