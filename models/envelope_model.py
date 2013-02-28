
import os
import errno
import sys
import time
import traceback
import numpy as np
import scipy

from sigvisa.database.signal_data import filter_and_sort_template_params
from sigvisa.database.dataset import *
from sigvisa.database import db

from sigvisa.infer.optimize.optim_utils import minimize_matrix, construct_optim_params

from optparse import OptionParser

from sigvisa import Sigvisa, NestedDict

from sigvisa.models import TimeSeriesDist, ConditionalDist
from sigvisa.models.noise.noise_util import get_noise_model

class EnvelopeNode(Node):

    """

    Compute the probability of a set of segments (signal
    envelopes), given a set of events. This is done by either
    maximizing or integrating the probability over the template
    parameters describing the signals.

    """

    def __init__(self, model_waveform, nm_type="ar", observed=True, **kwargs):

        super(Node, self).__init__(model=None, initial_value=model_waveform.data, fixed_value=observed, **kwargs)

        self.sigvisa = Sigvisa()
        self.nm_type = nm_type

        self.nm, self.nmid, _ = get_noise_model(waveform=model_waveform, model_type=self.nm_type, return_details=True)

        self.srate = model_waveform['srate']
        self.st = model_waveform['stime']
        self.et = model_waveform['etime']
        self.npts = model_waveform['npts']

    def assem_signal(self):
        signal = np.zeros((self.npts,))

        parent_templates = [tm for tm in self.parents.values() if tm.label.startswith("template_")]

        for tm in parent_templates:
            key = tm.label[10:]
            v = tm.get_value()

            arr_time = v[0]
            start = (arr_time - self.st) * self.srate
            start_idx = int(np.floor(start))
            if start_idx >= self.npts:
                continue

            offset = start - start_idx
            phase_env = tm.abstract_logenv_raw(v, idx_offset=offset, srate=srate)
            end_idx = start_idx + len(phase_env)
            if end_idx <= 0:
                continue
            early = max(0, - start_idx)
            overshoot = max(0, end_idx - len(data))
            final_template = np.exp(phase_env[early:len(phase_env) - overshoot])

            wm = self.parents['wiggle_%s' % key]
            wiggle_offset_idx = int(wm.atime_offset_seconds * self.srate)
            wiggle_npts = len(final_template) - wiggle_offset_idx
            wiggle = wm.get_wiggle(npts=wiggle_npts)
            final_template[wiggle_offset_idx:] *= wiggle
            signal[start_idx + early:end_idx - overshoot] += final_template

        return signal

    def prior_predict(self):
        signal = self.assem_signal()
        noise = self.nm.mean(n=len(signal))
        self.value = signal + noise

    def prior_sample(self):
        signal = self.assem_signal()
        noise = self.nm.sample(n=len(signal))
        self.value = signal + noise

    def log_p(self, value=None):
        if value is None:
            value = self.value

        pred_signal = self.assem_signal()
        diff = value - pred_signal
        return self.nm.log_p(diff)

