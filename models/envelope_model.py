
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
from sigvisa.signals.common import Waveform
from sigvisa.models.noise.noise_util import get_noise_model
from sigvisa.models.noise.noise_model import NoiseModel
from sigvisa.graph.nodes import Node

class EnvelopeNode(Node):

    """

    Compute the probability of a set of segments (signal
    envelopes), given a set of events. This is done by either
    maximizing or integrating the probability over the template
    parameters describing the signals.

    """

    def __init__(self, model_waveform, nm_type="ar", nmid=None, observed=True, **kwargs):

        super(EnvelopeNode, self).__init__(model=None, initial_value=model_waveform.data, fixed=observed, **kwargs)


        self.mw = model_waveform
        self.filter_str = model_waveform['filter_str']
        self.sta = model_waveform['sta']
        self.chan = model_waveform['chan']
        self.band = model_waveform['band']
        self.srate = model_waveform['srate']
        self.st = model_waveform['stime']
        self.et = model_waveform['etime']
        self.npts = model_waveform['npts']

        self.set_noise_model(nm_type=nm_type, nmid=nmid)


    def set_value(self, value):
        assert(len(value) == len(self._value))
        super(EnvelopeNode, self).set_value(value)

    def get_wave(self):
        return Waveform(data=self.get_value(), segment_stats=self.mw.segment_stats.copy(), my_stats=self.mw.my_stats.copy())

    def set_noise_model(self, nm_type="ar", nmid=None):
        if nmid is None:
            self.nm_type = nm_type
            self.nm, self.nmid, _ = get_noise_model(waveform=self.mw, model_type=self.nm_type, return_details=True)
        else:
            self.nmid = nmid
            self.nm = NoiseModel.load_by_nmid(Sigvisa().dbconn, self.nmid)
            self.nm_type = self.nm.noise_model_type()


    def assem_signal(self, include_wiggles=True, parent_templates=None, parent_values = None):
        signal = np.zeros((self.npts,))

        # we allow specifying the list of parents in order to generate
        # signals with a subset of arriving phases (used e.g. in
        # wiggle extraction)
        if parent_templates is None:
            parent_templates = [tm for tm in self.parents.values() if tm.label.startswith("template_")]

        for tm in parent_templates:
            key = tm.label[9:]
            v = parent_values[tm.label] if parent_values else tm.get_value()

            arr_time = v['arrival_time']
            start = (arr_time - self.st) * self.srate
            start_idx = int(np.floor(start))
            if start_idx >= self.npts:
                continue

            offset = start - start_idx
            phase_env = np.exp(tm.abstract_logenv_raw(v, idx_offset=offset, srate=self.srate))
            if include_wiggles:
                wm = self.parents['wiggle_%s' % key]
                wiggle = wm.get_wiggle(value = parent_values[wm.label] if parent_values else None)
                wiggle_len = min(len(wiggle), len(phase_env))
                phase_env[:wiggle_len] *= wiggle[:wiggle_len]

            end_idx = start_idx + len(phase_env)
            if end_idx <= 0:
                continue
            early = max(0, - start_idx)
            overshoot = max(0, end_idx - len(signal))
            final_template = phase_env[early:len(phase_env) - overshoot]

            signal[start_idx + early:end_idx - overshoot] += final_template

            if not np.isfinite(signal).all():
                raise ValueError("invalid (non-finite) signal generated for %s!" % self.mw)

        return signal

    def prior_predict(self):
        signal = self.assem_signal()
        noise = self.nm.predict(n=len(signal))
        self.set_value(signal + noise)

    def prior_sample(self):
        signal = self.assem_signal()
        noise = self.nm.sample(n=len(signal))
        self.set_value(signal + noise)

    def log_p(self, value=None, parent_values=None):
        if value is None:
            value = self.get_value()



        pred_signal = self.assem_signal(parent_values=parent_values)
        diff = value - pred_signal
        lp = self.nm.log_p(diff)
#        import hashlib
#        fname = hashlib.sha1(str(lp) + str(self.nmid)).hexdigest()
#        np.savetxt(fname, diff)
#        print "wave logp %f, nmid %d, saving diff to %s" % (lp, self.nmid, fname)


        return lp

    def deriv_log_p(self, value=None, parent_values=None, parent_name=None, parent_key=None, lp0=None, eps=1e-4):
        parent_values = parent_values if parent_values else self._parent_values()
        lp0 = lp0 if lp0 else self.log_p(value=value, parent_values=parent_values)
        pv = parent_values[parent_name]
        pv[parent_key] += eps
        parent_values[parent_name] = pv
        deriv = ( self.log_p(value=value, parent_values=parent_values) - lp0 ) / eps
        return deriv
