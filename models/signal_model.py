
import os
import errno
import sys
import time
import traceback
import numpy as np
import scipy
import re

from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.models.noise.noise_util import get_noise_model
from sigvisa.models.noise.noise_model import NoiseModel
from sigvisa.graph.nodes import Node
from sigvisa.graph.graph_utils import get_parent_value, create_key


def update_arrivals(parent_values):
    arrivals = set()
    r = re.compile("(\d+);(.+);(.+);(.+);(.+);(.+)")
    for k in parent_values.keys():
        m = r.match(k)
        if not m: raise ValueError("could not parse parent key %s" % k)
        eid = int(m.group(1))
        phase = m.group(2)
        arrivals.update((eid, phase))
    return arrivals

class ObservedSignalNode(Node):
    """

    Compute the probability of a set of segments (signal
    envelopes), given a set of events. This is done by either
    maximizing or integrating the probability over the template
    parameters describing the signals.

    """

    def __init__(self, model_waveform, nm_type="ar", nmid=None, observed=True, **kwargs):

        key = create_key(param="signal_%.2f_%.2f" % (model_waveform['stime'], model_waveform['etime'] ), sta=model_waveform['sta'], chan=model_waveform['band'], band=model_waveform['band'])

        super(ObservedSignalNode, self).__init__(model=None, initial_value=model_waveform.data, keys=[key,], fixed=observed, **kwargs)

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
        assert(len(value) == len(self.get_value(self.single_key)))
        super(ObservedSignalNode, self).set_value(value=value, key=self.single_key)

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

    def assem_signal(self, include_wiggles=True, arrivals=None, parent_values = None):
        signal = np.zeros((self.npts,))

        # we allow specifying the list of parents in order to generate
        # signals with a subset of arriving phases (used e.g. in
        # wiggle extraction)
        if arrivals is None:
            arrivals = update_arrivals(parent_values)

        for (eid, phase) in arrivals:
            tg = self.template_generator[phase]

            v = dict([(p, self.get_parent_value(eid, phase, p, parent_values)) for p in tg.params()])

            arr_time = self.get_parent_value(eid, phase, "arrival time", parent_values)
            start = (arr_time - self.st) * self.srate
            start_idx = int(np.floor(start))
            if start_idx >= self.npts:
                continue

            offset = start - start_idx
            phase_env = np.exp(tg.abstract_logenv_raw(v, idx_offset=offset, srate=self.srate))
            if include_wiggles:
                wg = self.wiggle_generator[phase]
                wiggle_params = self.get_parent_value(eid, phase, "wiggle", parent_values)
                wiggle = wg.get_wiggle(value = wiggle_params)
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

    def prior_predict(self, parent_values=None):
        parent_values = parent_values if parent_values else self._parent_values()
        signal = self.assem_signal(parent_values=parent_values)
        noise = self.nm.predict(n=len(signal))
        self.set_value(signal + noise)

    def prior_sample(self, parent_values=None):
        parent_values = parent_values if parent_values else self._parent_values()
        signal = self.assem_signal(parent_values=parent_values)
        noise = self.nm.sample(n=len(signal))
        self.set_value(signal + noise)

    def log_p(self, parent_values=None):
        parent_values = parent_values if parent_values else self._parent_values()
        value = self.get_value()

        pred_signal = self.assem_signal(parent_values=parent_values)
        diff = value - pred_signal
        lp = self.nm.log_p(diff)
#        import hashlib
#        fname = hashlib.sha1(str(lp) + str(self.nmid)).hexdigest()
#        np.savetxt(fname, diff)
#        print "wave logp %f, nmid %d, saving diff to %s" % (lp, self.nmid, fname)

        return lp

    def deriv_log_p(self, parent_values=None, parent_key=None, lp0=None, eps=1e-4):
        parent_values = parent_values if parent_values else self._parent_values()
        lp0 = lp0 if lp0 else self.log_p(parent_values=parent_values)
        parent_values[parent_key] += eps
        deriv = ( self.log_p(parent_values=parent_values) - lp0 ) / eps
        return deriv

    def get_parent_value(self, eid, phase, param_name, parent_values):
        return get_parent_value(eid=eid, phase=phase, sta=self.sta, chan=self.chan, band=self.band, param_name=param_name, parent_values=parent_values)
