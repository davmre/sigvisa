
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

from numba import double
from numba.decorators import autojit

def update_arrivals(parent_values):
    arrivals = set()
    r = re.compile("([-\d]+);(.+);(.+);(.+);(.+);(.+)")
    for k in parent_values.keys():
        m = r.match(k)
        if not m: raise ValueError("could not parse parent key %s" % k)
        eid = int(m.group(1))
        phase = m.group(2)
        arrivals.add((eid, phase))
    return arrivals

class ObservedSignalNode(Node):
    """

    Compute the probability of a set of segments (signal
    envelopes), given a set of events. This is done by either
    maximizing or integrating the probability over the template
    parameters describing the signals.

    """

    def __init__(self, model_waveform, graph, nm_type="ar", nmid=None, observed=True, **kwargs):

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

        self.graph = graph

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

    def assem_signal_usejit(self, include_wiggles=True, arrivals=None):

        # we allow specifying the list of parents in order to generate
        # signals with a subset of arriving phases (used e.g. in
        # wiggle extraction)
        if arrivals is None:
            arrivals = self.arrivals()

        arrivals = list(arrivals)
        sidxs = []
        logenvs = []
        wiggles = []

        for (eid, phase) in arrivals:
            v, tg = self.get_template_params_for_arrival(eid=eid, phase=phase)
            start = (v['arrival_time'] - self.st) * self.srate
            start_idx = int(np.floor(start))
            sidxs.append(start_idx)
            if start_idx >= self.npts:
                logenvs.append([])
                wiggles.append([])
                continue

            offset = start - start_idx
            logenv = tg.abstract_logenv_raw(v, idx_offset=offset, srate=self.srate)
            logenvs.append(logenv)

            if include_wiggles:
                wiggle = self.get_wiggle_for_arrival(eid=eid, phase=phase)
                wiggles.append(wiggle)
            else:
                wiggles.append([])

        signal = assem_signal_jit(arrivals, sidxs, logenvs, wiggles, self.npts)

        if not np.isfinite(signal).all():
            raise ValueError("invalid (non-finite) signal generated for %s!" % self.mw)
        return signal

    def assem_signal(self, include_wiggles=True, arrivals=None):
        signal = np.zeros((self.npts,))

        # we allow specifying the list of parents in order to generate
        # signals with a subset of arriving phases (used e.g. in
        # wiggle extraction)
        if arrivals is None:
            arrivals = self.arrivals()

        for (eid, phase) in arrivals:
            v, tg = self.get_template_params_for_arrival(eid=eid, phase=phase)
            start = (v['arrival_time'] - self.st) * self.srate
            start_idx = int(np.floor(start))
            if start_idx >= self.npts:
                continue

            offset = start - start_idx
            phase_env = np.exp(tg.abstract_logenv_raw(v, idx_offset=offset, srate=self.srate))
            if include_wiggles:
                wiggle = self.get_wiggle_for_arrival(eid=eid, phase=phase)
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

    def _parent_values(self):
        psc = self.parent_set_changed
        pvc = self.parent_value_changed
        pv = super(ObservedSignalNode, self)._parent_values()
        if psc:
            self._arrivals = update_arrivals(pv)
            self._tmpl_keys = dict()
            self._wiggle_keys = dict()
            # cache the list of tmpl/wiggle param keys for this arrival
            for (eid, phase) in self._arrivals:
                tg = self.graph.template_generator(phase=phase)
                self._tmpl_keys[(eid,phase)] = dict()
                for p in tg.params() + ('arrival_time',):
                    k, v = self.get_parent_value(eid, phase, p, pv, return_key=True)
                    self._tmpl_keys[(eid, phase)][p] = k

                wg = self.graph.wiggle_generator(phase=phase, srate=self.srate)
                self._wiggle_keys[(eid, phase)] = []
                for p in wg.params():
                    try:
                        k, v = self.get_parent_value(eid, phase, p, pv, return_key=True)
                        self._wiggle_keys[(eid, phase)].append(k)
                    except KeyError:
                        #print "WARNING: no wiggles for arrival (%d, %s) at (%s, %s, %s)" % (eid, phase, self.sta, self.band, self.chan)
                        k = None


                self._keymap=dict()
                for (p,k) in self._tmpl_keys[(eid, phase)].iteritems():
                    self._keymap[k] = (True, eid, phase, p)
                for (i,k) in enumerate(self._wiggle_keys[(eid, phase)]):
                    self._keymap[k] = (False, eid, phase, i)

        if pvc:
            self._tmpl_params = dict()
            self._wiggle_params = dict()
            for (eid, phase) in self._arrivals:
                self._tmpl_params[(eid,phase)] = dict([(p, pv[k]) for (p, k) in self._tmpl_keys[(eid, phase)].iteritems()])
                self._wiggle_params[(eid, phase)] = np.asarray([pv[k] for k in self._wiggle_keys[(eid, phase)]])

        return pv

    def arrivals(self):
        self._parent_values()
        return self._arrivals

    def parent_predict(self, parent_values=None):
        #parent_values = parent_values if parent_values else self._parent_values()
        signal = self.assem_signal()
        noise = self.nm.predict(n=len(signal))
        self.set_value(signal + noise)
        for child in self.children:
            child.parent_value_changed = True

    def parent_sample(self, parent_values=None):
        signal = self.assem_signal()
        noise = self.nm.sample(n=len(signal))
        self.set_value(signal + noise)
        for child in self.children:
            child.parent_value_changed = True

    def log_p(self, parent_values=None):
        parent_values = parent_values if parent_values else self._parent_values()
        value = self.get_value()

        pred_signal = self.assem_signal()
        diff = value - pred_signal
        lp = self.nm.log_p(diff)
#        import hashlib
#        fname = hashlib.sha1(str(lp) + str(self.nmid)).hexdigest()
#        np.savetxt(fname, diff)
#        print "wave logp %f, nmid %d, saving diff to %s" % (lp, self.nmid, fname)

        return lp

    def deriv_log_p(self, parent_key=None, lp0=None, eps=1e-4):
        parent_values = self._parent_values()
        lp0 = lp0 if lp0 else self.log_p(parent_values=parent_values)
        parent_values[parent_key] += eps
        is_tmpl, eid, phase, p = self._keymap[parent_key]
        if is_tmpl:
            self._tmpl_params[(eid, phase)][p] += eps
        else:
            self._wiggle_params[(eid, phase)][p] += eps
        deriv = ( self.log_p(parent_values=parent_values) - lp0 ) / eps
        parent_values[parent_key] -= eps
        if is_tmpl:
            self._tmpl_params[(eid, phase)][p] -= eps
        else:
            self._wiggle_params[(eid, phase)][p] -= eps
        return deriv

    def get_parent_value(self, eid, phase, param_name, parent_values, **kwargs):
         return get_parent_value(eid=eid, phase=phase, sta=self.sta, chan=self.chan, band=self.band, param_name=param_name, parent_values=parent_values, **kwargs)

    def get_wiggle_for_arrival(self, eid, phase, parent_values=None):
        parent_values = parent_values if parent_values else self._parent_values()
        wg = self.graph.wiggle_generator(phase, self.srate)
        if len(self._wiggle_params[(eid, phase)]) == wg.dimension():
            return wg.signal_from_features(features = self._wiggle_params[(eid, phase)])
        else:
            return np.ones((wg.npts,))

    def get_template_params_for_arrival(self, eid, phase, parent_values=None):
        tg = self.graph.template_generator(phase)
        return self._tmpl_params[(eid, phase)], tg

@autojit
def assem_signal_jit(arrivals, sidxs, logenvs, wiggles, npts):
    signal = np.zeros((npts,))

    for i in range(len(arrivals)):
        eid = arrivals[i][0]
        phase = arrivals[i][1]
        start_idx = sidxs[i]
        logenv = logenvs[i]
        wiggle = wiggles[i]

        wiggle_len = min(len(wiggle), len(logenv))

        end_idx = start_idx + len(logenv)
        if end_idx <= 0:
            continue
        early = max(0, - start_idx)
        overshoot = max(0, end_idx - len(signal))

        j = early
        total_wiggle_len = min(wiggle_len - early, len(logenv) - overshoot - early)
        for t in range(start_idx+early, start_idx+early+total_wiggle_len):
            signal[t] += np.exp(logenv[j]) * wiggle[j]
            j += 1
        for t in range(start_idx+early+total_wiggle_len, end_idx - overshoot):
            signal[t] += np.exp(logenv[j])
            j += 1

    return signal
