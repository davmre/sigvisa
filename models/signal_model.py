
import os
import errno
import sys
import time
import traceback
import numpy as np
import numpy.ma as ma
import scipy
import re

from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.models.noise.noise_util import get_noise_model
from sigvisa.models.noise.noise_model import NoiseModel
from sigvisa.graph.nodes import Node
from sigvisa.graph.graph_utils import get_parent_value, create_key


import scipy.weave as weave
from scipy.weave import converters

def get_new_arrivals(new_nodes, r):
    new_arrivals = set()
    for n in new_nodes:
        for k in n.keys():
            m = r.match(k)
            if not m: raise ValueError("could not parse parent key %s" % k)
            eid = int(m.group(1))
            phase = m.group(2)
            new_arrivals.add((eid, phase))
    return new_arrivals

def get_removed_arrivals(removed_keys, r):
    removed_arrivals = set()
    for k in removed_keys:
        m = r.match(k)
        if not m: raise ValueError("could not parse parent key %s" % k)
        eid = int(m.group(1))
        phase = m.group(2)
        removed_arrivals.add((eid, phase))
    return removed_arrivals

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

        self.signal_diff = np.empty((self.npts,))
        self.pred_signal = np.empty((self.npts,))

        self.set_noise_model(nm_type=nm_type, nmid=nmid)


        self._tmpl_params = dict()
        self._wiggle_params = dict()
        self._keymap = dict()
        self._arrivals = set()
        self.r = re.compile("([-\d]+);(.+);(.+);(.+);(.+);(.+)")

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

    def assem_signal(self, include_wiggles=True, arrivals=None):

        # we allow specifying the list of parents in order to generate
        # signals with a subset of arriving phases (used e.g. in
        # wiggle extraction)
        if arrivals is None:
            arrivals = self.arrivals()

        arrivals = list(arrivals)
        n = len(arrivals)
        sidxs = np.empty((n,))
        logenvs = [None] * n
        wiggles = [None] * n
        empty_array = np.reshape(np.array((), dtype=float), (0,))

        for (i, (eid, phase)) in enumerate(arrivals):
            v, tg = self.get_template_params_for_arrival(eid=eid, phase=phase)
            start = (v['arrival_time'] - self.st) * self.srate
            start_idx = int(np.floor(start))
            sidxs[i] = start_idx
            if start_idx >= self.npts:
                logenvs[i] = empty_array
                wiggles[i] = empty_array
                continue

            offset = start - start_idx
            logenv = tg.abstract_logenv_raw(v, idx_offset=offset, srate=self.srate)
            logenvs[i] = logenv

            if include_wiggles:
                wiggle = self.get_wiggle_for_arrival(eid=eid, phase=phase)
                wiggles[i] = wiggle
            else:
                wiggles[i] = empty_array

        npts = self.npts
        n = len(arrivals)
        signal = self.pred_signal
        code = """
      for(int i=0; i < npts; ++i) signal(i) = 0;
    for (int i = 0; i < n; ++i) {
        int start_idx = sidxs(i);

        PyArrayObject* logenv_arr = convert_to_numpy(PyList_GetItem(logenvs,i), "logenv");
        conversion_numpy_check_type(logenv_arr,PyArray_DOUBLE, "logenv");
        double * logenv = (double *)logenv_arr->data;

        PyArrayObject* wiggle_arr = convert_to_numpy(PyList_GetItem(wiggles,i), "wiggle");
        conversion_numpy_check_type(wiggle_arr,PyArray_DOUBLE, "wiggle");
        double * wiggle =  (double *) wiggle_arr->data;

        int len_logenv = logenv_arr->dimensions[0];
        int len_wiggle = wiggle_arr->dimensions[0];

        int wiggle_len = std::min(len_wiggle, len_logenv);

        int end_idx = start_idx + len_logenv;
        if (end_idx <= 0) {
            continue;
        }
        int early = std::max(0, - start_idx);
        int overshoot = std::max(0, end_idx - npts);

        int j = early;
        int total_wiggle_len = std::min(wiggle_len - early, len_logenv - overshoot - early);
        for(j=early; j < early + total_wiggle_len; ++j) {
            signal(j+start_idx) += exp(logenv[j]) * wiggle[j];
        }
        for(; j < len_logenv - overshoot; ++j) {
            signal(j + start_idx) += exp(logenv[j]);
        }
    }
"""
        weave.inline(code,['n', 'npts', 'sidxs', 'logenvs', 'wiggles', 'signal'],type_converters = converters.blitz,verbose=2,compiler='gcc')

        if not np.isfinite(signal).all():
            raise ValueError("invalid (non-finite) signal generated for %s!" % self.mw)
        return signal

    def _parent_values(self):
        parent_keys_removed = self.parent_keys_removed
        parent_keys_changed = self.parent_keys_changed
        parent_nodes_added = self.parent_nodes_added
        pv = super(ObservedSignalNode, self)._parent_values()

        new_arrivals = get_new_arrivals(parent_nodes_added, self.r)
        removed_arrivals = get_removed_arrivals(parent_keys_removed, self.r)
        self._arrivals.update(new_arrivals)
        self._arrivals.difference_update(removed_arrivals)

        # cache the list of tmpl/wiggle param keys for the new arrivals
        for (eid, phase) in new_arrivals:
            tg = self.graph.template_generator(phase=phase)
            self._tmpl_params[(eid,phase)] = dict()
            for p in tg.params() + ('arrival_time',):
                k, v = self.get_parent_value(eid, phase, p, pv, return_key=True)
                self._tmpl_params[(eid,phase)][p] = v
                self._keymap[k] = (True, eid, phase, p)

            wg = self.graph.wiggle_generator(phase=phase, srate=self.srate)
            self._wiggle_params[(eid, phase)] = np.empty((wg.dimension(),))
            for (i, p) in enumerate(wg.params()):
                try:
                    k, v = self.get_parent_value(eid, phase, p, pv, return_key=True)
                    self._wiggle_params[(eid,phase)][i] = v
                    self._keymap[k] = (False, eid, phase, i)
                except KeyError:
                    #print "WARNING: no wiggles for arrival (%d, %s) at (%s, %s, %s)" % (eid, phase, self.sta, self.band, self.chan)
                    k = None

        for k in parent_keys_removed:
            try:
                tmpl, eid, phase, p = self._keymap[k]
                if tmpl:
                    del self._tmpl_params[(eid, phase)]
                else:
                    del self._wiggle_params[(eid, phase)]
                del self._keymap[k]
            except KeyError:
                continue

        for (key, node) in parent_keys_changed:
            tmpl, eid, phase, p = self._keymap[key]
            if tmpl:
                self._tmpl_params[(eid,phase)][p] = pv[key]
            else:
                self._wiggle_params[(eid,phase)][p] = pv[key]

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
            child.parent_keys_changed.add(self.single_key)

    def parent_sample(self, parent_values=None):
        signal = self.assem_signal()
        noise = self.nm.sample(n=len(signal))
        self.set_value(signal + noise)
        for child in self.children:
            child.parent_keys_changed.add((self.single_key), self)

    def log_p(self, parent_values=None):
        parent_values = parent_values if parent_values else self._parent_values()
        v = self.get_value()
        value = v.data
        mask = v.mask

        pred_signal = self.assem_signal()
        signal_diff = self.signal_diff
        npts = self.npts
        code = """
for(int i=0; i < npts; ++i) {
signal_diff(i) =value(i) - pred_signal(i);
}
"""
        weave.inline(code,['npts', 'signal_diff', 'value', 'pred_signal'],type_converters = converters.blitz,verbose=2,compiler='gcc')

        lp = self.nm.log_p(signal_diff, mask=mask)
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
