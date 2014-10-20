
import os
import errno
import sys
import time
import traceback
import numpy as np
import numpy.ma as ma
import scipy
import re
import collections

from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.models.noise.noise_util import get_noise_model
from sigvisa.models.noise.noise_model import NoiseModel
from sigvisa.graph.nodes import Node
from sigvisa.graph.graph_utils import get_parent_value, create_key


import scipy.weave as weave
from scipy.weave import converters

def extract_arrival_from_key(k, r):
    m = r.match(k)
    if not m: raise ValueError("could not parse parent key %s" % k)
    eid = int(m.group(1))
    phase = m.group(2)
    return (eid, phase)

def get_new_arrivals(new_nodes, r):
    new_arrivals = set()
    for n in new_nodes:
        for k in n.keys():
            new_arrivals.add(extract_arrival_from_key(k, r))
    return new_arrivals

def get_removed_arrivals(removed_keys, r):
    removed_arrivals = set()
    for k in removed_keys:
        removed_arrivals.add(extract_arrival_from_key(k, r))
    return removed_arrivals

def update_arrivals(parent_values):
    arrivals = set()
    r = re.compile("([-\d]+);(.+);(.+);(.+);(.+);(.+)")
    for k in parent_values.keys():
        if k=="prefix": continue
        arrivals.add(extract_arrival_from_key(k, r))
    return arrivals

def unify_windows(w1, w2):
    start_idx1, end_idx1 = w1
    start_idx2, end_idx2 = w2

    start_idx = min(start_idx1, start_idx2)
    end_idx = max(end_idx1, end_idx2)
    return (start_idx, end_idx)

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
        self.valid_len = model_waveform['valid_len']
        self.env = 'env' in self.filter_str

        self.signal_diff = np.empty((self.npts,))
        self.pred_signal = np.empty((self.npts,))

        self.set_noise_model(nm_type=nm_type, nmid=nmid)


        self._tmpl_params = dict()
        self._wiggle_params = dict()
        self._keymap = dict()
        self._arrivals = set()
        self.r = re.compile("([-\d]+);(.+);(.+);(.+);(.+);(.+)")

        self.graph = graph


    def __str__(self):
        s = "Signal node w/ model wave %s" % (self.mw)
        return s

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

    def assem_signal(self, include_wiggles=True, arrivals=None, window_start_idx=0, npts=None):
        """

        WARNING: returns a pointer to self.pred_signal, which will be
        overwritten on future calls. So if you want the returned value
        to persist, you need to make a copy. (np.copy())

        """

        # we allow specifying the list of parents in order to generate
        # signals with a subset of arriving phases (used e.g. in
        # wiggle extraction)
        if arrivals is None:
            arrivals = self.arrivals()

        arrivals = list(arrivals)
        n = len(arrivals)
        sidxs = np.empty((n,), dtype=int)
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

            offset = float(start - start_idx)
            logenv = tg.abstract_logenv_raw(v, idx_offset=offset, srate=self.srate)
            logenvs[i] = logenv

            if include_wiggles:
                wiggle = self.get_wiggle_for_arrival(eid=eid, phase=phase)
                wiggles[i] = wiggle
            else:
                wiggles[i] = empty_array

        npts = self.npts-window_start_idx if not npts else int(npts)
        n = len(arrivals)
        envelope = int(self.env or (not include_wiggles))
        signal = self.pred_signal
        code = """
      for(int i=window_start_idx; i < window_start_idx+npts; ++i) signal(i) = 0;
    for (int i = 0; i < n; ++i) {
        int start_idx = sidxs(i);

        PyArrayObject* logenv_arr = convert_to_numpy(PyList_GetItem(logenvs,i), "logenv");
        conversion_numpy_check_type(logenv_arr,PyArray_DOUBLE, "logenv");
        double * logenv = (double *)logenv_arr->data;
        Py_XDECREF(logenv_arr); // convert_to_numpy does an incref for us, so we need
                                // to decref; we do it here to make sure it doesn't get
                                // forgotten later. this is safe because there's already
                                // a reference held by the calling python code, so the
                                // logenv_arr will never go out of scope while we're running.

        PyArrayObject* wiggle_arr = convert_to_numpy(PyList_GetItem(wiggles,i), "wiggle");
        conversion_numpy_check_type(wiggle_arr,PyArray_DOUBLE, "wiggle");
        double * wiggle =  (double *) wiggle_arr->data;
        Py_XDECREF(wiggle_arr);

        int len_logenv = logenv_arr->dimensions[0];
        int len_wiggle = wiggle_arr->dimensions[0];

        int wiggle_len = std::min(len_wiggle, len_logenv);

        int end_idx = start_idx + len_logenv;
        if (end_idx <= 0) {
            continue;
        }
        int early = std::max(0, - start_idx);
        int overshoot = std::max(0, end_idx - (window_start_idx+npts));

        int j = early;
        int total_wiggle_len = std::min(wiggle_len - early, len_logenv - overshoot - early);
        for(j=early; j < early + total_wiggle_len; ++j) {
            signal(j+start_idx) += exp(logenv[j]) * wiggle[j];
        }
        if (envelope) {
            for(; j < len_logenv - overshoot; ++j) {
               signal(j + start_idx) += exp(logenv[j]);
            }
        }
    }

"""
        weave.inline(code,['n', 'window_start_idx', 'npts', 'sidxs', 'logenvs', 'wiggles', 'signal', 'envelope'],type_converters = converters.blitz,verbose=2,compiler='gcc')

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
                self._tmpl_params[(eid,phase)][p] = float(v)
                self._keymap[k] = (True, eid, phase, p)

            wg = self.graph.wiggle_generator(phase=phase, srate=self.srate)
            self._wiggle_params[(eid, phase)] = np.empty((wg.dimension(),))
            for (i, p) in enumerate(wg.params()):
                try:
                    k, v = self.get_parent_value(eid, phase, p, pv, return_key=True)
                    self._wiggle_params[(eid,phase)][i] = float(v)
                    self._keymap[k] = (False, eid, phase, i)
                except KeyError:
                    #print "WARNING: no wiggles for arrival (%d, %s) at (%s, %s, %s)" % (eid, phase, self.sta, self.band, self.chan)
                    k = None

        for k in parent_keys_removed:
            try:
                tmpl, eid, phase, p = self._keymap[k]
                del self._keymap[k]
                if tmpl:
                    del self._tmpl_params[(eid, phase)]
                else:
                    del self._wiggle_params[(eid, phase)]
            except KeyError:
                pass

        for (key, node) in parent_keys_changed:
            try:
                tmpl, eid, phase, p = self._keymap[key]
                if tmpl:
                    self._tmpl_params[(eid,phase)][p] = float(pv[key])
                else:
                    self._wiggle_params[(eid,phase)][p] = float(pv[key])
            except KeyError:
                continue

        del parent_keys_removed
        del parent_keys_changed
        del parent_nodes_added
        return pv

    def arrivals(self):
        self._parent_values()
        return self._arrivals

    def parent_predict(self, parent_values=None, **kwargs):
        #parent_values = parent_values if parent_values else self._parent_values()
        signal = self.assem_signal(**kwargs)
        noise = self.nm.predict(n=len(signal))
        self.set_value(ma.masked_array(data=signal + noise, mask=self.get_value().mask, copy=False))
        for child in self.children:
            child.parent_keys_changed.add(self.single_key)

    def parent_sample(self, parent_values=None):
        signal = self.assem_signal()
        noise = self.nm.sample(n=len(signal))
        self.set_value(ma.masked_array(data=signal + noise, mask=self.get_value().mask, copy=False))
        for child in self.children:
            child.parent_keys_changed.add((self.single_key), self)

    def log_p(self, parent_values=None, **kwargs):
        parent_values = parent_values if parent_values else self._parent_values()
        v = self.get_value()
        value = v.data
        mask = v.mask

        pred_signal = self.assem_signal(**kwargs)
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
        try:
            is_tmpl, eid, phase, p = self._keymap[parent_key]
        except KeyError:
            # if this key doesn't affect signals at this node
            return 0.0
        if is_tmpl:
            self._tmpl_params[(eid, phase)][p] += eps
        else:
            self._wiggle_params[(eid, phase)][p] += eps
        deriv = ( self.log_p() - lp0 ) / eps
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
        parent_values = parent_values if parent_values else self._parent_values()
        tg = self.graph.template_generator(phase)
        return self._tmpl_params[(eid, phase)], tg

    def cache_latent_signal_for_template_optimization(self, eid, phase, force_bounds=True):

        def window_logp(w):
            start_idx, end_idx = w

            if self._cache_latent_signal_arrival != (eid, phase):
                raise ValueError("inconsistent state in signal cache at node %s: trying to compute logp for template %s, but cache is for different template %s!" % (self.label, (eid, phase), self._cache_latent_signal_arrival))


            self._parent_values()

            # check to make sure the template is actually contained within the specified window
            if force_bounds:
                t_start, t_end = self.template_idx_window(eid, phase, pre_arrival_slack_s=0, post_fade_slack_s=0)
                if t_start < start_idx or t_end > end_idx:
                    print "WARNING: template indices %s are out of bounds for cached indices %s" % ((t_start, t_end), w)
                    return np.float('-inf')

            v = self._cached_latent_signal
            value = v.data
            mask = v.mask
            pred_signal = self.assem_signal(arrivals=((eid, phase),), window_start_idx = start_idx, npts=end_idx-start_idx)
            signal_diff = self.signal_diff
            code = """
    for(int i=start_idx; i < end_idx; ++i) {
    signal_diff(i) =value(i) - pred_signal(i);
    }
    """
            weave.inline(code,['signal_diff', 'value', 'pred_signal', 'start_idx', 'end_idx'],type_converters = converters.blitz,verbose=2,compiler='gcc')

            lp = self.nm.log_p(signal_diff[start_idx:end_idx], mask=mask[start_idx:end_idx] if isinstance(mask,collections.Sequence) else mask)

            return lp

        def window_logp_deriv(w, parent_key, lp0=None, eps=1e-4):
            parent_values = self._parent_values()
            lp0 = lp0 if lp0 else window_logp(w)
            parent_values[parent_key] += eps
            try:
                is_tmpl, eid, phase, p = self._keymap[parent_key]
            except KeyError:
                # if this key doesn't affect signals at this node
                return 0.0
            if is_tmpl:
                self._tmpl_params[(eid, phase)][p] += eps
            else:
                self._wiggle_params[(eid, phase)][p] += eps
            deriv = ( window_logp(w) - lp0 ) / eps
            parent_values[parent_key] -= eps
            if is_tmpl:
                self._tmpl_params[(eid, phase)][p] -= eps
            else:
                self._wiggle_params[(eid, phase)][p] -= eps
            return deriv

        arrivals = self.arrivals()
        other_arrivals = [a for a in arrivals if a != (eid, phase)]
        self._cached_latent_signal = self.get_value() - self.assem_signal(arrivals=other_arrivals)
        self._cache_latent_signal_arrival = (eid, phase)
        return window_logp, window_logp_deriv

    def template_idx_window(self, eid=None, phase=None, vals=None, pre_arrival_slack_s = 10.0, post_fade_slack_s = 10.0):
        if vals is not None:
            v = vals
            tg = self.graph.template_generator(phase)
        else:
            v, tg = self.get_template_params_for_arrival(eid=eid, phase=phase)
        start_idx = int((v['arrival_time']  - self.st) * self.srate)
        window_start_idx = max(0, int(start_idx - pre_arrival_slack_s*self.srate))

        logenv_len = tg.abstract_logenv_length(v, srate=self.srate)
        window_end_idx = min(self.npts, int(start_idx + logenv_len + post_fade_slack_s*self.srate))

        return (window_start_idx, window_end_idx)

    def cache_latent_signal_for_fixed_window(self, eid, phase, force_bounds=True, **kwargs):
        window_lps = self.cache_latent_signal_for_template_optimization(eid, phase, force_bounds=force_bounds)
        return self.window_lps_to_proxy_lps(window_lps)

    def window_lps_to_proxy_lps(self, window_lps, w=None):

        if window_lps is None:
            return None

        if w is None:
            eid, phase = self._cache_latent_signal_arrival
            w = self.template_idx_window(eid, phase)

        lp, deriv_lp = window_lps
        lpw = lambda : lp(w)
        def deriv_lp_w(*args, **kwargs):
            dlpw =  deriv_lp(w, *args, **kwargs)
            #print "dlpw", w, dlpw, args, kwargs
            return dlpw
        proxy_lps = {self.label: (lpw, deriv_lp_w)}
        return proxy_lps
