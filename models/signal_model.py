
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
        arrivals.add(extract_arrival_from_key(k, r))
    return arrivals

class ObservedSignalNode(Node):
    """

    Compute the probability of a set of segments (signal
    envelopes), given a set of events. This is done by either
    maximizing or integrating the probability over the template
    parameters describing the signals.

    """

    def __init__(self, model_waveform, graph, nm_type="ar", nmid=None, observed=True, **kwargs):

        key = create_key(param="signal_%.2f_%.2f" % (model_waveform['stime'], model_waveform['etime'] ), sta=model_waveform['sta'], chan=model_waveform['chan'], band=model_waveform['band'])

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

        self.signal_diff = np.empty((self.npts,))
        self.pred_signal = np.empty((self.npts,))

        self.set_noise_model(nm_type=nm_type, nmid=nmid)


        self._arrivals = set()
        self.r = re.compile("([-\d]+);(.+);(.+);(.+);(.+);(.+)")

        self._latent_arrivals = dict()
        self._arrival_times = dict()
        self._keymap = dict()

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

    def arrival_start_idx(self, eid, phase, skip_pv_call=False):
        if not skip_pv_call:
            self._parent_values()

        atime = self._arrival_times[(eid, phase)]
        start = (atime - self.st) * self.srate
        start_idx = int(np.floor(start))
        return start_idx


    def assem_signal(self, include_wiggles=True, arrivals=None):
        self._parent_values()

        # we allow specifying the list of parents in order to generate
        # signals with a subset of arriving phases (used e.g. in
        # wiggle extraction)
        if arrivals is None:
            arrivals = self.arrivals()

        arrivals = list(arrivals)
        n = len(arrivals)
        sidxs = np.empty((n,), dtype=int)
        latent_envs = [None] * n
        empty_array = np.reshape(np.array((), dtype=float), (0,))

        for (i, (eid, phase)) in enumerate(arrivals):
            sidxs[i] = self.arrival_start_idx(eid, phase, skip_pv_call=True)
            if sidxs[i] >= self.npts:
                logenvs[i] = empty_array
                continue

            latent_env = self.get_latent_arrival(eid, phase)

            latent_envs[i] = latent_env

        npts = self.npts
        n = len(arrivals)
        signal = self.pred_signal
        code = """
      for(int i=0; i < npts; ++i) signal(i) = 0;
    for (int i = 0; i < n; ++i) {
        int start_idx = sidxs(i);

        PyArrayObject* latent_env_arr = convert_to_numpy(PyList_GetItem(latent_envs,i), "latent_env");
        conversion_numpy_check_type(latent_env_arr,PyArray_DOUBLE, "latent_env");
        double * latent_env = (double *)latent_env_arr->data;
        Py_XDECREF(latent_env_arr); // convert_to_numpy does an incref for us, so we need
                                // to decref; we do it here to make sure it doesn't get
                                // forgotten later. this is safe because there's already
                                // a reference held by the calling python code, so the
                                // latent_env_arr will never go out of scope while we're running.

        int len_latent_env = latent_env_arr->dimensions[0];
        int end_idx = start_idx + len_latent_env;
        if (end_idx <= 0) {
            continue;
        }
        int early = std::max(0, - start_idx);
        int overshoot = std::max(0, end_idx - npts);
        int j = early;
        for(j=early; j < len_latent_env - overshoot; ++j) {
               signal(j + start_idx) += latent_env[j];
        }
    }

"""
        weave.inline(code,['n', 'npts', 'sidxs', 'latent_envs', 'signal',],type_converters = converters.blitz,verbose=2,compiler='gcc')

        if not np.isfinite(signal).all():
            raise ValueError("invalid (non-finite) signal generated for %s!" % self.mw)

        return signal


    """

    keep the following members in sync:
    - self._arrivals: set of (eid, phase) pairs
    - self._keymap: map from parent key to (boolean is_tmpl_param, eid, phase, param_name) tuple
    - self._tmpl_params: map from (eid, phase) to a dict of template params for that arrival
    - self._wiggle_params: map from (eid, phase) to a dict of repeatable wiggle params

    """

    def _parent_values(self):
        parent_keys_removed = self.parent_keys_removed
        parent_keys_changed = self.parent_keys_changed
        parent_nodes_added = self.parent_nodes_added

        pv = super(ObservedSignalNode, self)._parent_values()

        if parent_nodes_added:
            new_arrivals = get_new_arrivals(parent_nodes_added, self.r)
            self._arrivals.update(new_arrivals)
            for node in parent_nodes_added:
                if 'latent_arrival' in node.label:
                    self._latent_arrivals[(node.eid, node.phase)] = node
                if 'arrival_time' in node.label:
                    for key in node.keys():
                        self._keymap[key] = (node.eid, node.phase)
                        parent_keys_changed.update(((key, node),))


        if parent_keys_removed:
            removed_arrivals = get_removed_arrivals(parent_keys_removed, self.r)
            self._arrivals.difference_update(removed_arrivals)

        for (k, n) in parent_keys_changed:
            # we assume every parent change is an arrival time,
            # without checking. this should work because all parents
            # are either arrival times or latent arrival signals, but
            # the latent arrival nodes will never set
            # parent_keys_changed.
            self._arrival_times[(n.eid, n.phase)] = n.get_value(key=k)

        del parent_keys_removed
        del parent_keys_changed
        del parent_nodes_added
        return pv


    """
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
    """

    def arrivals(self):
        self._parent_values()
        return self._arrivals

    def get_latent_arrival(self, eid, phase):
        return self._latent_arrivals[(eid, phase)].get_value()


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
        if 'arrival_time' in parent_key:
            eid, phase = self._keymap[parent_key]
        else:
            raise NotImplementedError("I don't know how to compute a derivative at ObservedSignalNode with respect to key %s." % parent_key)
        self._arrival_times[(eid, phase)] += eps
        deriv = ( self.log_p(parent_values=parent_values) - lp0 ) / eps
        parent_values[parent_key] -= eps
        self._arrival_times[(eid, phase)] -= eps
        return deriv

    def get_parent_value(self, eid, phase, param_name, parent_values, **kwargs):
         return get_parent_value(eid=eid, phase=phase, sta=self.sta, chan=self.chan, band=self.band, param_name=param_name, parent_values=parent_values, **kwargs)
