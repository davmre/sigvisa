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
from sigvisa.models.noise.armodel.model import ARModel, ErrorModel
from sigvisa.graph.nodes import Node
from sigvisa.graph.graph_utils import get_parent_value, parse_key, create_key
from sigvisa.plotting.plot import subplot_waveform

MAX_LATENT_SIGNAL_LEN = 24000 # (10 minutes * 40hz)

class LatentArrivalNode(Node):

    def __init__(self, graph, eid, phase, sta, chan, band, srate, **kwargs):

        self.eid, self.phase, self.sta, self.chan, self.band, self.srate = eid, phase, sta, chan, band, srate

        key = create_key(param="latent_arrival", sta=self.sta, chan=self.chan, band=self.band, eid=self.eid, phase=self.phase)

        super(LatentArrivalNode, self).__init__(keys=[key,], label=key, **kwargs)

        # figure out what the keys ought to be for the template and
        # repeatable-wiggle parameters at this arrival.
        self.graph = graph
        self.tg = self.graph.template_generator(self.phase)
        self.wg = self.graph.wiggle_generator(self.phase, self.srate)
        self._tmpl_keys = dict()
        self._wiggle_keys = dict()
        self._keymap = dict()
        for param in self.tg.params() + ('arrival_time',):
            k = create_key(param=param, sta=self.sta,
                           chan=self.chan if param != 'arrival_time' else None,
                           band=self.band if param != 'arrival_time' else None,
                           eid=self.eid, phase=self.phase)
            self._tmpl_keys[param] = k
            self._keymap[k] = (True, param)

        for param in self.wg.params():
            k = create_key(param=param, sta=self.sta, chan=self.chan, band=self.band, eid=self.eid, phase=self.phase)
            self._wiggle_keys[param] = k
            self._keymap[k] = (False, param)
        self._tmpl_params = dict()
        self._wiggle_params = dict()

        self._tmpl_shape_cache = np.array(())
        self._repeatable_wiggle_cache = np.array(())
        self._empirical_wiggle_cache = np.array(())

        #parent_tmpls_tmp = [(k, graph.nodes_by_key[k]) for k in self._tmpl_keys.values()]
        #parent_wiggles_tmp = [(k, graph.nodes_by_key[k]) for k in self._wiggle_keys.values()]
        #self.parent_keys_changed.update(parent_tmpls_tmp, parent_wiggles_tmp)

        # TEMPORARY HACK until I actually figure out how to train the AR wiggle models
        em = ErrorModel(mean=0.0, std=0.01)
        self.arwm = ARModel(params=(.7,.2,), c=1.0, em = em, sf=self.srate)

    def parent_predict(self):
        self.set_value(self._predict_value())
        self._parent_values(force_update_wiggle=True)

    def _predict_value(self):
        self._parent_values()
        env = self._tmpl_shape_cache
        wiggle_repeatable = self._repeatable_wiggle_cache
        wiggle = self.arwm.predict(n=len(env))
        wiggle[:len(wiggle_repeatable)] += wiggle_repeatable[:len(env)]
        wiggle *= env
        return wiggle

    def parent_sample(self):
        self.set_value(self._sample_value())
        self._parent_values(force_update_wiggle=True)

    def _sample_value(self):
        self._parent_values()
        env = self._tmpl_shape_cache
        wiggle_repeatable = self._repeatable_wiggle_cache
        wiggle = self.arwm.sample(n=len(env))
        wiggle[:len(wiggle_repeatable)] += wiggle_repeatable[:len(env)]
        wiggle *= env
        return wiggle

    def compute_empirical_wiggle(self, env, tmpl_shape_env=None, repeatable_wiggle=None, start_idx=0):
        """
        Given:
           env: latent signal envelope, or a substring of the envelope
           tmpl_shape_env: the predicted template shape (default is to use the current cached value)
           repeatable_wiggle: the repeatable wiggle component (default is to use the current cached value)
           start_idx: index (wrt to the arrival time of this node's signal) at which to start extracting the wiggle.
                      We assume that env begins at this index.
        """
        # if we're predicting a signal that bottoms out much
        # earlier than our current signal, we won't be able to get
        # any useful information about the wiggles after the
        # bottoming-out point (since we're essentially dividing by
        # zero). so we just look at the time range where both
        # signals are well-defined.
        # similarly, if the predicted signal is longer than the
        # current signal we're representing, we still don't know
        # anything about the latent wiggles past the point where the
        # current signal ends.

        if tmpl_shape_env is None or repeatable_wiggle is None:
            self._parent_values()
        if tmpl_shape_env is None:
            tmpl_shape_env = self._tmpl_shape_cache
        if repeatable_wiggle is None:
            repeatable_wiggle = self._repeatable_wiggle_cache

        resolveable_latent_wiggle_len = min(len(env), len(tmpl_shape_env)-start_idx, MAX_LATENT_SIGNAL_LEN)
        end_idx = start_idx + resolveable_latent_wiggle_len

        empirical_wiggle = env[:resolveable_latent_wiggle_len] / tmpl_shape_env[start_idx:end_idx]
        empirical_wiggle[:len(repeatable_wiggle) - start_idx] -= repeatable_wiggle[start_idx:end_idx]

        return empirical_wiggle


    def get_signal_components(self):
        self._parent_values()
        return self._empirical_wiggle_cache, self._tmpl_shape_cache, self._repeatable_wiggle_cache

    def log_p(self):
        # compute the log probability of the current latent signal
        # hypothesis, conditioned on all the parent parameters
        # (template shape and repeatable wiggle).

        # a slightly tricky issue is that the current shape parameters
        # might suggest a signal length different from the current
        # latent value.
        self._parent_values()
        empirical_wiggle = self._empirical_wiggle_cache
        resolveable_latent_wiggle_len = len(empirical_wiggle)

        observed_wiggle_logp = self.arwm.log_p(empirical_wiggle)
        entropy = self.arwm.em.entropy + 0.01 # HACK to discourage long latent signals
        latent_wiggle_expected_logp =  entropy * (MAX_LATENT_SIGNAL_LEN - resolveable_latent_wiggle_len)
        #print "computing logp over resolveable len %d. observed logp %f, latent %f, total %f" % (resolveable_latent_wiggle_len, observed_wiggle_logp, latent_wiggle_expected_logp, observed_wiggle_logp + latent_wiggle_expected_logp)
        return observed_wiggle_logp + latent_wiggle_expected_logp

    def set_from_child_signal(self, start_idx=0):
        self._parent_values()
        latent_len = len(self.get_value())

        child_wn = list(self.children)[0]
        latent_start_idx = child_wn.arrival_start_idx(self.eid, self.phase)
        latent_end_idx = latent_start_idx + latent_len

        arrivals_without_me = child_wn.arrivals() - set(((self.eid, self.phase),))
        predicted_without_me = child_wn.assem_signal(arrivals=arrivals_without_me)
        observed_signal = child_wn.get_value()

        # indexes into the observed signal, to extract
        extract_start_idx = max(latent_start_idx, 0)
        extract_end_idx = min(latent_end_idx, child_wn.npts)

        # indexes into the latent signal: the destination for the extracted signal
        extract_start_offset = extract_start_idx - latent_start_idx
        extract_end_offset = extract_end_idx - latent_start_idx

        self._dict[self.single_key][extract_start_offset:extract_end_offset] = observed_signal[extract_start_idx:extract_end_idx] - predicted_without_me[extract_start_idx:extract_end_idx]
        self._parent_values(force_update_wiggle=True)

    def _parent_values(self, force_update_wiggle=False):
        parent_keys_changed = self.parent_keys_changed
        for node in self.parent_nodes_added:
            for key in node.keys():
                parent_keys_changed.update(((key, node),))

        pv = super(LatentArrivalNode, self)._parent_values()

        tmpl_changed = False
        wiggle_changed = False
        for (key, node) in parent_keys_changed:
            try:
                tmpl, p = self._keymap[key]
                if tmpl:
                    self._tmpl_params[p] = float(pv[key])
                    tmpl_changed = True
                else:
                    self._wiggle_params[p] = float(pv[key])
                    wiggle_changed = True
            except KeyError:
                continue

        if tmpl_changed:

            self._tmpl_shape_cache = np.exp(self._recompute_cached_logenv())

            # keep the node value at the same length as the predicted shape
            v = self.get_value()
            v_len = len(v) if v is not None else 0
            new_tmpl_len = len(self._tmpl_shape_cache)
            if v_len > new_tmpl_len:
                self._dict[self.single_key] = v[:new_tmpl_len]
            elif v_len < new_tmpl_len:
                new_v = np.zeros((new_tmpl_len,))
                new_v[:v_len] = v
                new_v[v_len:] = self._tmpl_shape_cache[v_len:]
                self._dict[self.single_key] = new_v[:MAX_LATENT_SIGNAL_LEN]
        if wiggle_changed or len(self._repeatable_wiggle_cache) == 0:
            self._repeatable_wiggle_cache = self._recompute_cached_wiggle()
        if tmpl_changed or wiggle_changed or force_update_wiggle:
            v = self.get_value()
            if v is None:
                v = self._predict_value()
                self._dict[self.single_key] = v
            self._empirical_wiggle_cache = self.compute_empirical_wiggle(v, self._tmpl_shape_cache, self._repeatable_wiggle_cache)


        del parent_keys_changed
        return pv

    def _recompute_cached_wiggle(self, parent_values=None):
        # assumes self._wiggle_params is up to date
        wg = self.graph.wiggle_generator(self.phase, self.srate)
        if len(self._wiggle_params) == wg.dimension():
            return wg.signal_from_features(features = self._wiggle_params)
        else:
            return np.ones((wg.npts,))

    def _recompute_cached_logenv(self):
        # assumes self._tmpl_params is up to date
        v  = self._tmpl_params
        offset = (v['arrival_time'] - int(np.floor(v['arrival_time']))) * self.srate
        offset = offset - int(np.floor(offset))
        logenv = self.tg.abstract_logenv_raw(v, idx_offset=offset, srate=self.srate)[:MAX_LATENT_SIGNAL_LEN]
        return logenv

    def get_template_params(self):
        self._parent_values()
        return self._tmpl_params

    def deriv_log_p(self, parent_key=None, lp0=None, eps=1e-4):
        self._parent_values()
        lp0 = lp0 if lp0 else self.log_p()
        try:
            is_tmpl, p = self._keymap[parent_key]
        except KeyError:
            # if this key doesn't affect signals at this node
            return 0.0
        if is_tmpl:
            self._tmpl_params[p] += eps
        else:
            self._wiggle_params[p] += eps
        deriv = ( self.log_p() - lp0 ) / eps
        if is_tmpl:
            self._tmpl_params[p] -= eps
        else:
            self._wiggle_params[p] -= eps
        return deriv


    def get_wave(self):
        return Waveform(data=self.get_value(), stime=self._tmpl_params['arrival_time'], sta=self.sta, srate=self.srate, chan=self.chan, filter_str='env;' + self.band)


    def debugging_plot(self, ax, plot_mode="full"):
        self._parent_values()

        if plot_mode == "wiggle":
            wiggle = self._empirical_wiggle_cache
            wiggle_wave = Waveform(data=wiggle, stime=self._tmpl_params['arrival_time'], sta=self.sta, srate=self.srate, chan=self.chan, filter_str='')
            subplot_waveform(wiggle_wave, ax, plot_dets=False, c='black')
            return

        subplot_waveform(self.get_wave(), ax, plot_dets=False, c='black')

        if plot_mode == "shape":
            shape_wave = Waveform(data=self._tmpl_shape_cache, stime=self._tmpl_params['arrival_time'], sta=self.sta, srate=self.srate, chan=self.chan, filter_str='env;' + self.band)
            subplot_waveform(shape_wave, ax, plot_dets=False, c='green')

        if plot_mode == "predict" or plot_mode=="full":
            env_wave = Waveform(data=self._predict_value(), stime=self._tmpl_params['arrival_time'], sta=self.sta, srate=self.srate, chan=self.chan, filter_str='env;' + self.band)
            subplot_waveform(env_wave, ax, plot_dets=False, c='red')
        if plot_mode == "sample" or plot_mode=="full":
            env_wave = Waveform(data=self._sample_value(), stime=self._tmpl_params['arrival_time'], sta=self.sta, srate=self.srate, chan=self.chan, filter_str='env;' + self.band)
            subplot_waveform(env_wave, ax, plot_dets=False, c='purple')



    ####################################################
# hack: redefining parent methods for efficiency

    def set_value(self, value, key=None):
        key = key if key else self.single_key
        if self._mutable[key]:
            self._dict[key] = value
            # we explicitly don't set parent_keys_changed and
            # recompute deterministic values for our children, because
            # the only children in practice will be
            # ObservedSignalNodes which don't expect these things.

        self._parent_values(force_update_wiggle=True)

    def set_dict(self, value, override_fixed=False):
        assert(set(value.iterkeys()) == set(self._mutable.iterkeys()))
        if override_fixed:
            self._dict = value
        else:
            self._dict = {k : value[k] if self._mutable[k] else self._dict[k] for k in value.iterkeys() }


##########################################################
#

    def mutable_dimension(self):
        v = self.get_value()
        return len(v)

    def get_mutable_values(self):
        return self.get_value()

    def set_mutable_values(self, values):
        current_value = self.get_value()
        assert(len(values) == len(current_value))
        self.set_value(np.asarray(values))

    def set_index(self, val, i):
        self._dict[self.single_key][i] = val
        self._parent_values(force_update_wiggle=True)

    def _debug_grad(self):
        self._parent_values()
        grad = np.zeros((len(self._empirical_wiggle_cache),))
        self.update_mutable_grad(grad, 0)
        return grad

    def low_bounds(self):
        v = self.get_value()
        return np.zeros(v.shape)

    def high_bounds(self):
        v = self.get_value()
        return np.ones(v.shape) * float('inf')

    def update_mutable_grad(self, grad, i, eps=None, initial_lp=None):
        self._parent_values()
        empirical_wiggle = self._empirical_wiggle_cache
        predicted_shape = self._tmpl_shape_cache
        repeatable_wiggle = self._repeatable_wiggle_cache

        mygrad = self.arwm.log_p_grad(empirical_wiggle) / predicted_shape

        latent_len = len(empirical_wiggle)

        child_wn = list(self.children)[0]
        latent_start_idx = child_wn.arrival_start_idx(self.eid, self.phase)
        latent_end_idx = latent_start_idx + latent_len

        predicted_signal = child_wn.assem_signal()
        observed_signal = child_wn.get_value()

        arnm = child_wn.nm

        # indexes into the observed signal, to extract
        extract_start_idx = max(latent_start_idx, 0)
        extract_end_idx = min(latent_end_idx, child_wn.npts)

        # indexes into the latent signal: the destination for the extracted signal
        extract_start_offset = extract_start_idx - latent_start_idx
        extract_end_offset = extract_end_idx - latent_start_idx

        ar_padding = min(extract_start_idx, arnm.p) + 2
        station_noise = observed_signal[extract_start_idx-ar_padding:extract_end_idx] - predicted_signal[extract_start_idx-ar_padding:extract_end_idx]

        # the gradient we compute wrt station noise is the negative of
        # the true gradient wrt latent signal, since an increase in
        # station noise corresponds to a *decrease* in latent signal,
        # and voice versa
        obs_neg_grad = arnm.log_p_grad(station_noise)[ar_padding:]
        mygrad[extract_start_offset:extract_end_offset] -= obs_neg_grad

        ni = len(mygrad)
        grad[i:i+ni] += mygrad
        return ni


    def set_nonrepeatable_wiggle(self, x, start_idx=None, set_signal_length=False):
        """
        If set_signal_length is True, the latent signal is modified to have length len(x).
        Otherwise, the latent signal retains its current length:
           - if len(x) < len(self.get_value()), then only the initial portion of the latent signal is modified.
           - if len(x) > len(self.get_value()), then the trailing portion of x is ignored.

        """

        self._parent_values()

        if start_idx is None:
            start_idx = 0

        N = len(x)

        shape_env = self._tmpl_shape_cache[start_idx:start_idx+N]
        repeatable_wiggle = self._repeatable_wiggle_cache[start_idx:start_idx+N]

        x[:len(repeatable_wiggle)] += repeatable_wiggle
        x[:len(shape_env)] *= shape_env

        if start_idx==0 and set_signal_length:
            # replace the entire signal with our new signal
            self._dict[self.single_key] = x[:MAX_LATENT_SIGNAL_LEN]
        else:
            self._dict[self.single_key][start_idx:start_idx+N] = x[:len(self._dict[self.single_key])-start_idx]
        self._parent_values(force_update_wiggle=True)
