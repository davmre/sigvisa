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

        #parent_tmpls_tmp = [(k, graph.nodes_by_key[k]) for k in self._tmpl_keys.values()]
        #parent_wiggles_tmp = [(k, graph.nodes_by_key[k]) for k in self._wiggle_keys.values()]
        #self.parent_keys_changed.update(parent_tmpls_tmp, parent_wiggles_tmp)

        # TEMPORARY HACK until I actually figure out how to train the AR wiggle models
        em = ErrorModel(mean=0.0, std=0.1)
        self.arwm = ARModel(params=(.2, .7), c=1.0, em = em, sf=self.srate)

    def parent_predict(self):
        self.set_value(self._predict_value())

    def _predict_value(self):
        env = np.exp(self.get_template_logenv())
        wiggle_repeatable = self.get_wiggle()
        wiggle = self.arwm.predict(n=len(env))
        wiggle[:len(wiggle_repeatable)] += wiggle_repeatable[:len(env)]
        env *= wiggle
        return env

    def parent_sample(self):
        self.set_value(self._sample_value())

    def _sample_value(self):
        env = np.exp(self.get_template_logenv())
        wiggle_repeatable = self.get_wiggle()
        wiggle = self.arwm.sample(n=len(env))
        wiggle[:len(wiggle_repeatable)] += wiggle_repeatable[:len(env)]

        env *= wiggle
        return env

    def _empirical_wiggle(self, return_template_prediction=False):
        env = self.get_value()
        tmpl_shape_env = np.exp(self.get_template_logenv())

        resolveable_latent_wiggle_len = min(len(env), len(tmpl_shape_env), MAX_LATENT_SIGNAL_LEN)
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
        env = env[:resolveable_latent_wiggle_len]
        tmpl_shape_env = tmpl_shape_env[:resolveable_latent_wiggle_len]
        assert(len(env) == len(tmpl_shape_env))

        # avoid 0/0 issues for the first envelope value
        tmpl_shape_env[0] = 1
        try:
            empirical_wiggle = env / tmpl_shape_env
        except FloatingPointError:
            import pdb; pdb.set_trace()
        empirical_wiggle[0] = empirical_wiggle[1]

        wiggle_repeatable = self.get_wiggle()
        try:
            empirical_wiggle[:len(wiggle_repeatable)] -= wiggle_repeatable[:resolveable_latent_wiggle_len]
        except ValueError:
            import pdb; pdb.set_trace()

        if return_template_prediction:
            return (empirical_wiggle, tmpl_shape_env)
        else:
            return empirical_wiggle

    def log_p(self):
        # compute the log probability of the current latent signal
        # hypothesis, conditioned on all the parent parameters
        # (template shape and repeatable wiggle).

        # a slightly tricky issue is that the current shape parameters
        # might suggest a signal length different from the current
        # latent value.
        empirical_wiggle = self._empirical_wiggle()
        resolveable_latent_wiggle_len = len(empirical_wiggle)

        observed_wiggle_logp = self.arwm.log_p(empirical_wiggle)
        latent_wiggle_expected_logp = self.arwm.em.entropy * (MAX_LATENT_SIGNAL_LEN - resolveable_latent_wiggle_len)
        #print "computing logp over resolveable len %d. observed logp %f, altent %f, total %f" % (resolveable_latent_wiggle_len, observed_wiggle_logp, latent_wiggle_expected_logp, observed_wiggle_logp + latent_wiggle_expected_logp)
        return observed_wiggle_logp + latent_wiggle_expected_logp

    def set_from_child_signal(self, start_idx=0):
        current = self.get_value()
        if current is None:
            self.parent_predict()
            current = self.get_value()
        latent_len = len(current)

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


    def _parent_values(self):
        parent_keys_changed = self.parent_keys_changed
        for node in self.parent_nodes_added:
            for key in node.keys():
                parent_keys_changed.update(((key, node),))

        pv = super(LatentArrivalNode, self)._parent_values()

        for (key, node) in parent_keys_changed:
            try:
                tmpl, p = self._keymap[key]
                if tmpl:
                    self._tmpl_params[p] = float(pv[key])
                else:
                    self._wiggle_params[p] = float(pv[key])
            except KeyError:
                continue

        del parent_keys_changed
        return pv

    def get_wiggle(self, parent_values=None):
        parent_values = parent_values if parent_values else self._parent_values()
        wg = self.graph.wiggle_generator(self.phase, self.srate)
        if len(self._wiggle_params) == wg.dimension():
            return wg.signal_from_features(features = self._wiggle_params)
        else:
            return np.ones((wg.npts,))

    def get_template_logenv(self):
        v  = self.get_template_params()
        offset = (v['arrival_time'] - int(np.floor(v['arrival_time']))) * self.srate
        offset = offset - int(np.floor(offset))
        logenv = self.tg.abstract_logenv_raw(v, idx_offset=offset, srate=self.srate)
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

        subplot_waveform(self.get_wave(), ax, plot_dets=False, c='black')

        if plot_mode == "shape":
            shape_wave = Waveform(data=self.get_template_logenv(), stime=self._tmpl_params['arrival_time'], sta=self.sta, srate=self.srate, chan=self.chan, filter_str='env;' + self.band)
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

    def _debug_grad(self):
        empirical_wiggle = self._empirical_wiggle()
        grad = np.zeros((len(empirical_wiggle),))
        self.update_mutable_grad(grad, 0)
        return grad

    def low_bounds(self):
        v = self.get_value()
        return np.zeros(v.shape)

    def high_bounds(self):
        v = self.get_value()
        return np.ones(v.shape) * float('inf')

    def update_mutable_grad(self, grad, i, eps=None, initial_lp=None):
        (empirical_wiggle, predicted_shape) = self._empirical_wiggle(return_template_prediction=True)

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
