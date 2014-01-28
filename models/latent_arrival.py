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
        em = ErrorModel(mean=1.0, std=1.0)
        self.arwm = ARModel(params=(.2, .7), em = em, sf=self.srate)

    def parent_predict(self):
        env = np.exp(self.get_template_logenv())
        wiggle = self.get_wiggle()
        env[:len(wiggle)] *= wiggle
        self.set_value(env)

    def parent_sample(self):
        env = np.exp(self.get_template_logenv())
        wiggle_repeatable = self.get_wiggle()
        wiggle = self.arwm.sample(n=len(env))
        wiggle[:len(wiggle_repeatable)] += wiggle_repeatable
        env *= wiggle
        self.set_value(env)

    def log_p(self):
        env = self.get_value()
        tmpl_shape_env = np.exp(self.get_template_logenv())
        env /= tmpl_shape_env

        wiggle_repeatable = self.get_wiggle()
        env[:len(wiggle_repeatable)] -= wiggle_repeatable

        return self.arwm.log_p(env)

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

    def set_mutable_values(self, values):
        assert(len(values) == self.mutable_dimension())
        for (i,k) in enumerate(self._mutable_keys):
            self._dict[k] = values[i]
