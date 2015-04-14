import cPickle as pickle
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
import copy

from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.models.noise.noise_util import get_noise_model
from sigvisa.models.noise.noise_model import NoiseModel

#from sigvisa.models.statespace.transient import TransientCombinedSSM
#from sigvisa.models.statespace.ar import ARSSM
#from sigvisa.models.statespace.dummy import DummySSM
#from sigvisa.models.statespace.compact_support import CompactSupportSSM
from sigvisa.ssms_c import TransientCombinedSSM, ARSSM, CompactSupportSSM

from sigvisa.graph.dag import ParentConditionalNotDefined
from sigvisa.graph.nodes import Node
from sigvisa.graph.graph_utils import get_parent_value, create_key
from sigvisa.plotting.plot import subplot_waveform

import scipy.weave as weave
from scipy.weave import converters

TSSM_NOISE_PADDING=1e-6

def extract_arrival_from_key(k, r):
    m = r.match(k)
    if not m: raise ValueError("could not parse parent key %s" % k)
    eid = int(m.group(1))
    phase = m.group(2)
    return (eid, phase)

def get_new_arrivals(new_nodes, r):
    new_arrivals = set()
    for n in new_nodes:
        if "loc" in n.label or "mb" in n.label: continue
        for k in n.keys():
            new_arrivals.add(extract_arrival_from_key(k, r))
    return new_arrivals

def get_removed_arrivals(removed_keys, r):
    removed_arrivals = set()
    for k in removed_keys:
        try:
            removed_arrivals.add(extract_arrival_from_key(k, r))
        except ValueError:
            continue
    return removed_arrivals

def update_arrivals(parent_values):
    arrivals = set()
    r = re.compile("([-\d]+);(.+);(.+);(.+);(.+);(.+)")
    for k in parent_values.keys():
        if k=="prefix": continue
        if "lon" in k or "lat" in k or "depth" in k or "mb" in k: continue
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

    def __init__(self, model_waveform, graph, nm_type="ar", nmid=None, observed=True, wavelet_basis=None, wavelet_param_models=None, has_jointgp=False, **kwargs):

        key = create_key(param="signal_%.2f_%.2f" % (model_waveform['stime'], model_waveform['etime'] ), sta=model_waveform['sta'], chan=model_waveform['chan'], band=model_waveform['band'])

        # maintain the invariant that masked data is always NaN
        wave_data = model_waveform.data
        d = np.array(wave_data.data, dtype=np.float64)
        m = wave_data.mask
        if isinstance(m, np.ndarray):
            d[m] = np.nan
        wd = ma.masked_array(d, m)

        super(ObservedSignalNode, self).__init__(model=None, initial_value=wd, keys=[key,], fixed=observed, **kwargs)

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

        self.init_noise_model(nm_type=nm_type, nmid=nmid)

        self._tmpl_params = dict()
        self._ev_params = dict()
        self._keymap = dict()
        self._arrivals = set()
        self._arrival_phases = collections.defaultdict(set)
        self.r = re.compile("([-\d]+);(.+);(.+);(.+);(.+);(.+)")

        self.graph = graph

        self.arrival_ssms = dict()
        self.arrival_templates = dict()

        self.gps_by_phase = dict()

        self.iid_arssm = ARSSM(np.array((0,),  dtype=np.float), 1.0, 0.0, 0.0)
        self.tssm = TransientCombinedSSM([(self.noise_arssm, 0, self.npts, None),], TSSM_NOISE_PADDING)
        self.tssm_components =  [(None, None, None, 0, self.npts, "noise"),]

        self.wavelet_basis = wavelet_basis
        self.wavelet_param_models = wavelet_param_models

        self.has_jointgp = has_jointgp
        if self.has_jointgp:
            self.params_modeled_jointly = set()
            for phase in self.wavelet_param_models.keys():
                self.params_modeled_jointly = self.params_modeled_jointly | set(self.wavelet_param_models[phase])

        self.cached_logp = None
        self._coef_message_cache = None

        self.conditional_coef_distribution = False

    def __str__(self):
        try:
            s = "Signal node w/ model wave %s" % (self.mw)
        except AttributeError:
            s = "Uninitialized signal node"
        return s

    def set_value(self, value):
        assert(len(value) == len(self.get_value(self.single_key)))

        # represent missing data as NaN for easy processing
        d = np.array(value.data, dtype=np.float64)
        m = value.mask
        if isinstance(m, np.ndarray):
            d[m] = np.nan
        v = ma.masked_array(d, m)
        self.cached_logp = None
        self._coef_message_cache = None
        super(ObservedSignalNode, self).set_value(value=v, key=self.single_key)

    def get_wave(self):
        return Waveform(data=self.get_value(), segment_stats=self.mw.segment_stats.copy(), my_stats=self.mw.my_stats.copy())

    def init_noise_model(self, nm_type="ar", nmid=None):
        if nmid is None:
            self.nm_type = nm_type
            self.prior_nm, self.prior_nmid, _ = get_noise_model(waveform=self.mw, model_type=self.nm_type, return_details=True)
        else:
            self.prior_nmid = nmid
            self.prior_nm = NoiseModel.load_by_nmid(Sigvisa().dbconn, self.prior_nmid)
            self.nm_type = self.prior_nm.noise_model_type()

        assert(self.nm_type=="ar")
        self.noise_arssm = ARSSM(np.array(self.prior_nm.params, dtype=np.float), self.prior_nm.em.std**2, 0.0, self.prior_nm.c)
        self.nm = self.prior_nm.copy()
        self.nmid = self.prior_nmid

        ten_seconds = int(self.srate * 10.0)
        d = self.get_value().data
        for i in np.arange(0, self.npts, ten_seconds):
            window_max_10s = np.max(d[i:i+ten_seconds])
            window_sum_10s = np.sum(d[i:i+ten_seconds])
            #if np.isfinite(window_sum_10s) and window_max_10s < self.nm.c:
            #    raise Exception("signal max of %.3f between %d,%d is less than noise mean %.3f! Noise model is probably wrong." % (window_max_10s, i, i+ten_seconds, self.nm.c))



    def set_noise_model(self, arm, nmid=None):
        self.nm.params = arm.params
        self.nm.c = arm.c
        self.nm.sf = arm.sf
        self.nm.p = arm.p
        self.nm.em = arm.em

        self.noise_arssm.set_process(arm.params, arm.em.std**2, 0.0, arm.c)
        self.nmid = nmid


    def arrival_start_idx(self, eid, phase, skip_pv_call=False):
        if not skip_pv_call:
            self._parent_values()

        atime = self._arrival_times[(eid, phase)]
        start = (atime - self.st) * self.srate
        start_idx = int(np.floor(start))
        return start_idx


    def assem_signal(self, arrivals=None, window_start_idx=0, npts=None):
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
        empty_array = np.reshape(np.array((), dtype=float), (0,))

        for (i, (eid, phase)) in enumerate(arrivals):
            v, tg = self.get_template_params_for_arrival(eid=eid, phase=phase)
            start = (v['arrival_time'] - self.st) * self.srate
            start_idx = int(np.floor(start))
            sidxs[i] = start_idx
            if start_idx >= self.npts:
                logenvs[i] = empty_array
                continue

            offset = float(start - start_idx)
            logenv = tg.abstract_logenv_raw(v, idx_offset=offset, srate=self.srate)
            logenvs[i] = logenv


        npts = self.npts-window_start_idx if not npts else int(npts)
        n = len(arrivals)
        envelope = int(self.env)
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

        int len_logenv = logenv_arr->dimensions[0];

        int end_idx = start_idx + len_logenv;
        if (end_idx <= 0) {
            continue;
        }
        int early = std::max(0, - start_idx);
        int overshoot = std::max(0, end_idx - (window_start_idx+npts));

        int j = early;
        for(j=early; j < len_logenv - overshoot; ++j) {
               signal(j + start_idx) += exp(logenv[j]);
        }
    }

"""
        weave.inline(code,['n', 'window_start_idx', 'npts', 'sidxs', 'logenvs', 'signal', 'envelope'],type_converters = converters.blitz,verbose=2,compiler='gcc')

        if not np.isfinite(signal).all():
            raise ValueError("invalid (non-finite) signal generated for %s!" % self.mw)

        return signal


    """

    keep the following members in sync:
    - self._arrivals: set of (eid, phase) pairs
    - self._arrival_times: map from (eid, phase) to arrival times as floats
    - self._latent_arrivals: map from (eid, phase) to nodes representing the latent signal

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

        for eid, phase in new_arrivals:
            self._arrival_phases[eid].add(phase)
        for eid, phase in removed_arrivals:
            self._arrival_phases[eid].remove(phase)

        # cache the list of tmpl/wiggle param keys for the new arrivals
        for (eid, phase) in new_arrivals:
            tg = self.graph.template_generator(phase=phase)
            self._tmpl_params[(eid,phase)] = dict()
            for p in tg.params() + ('arrival_time',):
                k, v = self.get_parent_value(eid, phase, p, pv, return_key=True)
                self._tmpl_params[(eid,phase)][p] = float(v)
                self._keymap[k] = (True, eid, phase, p)

            if eid >= 0:
                self._ev_params[eid] = dict()
                for p in ("lon", "lat", "depth", "mb"):
                    k= "%d;%s" % (eid, p)
                    self._ev_params[eid][p] = pv[k]
                    self._keymap[k] = (False, eid, None, p)

        for k in parent_keys_removed:
            try:
                tmpl, eid, phase, p = self._keymap[k]
                del self._keymap[k]
                if tmpl:
                    del self._tmpl_params[(eid, phase)]
                else:
                    del self._ev_params[eid][p]

            except KeyError:
                pass

        evs_moved = set()
        for (key, node) in parent_keys_changed:
            try:
                tmpl, eid, phase, p = self._keymap[key]
                if tmpl:
                    self._tmpl_params[(eid,phase)][p] = float(pv[key])
                else:
                    self._ev_params[eid][p] = float(pv[key])
                    evs_moved.add(eid)
            except KeyError:
                continue


        # if we're recovering from an unpickling,
        # we'll need to regenerate all the SSMs.
        if not self.arrival_ssms:
            self.arrival_ssms = dict()
            for (eid, phase) in self._arrivals:
                new_arrivals.add((eid, phase))

        # recompute priors for new arrivals, or
        # arrivals whose event location has changed.
        for (eid, phase) in new_arrivals:
            self.arrival_ssms[(eid, phase)] = self.arrival_ssm(eid, phase)
        for (eid, phase) in removed_arrivals:
            del self.arrival_ssms[(eid, phase)]
        for eid in evs_moved:
            for phase in self._arrival_phases[eid]:
                self.arrival_ssms[(eid, phase)] = self.arrival_ssm(eid, phase)

        # if any arrival times or templates might have changed, recompute the tssm
        if len(new_arrivals) > 0 or len(removed_arrivals) > 0 or len(parent_keys_changed) > 0:
            self.tssm = self.transient_ssm(arrivals=self._arrivals, parent_values=pv)
            self.cached_logp = None
            self._coef_message_cache = None

        del parent_keys_removed
        del parent_keys_changed
        del parent_nodes_added
        return pv

    def pass_jointgp_messages(self, parent_values=None):
        parent_values = parent_values if parent_values else self._parent_values()
        if not self.has_jointgp:
            return

        try:
            self.tssm
        except AttributeError:
            self.tssm = self.transient_ssm(parent_values=parent_values)

        if self._coef_message_cache is None:

            prior_means, prior_vars = [], []
            for i, (eid, phase, scale, sidx, npts, component_type) in enumerate(self.tssm_components):
                if component_type != "wavelet": continue
                ssm = self.arrival_ssms[(eid, phase)]
                #pm, pv = ssm.get_coef_prior()

                pm, pv = zip(*[jgp.prior() for jgp in self.wavelet_param_models[phase]])
                pm, pv = np.asarray(pm, dtype=np.float64), np.asarray(pv, dtype=np.float64)
                ssm.set_coef_prior(pm, pv)
                #print "passing with vars", pv
                prior_means.append(pm)
                prior_vars.append(pv)

            d = self.get_value().data
            if len(prior_means) > 0:
                prior_means, prior_vars = np.concatenate(prior_means), np.concatenate(prior_vars)
                ell, marginals = self.tssm.all_filtered_cssm_coef_marginals(d)
                posterior_means, posterior_vars = zip(*marginals)
                posterior_means, posterior_vars = np.concatenate(posterior_means), np.concatenate(posterior_vars)
                assert( np.min(posterior_vars) >= 0 )
            else:
                ell = self.tssm.run_filter(d)
                prior_means, prior_vars = np.array(()), np.array(()),
                posterior_means, posterior_vars = np.array(()), np.array(())

            self._coef_message_cache = ell, prior_means, prior_vars, posterior_means, posterior_vars

        ell, prior_means, prior_vars, posterior_means, posterior_vars = self._coef_message_cache
        coef_idx = 0
        for i, (eid, phase, scale, sidx, npts, component_type) in enumerate(self.tssm_components):
            if component_type!="wavelet":continue
            evdict = self._ev_params[eid]
            n_coefs = len(self.wavelet_param_models[phase])
            for j in range(n_coefs):
                self.wavelet_param_models[phase][j].message_from_arrival(eid, evdict, prior_means[coef_idx+j], prior_vars[coef_idx+j], posterior_means[coef_idx+j], posterior_vars[coef_idx+j])
            coef_idx += n_coefs


    def upwards_message_normalizer(self, parent_values=None):
        parent_values = parent_values if parent_values else self._parent_values()
        assert(self.has_jointgp)

        if self._coef_message_cache is None:
            self.pass_jointgp_messages(parent_values=parent_values)

        if self.cached_logp is None:

            d = self.get_value().data
            ell, prior_means, prior_vars, posterior_means, posterior_vars = self._coef_message_cache

            #var_diff = prior_vars - posterior_vars
            #var_diff = np.where(var_diff > 1e-30, var_diff, np.ones(var_diff.shape)*1e-30)
            #c1 = np.sum(np.log(var_diff))
            #c2 = -np.sum(np.log(prior_vars))
            #c3 = np.sum(-.5*(posterior_means - prior_means)**2/(var_diff))
            #c4 = np.sum(-.5*np.log(2*np.pi * (var_diff)))
            #correction = c1+c2+c3+c4


            #correction2 = np.sum([ np.log(prior_vars[i]-posterior_vars[i]) - np.log(prior_vars[i]) +\
            #                       scipy.stats.norm(loc=posterior_means[i], \
            #                                        scale=np.sqrt(prior_vars[i]-posterior_vars[i])).logpdf(prior_means[i]) \
            #                       for i in range(len(prior_vars)) ])
            #print self.sta, ell, correction, ell-correction
            #assert(np.isfinite(correction))
            self.cached_logp = ell # - correction

        return self.cached_logp


    def arrival_ssm(self, eid, phase):

        if self.wavelet_basis is None:
            return None

        (start_idxs, end_idxs, identities, basis_prototypes, n_steps), iid_std, target_coef_var = self.wavelet_basis
        n_basis = len(start_idxs)

        prior_means = np.zeros((n_basis,))
        prior_vars = np.ones((n_basis,)) * target_coef_var

        if (eid, phase) in self.arrival_ssms:
            cssm = self.arrival_ssms[(eid, phase)]
            cssm.set_coef_prior(prior_means, prior_vars)
        else:
            cssm = CompactSupportSSM(start_idxs, end_idxs, identities, basis_prototypes, prior_means, prior_vars, 0.0, 0.0)
        return cssm

    def transient_ssm(self, arrivals=None, parent_values=None, save_components=True):

        # we allow specifying the list of parents in order to generate
        # signals with a subset of arriving phases (used e.g. in
        # wiggle extraction)
        if arrivals is None:
            arrivals = self.arrivals()

        arrivals = list(arrivals)
        n = len(arrivals)
        sidxs = np.empty((n,), dtype=int)
        envs = [None] * n
        min_logenv = max(-7.0, np.log(self.nm.c)-2)

        if self.wavelet_basis is not None:
            (start_idxs, end_idxs, identities, basis_prototypes, n_steps), iid_std, target_coef_var = self.wavelet_basis
            n_basis = len(start_idxs)
        else:
            n_basis = 0
            iid_std = np.ones((int(120*self.srate),))*0.5

        components = [(self.noise_arssm, 0, self.npts, None)]

        tssm_components = [(None, None, None, 0, self.npts, "noise"),]

        # TODO: can be smarter about this, and only regenerate the TSSM when arrival_time changes.
        # Any other template param change can be implemented by just updating the scale vector in
        # the current TSSM to the new envelope.
        for (i, (eid, phase)) in enumerate(arrivals):
            v, tg = self.get_template_params_for_arrival(eid=eid, phase=phase, parent_values=parent_values)
            start = (v['arrival_time'] - self.st) * self.srate
            start_idx = int(np.floor(start))
            sidxs[i] = start_idx
            if start_idx >= self.npts:
                continue

            offset = float(start - start_idx)
            env = np.exp(tg.abstract_logenv_raw(v, idx_offset=offset, srate=self.srate, min_logenv=min_logenv))
            if start_idx + len(env) < 0:
                continue

            wssm = self.arrival_ssms[(eid, phase)]
            mn_len = min(len(iid_std), len(env))
            mn_scale = env[:mn_len]*iid_std[:mn_len]
            if wssm is not None:
                npts = min(len(env), n_steps*2)
                components.append((wssm, start_idx, npts, env))
                tssm_components.append((eid, phase, env, start_idx, npts, "wavelet"))
                components.append((self.iid_arssm, start_idx, mn_len, mn_scale))
                tssm_components.append((eid, phase, mn_scale, start_idx, mn_len, "multnoise"))
            else:
                components.append((self.iid_arssm, start_idx, mn_len, mn_scale))
                tssm_components.append((eid, phase, mn_scale, start_idx, mn_len, "multnoise"))

            components.append((None, start_idx, len(env), env))
            tssm_components.append((eid, phase, env, start_idx, len(env), "template"))

        if save_components:
            self.tssm_components=tssm_components
        return TransientCombinedSSM(components, TSSM_NOISE_PADDING)

    def arrivals(self):
        self._parent_values()
        return self._arrivals

    def indep_block_bounds(self, target_eid, target_phase):
        """
        Return the start and end indices of an *independent signal block*
        containing the given arrival. This is defined as a block of signal such that
        the probability of arrivals outside the block are independent of the probability
        of arrivals inside the block, given the observed signal itself.
        In practice what this means is that the block is bookended by sections of noise,
        each long enough to be fully observed.
        """

        pv = self._parent_values()

        """ First get a list of non-noise blocks, and extract the block corresponding to the given event. """
        blocks = [(sidx, sidx+npts, eid, phase) for (eid, phase, scale, sidx, npts, label) in self.tssm_components if label=="template"]
        target_sidx = None
        for (sidx, eidx, eid, phase) in blocks:
            if eid==target_eid and phase==target_phase:
                target_sidx = sidx
                target_eidx = eidx
                break
        assert(target_sidx is not None)

        #Start with the hypothesis that the p steps after target_eidx are clear.
        p = self.nm.p
        block_eidx = target_eidx+p

        # then loop through all blocks ending after target_eidx,
        # (these are the only ones that could interfere with the training noise)
        # in order of their start time
        sorted_forward = sorted([(sidx, eidx) for (sidx, eidx, eid, phase) in blocks if eidx > target_eidx])
        for (sidx, eidx) in sorted_forward:
            if sidx > block_eidx:
                # at this point all remaining blocks will start after
                # our current safe period, so we are done.
                break
            if eidx > block_eidx-p:
                # if this block ends after the start of the safe
                # period (and by loop invariant, starts *before* the
                # end of the safe period), we need to extend the safe
                # period.
                block_eidx = eidx+p
        block_eidx = min(block_eidx,self.npts)

        # now do the reverse procedure to understand the beginning of the signal
        sorted_backward = sorted([(sidx, eidx) for (sidx, eidx, eid, phase) in blocks if sidx < target_sidx], reverse=True)
        block_sidx = target_sidx-p
        for (sidx, eidx) in sorted_backward:
            if eidx < block_sidx:
                break
            if sidx > block_sidx:
                block_sidx = sidx-p
        block_sidx = max(block_sidx,0)

        return (block_sidx, block_eidx)

    def parent_predict(self, parent_values=None, **kwargs):
        parent_values = parent_values if parent_values else self._parent_values()
        self._set_cssm_priors_from_model(parent_values=parent_values)

        v = self.tssm.mean_obs(self.npts)
        self.set_value(ma.masked_array(data=v, mask=self.get_value().mask, copy=False))
        for child in self.children:
            child.parent_keys_changed.add(self.single_key)

    def parent_sample(self, parent_values=None):
        parent_values = parent_values if parent_values else self._parent_values()
        self._set_cssm_priors_from_model(parent_values=parent_values)
        seed = np.random.randint(2**31)
        print "sampling w/ seed", seed
        v = self.tssm.prior_sample(self.npts, seed)
        self.set_value(ma.masked_array(data=v, mask=self.get_value().mask, copy=False))
        for child in self.children:
            child.parent_keys_changed.add((self.single_key), self)


    def conditional_log_p(self, parent_values=None):
        parent_values = parent_values if parent_values else self._parent_values()
        assert(self.has_jointgp)

        for (eid, phase) in self.arrivals():
            evdict = self._ev_params[eid]
            cssm = self.arrival_ssms[(eid, phase)]
            prior_means, prior_vars = zip(*[jgp.posterior(eid) for jgp in self.wavelet_param_models[phase]])
            prior_means = np.asarray(prior_means)
            prior_vars = np.asarray(prior_vars)
            #print "setting", pm, pv
            cssm.set_coef_prior(prior_means, prior_vars)

        d = self.get_value().data
        lp = self.tssm.run_filter(d)
        return lp

    def _set_cssm_priors_from_model(self, arrivals=None, parent_values=None):
        parent_values = parent_values if parent_values else self._parent_values()
        arrivals = arrivals if arrivals is not None else self.arrivals()

        for (eid, phase) in arrivals:
            evdict = self._ev_params[eid]
            cssm = self.arrival_ssms[(eid, phase)]
            if cssm is None: continue
            prior_means = np.array([gp.predict(cond=evdict) for gp in self.wavelet_param_models[phase]])
            prior_vars = np.array([gp.variance(cond=evdict) for gp in self.wavelet_param_models[phase]])
            cssm.set_coef_prior(prior_means, prior_vars)


    def log_p(self, parent_values=None, arrivals=None,  **kwargs):
        parent_values = parent_values if parent_values else self._parent_values()

        if self.has_jointgp:
            raise ParentConditionalNotDefined()

        if arrivals is None and self.cached_logp is not None:
            return self.cached_logp

        if arrivals is None:
            tssm = self.tssm
        else:
            tssm = self.transient_ssm(arrivals=arrivals, save_components=False)

        self._set_cssm_priors_from_model(arrivals=arrivals, parent_values=parent_values)

        d = self.get_value().data
        t0 = time.time()
        lp = tssm.run_filter(d)
        t1 = time.time()

        #print "logp", lp, "for", self.sta, "signal npts", self.npts, "arrivals", len(self.arrivals()), "in", t1-t0, "max dimension", self.tssm.max_dimension()

        if arrivals is None:
            self.cached_logp = lp

        return lp

    def ___log_p_old(self, parent_values=None, return_grad=False, **kwargs):
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

        if return_grad:
            lp, grad = self.nm.argrad(signal_diff)
            return lp, grad, pred_signal, signal_diff.copy()
        else:
            lp = self.nm.log_p(signal_diff, mask=mask)
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
        if not is_tmpl:
            raise Exception("don't know how to take signal probability derivatives wrt non-template parameters!")
        cache = self.cached_logp
        self.cached_logp = None
        old_tssm = self.tssm
        self._tmpl_params[(eid, phase)][p] += eps
        self.tssm = self.transient_ssm()
        deriv = ( self.log_p() - lp0 ) / eps
        parent_values[parent_key] -= eps
        self._tmpl_params[(eid, phase)][p] -= eps
        self.tssm = old_tssm
        self.cached_logp = cache
        return deriv

    def get_parent_value(self, eid, phase, param_name, parent_values, **kwargs):
         return get_parent_value(eid=eid, phase=phase, sta=self.sta, chan=self.chan, band=self.band, param_name=param_name, parent_values=parent_values, **kwargs)

    def get_template_params_for_arrival(self, eid, phase, parent_values=None):
        parent_values = parent_values if parent_values else self._parent_values()
        tg = self.graph.template_generator(phase)
        return self._tmpl_params[(eid, phase)], tg

    def signal_component_means(self, return_stds_instead=False):
        self.tssm = self.transient_ssm()

        d = self.get_value().data

        if return_stds_instead:
            means = [np.sqrt(z) for z in self.tssm.component_vars(d)]
        else:
            means = self.tssm.component_means(d)
        noise_mean = means[0]
        signal_mean = np.zeros((self.npts,))
        arrival_info = collections.defaultdict(dict)

        for mean, (eid, phase, scale, sidx, npts, component_type) in zip(means[1:], self.tssm_components[1:]):
            if scale is not None:
                arrival_info[(eid, phase)][component_type] = mean*scale[:len(mean)]
            else:
                arrival_info[(eid, phase)][component_type] = mean
            arrival_info[(eid, phase)]["stime"] = sidx/self.srate + self.st

            start_idx = max(sidx, 0)
            end_idx = min(sidx+npts, self.npts)

            src = arrival_info[(eid, phase)][component_type]
            signal_mean[start_idx:end_idx] += src[start_idx-sidx:end_idx-sidx]

        for k in arrival_info.keys():

            signals = [v for (kk,v) in arrival_info[k].items() if kk !="stime"]
            l = np.max([len(s) for s in signals])
            combined = np.zeros((l,))
            for s in signals:
                combined[:len(s)] += s
            arrival_info[k]["combined"] = combined


        arrival_info["noise"] = noise_mean
        arrival_info["signal"] = signal_mean

        return arrival_info


    def cache_latent_signal_for_template_optimization(self, eid, phase, force_bounds=True, return_llgrad=False):

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

        def window_logp_deriv_caching(w, parent_key, lp0=None, eps=1e-4):
            parent_values = self._parent_values()
            vals, tg = self.get_template_params_for_arrival(eid=eid, phase=phase)

            if vals == self._cached_window_logp_grad_vals:
                grad = self._cached_window_logp_grad
            else:
                ll, grad = window_logp_llgrad(w)
                self._cached_window_logp_grad_vals = copy.copy(vals)
                self._cached_window_logp_grad = grad

            if "arrival_time" in parent_key:
                r = grad[0]
            elif "peak_offset" in parent_key:
                r = grad[1]
            elif "coda_height" in parent_key:
                r = grad[2]
            elif "peak_decay" in parent_key:
                r = grad[3]
            elif "coda_decay" in parent_key:
                r = grad[4]
            else:
                raise Exception("don't know how to compute gradient wrt key %s" % parent_key)

            #r2 = window_logp_deriv(w, parent_key, lp0=lp0, eps=1e-6)
            #if "arrival_time" not in parent_key and np.abs(r) < 200 and np.abs(r-r2) > 0.1:
            #    import pdb; pdb.set_trace()
            return r

        def window_logp_llgrad(w):
            try:
                start_idx, end_idx = w

                # get tg from somewhere
                vals, tg = self.get_template_params_for_arrival(eid=eid, phase=phase)
                start = (vals['arrival_time'] - self.st) * self.srate
                tmpl_start_idx = int(np.floor(start))
                offset = float(start - tmpl_start_idx)
                pred_logenv, jacobian = tg.abstract_logenv_raw(vals, idx_offset=offset, srate=self.srate, return_jac_exp=True)
                pred_env = np.exp(pred_logenv)

                window_len = end_idx-start_idx
                pred_signal = self.pred_signal
                pred_signal[start_idx:end_idx]=0
                tmpl_start_idx_rel = tmpl_start_idx-start_idx
                tmpl_end_idx = tmpl_start_idx + len(pred_env)
                tmpl_end_idx_rel = tmpl_start_idx_rel + len(pred_env)
                early = max(0, -tmpl_start_idx_rel)
                overshoot = max(0, tmpl_end_idx_rel - window_len)
                if tmpl_end_idx-overshoot > early + tmpl_start_idx:
                    pred_signal[early + tmpl_start_idx:tmpl_end_idx-overshoot] = pred_env[early:len(pred_env)-overshoot]

                v = self._cached_latent_signal
                value = v.data
                mask = v.mask
                signal_diff = self.signal_diff
                code = """
                for(int i=start_idx; i < end_idx; ++i) {
                  signal_diff(i) =value(i) - pred_signal(i);
                }
                """
                weave.inline(code,['signal_diff', 'value', 'pred_signal', 'start_idx', 'end_idx'],type_converters = converters.blitz,verbose=2,compiler='gcc')

                lp, grad = self.nm.argrad(signal_diff[start_idx:end_idx])
                shifted_jacobian = np.zeros((window_len, 5))
                if tmpl_end_idx-overshoot > early + tmpl_start_idx:
                    shifted_jacobian[early + tmpl_start_idx_rel:tmpl_end_idx_rel-overshoot, :] = jacobian[early:len(pred_env)-overshoot,:]
                param_grad = np.dot(grad.reshape((1, -1)), shifted_jacobian).flatten() * -1


            except Exception as e:
                import pdb; pdb.set_trace()


            if np.isnan(param_grad).any():
                import pdb; pdb.set_trace()
            return lp, param_grad

        self._cached_latent_signal = self.unexplained_signal(eid, phase)
        self._cache_latent_signal_arrival = (eid, phase)
        self._cached_window_logp_grad_vals = None
        if return_llgrad:
            return window_logp, window_logp_deriv_caching, window_logp_llgrad
        else:
            return window_logp, window_logp_deriv_caching

    def unexplained_signal(self, eid, phase):
        arrivals = self.arrivals()
        other_arrivals = [a for a in arrivals if a != (eid, phase)]
        return self.get_value() - self.assem_signal(arrivals=other_arrivals)

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

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['tssm']
        del d['arrival_ssms']
        del d['noise_arssm']
        del d['iid_arssm']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        #if "uatemplate_wiggle_var" not in d:
        #    self.uatemplate_wiggle_var = 1.0
        #    self.graph.uatemplate_wiggle_var = self.uatemplate_wiggle_var

        self.noise_arssm = ARSSM(np.array(self.nm.params, dtype=np.float), self.nm.em.std**2, 0.0, self.nm.c)
        self.iid_arssm = ARSSM(np.array((0,),  dtype=np.float), 1.0, 0.0, 0.0)
        self.cached_logp = None
        self._coef_message_cache = None
        # don't try to regenerate other SSMs here because we might still be in
        # the middle of the unpickling process and can't depend on other program
        # components (e.g. self.graph.template_generator()) being functional.
        # We'll do this instead when parent_values is called.
        self.arrival_ssms = dict()
