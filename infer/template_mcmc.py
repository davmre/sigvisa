import numpy as np
import numpy.ma as ma
import sys
import os
import traceback
import pickle
import copy
import time

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa import Sigvisa
from sigvisa.utils.math import safe_log, safe_log_vec
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_event_station_chan
from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.models.distributions import Gaussian, PiecewiseLinear
from sigvisa.models.signal_model import extract_arrival_from_key, unify_windows
from sigvisa.models.noise.armodel.model import ARGradientException
from sigvisa.models.ttime import tt_predict
from sigvisa.infer.mcmc_basic import gaussian_propose, gaussian_MH_move, MH_accept, hmc_step, hmc_step_reversing, mh_accept_util
from sigvisa.graph.graph_utils import create_key,parse_key
from sigvisa.graph.dag import get_relevant_nodes
from sigvisa.plotting.plot import savefig, plot_with_fit, plot_waveform
from sigvisa.utils.counter import Counter
from sigvisa.utils.BayesianRidgeWithCov import BayesianRidgeWithCov
from matplotlib.figure import Figure

import scipy.weave as weave
from scipy.weave import converters

import numdifftools as nd

MERGE_PROPOSAL_WIDTH_S=10.0

class InvalidParamException(Exception):
    pass

def node_get_value(nodes, param):
    try:
        k, n = nodes[param]
    except TypeError as e:
        n = nodes[param]
        k = n.single_key
    return n.get_value(key=k)

def node_set_value(nodes, param, value):
    try:
        k, n = nodes[param]
    except TypeError as e:
        n = nodes[param]
        k = n.single_key
    n.set_value(key=k, value=value)




def get_env_based_amplitude_distribution2(sg, wn, prior_min, prior_max, prior_dist, tmvals, exclude_arrs=None):

    # propose from a linearly interpolated version of the "posterior" density,
    # using the noise-model log probability of envelope discrepancy as the 
    # approximate posterior. 
    # (that is, ignoring mean and variance from the wiggle model)

    atime = tmvals['arrival_time']
    peak_time = tmvals['arrival_time'] + np.exp(tmvals['peak_offset'])

    if exclude_arrs is None:
        pred_env = wn.assem_env()
        unexplained = wn.get_env().data - pred_env
    else:
        eid, phase = exclude_arrs[0]
        unexplained = wn.unexplained_env(eid, phase, addl_arrs=exclude_arrs[1:])

    peak_idx = int((peak_time - wn.st) * wn.srate)

    start_idx_true = int((atime - wn.st) * wn.srate)
    end_idx_true = int(peak_idx + 60*wn.srate)
    start_idx = max(0, start_idx_true)
    end_idx = min(wn.npts, end_idx_true)
    start_offset = start_idx - start_idx_true
    if end_idx-start_idx < wn.srate:
        # if less than 1s of available signal, don't even bother
        return None

    unexplained_local = unexplained[start_idx:end_idx]
    n = len(unexplained_local)

    try:
        peak_height = float(unexplained[peak_idx])
        env_height = max(peak_height - wn.nm_env.c, wn.nm_env.c/1000.0)
    except IndexError:
        # if the peak idx is past the end of the signal,
        # then the amplitude posterior will essentially 
        # just be the prior, so none of this really matters.
        # Let's just set a default value to avoid crashing.
        env_height = 1e-3

    data_min = np.log(env_height) - 2
    data_max = np.log(env_height) + 2
    prior_min = min(prior_min, 1)
    prior_max = max(prior_max, prior_min+1, -2)
    if np.isfinite(data_min):
        min_c = min(data_min, prior_min)
        max_c = max(data_max, prior_max)
        candidates = np.linspace(max(min_c, -5), min(max_c, 5), 20)
        candidates = np.array(sorted(list(candidates) + [np.log(env_height), np.log(env_height+wn.nm_env.c)]))
    else:
        candidates = np.linspace(max(prior_min, -4),  min(prior_max, 5), 20)
        

    provided_coda_height = tmvals['coda_height'] if "coda_height" in tmvals else None
    tg = sg.template_generator("P")
    lps = []
    def proxylp(candidate):
        tmvals['coda_height'] = candidate
        l = tg.abstract_logenv_raw(tmvals, srate=wn.srate, fixedlen=n+start_offset)
        diff = unexplained_local - np.exp(l[start_offset:])
        return wn.nm_env.log_p(diff) + prior_dist.log_p(candidate)

    lps = np.array([proxylp(candidate) for candidate in candidates])

    def bad_indices(lps):
        best_idx = np.argmax(lps)
        best_lp = np.max(lps)
        lp_diff = np.abs(np.diff(lps))

        # an lp is "significant" if it or its neighbor is above the threshold
        thresh = best_lp - 3
        significant_lps = ( lps[:-1] > thresh ) +  ( lps[1:] > thresh ) 

        # a "bad step" is where we have a sharp boundary next to a significant lp.
        # that is, the significant lps are the areas where it's important to
        # approximate the posterior well, and a large difference in lp between adjacent
        # candidates means we're not doing that. 
        badsteps = significant_lps * (lp_diff > 1)
        bad_idxs = np.arange(len(lps)-1)[badsteps]

        # also check lps at the edges. if either of these are bad, we'll need to expand the domain.
        if lps[0] > thresh:
            bad_idxs = np.concatenate((bad_idxs, (-1,)))
        if lps[-1] > thresh:
            bad_idxs = np.concatenate((bad_idxs, (len(lps),)))

        return bad_idxs

    bad_idxs = bad_indices(lps)
    while len(bad_idxs) > 0:
        new_candidates = []
        new_lps = []
        for idx in bad_idxs:
            if idx == -1:
                incr = max(0.05, candidates[1] - candidates[0])
                c = candidates[0] - incr
            elif idx == len(lps):
                incr = max(0.05, candidates[-1] - candidates[-2])
                c = candidates[-1] + incr
            else:
                c1 = candidates[idx]
                c2 = candidates[idx+1]
                c = c1 + (c2-c1)/2.0
            new_candidates.append(c)
            new_lps.append( proxylp(c))
        full_c = np.concatenate((candidates, new_candidates))
        full_lps = np.concatenate((lps, new_lps))
        perm = sorted(np.arange(len(full_c)), key = lambda i : full_c[i])
        candidates = np.array(full_c[perm])
        lps = np.array(full_lps[perm])
        bad_idxs = bad_indices(lps)

    assert( (np.diff(candidates) > 0).all() )

    if provided_coda_height is not None:
        tmvals['coda_height'] = provided_coda_height


    p = PiecewiseLinear(candidates, np.array(lps), mix_dist=prior_dist, mix_weight=0.001)


    return p
    

def get_env_based_amplitude_distribution(sg, wn, tmvals=None, peak_time=None, peak_period_s = 2.0, exclude_arr=None):

    if peak_time is None:
        peak_time = tmvals['arrival_time'] + np.exp(tmvals['peak_offset'])

    if exclude_arr is None:
        pred_env = wn.assem_env()
        unexplained = wn.get_env().data - pred_env
    else:
        eid, phase = exclude_arr
        unexplained = wn.unexplained_env(eid, phase)




    peak_idx = int((peak_time - wn.st) * wn.srate)
    peak_period_samples = int(peak_period_s * wn.srate)
    peak_data=unexplained[peak_idx - peak_period_samples:peak_idx + peak_period_samples]

    # if we land outside of the signal window, or during an unobserved (masked) portion,
    # we'll just sample from the event-conditional prior instead
    if ma.count(peak_data) == 0:
        return None

    #peak_height = peak_data.mean()
    peak_height = unexplained[peak_idx]

    env_height = max(peak_height - wn.nm_env.c, wn.nm_env.c/1000.0)

    peak_log_std = float(safe_log_vec(peak_data, default=np.nan).std())

    if np.isnan(peak_log_std):
        peak_log_std = 0.5
    std = max(peak_log_std, 0.1)

    # the log-amplitude model isn't very good near the noise floor,
    # since noise is additive and any amplitude up to the noise floor
    # is plausible. so we make sure our proposal distribution gives
    # the noise floor a z-score of at most 2.

    # if we're less than the noise floor, pretend we're at halfway between the noise floor and -7
    if env_height < wn.nm_env.c:
        env_height_floor=-7
    else:
        env_height_floor = np.log(env_height - wn.nm_env.c)
    env_height = np.exp((np.log(env_height) + env_height_floor)/2.0)

    std = max(std, (np.log(wn.nm_env.c) - np.log(env_height))/2.0)

    return Gaussian(mean=np.log(env_height), std=std)

#######################################################################

"""

Methods for sampling an arrival time independent of the current
arrival time, with probability proportional to the height of the
envelope at each point.

"""

def preprocess_signal_for_sampling(wave_env):

    d = wave_env**2
    """
    # sample locations where the envelope is increasing, relative to how fast it's increasing
    grad = np.gradient(wave_env)
    incr = (grad > 0)
    d = grad**2
    d[~incr] = max(np.min(d), 1e-3)
    """
    s = np.sum(d)

    if s <= 0:
        Sigvisa().logger.warning("tried to sample from envelope with no positive part, using uniform distribution instead")
        d = np.ones(wave_env.shape)
        d = np.sum(d)

    normalized_env = d/s
    cdf = np.concatenate([np.array((0,)), np.cumsum(normalized_env)])
    return cdf

def peak_log_p(cdf, stime, srate, peak_time):
    # compute the probability that sample_peak_time_from_cdf would
    # have sampled the current atime. formally this should be 0 for
    # all peak times that don't line up with an integer index, but we
    # just force an integer and hope that's okay.

    # we add one here since we created cdf with an initial
    # 0, so that the subtraction below works properly.
    # we add the .000001 to fix numerical issues, so that floor()
    # doesn't round down if we get .99999995103... or similar.
    idx = int(np.floor((peak_time - stime) * srate +.00001)) + 1

    """"in principle, we shouldn't be allowed to kill things outside of
    the signal window, but in practice it helps a lot.
    """
    #if (idx < 1) or (idx >= len(cdf)): return np.float('-inf')
    if (idx < 1) or (idx >= len(cdf)): return np.log(1.0/len(cdf))
    if (cdf[idx] - cdf[idx-1]) < .0000001:
        return np.log(1.0/len(cdf))

    #return np.log(1.0/len(cdf))
    discrete_lp = np.log(cdf[idx] - cdf[idx-1])
    return discrete_lp + np.log(srate)

def get_env_diff_positive_part(wn, arrival_set, remove_noise=False):
    env = wn.get_env().data
    pred_env = wn.assem_env(arrivals=arrival_set)

    npts = wn.npts
    env_diff_pos = wn.env_diff
    noise_mean = wn.nm_env.c
    if remove_noise:
        code = """
        for(int i=0; i < npts; ++i) {
        double v = fabs(env(i)) - fabs(pred_env(i)) - noise_mean;
        env_diff_pos(i) = v > 0 ? v : 0;
        }
        """
    else:
        code = """
        for(int i=0; i < npts; ++i) {
        double v = fabs(env(i)) - fabs(pred_env(i));
        env_diff_pos(i) = v > 0 ? v : 0;
        }
        """
    weave.inline(code,['npts', 'env_diff_pos', 'env', 'pred_env', 'noise_mean'],type_converters = converters.blitz,verbose=2,compiler='gcc')
    return env_diff_pos

def get_current_conditional_cdf(wn, arrival_set):
    env_diff_pos = get_env_diff_positive_part(wn, arrival_set)
    return preprocess_signal_for_sampling(env_diff_pos)


def sample_peak_time_from_cdf(cdf, stime, srate, return_lp=False):
    u = np.random.rand()
    idx = np.searchsorted(cdf, u)
    peak_time = stime + float(idx-1)/srate
    u2 = np.random.rand()
    peak_time += u / srate

    

    if return_lp:
        lp = np.log(cdf[idx]-cdf[idx-1])
    
        if np.isnan(lp):
            Sigvisa().logger.warning("nan lp when sampling from cdf %s, returning as if uniform" % str(cdf))
            lp = -np.log(len(cdf))

        lp += np.log(srate)
        return peak_time, lp
        #return peak_time, np.log(1.0/len(cdf))
    return peak_time

def indep_peak_move(sg, wn, tmnodes,
                    window_lps=None, std=None, **kwargs):


    arrival_key, arrival_node = tmnodes['arrival_time']
    offset_key, offset_node = tmnodes['peak_offset']
    relevant_nodes = [wn,]
    relevant_nodes += [arrival_node.parents[arrival_node.default_parent_key()],] if arrival_node.deterministic() else [arrival_node,]

    arr = extract_arrival_from_key(arrival_key, wn.r)
    other_arrs = wn.arrivals() - set(arr)


    current_atime = arrival_node.get_value(key=arrival_key)
    peak_offset = np.exp(offset_node.get_value(key=offset_key))

    cdf = get_current_conditional_cdf(wn, arrival_set=other_arrs)
    proposed_peak_time, proposal_lp =  sample_peak_time_from_cdf(cdf, wn.st,
                                                                  wn.srate,
                                                                  return_lp=True)
    backward_propose_lp = peak_log_p(cdf, wn.st,
                                     wn.srate,
                                     peak_time = current_atime + peak_offset)

    proposed_arrival_time = proposed_peak_time - peak_offset

    proxy_lps = None
    if window_lps is not None:
        eid, phase = arr
        w_start, w_end = wn.template_idx_window(eid, phase)

        proposed_idx_offset = int((proposed_arrival_time - current_atime) * wn.srate)
        proposed_start = max(0, w_start + proposed_idx_offset)
        proposed_end = min(wn.npts, w_end + proposed_idx_offset)
        w = unify_windows((w_start, w_end), (proposed_start, proposed_end))
        proxy_lps = wn.window_lps_to_proxy_lps(window_lps, w)

    return MH_accept(sg, keys=(arrival_key,),
                     oldvalues = (current_atime,),
                     newvalues = (proposed_arrival_time,),
                     log_qforward = proposal_lp,
                     log_qbackward = backward_propose_lp,
                     node_list = (arrival_node,),
                     relevant_nodes = relevant_nodes,
                     proxy_lps=proxy_lps)

######################################################################


def improve_offset_move_gaussian(sg, wn, tmnodes, std=0.5, window_lps=None, **kwargs):
    arrival_key, arrival_node = tmnodes['arrival_time']
    offset_key, offset_node = tmnodes['peak_offset']

    current_offset = offset_node.get_value(key=offset_key)
    proposed_offset = gaussian_propose(sg, keys=(offset_key,),
                                       node_list=(offset_node,),
                                       values=(current_offset,),
                                       std=std, **kwargs)[0]
    return improve_offset_move(sg, wn, tmnodes, proposed_offset, window_lps=window_lps)

def improve_offset_move_indep(sg, wn, tmnodes,  **kwargs):
    arrival_key, arrival_node = tmnodes['arrival_time']
    offset_key, offset_node = tmnodes['peak_offset']

    current_offset = offset_node.get_value(key=offset_key)
    reverse_lp = offset_node.log_p(v=current_offset)
    proposed_offset = offset_node.parent_sample(set_new_value=False)
    move_lp = offset_node.log_p(v=proposed_offset)
    return improve_offset_move(sg, wn, tmnodes, proposed_offset, move_lp=move_lp, reverse_lp=reverse_lp, **kwargs)


def improve_offset_move(sg, wn, tmnodes, proposed_offset, move_lp=0, reverse_lp=0, window_lps=None,  **kwargs):
    """
    Update the peak_offset while leaving the peak time constant, i.e.,
    adjust the arrival time to compensate for the change in offset.
    """

    proxy_lps = wn.window_lps_to_proxy_lps(window_lps)

    arrival_key, arrival_node = tmnodes['arrival_time']
    offset_key, offset_node = tmnodes['peak_offset']
    relevant_nodes = [wn,]
    relevant_nodes += [arrival_node.parents[arrival_node.default_parent_key()],] if arrival_node.deterministic() else [arrival_node,]
    relevant_nodes += [offset_node.parents[offset_node.default_parent_key()],] if offset_node.deterministic() else [offset_node,]

    current_offset = offset_node.get_value(key=offset_key)
    atime = arrival_node.get_value(key=arrival_key)
    new_atime = atime + (np.exp(current_offset) - np.exp(proposed_offset))

    node_list = [arrival_node,offset_node]
    keys = [arrival_key,offset_key]
    oldvalues = [atime,current_offset]
    newvalues = [new_atime,proposed_offset]

    accepted = MH_accept(sg=sg, keys=keys,
                         oldvalues=oldvalues,
                         newvalues = newvalues,
                         node_list = node_list,
                         relevant_nodes=relevant_nodes,
                         log_qforward=move_lp,
                         log_qbackward=reverse_lp,
                         proxy_lps=proxy_lps)
    return accepted

def improve_atime_move(sg, wn, tmnodes, std=1.0, window_lps=None, **kwargs):
    # here we re-implement get_relevant_nodes from sigvisa.graph.dag, with a few shortcuts
    k_atime, n_atime = tmnodes['arrival_time']
    eid, phase, sta, chan, band, param = parse_key(k_atime)

    # propose a new arrival time
    relevant_nodes = [wn,]
    relevant_nodes += [n_atime.parents[n_atime.default_parent_key()],] if n_atime.deterministic() else [n_atime,]

    old_atime = n_atime.get_value(k_atime)
    values = (old_atime,)
    atime_proposal = float(gaussian_propose(sg, keys=(k_atime,),
                                            node_list=(n_atime,),
                                            values=(values), std=std,
                                            **kwargs))

    node_list = [n_atime,]
    keys = [k_atime,]
    oldvalues = [old_atime,]
    newvalues = [atime_proposal,]

    proxy_lps = wn.window_lps_to_proxy_lps(window_lps)
    accepted = MH_accept(sg, keys, oldvalues, newvalues, node_list, relevant_nodes, proxy_lps=proxy_lps)
    return accepted

#just for debugging
"""
def do_atime_move(sg, wn, tmnodes, atime_offset):
    k_atime, n_atime = tmnodes['arrival_time']
    eid, phase, sta, chan, band, param = parse_key(k_atime)

    # propose a new arrival time
    relevant_nodes = [wn,]
    parent = n_atime.parents[n_atime.default_parent_key()]
    relevant_nodes.append(parent)
    old_atime = n_atime.get_value(k_atime)
    values = (old_atime,)
    atime_proposal = old_atime + atime_offset

    # adjust wiggles for that new time
    wg = sg.wiggle_generator(phase, wn.srate)
    wnodes = [(p, tmnodes[p]) for p in wg.params()]
    wiggle_vals = [n.get_value(k) for (p, (k,n)) in wnodes]
    wiggle_vals_new = np.array(wiggle_vals, copy=True)
    wg.timeshift_param_array(wiggle_vals_new, atime_offset)

    for (i, (p, (k, n))) in enumerate(wnodes):
        n.set_value(key=k, value=wiggle_vals_new[i])
    n_atime.set_value(key=k_atime, value=atime_proposal)
"""
#######################################################################

def get_sorted_arrivals(wn):
    # choose a template at random to try to split
    
    arrivals = wn.arrivals()
    arr_params = []
    for (eid, phase) in arrivals:
        if eid in wn.graph.fully_fixed_events: continue
        arr_params.append((wn.get_template_params_for_arrival(eid, phase)[0], eid, phase))
    sorted_arrs = sorted(arr_params, key = lambda x : x[0]['arrival_time'])
    return sorted_arrs

"""
def atime_window(wn, sorted_arrs, k):
    # when splitting an arrival k, the new arrival can be created anytime between arrivals k-1 and k+1

    n = len(sorted_arrs)
    atime_diff = np.float('inf')
    atime_window_start = wn.st
    atime_window_end = wn.et
    if k < n-1:
        atime_window_end = min(atime_window_end, sorted_arrs[k+1][0]['arrival_time'])
    if k > 0:
        atime_window_start = max(atime_window_start, sorted_arrs[k-1][0]['arrival_time'])
    atime_window_len = atime_window_end - atime_window_start

    return atime_window_start, atime_window_len
"""

def sample_arr_to_split(sg, wn):
    sorted_arrs = get_sorted_arrivals(wn)
    if len(sorted_arrs) < 1:
        return None, None, None
    n = len(sorted_arrs)
    k = np.random.randint(0, n)
    arr_to_split = sorted_arrs[k]

    return arr_to_split, k, sorted_arrs

def get_atime_index(sorted_arrs, atime):
    for k in range(len(sorted_arrs)):
        if sorted_arrs[k][0]['arrival_time'] == atime:
            break
    return k

def tmpl_move_logp(sg, wn, relevant_nodes, n=None):
    return sg.ntemplates_sta_log_p(wn, n=n) + sg.joint_logprob_keys(relevant_nodes)


def split_move_old(sg, wn, return_probs=False, force_accept=False, atime_width=MERGE_PROPOSAL_WIDTH_S):

    # figure out which arrival to split
    arr, k, sorted_arrs = sample_arr_to_split(sg, wn)
    if arr is None:
        return False
    n_arrs = len(sorted_arrs)
    tnodes = sg.get_template_nodes(eid=arr[1], phase=arr[2], sta=wn.sta, band=wn.band, chan=wn.chan)

    # account for possibly setting the coda_height of an event, or an unass template
    k,n = tnodes['coda_height']
    if n.deterministic():
        relevant_parent = [n.parents[n.default_parent_key()],]
    else:
        relevant_parent = [n,]
    lp_old = tmpl_move_logp(sg, wn, [wn,] + relevant_parent)

    # create the new uatemplate, with arrival time sampled uniformly
    current_atime = tnodes['arrival_time'][1].get_value()
    atime_dist = Laplacian(current_atime, atime_width)
    new_atime = atime_dist.sample()
    log_qforward = atime_dist.log_p(new_atime)

    new_tmpl = sg.create_unassociated_template(wn, atime=new_atime, nosort=True)
    sg._topo_sorted_list = new_tmpl.values() + sg._topo_sorted_list
    sg._gc_topo_sorted_nodes()

    # split off some fraction of the amplitude to the new node
    k,n = tnodes['coda_height']
    coda_height = n.get_value(key=k)
    u = np.random.rand()
    new_logheight = np.log(u) + coda_height
    new_tmpl['coda_height'].set_value(new_logheight)
    n.set_value( key=k, value=np.log(1-u) + coda_height)

    lp_new = tmpl_move_logp(sg, wn, [wn,] + relevant_parent + new_tmpl.values())

    new_tmpl_priorsampled = [n for (p, n) in new_tmpl.items() if p not in ['coda_height', 'tt_residual', 'arrival_time']]
    log_qforward += sg.joint_logprob_keys(new_tmpl_priorsampled)  - np.log(n_arrs)
    jacobian_determinant = 1.0/ (u * (1-u))

    # the reverse probability is the prob that we would have chosen to merge these two templates
    new_sorted_arrs = get_sorted_arrivals(wn)
    c1_idx = get_atime_index(new_sorted_arrs, current_atime)
    c2_idx = get_atime_index(new_sorted_arrs, new_atime)

    c1, _ = merge_target_distribution(sg, wn, sorted_arrs=new_sorted_arrs)
    c2 = merge_source_distribution(sg, wn, sorted_arrs=new_sorted_arrs, k=c1_idx, width=atime_width)

    log_qbackward = safe_log(c1[c1_idx]) + safe_log(c2[c2_idx])

    u = np.random.rand()
    if force_accept or ((lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant > np.log(u)):
        sg.logger.info("split template %d from %d: %.1f + %.5f - (%.1f + %.5f) + %f = %.1f vs %.1f" % (-new_tmpl["arrival_time"].tmid, arr[1], lp_new, log_qbackward, lp_old, log_qforward, jacobian_determinant, (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant, np.log(u)))

        if return_probs:
            return True, lp_new, lp_old, log_qforward, log_qbackward, jacobian_determinant
        else:
            return True
    else:
        sg.destroy_unassociated_template(new_tmpl, nosort=True)

        k,n = tnodes['coda_height']
        n.set_value(key=k, value=coda_height)

        # WARNING: this assumes the list hasn't been re-sorted by any
        # of our intermediate calls.
        sg._topo_sorted_list = sg._topo_sorted_list[len(new_tmpl):]
        sg._gc_topo_sorted_nodes()
        sg.next_uatemplateid -= 1
        #lp = sg.current_log_p()
        #assert(np.abs(lp - lp_old) < 1e-10)

        if return_probs:
            return False, lp_new, lp_old, log_qforward, log_qbackward, jacobian_determinant
        else:
            return False



def relevant_nodes_hack(tnodes):
    return [n if not n.deterministic() else n.parents[n.default_parent_key()] for (k,n) in tnodes.values()]

def merge_move_old(sg, wn, return_probs=False, split_atime_width=MERGE_PROPOSAL_WIDTH_S):

    # sample which two arrivals to merge
    n_arrs = len(wn.arrivals())
    arr1, arr2, merge_choice_prob, sorted_arrs = sample_arrivals_to_merge_twostep(sg, wn, width=split_atime_width)

    #arr1, arr2, merge_choice_prob = sample_arrivals_to_merge(sg, wn.sta)
    if arr1 is None:
        return False

    # get all relevant nodes for the arrivals we sampled
    t1nodes = sg.get_template_nodes(eid=arr1[1], phase=arr1[2], sta=wn.sta, band=wn.band, chan=wn.chan)
    t2nodes = sg.get_template_nodes(eid=arr2[1], phase=arr2[2], sta=wn.sta, band=wn.band, chan=wn.chan)

    # save the probability before we actually make the move
    log_qforward = 0.0
    ntemplates = len(sg.uatemplate_ids[(wn.sta, wn.chan, wn.band)])
    lp_old = tmpl_move_logp(sg, wn, [wn,] + relevant_nodes_hack(t1nodes) + relevant_nodes_hack(t2nodes), n=ntemplates)
    orig_topo_sorted = copy.copy(sg._topo_sorted_list)

    # combine the amplitudes of the two arrivals
    vals1, eid1, phase1 = arr1
    vals2, eid2, phase2 = arr2
    amp1 = np.exp(vals1['coda_height'])
    amp2 = np.exp(vals2['coda_height'])
    merged_logamp = np.log(amp1 + amp2)

    # if one arrival is an event arrival, we always keep it. otherwise, choose at random which to keep.
    if phase1 != "UA":
        keep1 = True
    elif phase2 != "UA":
        keep1 = False
    else:
        keep1 = (np.random.rand() < .5)
        log_qforward = np.log(.5)
    if keep1:
        keep_nodes = t1nodes
        lost_nodes = t2nodes
        keep_vals = copy.copy(vals1)
        lost_vals = copy.copy(vals2)
    else:
        keep_nodes = t2nodes
        lost_nodes = t1nodes
        keep_vals = copy.copy(vals2)
        lost_vals = copy.copy(vals1)


    k,n = keep_nodes['coda_height']
    n.set_value(key=k, value=merged_logamp)

    # remove all nodes from the deleted arrival
    for (key, node) in lost_nodes.values():
        sg.remove_node(node)
        sg._topo_sorted_list[node._topo_sorted_list_index] = None

    lp_new = tmpl_move_logp(sg, wn, [wn,] + relevant_nodes_hack(keep_nodes), n=ntemplates-1)

    log_qforward += np.log(merge_choice_prob)

    u = amp1/(amp1+amp2)
    # note (1-u) = amp2/(amp1+amp2), so despite appearances this
    # determinant is actually symmetric in amp1, amp2.
    jacobian_determinant = 1.0/(u * (1-u))

    # compute arrival time window for reverse (split) move
    k,n = keep_nodes['arrival_time']
    merged_atime = n.get_value(key=k)
    atime_dist = Laplacian(merged_atime, split_atime_width)
    log_qbackward = atime_dist.log_p(lost_vals['arrival_time'])

    # we can ignore keys and just take the node log_p because we know
    # the lost node is always going to be a single uatemplate (not an event, etc)
    new_tmpl_priorsampled = [n for (p, (k,n)) in lost_nodes.items() if p not in ['coda_height', 'arrival_time', 'tt_residual']]
    log_qbackward += sg.joint_logprob_keys(new_tmpl_priorsampled) - np.log(n_arrs-1)

    u = np.random.rand()
    if (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant > np.log(u):
        sg.logger.info( "merged templates: %.1f + %.5f - (%.1f + %.5f) + %f = %.1f vs %.1f" % (lp_new, log_qbackward, lp_old, log_qforward, jacobian_determinant, (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant, np.log(u)))

        uaid = lost_nodes['arrival_time'][1].tmid
        del sg.uatemplates[uaid]
        sg.uatemplate_ids[(wn.sta,wn.chan,wn.band)].remove(uaid)


        if return_probs:
            return True, lp_new, lp_old, log_qforward, log_qbackward, jacobian_determinant
        else:
            return True
    else:


        for (param, (key, node)) in lost_nodes.items():
            sg.add_node(node)
            node.addChild(wn)
        for (param, (key, node)) in keep_nodes.items():
            if param in keep_vals:
                node.set_value(key=key, value=keep_vals[param])
        wn.arrivals()
        sg._topo_sorted_list = orig_topo_sorted
        sg._gc_topo_sorted_nodes()


        #lp2 = sg.current_log_p()
        #assert(lp1 == lp2)

        #lp = sg.current_log_p()
        #assert(np.abs(lp - lp_old) < 1e-10)

        if return_probs:
            return False, lp_new, lp_old, log_qforward, log_qbackward, jacobian_determinant
        else:
            return False




#######################################################################


def birth_move(sg, wn,  **kwargs):
    lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move = birth_helper(sg, wn, **kwargs)
    result = mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=accept_move, revert_move=revert_move)
    #if result:
        #print "accepted birth move"
    return result

def birth_proposal(sg, wn, fix_result):
    """ compute the likelihood of proposing a certain fixed result under the dumb birth proposal. not used in inference, but helpful for evaluating the dumb vs optimizing proposals """

    peak_time = fix_result['arrival_time'] + np.exp(fix_result['peak_offset'])

    cdf = get_current_conditional_cdf(wn, arrival_set=wn.arrivals())
    lp = peak_log_p(cdf, wn.st,
                     wn.srate,
                     peak_time = peak_time)

    tg = sg.template_generator(phase="UA")
    logamplitude_proposal_dist = get_env_based_amplitude_distribution(sg, wn, peak_time=peak_time)
    if logamplitude_proposal_dist is None:
        logamplitude_proposal_dist = tg.unassociated_model('coda_height')
    lp += logamplitude_proposal_dist.log_p(fix_result['coda_height'])

    for param in ['peak_decay', 'coda_decay', 'peak_offset']:
        dist = tg.unassociated_model(param)
        lp += dist.log_p(fix_result[param])

    return fix_result, lp

def birth_helper(sg, wn,  **kwargs):
    #lp_old1 = sg.current_log_p()
    lp_old = tmpl_move_logp(sg, wn, [wn,])

    cdf = get_current_conditional_cdf(wn, arrival_set=wn.arrivals())
    peak_time, atime_proposal_lp =  sample_peak_time_from_cdf(cdf, wn.st,
                                                           wn.srate,
                                                           return_lp=True)
    plp = peak_log_p(cdf, wn.st,
                     wn.srate,
                     peak_time = peak_time)


    logamplitude_proposal_dist = get_env_based_amplitude_distribution(sg, wn, peak_time=peak_time)
    if logamplitude_proposal_dist is None:
        tg = sg.template_generator(phase="UA")
        logamplitude_proposal_dist = tg.unassociated_model('coda_height')
    proposed_logamp = logamplitude_proposal_dist.sample()
    proposed_logamp_lp = logamplitude_proposal_dist.log_p(proposed_logamp)

    tmpl = sg.create_unassociated_template(wn, peak_time, nosort=True, **kwargs)
    sg._topo_sorted_list = tmpl.values() + sg._topo_sorted_list
    sg._gc_topo_sorted_nodes()
    tmpl["arrival_time"].set_value(peak_time - np.exp(tmpl["peak_offset"].get_value()))
    tmpl["coda_height"].set_value(proposed_logamp)

    eid = -tmpl["arrival_time"].tmid
    lp_new = tmpl_move_logp(sg, wn, [wn,] + tmpl.values())

    # probability of this birth move is the product of probabilities
    # of all sampled params (including arrival time)
    log_qforward = atime_proposal_lp + proposed_logamp_lp
    for (key, node) in tmpl.items():
        if key == "arrival_time" or key=="coda_height": continue
        if key.startswith('phase') or key.startswith('amp'): continue
        log_qforward += node.log_p()

    # reverse (death) probability is just the probability of killing a
    # random template
    ntemplates = len([1 for (eid, phase) in wn.arrivals() if eid < 0])
    log_qbackward = np.log(1.0/ntemplates) # 0

    def accept_move():
        pass

    def revert_move():
        sg.destroy_unassociated_template(tmpl, nosort=True)
        # WARNING: this assumes the list hasn't been re-sorted by any
        # of our intermediate calls.
        sg._topo_sorted_list = sg._topo_sorted_list[len(tmpl):]
        sg._gc_topo_sorted_nodes()
        sg.next_uatemplateid -= 1
        #lp = sg.current_log_p()
        #assert(np.abs(lp - lp_old) < 1e-10)

    return lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move

def death_move(sg, wn):
    templates = [(eid, phase) for (eid, phase) in wn.arrivals() if eid < 0]
    if len(templates) < 1:
        return False

    u0 = np.random.rand()
    for i in range(len(templates)):
        if u0 <= float(i+1)/len(templates):
            tmpl_to_destroy = templates[i]
            break

    lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move = death_helper(sg, wn, tmpl_to_destroy)
    return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=accept_move, revert_move=revert_move)


def death_helper(sg, wn, tmpl_to_destroy):

    tnodes = sg.get_template_nodes(eid=tmpl_to_destroy[0], phase=tmpl_to_destroy[1], sta=wn.sta, band=wn.band, chan=wn.chan)

    ntemplates = len(sg.uatemplate_ids[(wn.sta, wn.chan, wn.band)])
    lp_old = tmpl_move_logp(sg, wn, [wn,] + [n for (k, n) in tnodes.values()], n=ntemplates)
    orig_topo_sorted = copy.copy(sg._topo_sorted_list)

    # it's debatable whether this should be log(1) or
    # log(1/N). argument for the former is that templates have
    # identities, and the birth move creates a template of a
    # particular identity so the matching death move should just kill
    # that identity, no choice required. (the choice is the choice of
    # a matching birth/death move pair, but that cancels out.) BUT I
    # don't believe this because we're actually treating the templates
    # as exchangeable, i.e. their identities don't matter in the
    # probability model. So I'm going for 1/N without actually
    # formally having worked this out.
    log_qforward = np.log(1.0/ntemplates)

    current_peak = tnodes['arrival_time'][1].get_value() + np.exp(tnodes['peak_offset'][1].get_value())
    eid = -tnodes["arrival_time"][1].tmid

    logamplitude_proposal_dist = get_env_based_amplitude_distribution(sg, wn, peak_time=current_peak)
    if logamplitude_proposal_dist is None:
        tg = sg.template_generator(phase="UA")
        logamplitude_proposal_dist = tg.unassociated_model('coda_height')
    current_logamp = tnodes['coda_height'][1].get_value()
    proposed_logamp_lp = logamplitude_proposal_dist.log_p(current_logamp)

    log_qbackward = proposed_logamp_lp
    for (param, (label, node)) in tnodes.items():
        if param != "arrival_time" and param != "coda_height":
            log_qbackward += node.log_p()
        sg.remove_node(node)
        sg._topo_sorted_list[node._topo_sorted_list_index] = None

    arrs = wn.arrivals()
    cdf = get_current_conditional_cdf(wn, arrival_set=arrs)
    peak_lp = peak_log_p(cdf, wn.st,
                         wn.srate,
                         peak_time = current_peak)
    log_qbackward += peak_lp

    lp_new = tmpl_move_logp(sg, wn, [wn,], n=ntemplates-1)

    def accept_move():
        uaid = -tmpl_to_destroy[0]
        del sg.uatemplates[uaid]
        sg.uatemplate_ids[(wn.sta,wn.chan,wn.band)].remove(uaid)

    def revert_move():

        for (param, (label, node)) in tnodes.items():
            sg.add_node(node)
            node.addChild(wn)
        wn.arrivals()
        sg._topo_sorted_list = orig_topo_sorted
        sg._gc_topo_sorted_nodes()

    return lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move



#####################################################################

from sigvisa.infer.optimize.optim_utils import construct_optim_params


from sigvisa.models.distributions import Uniform, Laplacian, Gaussian

import scipy.stats


def sort_node_list(node_list, sorted_params):
    sorted_node_list = []
    for p in sorted_params:
        for n in node_list:
            if p in n.label:
                sorted_node_list.append(n)
                break
    return sorted_node_list

def get_template_values(tmnodes, sorted_params):
    return np.array([tmnodes[p][1].get_value(tmnodes[p][0]) for p in sorted_params])

def set_template_values(tmnodes, vals, sorted_params):
    for param,v in zip(sorted_params, vals):
        key, node = tmnodes[param]
        node.set_value(key=key, value=v)

def set_template_values_from_dict(tmnodes, d):
    for param,v in d.items():
        key, node = tmnodes[param]
        node.set_value(key=key, value=v)

def template_logprob(nodes, vals, sorted_params, no_atime=False):
    v = get_template_values(nodes, sorted_params)
    set_template_values(nodes, vals, sorted_params)

    lp = 0
    for p in sorted_params:
        k, node = nodes[p]
        if no_atime and 'arrival_time' in k:
            continue
        lp += node.log_p()

    set_template_values(nodes, v, sorted_params)
    return lp

from sigvisa.models.signal_model import unify_windows

def template_overlap(wn, arr1, arr2):
    eid1, phase1 = arr1
    eid2, phase2 = arr2

    w1 = wn.template_idx_window(eid1, phase1)
    w2 = wn.template_idx_window(eid2, phase2)
    #joint_window = unify_windows(w1, w2)

    if w1[1] < w2[0] or w2[1] < w1[0]:
        # no overlap at all
        return 0

    sidx, eidx = w2

    env_with_arr1 = wn.assem_env(arrivals=(arr1,), window_start_idx = sidx, npts = (eidx-sidx))[sidx:eidx:10].copy()
    env_with_arr2 = wn.assem_env(arrivals=(arr2,), window_start_idx = sidx, npts = (eidx-sidx))[sidx:eidx:10]

    domination = env_with_arr1 > env_with_arr2
    total_s2 = np.sum(env_with_arr2)
    weighted_domination = domination * env_with_arr2
    if total_s2 > 0:
        expected_domination = np.sum(domination * env_with_arr2)/total_s2
    else:
        expected_domination = 0.0

    return expected_domination

def merge_target_distribution(sg, wn, sorted_arrs=None, snr_weight=1.0):
    # return a distribution over which template to attempt a merge into
    # currently this gives high probability to templates with lots of
    # other templates nearby, as well as to high-amplitude templates. the tradeoff
    # between these criteria is a parameter, snr_weight, which should probably
    # be optimized empirically.

    if sorted_arrs is None:
        sorted_arrs = get_sorted_arrivals(wn)

    n = len(sorted_arrs)
    if n < 2:
        return None, sorted_arrs

    c = Counter()
    vals = []
    for k in range(n):

        tvals, eid, phase = sorted_arrs[k]

        amp = np.exp(tvals['coda_height'])
        snr = amp / wn.nm_env.c

        atime = tvals['arrival_time']

        nearby_templates = 0
        atime_low = atime-10
        j = k-1
        while j >= 0:
            if sorted_arrs[j][0]['arrival_time'] < atime_low:
                break
            j -= 1
        nearby_templates += (k-j) - 1

        atime_high = atime + 30
        j = k+1
        while j < n:
            if sorted_arrs[j][0]['arrival_time'] > atime_high:
                break
            j += 1
        nearby_templates += (j-k) - 1

        c[k] = nearby_templates + snr_weight * snr

    c.normalize()
    return c, sorted_arrs

def merge_source_distribution(sg, wn, k, sorted_arrs, width = MERGE_PROPOSAL_WIDTH_S, failure_unnorm_prob=0.1):
    # given a target template, sample another template to attempt merging into that template
    # currently, this likelihood of sampling a template is Laplacian in the arrival time
    # (so templates with very nearby arrivals are more likely to be merged).

    n = len(sorted_arrs)

    atime1 = sorted_arrs[k][0]['arrival_time']

    arr1 = sorted_arrs[k][1:]

    c = Counter()
    c[None] = failure_unnorm_prob
    for j in range(n):
        if j==k: continue
        if sorted_arrs[j][2] != "UA": continue
        atime2 = sorted_arrs[j][0]['arrival_time']

        arr2 = sorted_arrs[j][1:]
        c[j] = max(0, template_overlap(wn, arr1, arr2) - 0.5)  \
               + np.exp(- np.abs(atime2-atime1)/width)
    c.normalize()
    return c

def sample_arrivals_to_merge_twostep(sg, wn, width=MERGE_PROPOSAL_WIDTH_S):
    c1, sorted_arrs = merge_target_distribution(sg, wn)
    if c1 is None:
        return None, None, 1.0, sorted_arrs

    target_k = c1.sample()
    arr_target = sorted_arrs[target_k]

    c2 = merge_source_distribution(sg, wn, target_k, sorted_arrs, width=width)
    source_k = c2.sample()
    if source_k is None:
        return None, None, None, None

    arr_source = sorted_arrs[source_k]
    merge_choice_prob = c1[target_k]*c2[source_k]

    return arr_target, arr_source, merge_choice_prob, sorted_arrs


def merge_move(sg, wn, split_atime_width=MERGE_PROPOSAL_WIDTH_S, split_noop_prob=0.9):
    arr1, arr2, merge_choice_prob, sorted_arrs = sample_arrivals_to_merge_twostep(sg, wn, width=split_atime_width)

    if arr1 is None:
        return False

    lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move = merge_helper(sg, wn, arr1, arr2, merge_choice_prob, split_atime_width=split_atime_width, split_noop_prob=split_noop_prob)
    #print "merge acceptance", (lp_new + log_qbackward) - (lp_old + log_qforward), "lp diff", lp_new-lp_old, "proposal lratio", log_qbackward - log_qforward
    return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=accept_move, revert_move=revert_move)

def remove_old_template(sg, wn, arr2, tmnodes2):
    for (key, node) in tmnodes2.values():
        sg.remove_node(node)
        sg._topo_sorted_list[node._topo_sorted_list_index] = None

    #sg.destroy_unassociated_template(dtmnodes2)

    return tmnodes2

def get_merge_distribution(sg, wn, keep_arr, keep_nodes, sorted_params,
                           mh_burnin_steps=500, mh_samples=500):
    v, eid, phase = keep_arr

    tmnode_list = [n for (k,n) in keep_nodes.values()]
    sorted_params = sorted_params[1:]

    window_lps = wn.cache_latent_env_for_template_optimization(eid, phase, force_bounds=False)

    proxy_lps = wn.window_lps_to_proxy_lps(window_lps)
    #proxy_lps =  wn.cache_latent_env_for_fixed_window(eid, phase, force_bounds=False)

    t0 = time.time()

    sg.joint_optimize_nodes(tmnode_list, optim_params=construct_optim_params("'eps': 1e-6, 'method': 'tnc'"),
                            proxy_lps = proxy_lps)
    optimized_vals = get_template_values(keep_nodes, sorted_params)

    from sigvisa.infer.run_mcmc import single_template_MH
    t1 = time.time()
    np.random.seed(1)
    sp, vals = single_template_MH(sg, wn, [(phase, keep_nodes)], steps=mh_burnin_steps+mh_samples, window_lps=window_lps, hamiltonian=False, sorted_params=sorted_params)
    t2 = time.time()
    sg.logger.debug( "template merge opt time %s mcmc time: %s" % (t1-t0, t2-t1))

    if (sp != sorted_params):
        print sp
        print sorted_params
        raise AssertionError("inconsistent params!")

    if mh_samples==0:
        # treat any MH burnin as just part of the optimization;
        # compute a Laplace approximation around the current value.
        try:
            m = vals[-1,:]
        except IndexError:
            m = optimized_vals

        import numdifftools as nd
        sorted_node_list = sort_node_list(tmnode_list, sorted_params)
        relevant_nodes = [n for n in sorted_node_list if not n.deterministic()] + [wn,]
        lb = lambda v : sg.joint_logprob(v, sorted_node_list, relevant_nodes, c=-1)
        lgrad = lambda v : sg.log_p_grad(v, sorted_node_list, relevant_nodes, c=-1)

        sg.logger.debug( "at objective %f" % lb(m))

        try:
            t1 = time.time()
            H = nd.Hessian(lb).hessian(m)
            t2 = time.time()

            def num_jac(f, x, eps=1e-4):
                v0 = f(x)
                n = len(x)
                m = len(v0) if isinstance(v0, np.ndarray) else 1
                jac = np.zeros((m, n))
                for i in range(n):
                    x[i] -= eps
                    v1 = f(x)
                    x[i] += 2*eps
                    v2 = f(x)
                    x[i] -= eps
                    jac[:, i] = (v2-v1)/(2*eps)
                return jac

            H2 = num_jac(lgrad, m)

            #H2 = nd.Jacobian(lgrad).jacobian(m)
            t3 = time.time()
            #print t3-t2, t2-t1, np.max(H2-H)

            #print H
            #print H2


            L = np.linalg.cholesky(H2)
        except Exception as e:
            import sys, traceback, pdb
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print e
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
            pdb.post_mortem(exc_traceback)
        invL = np.linalg.inv(L)
        Q = np.dot(invL.T, invL)

        #e_val, e_vec = np.linalg.eig(H.hessian(m))
        #C = np.dot(e_vec, np.dot(np.diag(np.abs(1.0/e_val)), e_vec.T))

    else:
        Q = np.cov(vals[mh_burnin_steps:,:].T)
        m = np.mean(vals[mh_burnin_steps:,:], axis=0)

    return scipy.stats.multivariate_normal(mean=m, cov=Q)

def propose_peak_offset(wn, tg, onset_env, signal_idx,
                        coda_height, peak_time, prior,
                        min_t=0.2, max_t=15.0, n=20,
                        fix_result=None):

    def onset_lp(log_offset_t):
        # given the unexplained signal preceding the peak, compute logprob of a given peak_offset.
        offset_t = np.exp(log_offset_t)
        atime = peak_time-offset_t
        start = ((atime - wn.st) * wn.srate)
        offset = start % 1.0
        start_idx = int(start)
        #peak_idx = int((peak_time - wn.st) * wn.srate)

        vals = {'arrival_time': atime, 'peak_offset': log_offset_t, 'coda_height': coda_height, 'coda_decay': 4.0, 'peak_decay': 4.0}
        pred_logenv = tg.abstract_logenv_raw(vals, idx_offset=offset, srate=wn.srate)

        #print len(pred_logenv), start_idx, signal_idx

        # arrival_offset_idx is where the onset starts, relative to the signal we're given
        # problem is this can be negative.
        arrival_offset_idx = start_idx-signal_idx
        if arrival_offset_idx >= 0:
            pred_env = np.zeros(onset_env.shape)
            pred_env_shifted = np.exp(pred_logenv[:len(onset_env)-arrival_offset_idx])
            pred_env[arrival_offset_idx:arrival_offset_idx+len(pred_env_shifted)] = pred_env_shifted
        else:
            pred_env = np.exp(pred_logenv[-arrival_offset_idx:len(onset_env)-arrival_offset_idx])
            if len(pred_env) < len(onset_env):
                tmp = np.zeros((len(onset_env),))
                tmp[:len(pred_env)] = pred_env
                pred_env = tmp

        diff = onset_env - pred_env

        lp = wn.nm_env.log_p(diff)
        return lp

    # evaluate a range of candidate offsets
    candidates = np.linspace(np.log(min_t), np.log(max_t), n)
    lps = [onset_lp(t) + prior.log_p(t) for t in candidates]
    ps = np.exp(lps - np.max(lps))
    ps /= np.sum(ps)

    # sample from the resulting discrete distribution
    # to determine a window of possible offsets
    # (which will just be the voronoi cell of the sampled point)
    if fix_result is None:
        peak_offset = None
        idx = np.random.choice(np.arange(n), p=ps)
    else:
        peak_offset = fix_result['peak_offset']
        i = np.searchsorted(candidates, peak_offset)
        if i==n:
            # if peak_offset is bigger than all candidates
            midpoint = np.inf
        else:
            midpoint = (candidates[i-1] + candidates[i])/2.0
        if peak_offset > midpoint:
            idx = i
        else:
            idx = i-1
    p = ps[idx]

    # if we sampled the first or last point, our window stretches
    # to zero or infinity respectively, so sample from some arbitrary
    # distribution on those ranges (exponential distributions in time
    # and in log-time, respectively).
    # otherwise, just sample from the uniform distribution on the vornoi
    # cell of the chosen point
    if idx==0:
        expw2 = np.exp((candidates[idx] + candidates[idx+1])/2.0)
        if peak_offset is None:
            # sample a time (instead of logtime) uniformly from the appropriate window
            sampled_t = np.random.rand() * expw2
            peak_offset = np.log(sampled_t)
        else:
            sampled_t = np.exp(peak_offset)
        p *= sampled_t/expw2
    elif idx == n-1:
        w1 = (candidates[idx-1] + candidates[idx])/2.0
        max_offset = np.log(MAX_PEAK_OFFSET_S)
        if peak_offset is None:
            peak_offset = w1 + np.random.rand() * (max_offset - w1)
        # TODO: get the correct probability here?
    elif peak_offset is None:
        w1 = (candidates[idx-1] + candidates[idx])/2.0
        w2 = (candidates[idx] + candidates[idx+1])/2.0
        peak_offset = w1 + np.random.rand()*(w2-w1)

    lp = onset_lp(peak_offset)

    assert(np.isfinite(prior.log_p(peak_offset)))

    assert(np.isfinite(lp))

    return peak_offset, p



def mask_increasing(env):
    derivs = np.zeros(len(env))
    derivs[1:] = np.diff(env)
    derivs[0] = 1 if np.max(derivs) <= 0 else derivs[1]
    positive_derivs = np.where(derivs>0, derivs, 0)
    return env * positive_derivs


def merge_distribution(peak_env, prior, return_debug=False, peak_detect=True,
                       uniform_lik_mass=0.2, smoothing=None, return_pdf=False):
    exp_peak_env = np.exp(peak_env - np.max(peak_env))
    if peak_detect:
        lik = mask_increasing(exp_peak_env)
    else:
        lik = exp_peak_env
    lik = lik / np.sum(lik)
    lik += uniform_lik_mass/float(len(lik))
    if smoothing is not None:
        window = np.ones((int(smoothing),))
        window = window/float(len(window))
        lik = np.convolve(lik, window, 'same')

    peak_dist = lik * prior
    peak_dist = peak_dist / np.sum(peak_dist)

    if return_pdf:
        peak_cdf = peak_dist
    else:
        peak_cdf = preprocess_signal_for_sampling(peak_dist)

    if return_debug:
        return peak_cdf, peak_dist, peak_env, exp_peak_env, positive_derivs, derivs, prior
    else:
        return peak_cdf

def merge_peak_distribution(wn, env_diff_pos, peak_time, window_len_s=20.0, **kwargs):

    peak_idx = int((peak_time-wn.st)*wn.srate)
    window_len_idx = window_len_s*wn.srate

    t = np.linspace(-window_len_s/2.0, window_len_s/2.0, int(window_len_idx))
    prior = np.exp(-np.abs(t/2.0))


    peak_window_start_idx = int(peak_idx - window_len_idx/2.0)
    peak_window_end_idx = int(peak_idx + window_len_idx/2.0)

    prior = prior[max(0, -peak_window_start_idx):]
    peak_window_start_idx = max(0, peak_window_start_idx)
    peak_window_end_idx = max(peak_window_end_idx, peak_window_start_idx)
    peak_env = env_diff_pos[peak_window_start_idx:peak_window_end_idx]
    prior = prior[:len(peak_env)]
    if len(peak_env) == 0:
        return np.linspace(0, 1, int(window_len_idx)), peak_window_start_idx # uniform distribution
    assert(len(peak_env)==len(prior))

    return merge_distribution(peak_env, prior, **kwargs), peak_window_start_idx

def birth_peak_distribution(wn, env_diff_pos, atime, **kwargs):

    atime_idx = int((atime-wn.st)*wn.srate)
    peak_window_end_idx = int(atime_idx + MAX_PEAK_OFFSET_S*wn.srate)

    atime_idx = max(0, atime_idx)

    peak_env = env_diff_pos[atime_idx:peak_window_end_idx]

    t = np.linspace(1.0/wn.srate, MAX_PEAK_OFFSET_S, len(peak_env))
    logt = np.log(t)
    prior = np.exp(-.5*((logt-.3)/1.1)**2)

    return merge_distribution(peak_env, prior, **kwargs)

def merge_proposal_distribution(sg, wn, eid, phase, fix_result=None, use_ar_noise=True):


    #node_list = [n for (k,n) in nodes.values()]
    arrivals = wn.arrivals()
    other_arrivals = [a for a in arrivals if a != (eid, phase)]
    env_diff_pos = get_env_diff_positive_part(wn, other_arrivals)

    # sample a good peak time
    current_vals, tg = wn.get_template_params_for_arrival(eid=eid, phase=phase)
    current_peak_time = current_vals['arrival_time'] + np.exp(current_vals['peak_offset'])
    peak_cdf, cdf_start_idx = merge_peak_distribution(wn, env_diff_pos, current_peak_time)
    cdf_stime = wn.st + (float(cdf_start_idx) /wn.srate)
    peak_time = sample_peak_time_from_cdf(peak_cdf, cdf_stime, wn.srate) \
        if fix_result is None \
        else fix_result['arrival_time'] + np.exp(fix_result['peak_offset'])
    lp = peak_log_p(peak_cdf, cdf_stime, wn.srate, peak_time)
    assert(np.isfinite(lp))

    # sample peak_offset (and thus, arrival time)
    (w_start_idx, w_end_idx) = wn.template_idx_window(eid, phase)
    unexplained_env = wn.unexplained_env(eid, phase)
    peak_idx = (peak_time-wn.st)*wn.srate
    if peak_idx > len(unexplained_env)-1 or peak_idx < 0:
        lp = -np.inf
        return fix_result, lp

    peak_idx_int = int(peak_idx) if unexplained_env[int(peak_idx)] > unexplained_env[int(peak_idx)+1] else int(peak_idx)+1
    w_start_idx = max(0, min(w_start_idx, peak_idx_int - int(10.0*wn.srate)))
    atime_env = unexplained_env[w_start_idx:peak_idx_int]
    if peak_idx_int <= 0:
        peak_offset = tg.unassociated_model("peak_offset").sample()
        offset_lp = tg.unassociated_model("peak_offset").log_p(peak_offset)
    else:
        coda_height_estimate = safe_log(atime_env[-1] - wn.nm_env.c, default=-4.0)
        peak_offset, offset_lp = propose_peak_offset(wn, tg, atime_env, w_start_idx,
                                                     coda_height_estimate, peak_time, tg.unassociated_model("peak_offset"),
                                                     fix_result=fix_result)
    lp += offset_lp


    # optimize the decay params
    w_end_idx = min(wn.npts, max(w_end_idx, peak_idx_int + int(25.0*wn.srate)))
    #print "fitting to window of len", w_end_idx - w_start_idx
    if use_ar_noise:
        decay_signal_idx = peak_idx_int
        (coda_height, peak_decay, coda_decay), decay_lp = amp_decay_proposal_laplace_ar(wn, tg, unexplained_env[w_start_idx:w_end_idx],
                                                                                        w_start_idx, peak_offset, peak_time,
                                                                                        current_vals,
                                                                                        fix_result=fix_result)
    else:
        decay_env = unexplained_env[peak_idx_int:w_end_idx]
        (coda_height, peak_decay, coda_decay), decay_lp = amp_decay_proposal_laplace_reparam(wn, tg, decay_env.data, wn.srate,
                                                                                fix_result=fix_result)
    lp += decay_lp


    atime = peak_time - np.exp(peak_offset)
    vals = {'arrival_time': atime, 'peak_offset': peak_offset, 'coda_height': coda_height,
            'peak_decay': peak_decay, 'coda_decay': coda_decay}
    return vals, lp

def amp_decay_proposal_laplace_ar(wn, tg, env_window, window_start_idx, peak_offset, peak_time, init_vals, fix_result=None):
    # propose amplitude and decay parameters from a laplace approximation on the (AR) signal likelihood

    coda_height_prior = tg.unassociated_model('coda_height')
    peak_decay_prior = tg.unassociated_model('peak_decay')
    coda_decay_prior = tg.unassociated_model('coda_decay')


    def window_logp_llgrad(x, return_debug=False):
        vals = {'arrival_time': peak_time-np.exp(peak_offset), 'peak_offset': peak_offset,
                'coda_height': x[0], 'peak_decay': x[1], 'coda_decay': x[2]}

        start = (vals['arrival_time'] - wn.st) * wn.srate
        tmpl_start_idx = int(np.floor(start))
        offset = float(start - tmpl_start_idx)
        try:
            pred_logenv, jacobian = tg.abstract_logenv_raw(vals, idx_offset=offset, srate=wn.srate, return_jac_exp=True)
        except ValueError:
            if return_debug:
                return np.inf, np.ones(x.shape), None, None, None, None, None, None, vals
            else:
                return np.inf, np.ones(x.shape)

        pred_env = np.exp(pred_logenv)

        tmpl_start_idx_rel = tmpl_start_idx-window_start_idx
        tmpl_end_idx = tmpl_start_idx + len(pred_env)
        tmpl_end_idx_rel = tmpl_start_idx_rel + len(pred_env)
        early = max(0, -tmpl_start_idx_rel)
        overshoot = max(0, tmpl_end_idx_rel - len(env_window))

        if tmpl_end_idx-overshoot > early + tmpl_start_idx:
            wn.pred_env[early + tmpl_start_idx:tmpl_end_idx-overshoot] = pred_env[early:len(pred_env)-overshoot]

        diff = env_window - wn.pred_env[window_start_idx:window_start_idx+len(env_window)]


        try:
            lp, grad = wn.nm_env.argrad(diff)
        except ARGradientException as e:
            raise e

        shifted_jacobian = np.zeros((len(env_window), 5))
        if tmpl_end_idx-overshoot > early + tmpl_start_idx:
            shifted_jacobian[early + tmpl_start_idx_rel:tmpl_end_idx_rel-overshoot, :] = jacobian[early:len(pred_env)-overshoot,:]

        lp += coda_height_prior.log_p(x[0])
        lp += peak_decay_prior.log_p(x[1])
        lp += coda_decay_prior.log_p(x[2])


        param_grad = np.dot(grad.reshape((1, -1)), shifted_jacobian).flatten() * -1
        param_grad[2] += coda_height_prior.deriv_log_p(x[0])
        param_grad[3] += peak_decay_prior.deriv_log_p(x[1])
        param_grad[4] += coda_decay_prior.deriv_log_p(x[2])

        if return_debug:
            return -lp, -param_grad[2:], pred_logenv, pred_env, diff, grad, jacobian, shifted_jacobian, vals
        return -lp, -param_grad[2:]

    def numeric_hessian(llgrad, x, eps=1e-4):
        ll0, g0 = llgrad(x)
        n = len(x)
        m = len(g0) if isinstance(g0, np.ndarray) else 1
        hess = np.zeros((m, n))
        for i in range(n):
            x[i] -= eps
            ll1, g1 = llgrad(x)
            x[i] += 2*eps
            ll2, g2 = llgrad(x)
            x[i] -= eps
            hess[:, i] = (g2-g1)/(2*eps)
        return hess

    x0 = np.array([init_vals['coda_height'], init_vals['peak_decay'], init_vals['coda_decay']])
    lbounds, hbounds = tg.low_bounds(), tg.high_bounds()
    bounds = [(lbounds['coda_height'], hbounds['coda_height']),
              (lbounds['peak_decay'], hbounds['peak_decay']),
              (lbounds['coda_decay'], hbounds['coda_decay']),]
    lb, hb = zip(*bounds)
    lb, hb = np.array(lb), np.array(hb)

    r = scipy.optimize.minimize(window_logp_llgrad, x0, jac=True, bounds=bounds)

    lp1, pg1, pred_logenv1, pred_env1, diff1, grad1, jacobian1, shifted_jacobian1, vals1 = window_logp_llgrad(x0, return_debug=True)
    lp2, pg2, pred_logenv2, pred_env2, diff2, grad2, jacobian2, shifted_jacobian2, vals2 = window_logp_llgrad(r.x, return_debug=True)

    H = numeric_hessian(window_logp_llgrad, r.x)
    try:
        cov = np.linalg.inv(H)
        rv = scipy.stats.multivariate_normal(r.x, cov)
    except (ValueError, np.linalg.linalg.LinAlgError):
        # non-posdef covariance usually means the optimization didn't
        # converge, or converged on the boundary. we'll just replace
        # it with an identity covariance which will likely be
        # rejected.
        cov = np.eye(3)
        rv = scipy.stats.multivariate_normal(r.x, cov)
    truncated_mass, j = scipy.stats.mvn.mvnun(lb,hb, r.x, cov)

    if fix_result is not None:
        sample = np.array([fix_result['coda_height'], fix_result['peak_decay'], fix_result['coda_decay']])
    else:
        sample = rv.rvs()
        lbounds_violation = sample < lb
        hbounds_violation = sample > hb
        trials = 0
        while lbounds_violation.any() or hbounds_violation.any():
            if trials > 20:
                sg.logger.warning("could not sample valid decay parameters")
                sample[lbounds_violation] = lb[lbounds_violation]
                sample[hbounds_violation] = hb[hbounds_violation]
                break
            sample = rv.rvs()
            lbounds_violation = sample < lb
            hbounds_violation = sample > hb
            trials += 1

    lprob = rv.logpdf(sample) - np.log(truncated_mass)

    def truncate_clfs(clfs):
        valid = np.ones(len(clfs))
        for i, clf in enumerate(clfs):
            mean = clf.coef_
            cov = clf.cov_

            #print i, mean, truncated_mass, j

        return valid

    assert(np.isfinite(lprob))

    return sample, lprob

def merge_helper(sg, wn, arr1, arr2, merge_choice_prob, split_atime_width, split_noop_prob, new_style=True, merge_dist_options={}):

    _, eid1, phase1 = arr1
    _, eid2, phase2 = arr2
    n_arrs = len(wn.arrivals())

    # by default, the param values are pointers to the cache
    # kept by the wave node, which will change as we optimize them.
    # so we have to make our own copy in order to revert if the
    # move is rejected.

    log_qforward = np.log(merge_choice_prob)

    n1 = sg.get_template_nodes(eid1, wn.sta, phase1, wn.band, wn.chan)
    n2 = sg.get_template_nodes(eid2, wn.sta, phase2, wn.band, wn.chan)

    # get parameter values of both templates in vector form
    sp1 = sorted([p for (p, (k, n)) in n1.items() if not n.deterministic()])
    sp2 = sorted([p for (p, (k, n)) in n2.items() if not n.deterministic()])

    orig_atime1 = n1['arrival_time'][1].get_value()
    orig_atime2 = n2['arrival_time'][1].get_value()

    #sorted_params = sorted([k for k in n1.keys() if k not in ('tt_residual', 'amp_transfer')])
    v1 = [n1[p][1].get_value() for p in sp1]
    v2 = [n2[p][1].get_value() for p in sp2]

    #lp_old_full = sg.current_log_p()
    lp_old = tmpl_move_logp(sg, wn, [wn,] + relevant_nodes_hack(n1) + relevant_nodes_hack(n2), n=n_arrs)

    orig_topo_sorted = copy.copy(sg._topo_sorted_list)

    # remove old template, and
    remove_old_template(sg, wn, arr2, n2)
    if new_style:
        pv, lp = merge_proposal_distribution(sg, wn, eid1, phase1, fix_result=None, use_ar_noise=True)
        if pv is not None:
            set_template_values_from_dict(n1, pv)
        log_qforward += lp
    else:
        proposal = get_merge_distribution(sg, wn, arr1, n1, sp1, **merge_dist_options)
        proposed_vals = proposal.rvs()
        log_qforward += proposal.logpdf(proposed_vals)
        set_template_values(n1, proposed_vals, sp1)

    lp_new = tmpl_move_logp(sg, wn, [wn,] + relevant_nodes_hack(n1), n=n_arrs-1)
    #lp_new_full = sg.current_log_p() - sg.ntemplates_sta_log_p(sta, n=n_arrs) + sg.ntemplates_sta_log_p(sta, n=n_arrs-1)

    proposed_atime = n1['arrival_time'][1].get_value()
    atime_dist = Laplacian(proposed_atime, split_atime_width)
    conditional_logprob_atime1 = atime_dist.log_p(orig_atime1)
    conditional_logprob_atime2 = atime_dist.log_p(orig_atime2)

    log_qbackward = template_logprob(n1, v1, sp1, no_atime=True) +  \
                    template_logprob(n2, v2, sp2, no_atime=True) -  \
                    np.log(n_arrs-1) + np.log(1-split_noop_prob)
    log_qbackward += conditional_logprob_atime1 + conditional_logprob_atime2

    def accept_move():
        uaid = n2['arrival_time'][1].tmid
        del sg.uatemplates[uaid]
        sg.uatemplate_ids[(wn.sta,wn.chan,wn.band)].remove(uaid)

    def revert_move():
        for (param, (key, node)) in n2.items():
            sg.add_node(node)
            node.addChild(wn)

        for (p, v) in zip(sp1, v1):
            k, n = n1[p]
            n.set_value(v, key=k)

        wn.arrivals()
        sg._topo_sorted_list = orig_topo_sorted
        sg._gc_topo_sorted_nodes()

    return lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move

def split_move(sg, wn, atime_width=MERGE_PROPOSAL_WIDTH_S, noop_prob=0.9):

    if np.random.rand() < noop_prob:
        return False

    arr, k, sorted_arrs = sample_arr_to_split(sg, wn)
    if arr is None:
        return False
    lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move = split_helper(sg, wn, arr, k, sorted_arrs, atime_width)
    return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=accept_move, revert_move=revert_move)

def split_helper(sg, wn, arr, k, sorted_arrs, atime_width, t1_values=None, t2_values=None, new_style_merge=True):
    old_vals_dict, split_eid, split_phase = arr
    old_vals_dict = copy.copy(old_vals_dict)

    n_arrs = len(sorted_arrs)
    log_qforward = -np.log(n_arrs)

    tnodes = sg.get_template_nodes(eid=arr[1], phase=arr[2], sta=wn.sta, band=wn.band, chan=wn.chan)
    sorted_params = [k for k in sorted(old_vals_dict.keys()) if k not in ('tt_residual', 'amp_transfer')]
    old_vals = [old_vals_dict[p] for p in sorted_params]

    t1_is_event = tnodes['coda_height'][1].deterministic()

    #lp_old_full = sg.current_log_p()
    lp_old = tmpl_move_logp(sg, wn, [wn,] + relevant_nodes_hack(tnodes), n=n_arrs)

    # resample first template from prior
    assert(tnodes['coda_decay'][1].single_key) # array stations are not supported until
                         # I figure out how to sample one station
                         # conditionally on the others
    atime_dist = Laplacian(tnodes['arrival_time'][1].get_value(), atime_width)
    for (p, (k,n)) in tnodes.items():
        if p=="arrival_time":
            if t1_values:
                new_time1 = t1_values['arrival_time']
            else:
                new_time1 = atime_dist.sample()
            n.set_value(new_time1, key=k)
            log_qforward += atime_dist.log_p(new_time1)
        else:
            if t1_values is not None:
                n.set_value(t1_values[p], key=k)
            else:
                n.parent_sample()
            if not n.deterministic():
                log_qforward += n.log_p()

    # starting with the newly-sampled template, find the optimal single template to get the reverse probability
    split_v1 = get_template_values(tnodes, sorted_params)

    if new_style_merge:
        pv, lp = merge_proposal_distribution(sg, wn, split_eid, split_phase, fix_result=old_vals_dict, use_ar_noise=True)
        log_qbackward = lp
    else:
        merge_dist = get_merge_distribution(sg, wn, arr, tnodes, sorted_params)
        log_qbackward = merge_dist.logpdf(old_vals)
    set_template_values(tnodes, split_v1, sorted_params)

    # sample second template from unassociated template prior
    new_time2 = atime_dist.sample()
    new_tmpl = sg.create_unassociated_template(wn, atime=new_time2, nosort=True)

    if t2_values:
        new_time2 = t2_values['arrival_time']
        for (p, n) in new_tmpl.items():
            n.set_value(t2_values[p])

    sg._topo_sorted_list = new_tmpl.values() + sg._topo_sorted_list
    sg._gc_topo_sorted_nodes()
    new_tmpl_priorsampled = [n for (p, n) in new_tmpl.items() if p not in ['arrival_time']]
    log_qforward += sg.joint_logprob_keys(new_tmpl_priorsampled)
    log_qforward += atime_dist.log_p(new_time2)

    lp_new = tmpl_move_logp(sg, wn, [wn,] + relevant_nodes_hack(tnodes) + new_tmpl.values(), n=n_arrs+1)


    # the reverse probability incorporates the prob that we would have chosen to merge these two templates
    new_sorted_arrs = get_sorted_arrivals(wn)
    c1_idx = get_atime_index(new_sorted_arrs, new_time1)
    c2_idx = get_atime_index(new_sorted_arrs, new_time2)

    c1, _ = merge_target_distribution(sg, wn, sorted_arrs=new_sorted_arrs)
    c2 = merge_source_distribution(sg, wn, sorted_arrs=new_sorted_arrs, k=c1_idx, width=atime_width)

    log_qbackward += safe_log(c1[c1_idx]) + safe_log(c2[c2_idx])

    def accept_move():
        pass

    def revert_move():
        sg.destroy_unassociated_template(new_tmpl, nosort=True)

        set_template_values(tnodes, old_vals, sorted_params)

        # WARNING: this assumes the list hasn't been re-sorted by any
        # of our intermediate calls.
        sg._topo_sorted_list = sg._topo_sorted_list[len(new_tmpl):]
        sg._gc_topo_sorted_nodes()
        sg.next_uatemplateid -= 1



    return lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move



#####################################################################

"""
def adjacent_pair_distribution(sg, wn):
    sorted_arrs = get_sorted_arrivals(wn)
    n = len(sorted_arrs)
    if n < 2:
        return None, sorted_arrs

    c = Counter()
    for k in range(n-1):
        if sorted_arrs[k][1] < 0 and sorted_arrs[k+1][1] < 0:
            # no point in swapping two uatemplates
            continue
        c[k] = 1.0 / (sorted_arrs[k+1][0]['arrival_time'] - sorted_arrs[k][0]['arrival_time'] + 1.0/wn.srate)
    c.normalize()
    return c, sorted_arrs

def sample_adjacent_pair_to_swap(sg, wn):
    c, sorted_arrs = adjacent_pair_distribution(sg, wn)
    if c is None or len(c) == 0:
        return None, None, None, None
    k = c.sample()
    return sorted_arrs[k], sorted_arrs[k+1], k, c[k]

def get_pair_prob(sg, wn, first_atime):
    c, sorted_arrs = adjacent_pair_distribution(sg, wn)
    k = get_atime_index(sorted_arrs, first_atime)
    return c[k]
"""


"""
def get_uniform_pair_prob(first_atime, second_atime, adjacency_decay=0.8):
    c, sorted_arrs = adjacent_pair_distribution(sg, wn)
    n = len(sorted_arrs)
    k1 = get_atime_index(sorted_arrs, first_atime)
    k2 = get_atime_index(sorted_arrs, second_atime)

    adjacency = k2-k1
    adjacency_prob = adjacency_decay * (1-adjacency_decay)**(adjacency-1)
    choice_prob = 1.0/(n-adjacency)
    return adjacency_prob*choice_prob
"""


def hamiltonian_template_move(sg, wn, tmnodes, window_lps=None,
                              log_eps_mean=3, log_eps_std=5,
                              epsL=0.2,
                              reverse_block_size=5,
                              reverse_block_min_std=1,
                              reverse_block_max_std=1000,
                              **kwargs):

    params = tmnodes.keys()

    node_list = [n for (k,n) in tmnodes.values()]
    relevant_nodes = node_list + [wn,]

    vals = sg.get_all(node_list)

    proxy_lps = wn.window_lps_to_proxy_lps(window_lps)

    class call_counter:
        pdf_calls = 0
        grad_calls = 0

    def logpdf(x):
        call_counter.pdf_calls += 1
        return sg.joint_logprob(x, node_list, relevant_nodes, proxy_lps)

    def logpdf_grad(x):
        call_counter.grad_calls += 1
        dx =  sg.log_p_grad(x, node_list, relevant_nodes, proxy_lps)
        return dx

    eps = np.exp(- (np.random.randn()*log_eps_std + log_eps_mean)  )
    L = epsL/eps
    L_blocks = int(np.ceil(L/reverse_block_size))
    #L = int(np.random.rand() * (L_max-L_min)) + L_min


    try:
        if reverse_block_size > 1:
            new_vals, new_p, old_p, accept_lp = hmc_step_reversing(vals, logpdf, logpdf_grad, L_blocks, eps, block_size=reverse_block_size, min_block_std=reverse_block_min_std, max_block_std=reverse_block_max_std)
        else:
            new_vals, new_p, old_p, accept_lp = hmc_step(vals, logpdf, logpdf_grad, int(np.ceil(L)), eps)
    except:
       accept_lp = float("-inf")

    if np.log(np.random.rand()) < accept_lp:
        accepted = True
        sg.set_all(values=new_vals, node_list=node_list)
        #print "hmc move eps %f L %f pdfs %d grad %d lp %f accepted? %s" % (eps, L,  call_counter.pdf_calls, call_counter.grad_calls, accept_lp, accepted)

    else:
        accepted = False
        sg.set_all(values=vals, node_list=node_list)


    #with open('/home/dmoore/hamiltonian_eps.txt', 'a') as f:
    #    f.write("%f, %s\n" % (eps, accepted))

    #print "hmc move eps %f L %f pdfs %d grad %d lp %f accepted? %s" % (eps, L,  call_counter.pdf_calls, call_counter.grad_calls, accept_lp, accepted)



    """
    hacky code for adaptive setting of eps, as in the NUTS paper.

    mu = np.log(0.01)
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    t = step
    H = 0.65 - np.exp(accept_lp)
    total_H += H

    if t==0:
        return accepted

    nu_t = t**(-kappa)
    x_old = np.log(eps)

    x_new = mu - t * np.sqrt(t) / (gamma * (t + t0)) * total_H
    xbar = x_new * nu_t + x_old * (1-nu_t)
    eps = np.exp(xbar)

    s.eps = eps
    s.total_H = total_H
#    print "adapting at %d: eps %f H %f nu %f x_old %f x_new %f xbar %f" % (t, eps, H, nu_t, x_old, x_new, xbar)
    """

    return accepted

def hamiltonian_move_reparameterized(sg, wn, tmnodes, window_lps=None, **kwargs):

    params = ['arrival_time', 'peak_offset', 'coda_height', 'peak_decay', 'coda_decay']

    rparams = ['arrival_time', 'peak_time', 'coda_height', 'peak_decay', 'coda_auc']

    node_list = [tmnodes[p][1] for p in params]
    relevant_nodes = node_list + [wn,]

    vals = np.array([n.get_value() for n in node_list])

    proxy_lps = wn.window_lps_to_proxy_lps(window_lps)

    def reparametrize(x):
        arrival_time, peak_offset, coda_height, peak_decay, coda_decay = x

        peak_time = arrival_time + np.exp(peak_offset)
        coda_auc = coda_height - coda_decay

        return np.array((arrival_time, peak_time, coda_height, peak_decay, coda_auc))

    def reparametrize_inv(rx):
        arrival_time, peak_time, coda_height, peak_decay, coda_auc = rx

        d = peak_time - arrival_time
        if d < 0:
            sg.logger.warning("hamiltonian move: BAD peak_time")
            d = 0.1

        peak_offset = np.log(d)
        coda_decay =  coda_height - coda_auc

        return np.array((arrival_time, peak_offset, coda_height, peak_decay, coda_decay))

    def logpdf(rx):
        x = reparametrize_inv(rx)
        return sg.joint_logprob(x, node_list, relevant_nodes, proxy_lps)

    def logpdf_grad(rx, eps=1e-4):
        # note: we compute a numerical gradient in the reparameterized
        # space, rather than transforming the gradient from the
        # original space. I *think* this should give better results
        # (the numerical gradient can take advantage of the nicer
        # shape of the reparameterized space) but haven't actually
        # done the comparison.


        #J = nd.Jacobian(logpdf)
        l0 = logpdf(rx)

        n = len(rx)
        g = np.zeros((n,))
        for i in range(n):
            rx[i] += eps
            g[i] = (logpdf(rx) - l0)/eps
            rx[i] -= eps

        return g

    rvals = reparametrize(vals)

    eps = np.random.rand() * 0.001

    L = int(np.random.rand() * 50) + 5

    # encode constraints that peak_time > arr_time, and coda_auc > 0
    ninf = float("-inf")
    lbounds = np.array([ninf, rvals[0], ninf, ninf, 0])

    new_rvals, accepted = hmc_step(rvals, logpdf, logpdf_grad, L, eps, lbounds=lbounds)

    new_vals = reparametrize_inv(new_rvals)

    sg.set_all(values=new_vals, node_list=node_list)

    return accepted

##############################################################################################
# optimizing birth/death moves
####################################################################################

from sigvisa.signals.common import smooth
from sigvisa.models.templates.lin_polyexp import MAX_PEAK_OFFSET_S

def sta_lta_cdf2(env_diff_pos, short_idx=2, long_idx=30, smooth_idx=7,
                 shift_idx=3, distsmooth_idx=3, sta_lta_power=3):

    #from obspy.signal.trigger import classicSTALTAPy as classicSTALTA
    # steal this definition from obspy since for some reason it's breaking on
    # python-dbg builds
    def classicSTALTAPy(a, nsta, nlta):
        # The cumulative sum can be exploited to calculate a moving average (the
        # cumsum function is quite efficient)
        sta = np.cumsum(a ** 2)

        # Convert to float
        sta = np.require(sta, dtype=np.float)

        # Copy for LTA
        lta = sta.copy()

        # Compute the STA and the LTA
        sta[nsta:] = sta[nsta:] - sta[:-nsta]
        sta /= nsta
        lta[nlta:] = lta[nlta:] - lta[:-nlta]
        lta /= nlta

        # Pad zeros
        sta[:nlta - 1] = 0

        # Avoid division by zero by setting zero values to tiny float
        dtiny = np.finfo(0.0).tiny
        idx = lta < dtiny
        lta[idx] = dtiny

        return sta / lta
    classicSTALTA = classicSTALTAPy

    if smooth_idx is not None:
        smoothed = smooth(env_diff_pos, smooth_idx)
    else:
        smoothed = env_diff_pos

    short_idx = max(1, short_idx)
    long_idx = max(5, long_idx)
    shift_idx = max(1, shift_idx)

    extended = np.zeros(long_idx + len(smoothed))
    extended[long_idx:] = smoothed
    extended[:long_idx] = np.mean(smoothed)

    sta_lta = classicSTALTA(extended, short_idx, long_idx)[long_idx:]
    sta_lta = sta_lta ** sta_lta_power
    d = np.where(sta_lta > 1, sta_lta - 0.999, 0.001)
    result = np.zeros(d.shape)
    result[:-shift_idx] = d[shift_idx:]

    if distsmooth_idx is not None:
        distsmooth=smooth(result, distsmooth_idx)
    else:
        distsmooth=result

    return preprocess_signal_for_sampling(result)

def peak_time_proposal_dist(wn, env_diff_pos, atime, return_debug=False):

    atime_idx = int((atime-wn.st)*wn.srate)
    peak_window_end_idx = int(atime_idx + MAX_PEAK_OFFSET_S*wn.srate)

    atime_idx = max(0, atime_idx)

    peak_env = env_diff_pos[atime_idx:peak_window_end_idx]
    if len(peak_env) == 0:
        # hack
        return np.linspace(0, 1, int(wn.srate*5.0))

    exp_peak_env = np.exp(peak_env - np.max(peak_env))

    derivs = np.zeros(len(exp_peak_env))
    derivs[1:] = np.diff(peak_env)
    derivs[0] = 1 if np.max(derivs) <= 0 else derivs[1]
    positive_derivs = np.where(derivs>0, derivs, 0)

    t = np.linspace(1.0/wn.srate, MAX_PEAK_OFFSET_S, len(exp_peak_env))
    logt = np.log(t)
    prior = np.exp(-.5*((logt-.3)/1.1)**2)

    peak_dist = exp_peak_env * positive_derivs * prior
    peak_dist = peak_dist / np.sum(peak_dist)

    peak_cdf = preprocess_signal_for_sampling(peak_dist)

    if return_debug:
        return peak_cdf, peak_dist, peak_env, exp_peak_env, positive_derivs, derivs, prior
    else:
        return peak_cdf

def splitpoint_lik(y, x1, x2, idx):

    y1 = y[:idx]
    y2 = y[idx:]


    clf = BayesianRidgeWithCov(compute_score=True, fit_intercept=False, normalize=True)
    clf.fit(x2[idx:, :], y2)
    score2 = clf.scores_[-1]

    clf.fit(x1[:idx, :], y1)
    score1 = clf.scores_[-1]

    return score1+score2, clf

def get_fit_window_bayes(wn, peak_time, env_diff_pos, incr_s=3.0, max_s=45.0, min_s=3.0):

    sidx = int((peak_time-wn.st)*wn.srate)
    jump_idx = int(incr_s * wn.srate)
    min_idx = int(min_s * wn.srate)
    max_idx = min(len(env_diff_pos) - sidx, int(max_s * wn.srate))

    if sidx >= len(env_diff_pos)-min_idx:
        return None

    lw = np.log(env_diff_pos)[sidx:sidx+max_idx]

    tlen = max_idx/wn.srate
    n = max_idx
    t = np.linspace(0, tlen, n).reshape((-1, 1))
    logt = np.log(t+1.0)
    b = np.ones(t.shape)
    x1 = np.hstack([b, logt, t])
    x2 = np.hstack([b,])

    idx_range = range(min_idx, max_idx, jump_idx)
    split_fits = [splitpoint_lik(lw, x1, x2, i) for i in idx_range]
    split_ll = [ll for (ll, clf) in split_fits]
    best_split = idx_range[np.argmax(split_ll)]
    fit_window = env_diff_pos[sidx:sidx+best_split].copy()
    return fit_window

def get_fit_window(wn, peak_time, env_diff_pos, incr_s=5.0, max_s=60.0, smoothing_s=15.0, min_phase_gap_s=15.0):
    # amplitude and decay parameters, conditioned on peak time

    peak_s = peak_time - wn.st
    if (peak_s < 0):
        raise Exception("cannot extract a fit window because peak precedes wn start")

    peak_idx1 = int(np.floor(peak_s * wn.srate))
    peak_idx2 = int(np.ceil(peak_s * wn.srate))
    try:
        sidx = peak_idx1 if env_diff_pos[peak_idx1] > env_diff_pos[peak_idx2] else peak_idx2
    except IndexError:
        return None

    jump_idx = int(incr_s * wn.srate)
    eidx = sidx + jump_idx

    if sidx >= len(env_diff_pos)-2:
        return None

    if smoothing_s is not None:
        s = smooth(env_diff_pos, int(smoothing_s*wn.srate))
    else:
        s = env_diff_pos

    min_phase_gap_idx = int(min_phase_gap_s * wn.srate)

    current_mean = np.mean(s[sidx:eidx])
    while (eidx < min(len(s), sidx + max_s * wn.srate)):
        next_mean = np.mean(s[eidx:eidx+jump_idx]) if eidx < len(s)-1 else current_mean

        # cut the window early if we are near the noise floor
        if next_mean < wn.nm_env.c * 1.1:
            break
        # OR if we are past a minimum length, and the signal seems to be increasing
        if eidx > sidx + min_phase_gap_idx and next_mean > current_mean * 1.1:
            break

        eidx += jump_idx
        current_mean = next_mean
    fitting_window = env_diff_pos[sidx:eidx].copy()
    return fitting_window

from sigvisa.models.templates.lin_polyexp import LinPolyExpTemplateGenerator
lbounds_dict = LinPolyExpTemplateGenerator.low_bounds()
hbounds_dict = LinPolyExpTemplateGenerator.high_bounds()
lbounds = np.array((lbounds_dict['coda_height'], lbounds_dict['peak_decay'], lbounds_dict['coda_decay']))
hbounds = np.array((hbounds_dict['coda_height'], hbounds_dict['peak_decay'], hbounds_dict['coda_decay']))

lbounds_exp = np.array((lbounds_dict['coda_height'], -np.exp(hbounds_dict['peak_decay']), -np.exp(hbounds_dict['coda_decay'])))
hbounds_exp = np.array((hbounds_dict['coda_height'], -np.exp(lbounds_dict['peak_decay']), -np.exp(lbounds_dict['coda_decay'])))

bounds = zip(lbounds, hbounds)

def amp_decay_proposal(*args, **kwargs):

    try:
        return amp_decay_proposal_linear(*args, **kwargs)
    except InvalidParamException:
        Sigvisa().logger.warning("amp decay proposal falling back to laplace")
        return amp_decay_proposal_laplace(*args, **kwargs)

def get_fit_window_dist_bayes(wn, peak_time, env_diff_pos, incr_s=2.0, max_s=60.0, min_s=5.0):

    sidx = int((peak_time-wn.st)*wn.srate)
    jump_idx = int(incr_s * wn.srate)
    min_idx = int(min_s * wn.srate)
    max_idx = min(len(env_diff_pos) - sidx, int(max_s * wn.srate))

    if sidx >= len(env_diff_pos)-min_idx:
        return None

    #s = smooth(env_diff_pos, 15.0)
    s = env_diff_pos
    lw = np.log(s)[sidx:sidx+max_idx]

    tlen = max_idx/wn.srate
    n = max_idx
    t = np.linspace(0, tlen, n).reshape((-1, 1))
    logt = np.log(t+1.0)
    b = np.ones(t.shape)
    x1 = np.hstack([b, logt, t])
    x2 = np.hstack([b,])

    idx_range = range(min_idx, max_idx-3, jump_idx)
    split_fits = [splitpoint_lik(lw, x1, x2, i) for i in idx_range]
    #split_ll = [ll for (ll, clf) in split_fits]
    #best_split = idx_range[np.argmax(split_ll)]
    #fit_window = env_diff_pos[sidx:sidx+best_split].copy()
    return split_fits, sidx

def amp_decay_mixture_proposal(wn, peak_time, env_diff_pos, fix_result=None):

    def truncate_clfs(clfs):
        valid = np.ones(len(clfs))
        for i, clf in enumerate(clfs):
            mean = clf.coef_
            cov = clf.cov_
            truncated_mass, j = scipy.stats.mvn.mvnun(lbounds_exp,hbounds_exp, mean, cov)
            #print i, mean, truncated_mass, j

            if truncated_mass < 0.05:
                valid[i] = 0
            clf.truncated_mass = truncated_mass
        return valid


    def sample_clf_posterior(clf):
        mean = clf.coef_
        cov = clf.cov_
        rv = scipy.stats.multivariate_normal(mean, cov)
        sample = rv.rvs(1)
        while (sample <= lbounds_exp).any() or (sample >= hbounds_exp).any():
            sample = rv.rvs(1)
        return sample

    def clf_ll(clf, x):
        mean = clf.coef_
        cov = clf.cov_
        rv = scipy.stats.multivariate_normal(mean, cov)
        return rv.logpdf(sample) - np.log(clf.truncated_mass)

    split_fits, sidx = get_fit_window_dist_bayes(wn, peak_time, env_diff_pos)
    split_ll, clfs = zip(*split_fits)
    split_dist = np.exp(split_ll - np.max(split_ll))

    valid = truncate_clfs(clfs)
    split_dist = np.where(valid, split_dist, 0)
    split_dist /= np.sum(split_dist)

    if fix_result is None:
        split_idx = np.random.choice(len(split_dist), p=split_dist)
        sample = sample_clf_posterior(clfs[split_idx])
    else:
        sample = fix_result
        sample[1] = -np.exp(sample[1])
        sample[2] = -np.exp(sample[2])

    overall_lp = -np.inf
    for i in range(len(split_dist)):
        if split_dist[i] < 1e-20:
            continue
        split_lp = np.log(split_dist[i])
        param_lp = clf_ll(clfs[i], sample)
        joint_lp = split_lp + param_lp
        overall_lp = np.logaddexp(overall_lp, joint_lp)

        #print joint_lp,
    #print " =", overall_lp

    sample[1] = np.log(-sample[1])
    sample[2] = np.log(-sample[2])
    overall_lp += sample[1] + sample[2]


    return sample, overall_lp



def amp_decay_proposal_linear(env, true_srate, downsample_by=1, fix_result=None, return_debug=False):

    if downsample_by != 1:
        downsampled = env[::downsample_by]
    else:
        downsampled = env
    srate = true_srate / float(downsample_by)

    downsampled[downsampled < 0.0001] = 0.0001
    lw = np.log(downsampled)

    n = len(downsampled)
    tlen = n / srate
    t = np.linspace(0, tlen, n).reshape((-1, 1))
    logt = np.log(t+1)

    clf = BayesianRidgeWithCov(compute_score=False, fit_intercept=False, normalize=False, tol=1e-10)

    t1 = np.hstack([np.ones(t.shape), np.log(t+1.0), t])

    clf.fit(t1, lw)

    mean = clf.coef_
    cov = clf.cov_
    noise_prec = clf.alpha_

    # because we're treating the decay rates in their linear
    # parameterizations, we might infer invalid (positive) values.  if
    # the variance is high enough, we can work around using rejection
    # sampling to sample from the truncated distribution in which the
    # rates are constrained to be negative. if rejection sampling
    # seems likely to fail, we can a) fall back to the
    # optimize+laplace approach, which guarantees valid params, or b)
    # fail entirely and hope that other windows succeed.

    truncated_mass, i = scipy.stats.mvn.mvnun(lbounds_exp,hbounds_exp, mean, cov)
    if i != 0 or truncated_mass < 0.1:
        raise InvalidParamException(mean, cov, truncated_mass)
    else:
        rv = scipy.stats.multivariate_normal(mean, cov)

    if fix_result is not None:
        sample = fix_result
        sample[1] = -np.exp(sample[1])
        sample[2] = -np.exp(sample[2])
    else:
        # rejection sample
        sample = rv.rvs(1)
        while (sample <= lbounds_exp or sample >= hbounds_exp).any():
            sample = rv.rvs(1)

    # renormalize pdf to account for truncation
    logp = rv.logpdf(sample) - np.log(truncated_mass)

    # convert decay rates to logscale parameterization
    sample[1] = np.log(-sample[1])
    sample[2] = np.log(-sample[2])

    if not np.isfinite(sample).all():
        raise ValueError("proposed bad template params %s" % (samples))

    # change of variables:
    # phi is the reverse transform: u -> [u1, e^u2, e^u3]
    # Jacobian of phi is
    # 1 * e^u2 * e^u3
    # and log jacobian is therefore
    # u2 + u3
    logp += sample[1] + sample[2]

    if return_debug:
        return sample, logp, mean, cov, None
    else:
        return sample, logp


def amp_decay_proposal_laplace(env, true_srate, downsample_by=1, fix_result=None, return_debug=False):

    if downsample_by != 1:
        downsampled = env[::downsample_by]
    else:
        downsampled = env
    srate = true_srate / float(downsample_by)

    downsampled[downsampled < 0.0001] = 0.0001
    lw = np.log(downsampled)

    n = len(downsampled)
    tlen = n / srate


    t = np.linspace(0, tlen, n)
    logt = np.log(t+1)

    def lik_grad(x, return_prec=False):
        # compute log-likelihood and derivatives under an iid Gaussian noise model.

        coda_height = x[0]
        peak_decay = np.exp(x[1])
        coda_decay = np.exp(x[2])

        pred_log_env = coda_height - peak_decay * logt - t * coda_decay
        residuals = pred_log_env - lw
        ll = .5 * np.sum(residuals**2)

        d_dA = np.sum(residuals)
        d_dpeak = - np.sum(residuals * logt * peak_decay)
        d_dcoda = -np.sum(residuals * t * coda_decay)
        grad = np.array((d_dA, d_dpeak, d_dcoda))

        if return_prec:
            # return the posterior covariance under a Laplace approximation,
            # using the maximum-likelihood estimate of the noise variance

            T = len(t)
            sum_logt = np.sum(logt)
            sum_t = T*(T+1)/2
            H = np.zeros((3,3))

            H[0,0] = T
            H[0,1] = -sum_logt * peak_decay
            H[1,0] = H[0,1]
            H[0,2] = -sum_t * coda_decay
            H[2,0] = H[0,2]

            H[1,1] = np.sum( (logt * peak_decay)**2 - logt*peak_decay*residuals)
            H[1,2] = np.sum( logt * t * peak_decay*coda_decay)
            H[2,1] = H[1,2]
            H[2,2] = np.sum( (t * coda_decay)**2 - residuals*t*coda_decay)

            #mle_noise_var = np.var(residuals)
            mean_deviation = np.mean(np.abs(residuals[:int(len(residuals)/2)]))
            robust_std = mean_deviation * 1.25 # formally, sqrt(pi/2) is the ratio of STD to MAD for a Gaussian
            robust_var = robust_std**2
            #print "estimated var", robust_var
            H /= robust_var

            return ll, grad, H

        else:
            return ll, grad

    # initialize optimization at sensible defaults:
    init_amp = lw[0]  # amplitude estimate is the height at the beginning of the window
    if not np.isfinite(init_amp):
        init_amp = -2.0

    try:
        init_coda_decay = np.log(-(lw[-1]-lw[0])/tlen) # estimated decay rate is the empirical slope of the log signal
    except FloatingPointError:
        init_coda_decay = np.nan

    if not np.isfinite(init_coda_decay):
        init_coda_decay = 0.0

    init_peak_decay = init_coda_decay # coda and peak decay are empirically correlated,
                                      # so we'll start by assuming they're the same.
    x0 = np.array((init_amp, init_peak_decay, init_coda_decay))

    # find the most likely set of parameters

    r = scipy.optimize.minimize(lik_grad, x0, jac=True, bounds=bounds)

    # propose parameters from a Laplace approximation (Gaussian distribution
    # with covariance given by the inverse Hessian) around the mode
    x = r.x
    l, g, prec = lik_grad(x, return_prec=True)

    try:
        cholprec = np.linalg.cholesky(prec)
    except np.linalg.LinAlgError:
        # if Hessian is not positive definite, then
        # optimzation failed (we didn't find a maximum).
        # so just use our initial guess instead.
        # (this will likely be rejected...)
        x = x0
        prec = np.eye(3)
        cholprec = np.linalg.cholesky(prec)

    if fix_result is not None:
        sample = fix_result
        z = np.dot(cholprec.T, (sample-x))
    else:
        z = np.random.randn(3)
        sample = x + np.dot(np.linalg.inv(cholprec.T), z).flatten()



    half_logdet = -np.sum(np.log(np.diag(cholprec)))
    logp = -.5 * np.sum(z**2) - half_logdet - 3/2.0 * np.log(2*np.pi)


    #print "*******************"
    #print "mean", x
    #print "cov", np.linalg.inv(prec)
    #print "sample", sample
    #print "z", z
    #print "logp", logp


    # truncate the proposals at reasonable bounds.  technically this
    # is invalid since I'm not accounting for the truncation in the
    # returned probability, but it shouldn't make a big difference
    # since the bounds are quite wide.

    lbounds_violation = sample < lbounds
    sample[lbounds_violation] = lbounds[lbounds_violation]

    hbounds_violation = sample > hbounds
    sample[hbounds_violation] = hbounds[hbounds_violation]

    if return_debug:
        return sample, logp, x, prec, lik_grad
    else:
        return sample, logp

from algopy import UTPM, exp
import algopy
def amp_decay_proposal_laplace_reparam(wn, tg, env, true_srate, downsample_by=1, fix_result=None, return_debug=False):

    if downsample_by != 1:
        w = env[::downsample_by].copy()
    else:
        w = env.copy()
    srate = true_srate / float(downsample_by)

    n = len(w)
    tlen = n / srate
    t = np.linspace(0, tlen, n)
    logt = np.log(t+1)
    b = np.ones(t.shape)

    coda_height_prior = tg.unassociated_model('coda_height')
    peak_decay_prior = tg.unassociated_model('peak_decay')
    coda_decay_prior = tg.unassociated_model('coda_decay')
    noise_mean_prior = Gaussian(wn.nm_env.c, wn.nm_env.c/10.0)
    noise_var_prior = Gaussian(np.log(wn.nm_env.c), 1.0)


    def lik_grad(x, grad=False, return_all=False):
        # compute log-likelihood and derivatives under an iid Gaussian noise model.

        alpha = 0.0 # multiplicative noise contribution: should optimize this too...

        coda_height = x[0]
        peak_decay = np.exp(x[1])
        coda_decay  = np.exp(x[2])
        noise_mean = x[3] #np.exp(x[3])
        noise_var = np.exp(x[4])

        pred_log_env = coda_height*b - peak_decay * logt - t * coda_decay
        pred_env = np.exp(pred_log_env)
        residuals = w - pred_env - noise_mean*b
        var = noise_var*b + alpha*pred_env
        nll = np.sum(np.log(var) + residuals**2/var)

        nll -= coda_height_prior.log_p(x[0])
        nll -= peak_decay_prior.log_p(x[1])
        nll -= coda_decay_prior.log_p(x[2])
        nll -= noise_mean_prior.log_p(x[3])
        nll -= noise_var_prior.log_p(x[4])

        if grad:
            ds_dA = pred_env
            ds_dP = -pred_env * peak_decay * logt
            ds_dC = -pred_env * t * coda_decay

            dll_ds = alpha/(var) - (2*var*residuals + alpha*residuals**2)/var**2
            dll_dm = np.sum(-2* (residuals)/var)
            dll_dv = np.sum(noise_var* (alpha*pred_env + noise_var - residuals**2) / (var**2))
            grad = np.array((np.sum(dll_ds * ds_dA) - coda_height_prior.deriv_log_p(x[0]),
                             np.sum(dll_ds * ds_dP) - peak_decay_prior.deriv_log_p(x[1]),
                             np.sum(dll_ds * ds_dC) - coda_decay_prior.deriv_log_p(x[2]),
                             dll_dm - noise_mean_prior.deriv_log_p(x[3]),
                             dll_dv - noise_var_prior.deriv_log_p(x[4])))
            return nll, grad

        if return_all:
            return nll, pred_log_env, pred_env, residuals, var

        return nll

    # initialize optimization at sensible defaults:
    lw = safe_log_vec(w)
    try:
        init_amp = lw[0]  # amplitude estimate is the height at the beginning of the window
    except:
        init_amp = np.nan
    if not np.isfinite(init_amp):
        init_amp = -2.0

    try:
        init_coda_decay = np.log(-(lw[-1]-lw[0])/tlen) # estimated decay rate is the empirical slope of the log signal
    except:
        init_coda_decay = np.nan

    if not np.isfinite(init_coda_decay):
        init_coda_decay = 0.0
    init_peak_decay = init_coda_decay # coda and peak decay are empirically correlated,
                                      # so we'll start by assuming they're the same.
    init_noise_mean = np.log(wn.nm_env.c)
    try:
        init_noise_var = .1 * np.log(np.var(w))
    except:
        init_noise_var = 1.0
    x0 = np.array((init_amp, init_peak_decay, init_coda_decay, init_noise_mean, init_noise_var))

    # find the most likely set of parameters
    r = scipy.optimize.minimize(lik_grad, x0, jac=True, args=(True,), bounds=bounds + [(-5., 3.), (-5., 3.)])

    # propose parameters from a Laplace approximation (Gaussian distribution
    # with covariance given by the inverse Hessian) around the mode.
    # Find the Hessian using automatic differentiation because I'm bad at calculus.
    cg = algopy.CGraph()
    x = algopy.Function(x0)
    y = lik_grad(x)
    cg.trace_off()
    cg.independentFunctionList = [x]
    cg.dependentFunctionList = [y]
    H = cg.hessian(r.x)

    cov = np.linalg.inv(H)
    marginal_cov = cov[0:3, 0:3]
    prec = np.linalg.inv(marginal_cov)
    x = r.x[:3]

    try:
        cholprec = np.linalg.cholesky(prec)
    except np.linalg.LinAlgError:
        # if Hessian is not positive definite, then
        # optimzation failed (we didn't find a maximum).
        # so just use our initial guess instead.
        # (this will likely be rejected...)
        x = x0[:3]
        prec = np.eye(3)
        cholprec = np.linalg.cholesky(prec)

    if fix_result is not None:
        sample = fix_result
        z = np.dot(cholprec.T, (sample-x))
    else:
        z = np.random.randn(3)
        sample = x + np.dot(np.linalg.inv(cholprec.T), z).flatten()

    half_logdet = -np.sum(np.log(np.diag(cholprec)))
    logp = -.5 * np.sum(z**2) - half_logdet - 3/2.0 * np.log(2*np.pi)

    lbounds_violation = sample < lbounds
    sample[lbounds_violation] = lbounds[lbounds_violation]

    hbounds_violation = sample > hbounds
    sample[hbounds_violation] = hbounds[hbounds_violation]

    #if logp < -50:
    #    import pdb; pdb.set_trace()



    if return_debug:
        return sample, logp, x, prec, lik_grad
    else:
        return sample, logp


def propose_from_prior(sg, wn, x):
    # in the edge case that the entire decay region is off the
    # edge of the signal, just propose from the prior.
     tg = sg.template_generator(phase="UA")
     if x is None:
         coda_height = tg.unassociated_model("coda_height").sample()
         peak_decay = tg.unassociated_model("peak_decay").sample()
         coda_decay = tg.unassociated_model("coda_decay").sample()
     else:
         coda_height, peak_decay, coda_decay = x
     proposal_lp = tg.unassociated_model("coda_height").log_p(coda_height)
     proposal_lp += tg.unassociated_model("peak_decay").log_p(peak_decay)
     proposal_lp += tg.unassociated_model("coda_decay").log_p(coda_decay)
     return (coda_height, peak_decay, coda_decay), proposal_lp

def optimizing_birth_proposal(sg, wn, fix_result=None, return_debug=False, laplace_proposal=False, laplace_reparam=True, mixture_proposal=False):

    # arrival time proposal
    env_diff_pos = get_env_diff_positive_part(wn, arrival_set=wn.arrivals(), remove_noise=False)
    cdf2 = sta_lta_cdf2(env_diff_pos, short_idx = int(2*wn.srate), long_idx=int(30*wn.srate), smooth_idx=int(7*wn.srate), distsmooth_idx=int(3*wn.srate), shift_idx=int(3*wn.srate))
    if fix_result is not None:
        atime  = fix_result['arrival_time']
        atime_proposal_lp = peak_log_p(cdf2, wn.st + 1/wn.srate, wn.srate, atime)
    else:
        atime, atime_proposal_lp =  sample_peak_time_from_cdf(cdf2, wn.st, wn.srate, return_lp=True)

    # peak time conditioned on atime
    if return_debug:
        peak_cdf, peak_dist, peak_env, exp_peak_env, positive_derivs, derivs, prior = peak_time_proposal_dist(wn, env_diff_pos, atime, return_debug=True)
    else:
        peak_cdf = peak_time_proposal_dist(wn, env_diff_pos, atime)
    if fix_result is not None:
        peak_offset = fix_result['peak_offset']
        peak_time = atime + np.exp(peak_offset)
        peak_time_proposal_lp =  peak_log_p(peak_cdf, atime, wn.srate, peak_time)
    else:
        peak_time, peak_time_proposal_lp =  sample_peak_time_from_cdf(peak_cdf, atime, wn.srate, return_lp=True)
        peak_offset = np.log(peak_time - atime) if peak_time-atime > 0 else np.log(1.0/wn.srate)
        # print "sampled times", peak_time, atime, peak_offset
    # jacobian transformation for change of variables:
    # let y = log t
    # p_t(t) dt = p_t(e^y) dt/dy dy
    #           = p_y(y)   e^y   dy
    # so to convert a dt log-density into a dy log-density, we add the log of e^y, i.e., y
    peak_offset_proposal_lp = peak_time_proposal_lp + peak_offset

    if fix_result is not None:
        x = np.array([fix_result['coda_height'], fix_result['peak_decay'], fix_result['coda_decay']])
    else:
        x = None

    if mixture_proposal:

        (coda_height, peak_decay, coda_decay), proposal_lp = amp_decay_mixture_proposal(wn, peak_time, env_diff_pos, fix_result=x)
        fitting_window = None
    else:
        fitting_window = None
        try:
            fitting_window = get_fit_window(wn, peak_time, env_diff_pos)
        except:
            pass

        if fitting_window is None:
            (coda_height, peak_decay, coda_decay), proposal_lp = propose_from_prior(sg, wn, x)
        else:
            if laplace_proposal:
                if return_debug:
                    (coda_height, peak_decay, coda_decay), proposal_lp, proposal_mean, proposal_prec, lik_grad = amp_decay_proposal_laplace(fitting_window, wn.srate, fix_result=x, return_debug=True)
                    proposal_cov = np.linalg.inv(proposal_prec)
                else:
                    (coda_height, peak_decay, coda_decay), proposal_lp = amp_decay_proposal_laplace(fitting_window, wn.srate, fix_result=x)
            elif laplace_reparam:
                tg = sg.template_generator(phase="UA")
                if return_debug:
                    (coda_height, peak_decay, coda_decay), proposal_lp, proposal_mean, proposal_prec, lik_grad = amp_decay_proposal_laplace_reparam(wn, tg, fitting_window, wn.srate, fix_result=x, return_debug=True)
                    proposal_cov = np.linalg.inv(proposal_prec)
                else:
                    (coda_height, peak_decay, coda_decay), proposal_lp = amp_decay_proposal_laplace_reparam(wn, tg, fitting_window, wn.srate, fix_result=x)
            else:
                if return_debug:
                    (coda_height, peak_decay, coda_decay), proposal_lp, proposal_mean, proposal_cov, lik_grad = amp_decay_proposal(fitting_window, wn.srate, fix_result=x, return_debug=True)
                else:
                    (coda_height, peak_decay, coda_decay), proposal_lp = amp_decay_proposal(fitting_window, wn.srate, fix_result=x)

    initial_vals = {'arrival_time': atime, 'peak_offset': peak_offset, 'coda_height': coda_height,
                    'coda_decay': coda_decay, 'peak_decay': peak_decay, 'mult_wiggle_std':  0.6}



    log_qforward = atime_proposal_lp + peak_time_proposal_lp + proposal_lp

    if return_debug:
        debug_info={'env_diff_pos': env_diff_pos,
                    'cdf2': cdf2,
                    'peak_cdf': peak_cdf,
                    'fitting_window': fitting_window,
                    'atime_proposal_lp': atime_proposal_lp,
                    'peak_time_proposal_lp': peak_time_proposal_lp,
                    'proposal_lp': proposal_lp,
                    'proposal_mean': proposal_mean,
                    'proposal_cov': proposal_cov,
                    'lik_grad': lik_grad,
                    'x': x,
                    'peak_dist': peak_dist,
                    'peak_env': peak_env,
                    'exp_peak_env': exp_peak_env,
                    'positive_derivs': positive_derivs,
                    'derivs': derivs,
                    'prior': prior}

        return initial_vals, log_qforward, debug_info
    else:
        return initial_vals, log_qforward

def optimizing_birth_helper(sg, wn, return_debug=False, **kwargs):
    lp_old = tmpl_move_logp(sg, wn, [wn,])

    if return_debug:
        r = optimizing_birth_proposal(sg, wn, return_debug=True, **kwargs)
        vals, tmpl_log_qforward = r[0:2]
    else:
        vals, tmpl_log_qforward = optimizing_birth_proposal(sg, wn, **kwargs)

    tmpl = sg.create_unassociated_template(wn, vals['arrival_time'], nosort=True, initial_vals=vals)
    sg._topo_sorted_list = tmpl.values() + sg._topo_sorted_list
    sg._gc_topo_sorted_nodes()

    eid = -tmpl["arrival_time"].tmid
    lp_new = tmpl_move_logp(sg, wn, [wn,] + tmpl.values())

    log_qforward = tmpl_log_qforward

    # reverse (death) probability is just the probability of killing a
    # random template
    c = template_kill_distribution(wn)
    log_qbackward = np.log(c[(eid, 'UA')])

    def accept_move():
        pass

    def revert_move():
        sg.destroy_unassociated_template(tmpl, nosort=True)
        # WARNING: this assumes the list hasn't been re-sorted by any
        # of our intermediate calls.
        sg._topo_sorted_list = sg._topo_sorted_list[len(tmpl):]
        sg._gc_topo_sorted_nodes()
        sg.next_uatemplateid -= 1
        #lp = sg.current_log_p()
        #assert(np.abs(lp - lp_old) < 1e-10)
    if return_debug:
        return lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move, r
    else:
        return lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move

def optimizing_birth_move(sg, wn,  **kwargs):
    lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move = optimizing_birth_helper(sg, wn, **kwargs)

    return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=accept_move, revert_move=revert_move)


def template_kill_distribution(wn):
    templates = [(eid, phase) for (eid, phase) in wn.arrivals() if eid < 0]
    if len(templates) == 0:
        return None

    c = Counter()
    for (i, (eid, phase)) in enumerate(templates):
        vals, _ = wn.get_template_params_for_arrival(eid, phase)
        c[(eid, phase)] = 1.0/np.exp(vals['coda_height'])
    c.normalize()

    return c

def death_move_for_optimizing_birth(sg, wn):
    c = template_kill_distribution(wn)
    if c is None:
        return False

    tmpl_to_destroy = c.sample()
    kill_choice_prob = c[tmpl_to_destroy]

    lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move = death_helper_for_optimizing_birth(sg, wn, tmpl_to_destroy, kill_choice_prob)
    return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=accept_move, revert_move=revert_move)


def death_helper_for_optimizing_birth(sg, wn, tmpl_to_destroy, kill_choice_prob):

    tnodes = sg.get_template_nodes(eid=tmpl_to_destroy[0], phase=tmpl_to_destroy[1], sta=wn.sta, band=wn.band, chan=wn.chan)

    ntemplates = len(sg.uatemplate_ids[(wn.sta, wn.chan, wn.band)])
    lp_old = tmpl_move_logp(sg, wn, [wn,] + [n for (k, n) in tnodes.values() ], n=ntemplates)
    orig_topo_sorted = copy.copy(sg._topo_sorted_list)

    log_qforward = np.log(kill_choice_prob)

    current_peak = tnodes['arrival_time'][1].get_value() + np.exp(tnodes['peak_offset'][1].get_value())
    eid = -tnodes["arrival_time"][1].tmid

    tvals = sg.get_template_vals(eid=tmpl_to_destroy[0], phase=tmpl_to_destroy[1], sta=wn.sta, band=wn.band, chan=wn.chan)

    for (param, (label, node)) in tnodes.items():
        sg.remove_node(node)
        sg._topo_sorted_list[node._topo_sorted_list_index] = None

    _, tmpl_log_qbackward = optimizing_birth_proposal(sg, wn, fix_result=tvals)
    log_qbackward = tmpl_log_qbackward

    lp_new = tmpl_move_logp(sg, wn, [wn,], n=ntemplates-1)

    def accept_move():
        uaid = -tmpl_to_destroy[0]
        del sg.uatemplates[uaid]
        sg.uatemplate_ids[(wn.sta,wn.chan,wn.band)].remove(uaid)

    def revert_move():

        for (param, (label, node)) in tnodes.items():
            sg.add_node(node)
            node.addChild(wn)
        wn.arrivals()
        sg._topo_sorted_list = orig_topo_sorted
        sg._gc_topo_sorted_nodes()

    return lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move
