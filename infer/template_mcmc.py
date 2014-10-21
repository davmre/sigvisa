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
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_event_station_chan
from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.models.distributions import Gaussian
from sigvisa.models.signal_model import extract_arrival_from_key, unify_windows
from sigvisa.models.wiggles.wiggle import extract_phase_wiggle_for_proposal
from sigvisa.infer.mcmc_basic import get_node_scales, gaussian_propose, gaussian_MH_move, MH_accept, hmc_step, hmc_step_reversing
from sigvisa.graph.graph_utils import create_key,parse_key
from sigvisa.graph.dag import get_relevant_nodes
from sigvisa.plotting.plot import savefig, plot_with_fit, plot_waveform
from sigvisa.utils.counter import Counter
from matplotlib.figure import Figure

import scipy.weave as weave
from scipy.weave import converters

import numdifftools as nd

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


######################################################################

def get_signal_based_amplitude_distribution(sg, sta, tmvals=None, peak_time=None, peak_period_s = 2.0):
    wn = sg.station_waves[sta][0]

    if peak_time is None:
        peak_time = tmvals['arrival_time'] + np.exp(tmvals['peak_offset'])
    peak_idx = int((peak_time - wn.st) * wn.srate)
    peak_period_samples = int(peak_period_s * wn.srate)
    peak_data=wn.get_value()[peak_idx - peak_period_samples:peak_idx + peak_period_samples]

    # if we land outside of the signal window, or during an unobserved (masked) portion,
    # we'll just sample from the event-conditional prior instead
    if ma.count(peak_data) == 0:
        return None

    peak_height = peak_data.mean()

    env_height = max(peak_height - wn.nm.c, wn.nm.c/100.0)

    peak_log_std = np.log(peak_data).std()

    return Gaussian(mean=np.log(env_height), std = max(peak_log_std, 0.1))

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
    normalized_env = d/s
    cdf = np.concatenate([np.array((0,)), np.cumsum(normalized_env)])
    return cdf

def peak_log_p(cdf, stime, srate, peak_time):
    # compute the probability that sample_peak_time_from_signal would
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
    return np.log(cdf[idx] - cdf[idx-1])

def get_signal_diff_positive_part(wave_node, arrival_set):
    value = wave_node.get_value().data
    pred_signal = wave_node.assem_signal(arrivals=arrival_set, include_wiggles=False)

    npts = wave_node.npts
    signal_diff_pos = wave_node.signal_diff
    code = """
for(int i=0; i < npts; ++i) {
double v = fabs(value(i)) - fabs(pred_signal(i));
signal_diff_pos(i) = v > 0 ? v : 0;
}
"""
    weave.inline(code,['npts', 'signal_diff_pos', 'value', 'pred_signal'],type_converters = converters.blitz,verbose=2,compiler='gcc')
    return signal_diff_pos

def get_current_conditional_cdf(wave_node, arrival_set):
    signal_diff_pos = get_signal_diff_positive_part(wave_node, arrival_set)
    return preprocess_signal_for_sampling(signal_diff_pos)


def sample_peak_time_from_signal(cdf, stime, srate, return_lp=False):
    u = np.random.rand()
    idx = np.searchsorted(cdf, u)
    peak_time = stime + float(idx-1)/srate

    if return_lp:
        return peak_time, np.log(cdf[idx]-cdf[idx-1])
        #return peak_time, np.log(1.0/len(cdf))
    return peak_time

def indep_peak_move(sg, wave_node, tmnodes, window_lps=None, std=None):
    arrival_key, arrival_node = tmnodes['arrival_time']
    offset_key, offset_node = tmnodes['peak_offset']
    relevant_nodes = [wave_node,]
    relevant_nodes += [arrival_node.parents[arrival_node.default_parent_key()],] if arrival_node.deterministic() else [arrival_node,]

    arr = extract_arrival_from_key(arrival_key, wave_node.r)
    other_arrs = wave_node.arrivals() - set(arr)


    current_atime = arrival_node.get_value(key=arrival_key)
    peak_offset = np.exp(offset_node.get_value(key=offset_key))

    cdf = get_current_conditional_cdf(wave_node, arrival_set=other_arrs)
    proposed_peak_time, proposal_lp =  sample_peak_time_from_signal(cdf, wave_node.st,
                                                                  wave_node.srate,
                                                                  return_lp=True)
    backward_propose_lp = peak_log_p(cdf, wave_node.st,
                                     wave_node.srate,
                                     peak_time = current_atime + peak_offset)

    proposed_arrival_time = proposed_peak_time - peak_offset

    proxy_lps = None
    if window_lps is not None:
        eid, phase = arr
        w_start, w_end = wave_node.template_idx_window(eid, phase)

        proposed_idx_offset = int((proposed_arrival_time - current_atime) * wave_node.srate)
        proposed_start = max(0, w_start + proposed_idx_offset)
        proposed_end = min(wave_node.npts, w_end + proposed_idx_offset)
        w = unify_windows((w_start, w_end), (proposed_start, proposed_end))
        proxy_lps = wave_node.window_lps_to_proxy_lps(window_lps, w)

    return MH_accept(sg, keys=(arrival_key,),
                     oldvalues = (current_atime,),
                     newvalues = (proposed_arrival_time,),
                     log_qforward = proposal_lp,
                     log_qbackward = backward_propose_lp,
                     node_list = (arrival_node,),
                     relevant_nodes = relevant_nodes,
                     proxy_lps=proxy_lps)

######################################################################

def update_wiggle_submove(sg, wave_node, tmnodes, atime_key,
                          atime_node, old_atime, new_atime):
    # this function factors out the common bit of improve_offset_move
    # and improve_atime_move, which shifts the wiggles in time to
    # correspond for the new template arrival time

    eid, phase, sta, chan, band, param = parse_key(atime_key)

    # adjust wiggles for that new time
    wg = sg.wiggle_generator(phase, wave_node.srate)
    wnodes = [(p, tmnodes[p]) for p in wg.params()]
    wiggle_vals = [n.get_value(k) for (p, (k,n)) in wnodes]
    wiggle_vals_new = np.array(wiggle_vals, copy=True)
    wg.timeshift_param_array(wiggle_vals_new, new_atime-old_atime)
    wiggle_vals_new = list(wiggle_vals_new)

    # consider the proposed arrival time along with new adjusted wiggles.
    # HACK note: we hard-code the assumption of fourier wiggles, where we
    # know that a timeshift will only change the phase parameters (the latter
    # half of the param array).
    d2 = wg.dimension()/2
    phase_wnodes = wnodes[d2:]
    phase_nodes = [n for (p,(k, n)) in phase_wnodes]
    phase_keys = [k for (p,(k, n)) in phase_wnodes]

    relevant_nodes = phase_nodes
    node_list = [atime_node,] + phase_nodes
    keys = [atime_key,] + phase_keys
    oldvalues = [old_atime,] + wiggle_vals[d2:]
    newvalues = [new_atime,] + wiggle_vals_new[d2:]

    return relevant_nodes, node_list, keys, oldvalues, newvalues

def improve_offset_move_gaussian(sg, wave_node, tmnodes, std=0.5, window_lps=None, **kwargs):
    arrival_key, arrival_node = tmnodes['arrival_time']
    offset_key, offset_node = tmnodes['peak_offset']

    current_offset = offset_node.get_value(key=offset_key)
    proposed_offset = gaussian_propose(sg, keys=(offset_key,),
                                       node_list=(offset_node,),
                                       values=(current_offset,),
                                       std=std, **kwargs)[0]
    return improve_offset_move(sg, wave_node, tmnodes, proposed_offset, window_lps=window_lps)

def improve_offset_move_indep(sg, wave_node, tmnodes,  **kwargs):
    arrival_key, arrival_node = tmnodes['arrival_time']
    offset_key, offset_node = tmnodes['peak_offset']

    current_offset = offset_node.get_value(key=offset_key)
    reverse_lp = offset_node.log_p(v=current_offset)
    proposed_offset = offset_node.parent_sample(set_new_value=False)
    move_lp = offset_node.log_p(v=proposed_offset)
    return improve_offset_move(sg, wave_node, tmnodes, proposed_offset, move_lp=move_lp, reverse_lp=reverse_lp, **kwargs)


def improve_offset_move(sg, wave_node, tmnodes, proposed_offset, move_lp=0, reverse_lp=0, window_lps=None,  **kwargs):
    """
    Update the peak_offset while leaving the peak time constant, i.e.,
    adjust the arrival time to compensate for the change in offset.
    """

    proxy_lps = wave_node.window_lps_to_proxy_lps(window_lps)

    arrival_key, arrival_node = tmnodes['arrival_time']
    offset_key, offset_node = tmnodes['peak_offset']
    relevant_nodes = [wave_node,]
    relevant_nodes += [arrival_node.parents[arrival_node.default_parent_key()],] if arrival_node.deterministic() else [arrival_node,]
    relevant_nodes += [offset_node.parents[offset_node.default_parent_key()],] if offset_node.deterministic() else [offset_node,]

    current_offset = offset_node.get_value(key=offset_key)
    atime = arrival_node.get_value(key=arrival_key)
    new_atime = atime + (np.exp(current_offset) - np.exp(proposed_offset))

    rn_tmp, node_list, keys, oldvalues, newvalues = update_wiggle_submove(sg, wave_node, tmnodes,
                                                                          arrival_key, arrival_node,
                                                                          atime, new_atime)
    relevant_nodes += rn_tmp
    node_list.append(offset_node)
    newvalues.append(proposed_offset)
    oldvalues.append(current_offset)
    keys.append(offset_key)


    accepted = MH_accept(sg=sg, keys=keys,
                         oldvalues=oldvalues,
                         newvalues = newvalues,
                         node_list = node_list,
                         relevant_nodes=relevant_nodes,
                         log_qforward=move_lp,
                         log_qbackward=reverse_lp,
                         proxy_lps=proxy_lps)
    return accepted

def improve_atime_move(sg, wave_node, tmnodes, std=1.0, window_lps=None, **kwargs):
    # here we re-implement get_relevant_nodes from sigvisa.graph.dag, with a few shortcuts
    k_atime, n_atime = tmnodes['arrival_time']
    eid, phase, sta, chan, band, param = parse_key(k_atime)

    # propose a new arrival time
    relevant_nodes = [wave_node,]
    relevant_nodes += [n_atime.parents[n_atime.default_parent_key()],] if n_atime.deterministic() else [n_atime,]

    old_atime = n_atime.get_value(k_atime)
    values = (old_atime,)
    atime_proposal = float(gaussian_propose(sg, keys=(k_atime,),
                                            node_list=(n_atime,),
                                            values=(values), std=std,
                                            **kwargs))

    rn_tmp, node_list, keys, oldvalues, newvalues = update_wiggle_submove(sg, wave_node, tmnodes,
                                                                          k_atime, n_atime,
                                                                          old_atime, atime_proposal)
    relevant_nodes += rn_tmp

    proxy_lps = wave_node.window_lps_to_proxy_lps(window_lps)
    accepted = MH_accept(sg, keys, oldvalues, newvalues, node_list, relevant_nodes, proxy_lps=proxy_lps)
    return accepted

#just for debugging
"""
def do_atime_move(sg, wave_node, tmnodes, atime_offset):
    k_atime, n_atime = tmnodes['arrival_time']
    eid, phase, sta, chan, band, param = parse_key(k_atime)

    # propose a new arrival time
    relevant_nodes = [wave_node,]
    parent = n_atime.parents[n_atime.default_parent_key()]
    relevant_nodes.append(parent)
    old_atime = n_atime.get_value(k_atime)
    values = (old_atime,)
    atime_proposal = old_atime + atime_offset

    # adjust wiggles for that new time
    wg = sg.wiggle_generator(phase, wave_node.srate)
    wnodes = [(p, tmnodes[p]) for p in wg.params()]
    wiggle_vals = [n.get_value(k) for (p, (k,n)) in wnodes]
    wiggle_vals_new = np.array(wiggle_vals, copy=True)
    wg.timeshift_param_array(wiggle_vals_new, atime_offset)

    for (i, (p, (k, n))) in enumerate(wnodes):
        n.set_value(key=k, value=wiggle_vals_new[i])
    n_atime.set_value(key=k_atime, value=atime_proposal)
"""
#######################################################################

def get_sorted_arrivals(wave_node):
    # choose a template at random to try to split
    arrivals = wave_node.arrivals()
    arr_params = []
    for (eid, phase) in arrivals:
        arr_params.append((wave_node.get_template_params_for_arrival(eid, phase)[0], eid, phase))
    sorted_arrs = sorted(arr_params, key = lambda x : x[0]['arrival_time'])
    return sorted_arrs

"""
def atime_window(wave_node, sorted_arrs, k):
    # when splitting an arrival k, the new arrival can be created anytime between arrivals k-1 and k+1

    n = len(sorted_arrs)
    atime_diff = np.float('inf')
    atime_window_start = wave_node.st
    atime_window_end = wave_node.et
    if k < n-1:
        atime_window_end = min(atime_window_end, sorted_arrs[k+1][0]['arrival_time'])
    if k > 0:
        atime_window_start = max(atime_window_start, sorted_arrs[k-1][0]['arrival_time'])
    atime_window_len = atime_window_end - atime_window_start

    return atime_window_start, atime_window_len
"""

def sample_arr_to_split(sg, wave_node):
    sorted_arrs = get_sorted_arrivals(wave_node)
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

def tmpl_move_logp(sg, sta, relevant_nodes, n=None):
    return sg.ntemplates_sta_log_p(sta, n=n) + sg.joint_logprob_keys(relevant_nodes)

"""
def split_move(sg, wave_node, return_probs=False, force_accept=False):

    # figure out which arrival to split
    arr, k, sorted_arrs = sample_arr_to_split(sg, wave_node)
    if arr is None:
        return False
    atime_window_start, atime_window_len = atime_window(wave_node, sorted_arrs, k)
    n_arrs = len(sorted_arrs)
    tnodes = sg.get_template_nodes(eid=arr[1], phase=arr[2], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)

    # account for possibly setting the coda_height of an event, or an unass template
    k,n = tnodes['coda_height']
    if n.deterministic():
        relevant_parent = [n.parents[n.default_parent_key()],]
    else:
        relevant_parent = [n,]
    lp_old = tmpl_move_logp(sg, wave_node.sta, [wave_node,] + relevant_parent)

    # create the new uatemplate, with arrival time sampled uniformly
    eps_atime = np.random.rand()*atime_window_len
    new_atime = atime_window_start + eps_atime
    new_tmpl = sg.create_unassociated_template(wave_node, atime=new_atime, nosort=True, sample_wiggles=True)
    sg._topo_sorted_list = new_tmpl.values() + sg._topo_sorted_list
    sg._gc_topo_sorted_nodes()

    # split off some fraction of the amplitude to the new node
    k,n = tnodes['coda_height']
    coda_height = n.get_value(key=k)
    u = np.random.rand()
    new_logheight = np.log(u) + coda_height
    new_tmpl['coda_height'].set_value(new_logheight)
    n.set_value( key=k, value=np.log(1-u) + coda_height)

    lp_new = tmpl_move_logp(sg, wave_node.sta, [wave_node,] + relevant_parent + new_tmpl.values())

    new_tmpl_priorsampled = [n for (p, n) in new_tmpl.items() if p not in ['coda_height', 'tt_residual', 'arrival_time']]
    log_qforward = sg.joint_logprob_keys(new_tmpl_priorsampled) - np.log(atime_window_len) - np.log(n_arrs)
    jacobian_determinant = 1.0/ (u * (1-u))

    # the reverse probability is the prob that we would have chosen to merge these two templates
    new_sorted_arrs = get_sorted_arrivals(wave_node)
    k,n = tnodes['arrival_time']
    old_atime = n.get_value(key=k)
    first_atime = min(old_atime, new_atime)
    log_qbackward = np.log(arrival_merge_prob(sg, wave_node.sta,
                                              get_atime_index(new_sorted_arrs, first_atime),
                                              sorted_arrs=new_sorted_arrs))
    # if we are splitting an unass template, the reverse merge move
    # would have had to decide to merge into *this* template rather
    # than the other one
    if not tnodes['coda_height'][1].deterministic():
        log_qbackward += np.log(.5)

    u = np.random.rand()
    if force_accept or ((lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant > np.log(u)):
        print "split template %d from %d: %.1f + %.5f - (%.1f + %.5f) + %f = %.1f vs %.1f" % (-new_tmpl["arrival_time"].tmid, arr[1], lp_new, log_qbackward, lp_old, log_qforward, jacobian_determinant, (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant, np.log(u))

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
"""

#def arrivals_merge_distribution(sg, sta, sorted_arrs=None):
#    """
#
#    We consider merging any two adjacent arrivals, as long as at least
#    one of them is unassociated.  The proposal probability for merging
#    any particular pair of arrivals is inversely proportional to the
#    difference in arrival times.
#
#    """
#    assert(len(sg.station_waves[sta]) == 1)
#    wave_node = list(sg.station_waves[sta])[0]

#    if sorted_arrs is None:
#        sorted_arrs = get_sorted_arrivals(wave_node)
#    n = len(sorted_arrs)
#    if n < 2:
#        return None, sorted_arrs

#    c = Counter()
#    for k in range(n-1):
#        if sorted_arrs[k][1] >= 0 and sorted_arrs[k+1][1] >= 0:
#            # can't merge two event arrivals
#            continue
#        c[k] = 1.0 / (sorted_arrs[k+1][0]['arrival_time'] - sorted_arrs[k][0]['arrival_time'] + 1.0/wave_node.srate)
#    c.normalize()

#    return c, sorted_arrs

"""
def sample_arrivals_to_merge(sg, sta, **kwargs):
    c, sorted_arrs = arrivals_merge_distribution(sg, sta, **kwargs)
    if c is None or len(c) == 0:
        return None, None, None

    k = c.sample()
    return sorted_arrs[k], sorted_arrs[k+1], c[k]

def arrival_merge_prob(sg, sta, k, **kwargs):
    c, sorted_arrs = arrivals_merge_distribution(sg, sta, **kwargs)
    if c is None:
        return None
    return c[k]
"""

def relevant_nodes(tnodes):
    return [n if not n.deterministic() else n.parents[n.default_parent_key()] for (k,n) in tnodes.values()]

"""
def merge_move(sg, wave_node, return_probs=False):

    #lp1 = sg.current_log_p()

    # sample which two arrivals to merge
    n_arrs = len(wave_node.arrivals())
    arr1, arr2, merge_choice_prob = sample_arrivals_to_merge(sg, wave_node.sta)
    if arr1 is None:
        return False

    # get all relevant nodes for the arrivals we sampled
    t1nodes = sg.get_arrival_nodes(eid=arr1[1], phase=arr1[2], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)
    t2nodes = sg.get_arrival_nodes(eid=arr2[1], phase=arr2[2], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)

    # save the probability before we actually make the move
    log_qforward = 0.0
    ntemplates = len(sg.uatemplate_ids[(wave_node.sta, wave_node.chan, wave_node.band)])
    lp_old = tmpl_move_logp(sg, wave_node.sta, [wave_node,] + relevant_nodes(t1nodes) + relevant_nodes(t2nodes), n=ntemplates)
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

    lp_new = tmpl_move_logp(sg, wave_node.sta, [wave_node,] + relevant_nodes(keep_nodes), n=ntemplates-1)

    log_qforward += np.log(merge_choice_prob)

    u = amp1/(amp1+amp2)
    # note (1-u) = amp2/(amp1+amp2), so despite appearances this
    # determinant is actually symmetric in amp1, amp2.
    jacobian_determinant = 1.0/(u * (1-u))

    # compute arrival time window for reverse (split) move
    k,n = keep_nodes['arrival_time']
    merged_atime = n.get_value(key=k)
    sorted_arrs = get_sorted_arrivals(wave_node)
    idx = get_atime_index(sorted_arrs, merged_atime)
    atime_window_start, atime_window_len = atime_window(wave_node, sorted_arrs, idx)

    # we can ignore keys and just take the node log_p because we know
    # the lost node is always going to be a single uatemplate (not an event, etc)
    new_tmpl_priorsampled = [n for (p, (k,n)) in lost_nodes.items() if p not in ['coda_height', 'arrival_time', 'tt_residual']]
    log_qbackward = sg.joint_logprob_keys(new_tmpl_priorsampled) - np.log(atime_window_len) - np.log(n_arrs-1)

    u = np.random.rand()
    if (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant > np.log(u):
        print "merged templates: %.1f + %.5f - (%.1f + %.5f) + %f = %.1f vs %.1f" % (lp_new, log_qbackward, lp_old, log_qforward, jacobian_determinant, (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant, np.log(u))

        uaid = lost_nodes['arrival_time'][1].tmid
        del sg.uatemplates[uaid]
        sg.uatemplate_ids[(wave_node.sta,wave_node.chan,wave_node.band)].remove(uaid)


        if return_probs:
            print "proposed", arr1, arr2, keep1
            return True, lp_new, lp_old, log_qforward, log_qbackward, jacobian_determinant
        else:
            return True
    else:


        for (param, (key, node)) in lost_nodes.items():
            sg.add_node(node)
            node.addChild(wave_node)
        for (param, (key, node)) in keep_nodes.items():
            if param in keep_vals:
                node.set_value(key=key, value=keep_vals[param])
        wave_node.arrivals()
        sg._topo_sorted_list = orig_topo_sorted
        sg._gc_topo_sorted_nodes()


        #lp2 = sg.current_log_p()
        #assert(lp1 == lp2)

        #lp = sg.current_log_p()
        #assert(np.abs(lp - lp_old) < 1e-10)

        if return_probs:
            print "proposed", arr1, arr2, keep1
            return False, lp_new, lp_old, log_qforward, log_qbackward, jacobian_determinant
        else:
            return False
"""



#######################################################################

def get_wiggles_from_signal(eid, phase, wave_node, wg, peak_offset):
    arrivals = wave_node.arrivals()
    arr = (eid, phase)
    signal_data = extract_phase_wiggle_for_proposal(arr, arrivals, wave_node, wg)
    features = wg.features_from_signal(signal_data, return_array=True)
    new_signal = wg.signal_from_features(features)

    return features

def propose_wiggles_from_signal(eid, phase, wave_node, wg, nodes, amp_std=0.02, phase_std=0.02):
    # the proposal distribution for wiggles is a Gaussian centered around an FFT of the current
    # unexplained signal

    if wg.basis_type() == "dummy":
        return 0.0

    features = get_wiggles_from_signal(eid, phase, wave_node, wg, peak_offset=np.exp(node_get_value(nodes, 'peak_offset')))
    assert(wg.basis_type() == "fourier")
    dim = wg.dimension()

    amp_residuals = np.random.randn(dim/2) * amp_std
    phase_residuals = np.random.randn(dim/2) * phase_std
    residuals = np.concatenate([amp_residuals, phase_residuals])
    features += residuals
    features[dim/2:] = features[dim/2:] % (1.0)
    for (i,param) in enumerate(wg.params()):
        node_set_value(nodes, param, value=features[i])

    lp = -.5 * np.log(2*np.pi*amp_std**2) +  - .5 * np.sum( (amp_residuals)**2 ) / amp_std**2
    lp += -.5 * np.log(2*np.pi*phase_std**2) - .5 * np.sum( (phase_residuals)**2 ) / phase_std**2

    return lp


def wiggle_proposal_lprob_from_signal(eid, phase, wave_node, wg, wnodes=None, wvals=None, amp_std=0.02, phase_std=0.02):
    if wg.basis_type() == "dummy":
        return 0.0

    features = np.empty((wg.dimension(),))
    for (i,param) in enumerate(wg.params()):
        if wvals is not None:
            features[i] = wvals[param]
        else:
            features[i] = node_get_value(wnodes, param)

    assert(wg.basis_type() == "fourier")
    dim = wg.dimension()

    peak_offset = np.exp(wvals['peak_offset'] if wvals is not None else node_get_value(wnodes, 'peak_offset'))

    signal_features = get_wiggles_from_signal(eid, phase, wave_node, wg, peak_offset=peak_offset)
    residuals = signal_features - features
    amp_residuals = residuals[:dim/2]

    # compute distances wrapping around the unit circle
    phase_residuals = residuals[dim/2:] % 1
    phase_wraparound = (phase_residuals > .5)
    phase_residuals[phase_wraparound] = 1 - phase_residuals[phase_wraparound]

    lp = -.5 * np.log(2*np.pi*amp_std**2) - .5 * np.sum((amp_residuals)**2) / amp_std**2
    lp += -.5 * np.log(2*np.pi*phase_std**2) - .5 * np.sum((phase_residuals)**2) / phase_std**2
    return lp

def birth_move(sg, wave_node,  **kwargs):
    lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move = birth_helper(sg, wave_node, **kwargs)
    return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=accept_move, revert_move=revert_move)

def birth_helper(sg, wave_node,  **kwargs):
    #lp_old1 = sg.current_log_p()
    lp_old = tmpl_move_logp(sg, wave_node.sta, [wave_node,])

    cdf = get_current_conditional_cdf(wave_node, arrival_set=wave_node.arrivals())
    peak_time, atime_proposal_lp =  sample_peak_time_from_signal(cdf, wave_node.st,
                                                           wave_node.srate,
                                                           return_lp=True)
    plp = peak_log_p(cdf, wave_node.st,
                     wave_node.srate,
                     peak_time = peak_time)


    logamplitude_proposal_dist = get_signal_based_amplitude_distribution(sg, wave_node.sta, peak_time=peak_time)
    if logamplitude_proposal_dist is None:
        tg = sg.template_generator(phase="UA")
        logamplitude_proposal_dist = tg.unassociated_model('coda_height')
    proposed_logamp = logamplitude_proposal_dist.sample()
    proposed_logamp_lp = logamplitude_proposal_dist.log_p(proposed_logamp)

    tmpl = sg.create_unassociated_template(wave_node, peak_time, nosort=True, **kwargs)
    sg._topo_sorted_list = tmpl.values() + sg._topo_sorted_list
    sg._gc_topo_sorted_nodes()
    tmpl["arrival_time"].set_value(peak_time - np.exp(tmpl["peak_offset"].get_value()))
    tmpl["coda_height"].set_value(proposed_logamp)

    eid = -tmpl["arrival_time"].tmid
    wg = sg.wiggle_generator(phase="UA", srate=wave_node.srate)
    wiggle_proposal_lp = propose_wiggles_from_signal(eid, 'UA', wave_node, wg, tmpl)
    lp_new = tmpl_move_logp(sg, wave_node.sta, [wave_node,] + tmpl.values())

    # probability of this birth move is the product of probabilities
    # of all sampled params (including arrival time)
    log_qforward = atime_proposal_lp + proposed_logamp_lp + wiggle_proposal_lp
    for (key, node) in tmpl.items():
        if key == "arrival_time" or key=="coda_height": continue
        if key.startswith('phase') or key.startswith('amp'): continue
        log_qforward += node.log_p()

    # reverse (death) probability is just the probability of killing a
    # random template
    ntemplates = len([1 for (eid, phase) in wave_node.arrivals() if eid < 0])
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

def death_move(sg, wave_node):
    templates = [(eid, phase) for (eid, phase) in wave_node.arrivals() if eid < 0]
    if len(templates) < 1:
        return False

    u0 = np.random.rand()
    for i in range(len(templates)):
        if u0 <= float(i+1)/len(templates):
            tmpl_to_destroy = templates[i]
            break

    lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move = death_helper(sg, wave_node, tmpl_to_destroy)
    return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=accept_move, revert_move=revert_move)


def death_helper(sg, wave_node, tmpl_to_destroy):

    tnodes = sg.get_template_nodes(eid=tmpl_to_destroy[0], phase=tmpl_to_destroy[1], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)
    wnodes = sg.get_wiggle_nodes(eid=tmpl_to_destroy[0], phase=tmpl_to_destroy[1], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)

    ntemplates = len(sg.uatemplate_ids[(wave_node.sta, wave_node.chan, wave_node.band)])
    lp_old = tmpl_move_logp(sg, wave_node.sta, [wave_node,] + [n for (k, n) in tnodes.values() + wnodes.values()], n=ntemplates)
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
    wg = sg.wiggle_generator(phase="UA", srate=wave_node.srate)
    allnodes = dict([(p, (k,n)) for (p, (k,n)) in tnodes.items() + wnodes.items()])
    wiggle_proposal_lp = wiggle_proposal_lprob_from_signal(eid, 'UA', wave_node, wg, allnodes)

    logamplitude_proposal_dist = get_signal_based_amplitude_distribution(sg, wave_node.sta, peak_time=current_peak)
    if logamplitude_proposal_dist is None:
        tg = sg.template_generator(phase="UA")
        logamplitude_proposal_dist = tg.unassociated_model('coda_height')
    current_logamp = tnodes['coda_height'][1].get_value()
    proposed_logamp_lp = logamplitude_proposal_dist.log_p(current_logamp)

    log_qbackward = wiggle_proposal_lp + proposed_logamp_lp
    for (param, (label, node)) in tnodes.items():
        if param != "arrival_time" and param != "coda_height":
            log_qbackward += node.log_p()
        sg.remove_node(node)
        sg._topo_sorted_list[node._topo_sorted_list_index] = None
    for (param, (label, node)) in wnodes.items():
        sg.remove_node(node)
        sg._topo_sorted_list[node._topo_sorted_list_index] = None

    arrs = wave_node.arrivals()
    cdf = get_current_conditional_cdf(wave_node, arrival_set=arrs)
    peak_lp = peak_log_p(cdf, wave_node.st,
                         wave_node.srate,
                         peak_time = current_peak)
    log_qbackward += peak_lp

    lp_new = tmpl_move_logp(sg, wave_node.sta, [wave_node,], n=ntemplates-1)

    def accept_move():
        uaid = -tmpl_to_destroy[0]
        del sg.uatemplates[uaid]
        sg.uatemplate_ids[(wave_node.sta,wave_node.chan,wave_node.band)].remove(uaid)

    def revert_move():

        for (param, (label, node)) in tnodes.items() + wnodes.items():
            sg.add_node(node)
            node.addChild(wave_node)
        wave_node.arrivals()
        sg._topo_sorted_list = orig_topo_sorted
        sg._gc_topo_sorted_nodes()

    return lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move



#####################################################################

from sigvisa.infer.optimize.optim_utils import construct_optim_params


from sigvisa.models.distributions import Uniform, Laplacian, Gaussian

import scipy.stats

def mh_accept_util(lp_old, lp_new, log_qforward=0, log_qbackward=0, jacobian_determinant=0, accept_move=None, revert_move=None):
    print lp_new, lp_old, log_qbackward, log_qforward, jacobian_determinant, "FINAL", (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant
    u = np.random.rand()
    if (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant > np.log(u):
        if accept_move:
            accept_move()
        return True
    else:
        if revert_move:
            revert_move()
        return False

def get_template_values(tmnodes, sorted_params):
    return np.array([tmnodes[p][1].get_value(tmnodes[p][0]) for p in sorted_params])

def set_template_values(tmnodes, vals, sorted_params):
    for param,v in zip(sorted_params, vals):
        key, node = tmnodes[param]
        node.set_value(key=key, value=v)

def template_logprob(nodes, vals, sorted_params, no_atime=False):
    v = get_template_values(nodes, sorted_params)
    set_template_values(nodes, vals, sorted_params)
    lp = np.sum([node.log_p() for k,node in nodes.values() if not no_atime or 'arrivel_time' not in k])

    set_template_values(nodes, v, sorted_params)
    return lp

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
        snr = amp / wn.nm.c

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

def merge_source_distribution(sg, wn, k, sorted_arrs, width = 2.0):
    # given a target template, sample another template to attempt merging into that template
    # currently, this likelihood of sampling a template is Laplacian in the arrival time
    # (so templates with very nearby arrivals are more likely to be merged).

    n = len(sorted_arrs)

    atime1 = sorted_arrs[k][0]['arrival_time']

    c = Counter()
    for j in range(n):
        if j==k: continue
        if sorted_arrs[j][2] != "UA": continue
        atime2 = sorted_arrs[j][0]['arrival_time']
        c[j] = np.exp(- np.abs(atime2-atime1)/width)
    c.normalize()
    return c

def sample_arrivals_to_merge_twostep(sg, wn, width=2.0):
    c1, sorted_arrs = merge_target_distribution(sg, wn)
    if c1 is None:
        return None, None, 1.0, sorted_arrs

    target_k = c1.sample()
    arr_target = sorted_arrs[target_k]

    c2 = merge_source_distribution(sg, wn, target_k, sorted_arrs, width=width)
    source_k = c2.sample()
    arr_source = sorted_arrs[source_k]
    merge_choice_prob = c1[target_k]*c2[source_k]

    return arr_target, arr_source, merge_choice_prob, sorted_arrs


def merge_overall(sg, wave_node, split_atime_width=2.0):
    arr1, arr2, merge_choice_prob, sorted_arrs = sample_arrivals_to_merge_twostep(sg, wave_node, width=split_atime_width)

    print "merge called"

    if arr1 is None:
        return False

    lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move = merge_helper(sg, wave_node, arr1, arr2, merge_choice_prob, split_atime_width=split_atime_width)
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

    proxy_lps =  wn.cache_latent_signal_for_fixed_window(eid, phase, force_bounds=False)

    t0 = time.time()
    sg.joint_optimize_nodes(tmnode_list, optim_params=construct_optim_params(),
                            proxy_lps = proxy_lps)
    t1 = time.time()
    #optimized_vals = get_template_values(keep_nodes, sorted_params)

    from sigvisa.infer.run_mcmc import single_template_MH
    np.random.seed(1)
    sp, vals = single_template_MH(sg, wn, keep_nodes, phase=phase, steps=mh_burnin_steps+mh_samples, proxy_lps=proxy_lps)
    t2 = time.time()
    print "opt time: %s, mcmc time: %s" % (t1-t0, t2-t1)
    assert(sp == sorted_params)

    if mh_samples==0:
        # treat any MH burnin as just part of the optimization;
        # compute a Laplace approximation around the current value.
        m = vals[-1,:]

        import numdifftools as nd
        H = nd.Hessian(lb, step_nom=[1.0, 0.1, 0.1, 0.1, 0.1])
        e_val, e_vec = np.linalg.eig(H.hessian(m))
        C = np.dot(e_vec, np.dot(np.diag(np.abs(1.0/e_val)), e_vec.T))

    else:
        Q = np.cov(vals[mh_burnin_steps:,:].T)
        m = np.mean(vals[mh_burnin_steps:,:], axis=0)

    return scipy.stats.multivariate_normal(mean=m, cov=Q)


def merge_helper(sg, wn, arr1, arr2, merge_choice_prob, split_atime_width):

    vals1, eid1, phase1 = arr1
    vals2, eid2, phase2 = arr2
    n_arrs = len(wn.arrivals())

    # by default, the param values are pointers to the cache
    # kept by the wave node, which will change as we optimize them.
    # so we have to make our own copy in order to revert if the
    # move is rejected.
    vals1 = copy.copy(vals1)
    log_qforward = np.log(merge_choice_prob)

    n1 = sg.get_arrival_nodes(eid1, wn.sta, phase1, wn.band, wn.chan)
    n2 = sg.get_arrival_nodes(eid2, wn.sta, phase2, wn.band, wn.chan)

    # get parameter values of both templates in vector form
    sorted_params = sorted([k for k in n1.keys() if k not in ('tt_residual', 'amp_transfer')])
    v1 = [vals1[p] for p in sorted_params]
    v2 = [vals2[p] for p in sorted_params]

    #lp_old_full = sg.current_log_p()
    lp_old = tmpl_move_logp(sg, wn.sta, [wn,] + relevant_nodes(n1) + relevant_nodes(n2), n=n_arrs)

    orig_topo_sorted = copy.copy(sg._topo_sorted_list)


    # remove old template, and
    remove_old_template(sg, wn, arr2, n2)
    proposal = get_merge_distribution(sg, wn, arr1, n1, sorted_params)
    proposed_vals = proposal.rvs()
    log_qforward += proposal.logpdf(proposed_vals)

    set_template_values(n1, proposed_vals, sorted_params)


    lp_new = tmpl_move_logp(sg, wn.sta, [wn,] + relevant_nodes(n1), n=n_arrs-1)
    #lp_new_full = sg.current_log_p() - sg.ntemplates_sta_log_p(sta, n=n_arrs) + sg.ntemplates_sta_log_p(sta, n=n_arrs-1)


    atime_idx = sorted_params.index("arrival_time")
    atime_dist = Laplacian(proposed_vals[atime_idx], split_atime_width)
    conditional_logprob_atime1 = atime_dist.log_p(v1[atime_idx])
    conditional_logprob_atime2 = atime_dist.log_p(v2[atime_idx])

    log_qbackward = template_logprob(n1, v1, sorted_params, no_atime=True) +  \
                    template_logprob(n2, v2, sorted_params, no_atime=True) -  \
                    np.log(n_arrs-1)
    log_qbackward += conditional_logprob_atime1 + conditional_logprob_atime2

    def accept_move():
        uaid = n2['arrival_time'][1].tmid
        del sg.uatemplates[uaid]
        sg.uatemplate_ids[(wn.sta,wn.chan,wn.band)].remove(uaid)

    def revert_move():
        for (param, (key, node)) in n2.items():
            sg.add_node(node)
            node.addChild(wn)

        for (param, (key, node)) in n1.items():
            node.set_value(vals1[param], key=key)

        wn.arrivals()
        sg._topo_sorted_list = orig_topo_sorted
        sg._gc_topo_sorted_nodes()

    import pdb; pdb.set_trace()
    return lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move

def split_overall(sg, wave_node, atime_width=2.0):
    arr, k, sorted_arrs = sample_arr_to_split(sg, wave_node)
    if arr is None:
        return False
    lp_old, lp_new, log_qforward, log_qbackward, accept_move, revert_move = split_helper(sg, wave_node, arr, k, sorted_arrs, atime_width)
    return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=accept_move, revert_move=revert_move)

def split_helper(sg, wn, arr, k, sorted_arrs, atime_width, t1_values=None, t2_values=None):
    old_vals_dict, split_eidl, split_phase = arr
    old_vals_dict = copy.copy(old_vals_dict)

    n_arrs = len(sorted_arrs)
    log_qforward = -np.log(n_arrs)

    tnodes = sg.get_template_nodes(eid=arr[1], phase=arr[2], sta=wn.sta, band=wn.band, chan=wn.chan)
    sorted_params = [k for k in sorted(old_vals_dict.keys()) if k not in ('tt_residual', 'amp_transfer')]
    old_vals = [old_vals_dict[p] for p in sorted_params]

    t1_is_event = tnodes['coda_height'][1].deterministic()

    #lp_old_full = sg.current_log_p()
    lp_old = tmpl_move_logp(sg, wn.sta, [wn,] + relevant_nodes(tnodes), n=n_arrs)

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
            if t1_values:
                n.set_value(t1_values[p], key=k)
            else:
                n.parent_sample()
            log_qforward += n.log_p()


    # starting with the newly-sampled template, find the optimal single template to get the reverse probability
    split_v1 = get_template_values(tnodes, sorted_params)
    merge_dist = get_merge_distribution(sg, wn, arr, tnodes, sorted_params)
    log_qbackward = merge_dist.logpdf(old_vals)
    set_template_values(tnodes, split_v1, sorted_params)

    # sample second template from unassociated template prior
    new_time2 = atime_dist.sample()
    new_tmpl = sg.create_unassociated_template(wn, atime=new_time2, nosort=True, sample_wiggles=True)

    if t2_values:
        new_time2 = t2_values['arrival_time']
        for (p, n) in new_tmpl.items():
            n.set_value(t2_values[p])

    sg._topo_sorted_list = new_tmpl.values() + sg._topo_sorted_list
    sg._gc_topo_sorted_nodes()
    new_tmpl_priorsampled = [n for (p, n) in new_tmpl.items() if p not in ['arrival_time']]
    log_qforward += sg.joint_logprob_keys(new_tmpl_priorsampled)
    log_qforward += atime_dist.log_p(new_time2)

    lp_new = tmpl_move_logp(sg, wn.sta, [wn,] + relevant_nodes(tnodes) + new_tmpl.values(), n=n_arrs+1)


    # the reverse probability incorporates the prob that we would have chosen to merge these two templates
    new_sorted_arrs = get_sorted_arrivals(wn)
    c1_idx = get_atime_index(new_sorted_arrs, new_time1)
    c2_idx = get_atime_index(new_sorted_arrs, new_time2)

    c1, _ = merge_target_distribution(sg, wn, sorted_arrs=new_sorted_arrs)
    c2 = merge_source_distribution(sg, wn, sorted_arrs=new_sorted_arrs, k=c1_idx, width=atime_width)

    log_qbackward += np.log(c1[c1_idx]) + np.log(c2[c2_idx])

    def accept_move():
        assert(np.abs(sg.current_log_p() - lp_new) < 1e-10)
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

def adjacent_pair_distribution(sg, wave_node):
    sorted_arrs = get_sorted_arrivals(wave_node)
    n = len(sorted_arrs)
    if n < 2:
        return None, sorted_arrs

    c = Counter()
    for k in range(n-1):
        if sorted_arrs[k][1] < 0 and sorted_arrs[k+1][1] < 0:
            # no point in swapping two uatemplates
            continue
        c[k] = 1.0 / (sorted_arrs[k+1][0]['arrival_time'] - sorted_arrs[k][0]['arrival_time'] + 1.0/wave_node.srate)
    c.normalize()
    return c, sorted_arrs

def sample_adjacent_pair_to_swap(sg, wave_node):
    c, sorted_arrs = adjacent_pair_distribution(sg, wave_node)
    if c is None or len(c) == 0:
        return None, None, None, None
    k = c.sample()
    return sorted_arrs[k], sorted_arrs[k+1], k, c[k]

def get_pair_prob(sg, wave_node, first_atime):
    c, sorted_arrs = adjacent_pair_distribution(sg, wave_node)
    k = get_atime_index(sorted_arrs, first_atime)
    return c[k]

def swap_params(t1nodes, t2nodes):
    for (p, (k1, n1)) in t1nodes.items():
        if p=="amp_transfer" or p=="tt_residual":
            continue
        k2, n2 = t2nodes[p]
        v1 = n1.get_value(key=k1)
        v2 = n2.get_value(key=k2)
        n1.set_value(key=k1, value=v2)
        n2.set_value(key=k2, value=v1)

        if p == "arrival_time":
            atime1, atime2 = v1, v2
    return atime1, atime2

def swap_association_move(sg, wave_node):

    # sample from all pairs of adjacent templates in which not both are uatemplates
    arr1, arr2, k, pair_prob = sample_adjacent_pair_to_swap(sg, wave_node)
    if arr1 is None:
        return False

    # get all relevant nodes for the arrivals we sampled
    t1nodes = sg.get_template_nodes(eid=arr1[1], phase=arr1[2], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)
    t2nodes = sg.get_template_nodes(eid=arr2[1], phase=arr2[2], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)
    rn = list(set(relevant_nodes(t1nodes) + relevant_nodes(t2nodes)))
    lp_old = sg.joint_logprob_keys(rn)

    log_qforward = np.log(pair_prob)

    # switch their parameters
    atime1, atime2 = swap_params(t1nodes, t2nodes)
    min_atime = min(atime1, atime2)

    lp_new = sg.joint_logprob_keys(rn)
    log_qbackward = np.log(get_pair_prob(sg, wave_node, min_atime))

    u = np.random.rand()
    if (lp_new + log_qbackward) - (lp_old + log_qforward) > np.log(u):
        print "successful swap"
        return True
    else:
        atime1, atime2 = swap_params(t1nodes, t2nodes)
        return False


def hamiltonian_template_move(sg, wave_node, tmnodes, window_lps=None,
                              log_eps_mean=3, log_eps_std=5,
                              epsL=0.2,
                              reverse_block_size=5,
                              reverse_block_min_std=1,
                              reverse_block_max_std=1000,
                              **kwargs):

    node_list = [n for (k, n) in tmnodes.values()]
    relevant_nodes = node_list + [wave_node,]

    vals = np.array([n.get_value() for n in node_list])

    proxy_lps = wave_node.window_lps_to_proxy_lps(window_lps)

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

def hamiltonian_move_reparameterized(sg, wave_node, tmnodes, window_lps=None, **kwargs):

    params = ['arrival_time', 'peak_offset', 'coda_height', 'peak_decay', 'coda_decay']

    rparams = ['arrival_time', 'peak_time', 'coda_height', 'peak_decay', 'coda_auc']

    node_list = [tmnodes[p][1] for p in params]
    relevant_nodes = node_list + [wave_node,]

    vals = np.array([n.get_value() for n in node_list])

    proxy_lps = wave_node.window_lps_to_proxy_lps(window_lps)

    def reparametrize(x):
        arrival_time, peak_offset, coda_height, peak_decay, coda_decay = x

        peak_time = arrival_time + np.exp(peak_offset)
        coda_auc = coda_height - coda_decay

        return np.array((arrival_time, peak_time, coda_height, peak_decay, coda_auc))

    def reparametrize_inv(rx):
        arrival_time, peak_time, coda_height, peak_decay, coda_auc = rx

        d = peak_time - arrival_time
        if d < 0:
            print "WARNING BAD peak_time"
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
