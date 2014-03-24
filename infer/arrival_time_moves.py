import numpy as np
import numpy.ma as ma
import sys
import os
import traceback
import pickle
import copy
import time

from sigvisa import Sigvisa
from sigvisa.infer.mcmc_basic import get_node_scales, gaussian_propose, gaussian_MH_move, MH_accept
from sigvisa.infer.autoregressive_mcmc import gibbs_sweep
from sigvisa.graph.graph_utils import create_key,parse_key
from sigvisa.models.signal_model import extract_arrival_from_key
from sigvisa.graph.dag import get_relevant_nodes
from sigvisa.plotting.plot import savefig, plot_with_fit, plot_waveform

from matplotlib.figure import Figure

import scipy.weave as weave
from scipy.weave import converters


######################################################################

def get_signal_based_amplitude_distribution(sg, sta, tmvals=None, peak_time=None, peak_period_s = 1.0):
    wn = sg.station_waves[sta][0]

    if peak_time is None:
        peak_time = tmvals['arrival_time'] + tmvals['peak_offset']
    peak_idx = int((peak_time - wn.st) * wn.srate)
    peak_period_samples = int(peak_period_s * wn.srate)
    peak_data=wn.get_value()[peak_idx - peak_period_samples:peak_idx + peak_period_samples]

    # if we land outside of the signal window, or during an unobserved (masked) portion,
    # we'll just sample from the event-conditional prior instead
    if ma.count(peak_data) == 0:
        return None

    peak_height = peak_data.mean()

    env_height = max(peak_height - wn.nm.c, wn.nm.c/100.0)



    return Gaussian(mean=np.log(env_height), std = 0.1)



#######################################################################

"""

Methods for sampling an arrival time independent of the current
arrival time, with probability proportional to the height of the
envelope at each point.

"""

def preprocess_signal_for_sampling(wave_env):

    d = wave_env**4

    np.savetxt("wave_env.txt", wave_env)
    np.savetxt("d.txt", d)

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
    mask = wave_node.get_value().mask

    pred_signal = wave_node.assem_signal(arrivals=arrival_set, include_wiggles=False)

    npts = wave_node.npts
    signal_diff_pos = wave_node.signal_diff
    code = """
for(int i=0; i < npts; ++i) {
  if (mask(i)) {
      signal_diff_pos(i) = 0;
  }  else {
    double v = fabs(value(i)) - fabs(pred_signal(i));
    signal_diff_pos(i) = v > 0 ? v : 0;
  }
}
"""
    weave.inline(code,['npts', 'signal_diff_pos', 'value', 'mask', 'pred_signal'],type_converters = converters.blitz,verbose=2,compiler='gcc')
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

def indep_peak_move(sg, wave_node, tmnodes, std=None):
    arrival_key, arrival_node = tmnodes['arrival_time']
    offset_key, offset_node = tmnodes['peak_offset']
    k_latent, n_latent = tmnodes['latent_arrival']

    relevant_nodes = [wave_node,n_latent]
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

    # print "current peak", current_atime+peak_offset, "proposed peak", proposed_peak_time, "atime", proposed_arrival_time

    old_latent, resample_lp, reverse_lp, lp_old, lp_new = repropose_full_latent_signal(sg, n_latent, arrival_node, current_atime, proposed_arrival_time, relevant_nodes)

    reverse_lp += backward_propose_lp
    resample_lp += proposal_lp

    # do MH acceptance
    u = np.random.rand()
    if (lp_new + reverse_lp) - (lp_old + resample_lp) > np.log(u):
        print "accepted peak move: new atime", proposed_arrival_time, "peak time", proposed_peak_time
        return True
    else:
        keys = [arrival_key, k_latent]
        node_list = [arrival_node, n_latent]
        oldvalues = [current_atime, old_latent]
        for (key, val, n) in zip(keys, oldvalues, node_list):
            n.set_value(key=key, value=val)
        return False


######################################################################

def resample_initial_latent(sg, n_latent, n_atime, n_offset, peak_time, exp_old_offset, offset_proposal, relevant_nodes):

    # we need to know the peak time, and the old/new peak offsets


    lp_old = sg.joint_logprob_keys(relevant_nodes)
    old_latent = np.copy(n_latent.get_value())

    resample_padding = 2 * n_latent.srate
    old_onset_npts = int(exp_old_offset * n_latent.srate)

    # first we compute the resample probability of the current *onset period*
    reverse_lp = gibbs_sweep(n_latent, start_idx=0, end_idx =old_onset_npts+resample_padding, target_signal=old_latent[:old_onset_npts+resample_padding])

    # now we shift the signal by the amount of the change in
    # offsets. this will get us a new latent signal that matches the
    # existing signal exactly, except for the onset period

    exp_proposal = np.exp(offset_proposal)

    delta_offset_s = exp_proposal - exp_old_offset
    delta_offset_npts = int(delta_offset_s * n_latent.srate)

    decay_npts = int(len(old_latent) - old_onset_npts)
    shifted_npts = len(old_latent) + delta_offset_npts

    shifted_latent = np.zeros((shifted_npts,))
    shifted_latent[-decay_npts:] = old_latent[-decay_npts:]
    n_latent.set_value(shifted_latent)

    n_atime.set_value(peak_time - exp_proposal)
    n_offset.set_value(offset_proposal)

    # then we resample the onset period.
    resample_lp = gibbs_sweep(n_latent, start_idx=0, end_idx=int(shifted_npts-decay_npts)+resample_padding)

    lp_new = sg.joint_logprob_keys(relevant_nodes)

    return old_latent, resample_lp, reverse_lp, lp_old, lp_new


"""
def shift_and_propose_latent_signal(sg, n_latent, n_atime, old_atime, atime_proposal, relevant_nodes):
    lp_old = sg.joint_logprob_keys(relevant_nodes)
    old_latent = np.copy(n_latent.get_value())
    n = len(old_latent)
    old_wiggle, old_shape, old_repeatable = n_latent.get_signal_components()
    atime_shift_s = atime_proposal - old_atime
    atime_shift_idx = int(np.abs(atime_shift_s * n_latent.srate))

    resample_padding_idx = min(int(3 * n_latent.srate), n - atime_shift_idx)

    shifted_latent = np.zeros(old_latent.shape)
    if atime_shift_s > 0:
        # new atime is later than old atime

        # first compute the reverse probability of resampling the old wiggle

        # TODO: do we have to set the backshifted wiggle at all?
        backshifted_latent = np.ones(old_latent.shape)
        backshifted_latent[atime_shift_idx:] = old_latent[atime_shift_idx:]
        n_latent.set_value(backshifted_latent)
        reverse_lp = gibbs_sweep_python(n_latent, start_idx=0, end_idx=atime_shift_idx+resample_padding_idx, reverse=True, target_signal=old_latent[:atime_shift_idx], mask_unsampled=True)


        n_atime.set_value(atime_proposal)
        shifted_latent[:-atime_shift_idx] = old_latent[atime_shift_idx:]
        n_latent.set_value(shifted_latent)
        resample_lp = gibbs_sweep_python(n_latent, start_idx=n-atime_shift_idx-resample_padding_idx, end_idx=n, reverse=False, mask_unsampled=True)
        lp_new = sg.joint_logprob_keys(relevant_nodes)

    else:
        # new atime is earlier than old atime

        backshifted_latent = np.zeros(old_wiggle.shape)
        backshifted_latent[:-atime_shift_idx] = old_wiggle[:-atime_shift_idx]
        n_latent.set_value(backshifted_latent)
        reverse_lp = gibbs_sweep_python(n_latent, start_idx=n-atime_shift_idx-resample_padding_idx, end_idx=n, reverse=False, target_signal=old_latent[-atime_shift_idx-resample_padding_idx:], mask_unsampled=True)

        n_atime.set_value(atime_proposal)
        shifted_latent[atime_shift_idx:] = old_latent[:-atime_shift_idx]
        n_latent.set_value(shifted_latent)

        resample_lp = gibbs_sweep_python(n_latent, start_idx=0, end_idx=atime_shift_idx+resample_padding_idx, reverse=True, mask_unsampled=True)
        lp_new = sg.joint_logprob_keys(relevant_nodes)

    return old_latent, resample_lp, reverse_lp, lp_old, lp_new
"""

def repropose_full_latent_signal(sg, n_latent, n_atime, old_atime, atime_proposal, relevant_nodes):

    lp_old = sg.joint_logprob_keys(relevant_nodes)
    old_latent = np.copy(n_latent.get_value())
    n = len(old_latent)

    atime_shift_s = atime_proposal - old_atime
    atime_shift_idx = int(np.abs(atime_shift_s * n_latent.srate))


    # first compute the reverse probability of resampling the old wiggle
    reverse_lp = gibbs_sweep(n_latent, target_signal=old_latent)

    n_atime.set_value(atime_proposal)
    resample_lp = gibbs_sweep(n_latent)
    lp_new = sg.joint_logprob_keys(relevant_nodes)

    return old_latent, resample_lp, reverse_lp, lp_old, lp_new



#################################################################

# maybe bring this back when we re-introduce repeatable wiggles
"""
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
"""


def improve_offset_move(sg, wave_node, tmnodes, std=0.5, **kwargs):
    """
    Update the peak_offset while leaving the peak time constant, i.e.,
    adjust the arrival time to compensate for the change in offset.
    """

    arrival_key, arrival_node = tmnodes['arrival_time']
    offset_key, offset_node = tmnodes['peak_offset']
    relevant_nodes = [wave_node,]
    relevant_nodes += [arrival_node.parents[arrival_node.default_parent_key()],] if arrival_node.deterministic() else [arrival_node,]
    relevant_nodes += [offset_node.parents[offset_node.default_parent_key()],] if offset_node.deterministic() else [offset_node,]

    print "WARNING: BROKEN relevant_nodes in improve_offset_move"

    current_offset = offset_node.get_value(key=offset_key)
    exp_offset = np.exp(current_offset)
    atime = arrival_node.get_value(key=arrival_key)
    proposed_offset = gaussian_propose(sg, keys=(offset_key,),
                                       node_list=(offset_node,),
                                       values=(current_offset,),
                                       std=std, **kwargs)[0]
    exp_proposed = np.exp(proposed_offset)
    new_atime = atime + (exp_offset - exp_proposed)

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
                         relevant_nodes=relevant_nodes)
    return accepted

def improve_atime_move(sg, wave_node, tmnodes, std=1.0, **kwargs):
    # here we re-implement get_relevant_nodes from sigvisa.graph.dag, with a few shortcuts
    k_atime, n_atime = tmnodes['arrival_time']
    k_latent, n_latent = tmnodes['latent_arrival']

    relevant_nodes = [wave_node,n_latent]
    relevant_nodes += [n_atime.parents[n_atime.default_parent_key()],] if n_atime.deterministic() else [n_atime,]


    # propose a new arrival time
    old_atime = n_atime.get_value(k_atime)
    values = (old_atime,)
    atime_proposal = float(gaussian_propose(sg, keys=(k_atime,),
                                            node_list=(n_atime,),
                                            values=(values), std=std,
                                            **kwargs))


    old_latent, resample_lp, reverse_lp, lp_old, lp_new = repropose_full_latent_signal(sg, n_latent, n_atime, old_atime, atime_proposal, relevant_nodes)

    # do MH acceptance
    u = np.random.rand()
    if (lp_new + reverse_lp) - (lp_old + resample_lp) > np.log(u):
        return True
    else:
        keys = [k_atime, k_latent]
        node_list = [n_atime, n_latent]
        oldvalues = [old_atime, old_latent]
        for (key, val, n) in zip(keys, oldvalues, node_list):
            n.set_value(key=key, value=val)
        return False

#######################################################################################

def coda_height_move(sg, wave_node, tmnodes, std=1.0, **kwargs):
    # here we re-implement get_relevant_nodes from sigvisa.graph.dag, with a few shortcuts
    k_height, n_height = tmnodes['coda_height']
    k_latent, n_latent = tmnodes['latent_arrival']

    relevant_nodes = [wave_node,n_latent]
    relevant_nodes += [n_height.parents[n_height.default_parent_key()],] if n_height.deterministic() else [n_height,]


    # propose a new arrival time
    old_height = n_height.get_value(k_height)
    values = (old_height,)
    height_proposal = float(gaussian_propose(sg, keys=(k_height,),
                                             node_list=(n_height,),
                                             values=(values), std=std,
                                             **kwargs))

    old_latent = np.copy(n_latent)
    n_height.set_value(height_proposal)



    # do MH acceptance
    u = np.random.rand()
    if (lp_new + reverse_lp) - (lp_old + resample_lp) > np.log(u):
        return True
    else:
        keys = [k_atime, k_latent]
        node_list = [n_atime, n_latent]
        oldvalues = [old_atime, old_latent]
        for (key, val, n) in zip(keys, oldvalues, node_list):
            n.set_value(key=key, value=val)
        return False

#######################################################################################

def coda_decay_joint_move(sg, wave_node, tmnodes, std=1.0, **kwargs):
    # here we re-implement get_relevant_nodes from sigvisa.graph.dag, with a few shortcuts
    k_decay, n_decay = tmnodes['coda_decay']
    k_latent, n_latent = tmnodes['latent_arrival']

    relevant_nodes = [wave_node,n_latent]
    relevant_nodes +=  [n_decay,]

    lp_old = sg.joint_logprob_keys(relevant_nodes)

    # propose a new coda decay
    old_decay = n_decay.get_value(k_decay)
    values = (old_decay,)
    decay_proposal = float(gaussian_propose(sg, keys=(k_decay,),
                                             node_list=(n_decay,),
                                             values=(values), std=std,
                                             **kwargs))

    old_latent = np.copy(n_latent.get_value())
    ew, shape, rw = n_latent.get_signal_components()
    wiggle = np.copy(ew)
    n_decay.set_value(decay_proposal)
    n_latent.set_nonrepeatable_wiggle(wiggle)

    lp_new = sg.joint_logprob_keys(relevant_nodes)

    # do MH acceptance
    u = np.random.rand()
    if (lp_new) - (lp_old) > np.log(u):
        return True
    else:
        n_decay.set_value(key=k_decay, value=old_decay)
        n_latent.set_value(key=k_latent, value=old_latent)
        return False

def coda_decay_joint_gibbs_move(sg, wave_node, tmnodes, std=1.0, **kwargs):
    # here we re-implement get_relevant_nodes from sigvisa.graph.dag, with a few shortcuts
    k_decay, n_decay = tmnodes['coda_decay']
    k_latent, n_latent = tmnodes['latent_arrival']

    relevant_nodes = [wave_node,n_latent]
    relevant_nodes +=  [n_decay,]

    lp_old = sg.joint_logprob_keys(relevant_nodes)

    # propose a new coda decay
    old_decay = n_decay.get_value(k_decay)
    values = (old_decay,)
    decay_proposal = float(gaussian_propose(sg, keys=(k_decay,),
                                             node_list=(n_decay,),
                                             values=(values), std=std,
                                             **kwargs))

    old_latent = np.copy(n_latent.get_value())
    lp_reverse = gibbs_sweep(n_latent, target_signal=old_latent)
    n_decay.set_value(decay_proposal)
    lp_resample = gibbs_sweep(n_latent)


    lp_new = sg.joint_logprob_keys(relevant_nodes)

    # do MH acceptance
    u = np.random.rand()
    if (lp_new + lp_reverse) - (lp_old+lp_resample) > np.log(u):
        return True
    else:
        n_decay.set_value(key=k_decay, value=old_decay)
        n_latent.set_value(key=k_latent, value=old_latent)
        print lp_new, lp_old, lp_reverse, lp_resample
        print lp_new + lp_reverse, lp_old+lp_resample
        print lp_new + lp_reverse - lp_old-lp_resample
        return False
