import numpy as np

#######################################################################

"""

Methods for sampling an arrival time independent of the current
arrival time, with probability proportional to the height of the
envelope at each point.

"""

def preprocess_signal_for_sampling(wave_data):

    d = wave_data**2
    s = np.sum(d)
    normalized_data = d/s
    cdf = np.concatenate([np.array((0,)), np.cumsum(normalized_data)])
    return cdf

def atime_log_p(wave_node, arrival_node, offset_node):
    atime = arrival_node.get_value()
    offset = offset_node.get_value()
    peak = atime + offset
    return peak_log_p(wave_node.cdf, wave_node.st, wave_node.srate, peak)

def peak_log_p(cdf, stime, srate, peak_time):
    # compute the probability that sample_peak_time_from_signal would
    # have sampled the current atime. formally this should be 0 for
    # all peak times that don't line up with an integer index, but we
    # just force an integer and hope that's okay.

    # we add one here since we created cdf with an initial
    # 0, so that the subtraction below works properly
    idx = np.round((peak_time - stime) * srate) + 1

    if (idx < 1) or (idx >= len(cdf)): return np.float('-inf')
    return np.log(cdf[idx] - cdf[idx-1])

def sample_peak_time_from_signal(cdf, stime, srate, return_lp=False):
    u = np.random.rand()
    idx = np.searchsorted(cdf, u)
    peak_time = stime + float(idx-1)/srate
    if return_lp:
        return peak_time, np.log(cdf[idx]-cdf[idx-1])
    return peak_time

def indep_peak_move(sg, arrival_node, offset_node, wave_node):
    current_atime = arrival_node.get_value()
    peak_offset = offset_node.get_value()
    proposed_peak_time = sample_peak_time_from_signal(wave_node.cdf, wave_node.st, wave_node.srate)
    proposed_arrival_time = proposed_peak_time - peak_offset
    return MH_accept(sg, oldvalues = (current_atime,),
                     newvalues = (proposed_arrival_time,),
                     node_list = (arrival_node,),
                     relevant_nodes = (arrival_node, wave_node))


######################################################################

def indep_offset_move(sg, arrival_node, offset_node, wave_node):
    current_offset = offset_node.get_value()
    atime = arrival_node.get_value()
    proposed_offset = np.random.rand() * 40
    new_atime = atime + (current_offset - proposed_offset)
    accepted = MH_accept(sg=sg, oldvalues=(atime, current_offset),
                         newvalues = (new_atime, proposed_offset),
                         node_list = (arrival_node, offset_node),
                         relevant_nodes=(arrival_node, offset_node, wave_node))
    return accepted

def improve_offset_move(sg, arrival_node, offset_node, wave_node, **kwargs):
    """
    Update the peak_offset while leaving the peak time constant, i.e.,
    adjust the arrival time to compensate for the change in offset.
    """
    current_offset = offset_node.get_value()
    atime = arrival_node.get_value()
    proposed_offset = gaussian_propose(sg, node_list=(offset_node,), values=(current_offset,), **kwargs)[0]
    new_atime = atime + (current_offset - proposed_offset)
    accepted = MH_accept(sg=sg, oldvalues=(atime, current_offset),
                         newvalues = (new_atime, proposed_offset),
                         node_list = (arrival_node, offset_node),
                         relevant_nodes=(arrival_node, offset_node, wave_node))
    return accepted

#######################################################################

def get_node_scales(node_list):
    low_bounds = np.concatenate([node.low_bounds() for node in node_list])
    high_bounds = np.concatenate([node.high_bounds() for node in node_list])
    scales = (high_bounds - low_bounds)/2.0
    scaled = np.isfinite(scales)
    scales[~scaled] = 1.0
    return scales

def gaussian_propose(sg, node_list, values=None, scales=None, std=0.01, phase_wraparound=False):
    scales = scales if scales else get_node_scales(node_list)
    values = values if values else sg.get_all(node_list = node_list)
    n = len(values)

    gsample = np.random.normal(0, std, n)
    move = gsample * scales
    if phase_wraparound: # phases must be between 0 and 2pi
        return (values + move) % 6.283185307179586
    else:
        return values + move

def gaussian_MH_move(sg, node_list, relevant_nodes, scales=None, **kwargs):
    values = sg.get_all(node_list = node_list)
    proposal = gaussian_propose(sg, node_list, values=values, scales=scales, **kwargs)
    return MH_accept(sg, values, proposal, node_list, relevant_nodes)

def MH_accept(sg, oldvalues, newvalues, node_list, relevant_nodes, log_qforward=0.0, log_qbackward=0.0):
    lp_new = sg.joint_prob(values=newvalues, node_list=node_list, relevant_nodes=relevant_nodes)
    lp_old = sg.joint_prob(values=oldvalues, node_list=node_list, relevant_nodes=relevant_nodes)

    u = np.random.rand()
    if (lp_new + log_qbackward) - (lp_old + log_qforward) > np.log(u):
        sg.set_all(newvalues, node_list)
        return True
    else:
        return False
