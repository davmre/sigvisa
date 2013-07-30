import numpy as np


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
