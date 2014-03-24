import numpy as np
from functools32 import lru_cache

@lru_cache(maxsize=2048)
def get_node_scales(node_list):
    low_bounds = np.concatenate([node.low_bounds() for node in node_list])
    high_bounds = np.concatenate([node.high_bounds() for node in node_list])
    scales = (high_bounds - low_bounds)/2.0
    scaled = np.isfinite(scales)
    scales[~scaled] = 1.0
    return scales

def gaussian_propose(sg, keys, node_list, values=None, scales=None, std=0.01, phase_wraparound=False):
    scales = scales if scales is not None else get_node_scales(node_list)
    values = values if values is not None else np.array([n.get_value(k) for (n,k) in zip(node_list, keys)])
    n = len(values)

    gsample = np.random.normal(0, std, n)
    move = gsample * scales
    if phase_wraparound: # phases must be between 0 and 2pi
        return (values + move) % 6.283185307179586
    else:
        return values + move

def gaussian_MH_move(sg, keys, node_list, relevant_nodes, scales=None, **kwargs):
    #node_list = [sg.nodes_by_key(k) for k in keys]
    #_, relevant_nodes = get_relevant_nodes(node_list)

    values = np.array([n.get_value(k) for (n,k) in zip(node_list, keys)])
    proposal = gaussian_propose(sg, keys, node_list, values=values, scales=scales, **kwargs)
    return MH_accept(sg, keys, values, proposal, node_list, relevant_nodes)


def MH_accept(sg, keys, oldvalues, newvalues, node_list, relevant_nodes, log_qforward=0.0, log_qbackward=0.0):
    lp_old = sg.joint_logprob_keys(relevant_nodes) # assume oldvalues are still set
    lp_new = sg.joint_logprob_keys(keys=keys, values=newvalues, node_list=node_list, relevant_nodes=relevant_nodes)

    u = np.random.rand()
    if (lp_new + log_qbackward) - (lp_old + log_qforward) > np.log(u):
        return True
    else:
        for (key, val, n) in zip(keys, oldvalues, node_list):
            n.set_value(key=key, value=val)
        return False


def gaussian_MH_move_joint(sg, key, node, latent, relevant_nodes, scales=None, **kwargs):
    """
    propose a Gaussian tweak to a template parameter, while also proposing an update to the latent signal that holds the latent wiggle constant.
    """

    lp_old = sg.joint_logprob_keys(relevant_nodes)

    oldval  = node.get_value(key=key)
    old_latent = np.copy(latent.get_value())
    proposal = gaussian_propose(sg, (key,), (node,), values=np.array([oldval,]), scales=scales, **kwargs)[0]

    wiggle, shape, rw = latent.get_signal_components()
    wiggle = np.copy(wiggle)

    node.set_value(key=key, value=proposal)

    latent.set_nonrepeatable_wiggle(wiggle)

    lp_new = sg.joint_logprob_keys(relevant_nodes=relevant_nodes)

    u = np.random.rand()
    if (lp_new ) - (lp_old) > np.log(u):
        return True
    else:
        node.set_value(key=key, value=oldval)
        latent.set_value(old_latent)
        return False
