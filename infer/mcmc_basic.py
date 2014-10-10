import numpy as np
from functools32 import lru_cache

@lru_cache(maxsize=2048)
def get_node_scales(node_list):
    low_bounds = np.concatenate([node.low_bounds() for node in node_list])
    high_bounds = np.concatenate([node.high_bounds() for node in node_list])
    scales = (high_bounds - low_bounds)/2.0
    scaled = np.isfinite(scales)
    scaled *= ["arrival_time" not in n.label for n in node_list]
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

def gaussian_MH_move(sg, keys, node_list, relevant_nodes, scales=None, proxy_lps=None, **kwargs):
    #node_list = [sg.nodes_by_key(k) for k in keys]
    #_, relevant_nodes = get_relevant_nodes(node_list)

    values = np.array([n.get_value(k) for (n,k) in zip(node_list, keys)])
    proposal = gaussian_propose(sg, keys, node_list, values=values, scales=scales, **kwargs)
    return MH_accept(sg, keys, values, proposal, node_list, relevant_nodes, proxy_lps=proxy_lps)

def MH_accept(sg, keys, oldvalues, newvalues, node_list, relevant_nodes,
              log_qforward=0.0, log_qbackward=0.0, proxy_lps=None):
    lp_old = sg.joint_logprob_keys(relevant_nodes, proxy_lps=proxy_lps) # assume oldvalues are still set
    lp_new = sg.joint_logprob_keys(keys=keys, values=newvalues, node_list=node_list,
                                   relevant_nodes=relevant_nodes, proxy_lps=proxy_lps)


    u = np.random.rand()
    if (lp_new + log_qbackward) - (lp_old + log_qforward) > np.log(u):
        return True
    else:
        for (key, val, n) in zip(keys, oldvalues, node_list):
            n.set_value(key=key, value=val)
        return False

def hmc_step(x, logpdf, logpdf_grad, L, eps):

    x = x.flatten()
    n = len(x)

    p = np.random.randn(n) # momentum
    current_x = x
    current_p = p

    # half step for momentum
    p = p + eps * logpdf_grad(x)/2

    print "hmc", L, eps

    # alternate full steps for position and momentum
    for i in range(L+1):

        if np.isnan(p).any():
            raise ValueError("HMC move failed: momentum is NaN")

        # full step for position
        x = x + eps * p

        # full step for momentum except on the last iteration
        if i != L:
            p = p + eps * logpdf_grad(x)

    p = p + eps * logpdf_grad(x) / 2

    current_U = -logpdf(current_x)
    current_K = np.sum(current_p**2)/2

    proposed_U = -logpdf(x)
    proposed_K = np.sum(p**2)/2

    print "hmc step: init U,K=(%.1f, %.1f), proposed U,K=(%.1f, %.1f), accept prob %.3f" % (current_U, current_K, proposed_U, proposed_K, np.exp(current_U-proposed_U+current_K-proposed_K))
    print " initial x", current_x
    print " proposed x", x

    if np.log(np.random.rand()) < current_U-proposed_U+current_K-proposed_K:
        return x, True
    else:
        return current_x, False
