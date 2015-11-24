import numpy as np
from functools32 import lru_cache
from sigvisa.models.signal_model import ObservedSignalNode

"""
@lru_cache(maxsize=2048)
def get_node_scales(node_list):
    low_bounds = np.concatenate([node.low_bounds() for node in node_list])
    high_bounds = np.concatenate([node.high_bounds() for node in node_list])
    scales = (high_bounds - low_bounds)/2.0
    scaled = np.isfinite(scales)
    scaled *= ["arrival_time" not in n.label for n in node_list]
    scales[~scaled] = 1.0
    return scales
"""

def gaussian_propose(sg, keys, node_list, values=None, scales=None, std=0.01, phase_wraparound=False, **kwargs):
    #scales = scales if scales is not None else get_node_scales(node_list)
    values = values if values is not None else np.array([n.get_value(k) for (n,k) in zip(node_list, keys)])
    n = len(values)

    gsample = np.random.normal(0, std, n)
    move = gsample #* scales
    if phase_wraparound: # phases must be between 0 and 2pi
        return (values + move) % 6.283185307179586
    else:
        return values + move

def gaussian_MH_move(sg, keys, node_list, relevant_nodes, scales=None, proxy_lps=None, step=None, **kwargs):
    #node_list = [sg.nodes_by_key(k) for k in keys]
    #_, relevant_nodes = get_relevant_nodes(node_list)

    values = np.array([n.get_value(k) for (n,k) in zip(node_list, keys)])
    proposal = gaussian_propose(sg, keys, node_list, values=values, scales=scales, **kwargs)
    return MH_accept(sg, keys, values, proposal, node_list, relevant_nodes, proxy_lps=proxy_lps)


alldiffs = 0
def mh_accept_lp(sg, lp, oldvalues, newvalues):
    # assume "lp" sets the appropriate values and then computes a log-likelihood

    check=False

    lp_old = lp(oldvalues)
    lp_new = lp(newvalues)

    if check:
        lp_old1 = lp(oldvalues)
        lp_old_true = sg.current_log_p()
        lp_new1 = lp(newvalues)
        lp_new_true = sg.current_log_p()

        if np.isfinite(lp_new):
            try:
                assert(np.abs((lp_new_true-lp_old_true) - (lp_new-lp_old)) < 1e-3)
                print "check passed!"
            except AssertionError:
                import pdb; pdb.set_trace()

    u = np.random.rand()

    #global alldiffs
    #if u < 0.01:
    #    print "alldiffs", alldiffs

    if lp_new - lp_old > np.log(u):
        #print "accepting, diff", lp_new-lp_old
        #lp(newvalues)
        #alldiffs += lp_new-lp_old
        return True
    else:
        lp(oldvalues)
        #print "rejecting, diff", lp_new-lp_old
        return False


def MH_accept(sg, keys, oldvalues, newvalues, node_list, relevant_nodes,
              log_qforward=0.0, log_qbackward=0.0, proxy_lps=None):



    lp_old = sg.joint_logprob_keys(relevant_nodes, proxy_lps=proxy_lps) # assume oldvalues are still set

    #lp_old_true = sg.current_log_p()

    lp_new = sg.joint_logprob_keys(keys=keys, values=newvalues, node_list=node_list,
                                   relevant_nodes=relevant_nodes, proxy_lps=proxy_lps)


    #if np.isfinite(lp_new):
    #    lp_new_true = sg.current_log_p()
    #else:
    #    lp_new_true = lp_new

    #if np.isfinite(lp_new):
    #    assert(np.abs( (lp_new - lp_old) - (lp_new_true - lp_old_true) ) < 1e-8 )
    #    print "assertion passed", keys

    u = np.random.rand()
    if (lp_new + log_qbackward) - (lp_old + log_qforward) > np.log(u):
        return True
    else:
        for (key, val, n) in zip(keys, oldvalues, node_list):
            n.set_value(key=key, value=val)

        for n in relevant_nodes:
            if len(n.params_modeled_jointly) > 0:
                # pass the proper (original) messages
                # back up to the joint param nodes.
                # TODO: cache messages to avoid recomputation
                n.upwards_message_normalizer()

        return False

def mh_accept_util(lp_old, lp_new, log_qforward=0, log_qbackward=0, jacobian_determinant=0, accept_move=None, revert_move=None, force_outcome=None):
    # print lp_new, lp_old, log_qbackward, log_qforward, jacobian_determinant, "FINAL", (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant
    if force_outcome is None:
        logu =  np.log(np.random.rand())
    elif force_outcome == True:
        logu = -np.inf
    elif force_outcome == False:
        logu = np.inf

    if (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant > logu:
        if accept_move:
            accept_move()
        return True
    else:
        if revert_move:
            revert_move()
        return False


def hmc_step_reversing(q, logpdf, logpdf_grad, L_blocks, eps,
                       block_size=5, min_block_std=1,
                       max_block_std=1000, force_p=None):


    def final_state(states, backstates):
        blocks_remaining  = L_blocks-i_blocks
        legal_positive_blocks = len(states) - 1
        legal_negative_blocks = len(backstates) - 1

        traj_index = -legal_negative_blocks

        legal_blocks = legal_negative_blocks+legal_positive_blocks

        # each roundtrip traverses all legal blocks in both
        # directions, and uses an additional block on each side to
        # reverse direction
        leftover_after_roundtrips = blocks_remaining % (2*legal_blocks + 2)

        if leftover_after_roundtrips <= legal_blocks:
            final_index = traj_index + leftover_after_roundtrips
            final_dir = 1
        else:
            final_index = traj_index + legal_blocks - (leftover_after_roundtrips - legal_blocks - 1)
            final_dir = -1

        if final_index >= 0:
            p, q = states[final_index]
            p = p * final_dir
        else:
            p, q = backstates[-final_index]
            p = p * (-1 * final_dir)

        return p, q, final_index

    # standard deviation doesn't make sense on size-1 blocks...
    assert(block_size > 1)

    q = q.flatten()
    n = len(q)
    if force_p is not None:
        p = force_p
    else:
        p = np.random.randn(n) # momentum
    current_q = q
    current_p = p

    # keep track of where we are in the trajectory. positive index
    # indicates forward motion from initial state, negative index is
    # backward.
    traj_index = 0

    # we need to cache the trajectory in order to efficiently reverse when we discover a bad step size.
    # "states" stores initial state and all positive indices.
    states = []
    # "backstates" stores all negative indices
    backstates = []

    i_blocks = 0
    direction = 1
    Hs = np.zeros((block_size+1,))

    # start with half step for momentum
    p_step = eps * logpdf_grad(q)

    while i_blocks < L_blocks:

        block_p = p
        block_q = q

        if direction == 1:
            states.append((p,q))
        else:
            backstates.append((p,q))

        # do a bunch of moves in succession
        for i in range(block_size):
            Hs[i] =  -logpdf(q) + np.sum(p**2)/2

            # half step for momentum
            p = p + p_step/2

            # full step for position
            q = q + eps * p

            # half step for momentum
            p_step = eps * logpdf_grad(q)
            p = p + p_step/2
        Hs[block_size] =  -logpdf(q) + np.sum(p**2)/2

        # now test whether the simulation is good foer this block of moves
        i_blocks += 1
        block_H_std = np.std(Hs)

        if block_H_std < min_block_std or block_H_std > max_block_std:

            """
            # formally speaking, if we hit a bad block, we just go back to the
            # beginning of the block and reverse direction:

            p = -block_p
            q = block_q
            p_step = eps * logpdf_grad(q)
            direction *= -1
            print "reversing"
            continue
            """

            # if this is the first time we're reversing, go back to the original state
            if direction==1:
                blocks_remaining  = L_blocks-i_blocks
                blocks_to_reverse = min(blocks_remaining, i_blocks-1)
                i_blocks += blocks_to_reverse
                traj_index -= blocks_to_reverse

                p, q = states[traj_index]
                p = -p
                p_step = eps * logpdf_grad(q)
                direction = -1
            else:
                assert (traj_index == -len(backstates)+1)
                p, q, traj_index = final_state(states, backstates)
                break
        else:
            traj_index += direction

    if traj_index == 0:
        # if we didn't go anywhere, count this as a rejection
        return q, current_p, p, float("-inf")

    current_U = -logpdf(current_q)
    current_K = np.sum(current_p**2)/2

    proposed_U = -logpdf(q)
    proposed_K = np.sum(p**2)/2

    accept_lp = current_U-proposed_U+current_K-proposed_K
    return q, current_p, p, accept_lp


def hmc_step(q, logpdf, logpdf_grad, L, eps, force_p=None):

    q = q.flatten()
    n = len(q)

    if force_p is None:
        p = np.random.randn(n) # momentum
    else:
        p = force_p
    current_q = q
    current_p = p

    # half step for momentum
    p = p + eps * logpdf_grad(q)/2

    # alternate full steps for position and momentum
    for i in range(L+1):

        if np.isnan(p).any():
            raise ValueError("HMC move failed: momentum is NaN")

        # full step for position
        q = q + eps * p

        # full step for momentum except on the last iteration
        if i != L:
            p = p + eps * logpdf_grad(q)

    p = p + eps * logpdf_grad(q) / 2

    current_U = -logpdf(current_q)
    current_K = np.sum(current_p**2)/2

    proposed_U = -logpdf(q)
    proposed_K = np.sum(p**2)/2

    #print "hmc step: init U,K=(%.1f, %.1f), proposed U,K=(%.1f, %.1f), accept prob %.3f" % (current_U, current_K, proposed_U, proposed_K, np.exp(current_U-proposed_U+current_K-proposed_K))
    #print " initial q", current_q
    #print " initial grad", logpdf_grad(current_q)
    #print " proposed q", q

    accept_lp = current_U-proposed_U+current_K-proposed_K
    return q, current_p, p, accept_lp
