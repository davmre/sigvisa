import numpy as np
import sys
import os
import traceback
import pickle
import copy


from sigvisa import Sigvisa

from sigvisa.graph.sigvisa_graph import SigvisaGraph



from sigvisa.models.templates.load_by_name import load_template_generator
from sigvisa.models.ttime import tt_predict
from sigvisa.infer.event_mcmc import *
from sigvisa.infer.propose_hough import hough_location_proposal
from sigvisa.infer.propose_lstsqr import overpropose_new_locations

from sigvisa.infer.event_birthdeath import ev_birth_executor,ev_death_executor

from sigvisa.infer.template_mcmc import *
from sigvisa.plotting.plot import plot_with_fit, plot_with_fit_shapes, plot_pred_atimes
from sigvisa.signals.common import Waveform
from sigvisa.utils.fileutils import clear_directory, mkdir_p




def sample_events_to_swap(sg, n_events=2, fix_result=None):
    # fix_results is a 2-tuple of eids
    # if set, return lp
    # else, return eid1, eid2, lp

    eids = sorted([eid for eid in sg.evnodes.keys() if eid not in sg.fixed_events])
    n = len(eids)

    if len(eids) < n_events:
        if fix_result is None:
            return None, -np.inf
        else:
            return -np.inf

    combinations = scipy.misc.comb(n, n_events)

    lp = -np.log(combinations)

    if fix_result is None:
        chosen_eids = []
        for i in range(n_events):
            eid = np.random.choice(eids)
            eids.remove(eid)
            chosen_eids.append(eid)

        chosen_eids = sorted(chosen_eids)
        return chosen_eids, lp
    else:
        if np.all([eid in eids for eid in fix_result]):
            return lp
        else:
            return -np.inf

def repropose_event_move_hough(sg, **kwargs):
    return swap_events_move(sg, n_events=1, location_proposal=hough_location_proposal, **kwargs)

def repropose_event_move_lstsqr(sg, **kwargs):
    return swap_events_move(sg, n_events=1, location_proposal=overpropose_new_locations, **kwargs)

def swap_threeway_lstsqr(sg, **kwargs):
    return swap_events_move(sg, n_events=3, location_proposal=overpropose_new_locations, **kwargs)

def swap_threeway_hough(sg, **kwargs):
    return swap_events_move(sg, n_events=3, location_proposal=hough_location_proposal, **kwargs)


def swap_events_move(sg, location_proposal, n_events=2, log_to_run_dir=None, return_probs=False):
    eids, lp_swap = sample_events_to_swap(sg, n_events=n_events)
    if eids is None:
        return False
    lp_new, lp_old, log_qforward, log_qbackward, revert_move = rebirth_events_helper(sg, eids, location_proposal=location_proposal)
    log_qforward += lp_swap
    log_qbackward += sample_events_to_swap(sg, fix_result=eids)

    if return_probs:
        return lp_old, lp_new, log_qforward, log_qbackward, revert_move
    else:
        return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=None, revert_move=revert_move)

def swap_events_move_hough(*args, **kwargs):
    kwargs['location_proposal']=hough_location_proposal
    return swap_events_move(*args, **kwargs)

def swap_events_move_lstsqr(*args, **kwargs):
    kwargs['location_proposal']=overpropose_new_locations
    return swap_events_move(*args, **kwargs)

def rebirth_events_helper(sg, eids, location_proposal, 
                          forward_birth_type="dumb", 
                          reverse_birth_type="dumb"):
    """
    Given a list of events, propose killing them all and rebirthing
    new events from the resulting uatemplates.

    dumb_birth_forward asks whether we should propose the new event 
    from a dumb birth distribution. This is more reasonable than you'd
    think since all the templates that really need to fit the signal are
    hopefully being re-associated (having been de-associated by the death 
    move) so the stuff that's left is either high-noise signals or 
    low-amplitude templates. A potential problem is cases where we 
    killed a small template in the death move, but the dumb birth tries to
    propose a big template. 

    dumb_birth_reverse asks whether we should use a dumb proposal to evaluate
    the reverse probability of rebirthing the current event. 

    note that for valid MCMC, we need either:
    - dumb_birth_forward and dumb_birth_reverse are the same (so this move correctly computes its own reverse probability)
    - they are different, but we choose the ordering uniformly at random, i.e. flip a coin to decide which one is true. this is smart/dumb, dumb/smart. 

    """

    old_evs = [sg.get_event(eid) for eid in eids]

    lp_old = sg.current_log_p()
    log_qforward = 0
    log_qbackward = 0
    revert_moves = []

    seeds = [np.random.choice(1000) for eid in eids]

    lp_old = None

    for i, eid in enumerate(eids):
        lp = lambda *args, **kwargs : location_proposal(*args, proposal_dist_seed=seeds[i], **kwargs)

        r = ev_death_executor(sg, force_kill_eid=eid, location_proposal=lp, 
                              birth_type=reverse_birth_type)
        if r is None:
            revert_moves.reverse()
            for r in revert_moves:
                r()
            return False
        lp_intermediate, lpo, lqf_old, lqb_old, redeath_old, rebirth_old, proposal_extra = r
        if lp_old is None:
            lp_old = lpo

        redeath_old()
        # leave the graph in the dead state, so the new death proposals are with respect to
        # this one already having happened...

        print "death %d lqf %f lqb %f" % (eid, lqf_old, lqb_old)
        log_qforward += lqf_old
        log_qbackward += lqb_old
        revert_moves.append(rebirth_old)

    for i, eid in enumerate(eids):
        lp = lambda *args, **kwargs : location_proposal(*args, proposal_dist_seed=seeds[i], **kwargs)
        r = ev_birth_executor(sg, location_proposal=lp, force_eid=eid, 
                              proposal_type=forward_birth_type)
        if r is None:
            revert_moves.reverse()
            for r in revert_moves:
                r()
            return False

        lp_new, lp_int, lqf_new, lqb_new, rebirth_new, redeath_new, proposal_extra = r
        rebirth_new()

        print "birth %d lqf %f lqb %f" % (eid, lqf_new, lqb_new)
        log_qforward += lqf_new
        log_qbackward += lqb_new
        revert_moves.append(redeath_new)

    revert_moves.reverse()
    def revert_move():
        for r in revert_moves:
            r()

    return lp_new, lp_old, log_qforward, log_qbackward, revert_move
