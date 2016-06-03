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
from sigvisa.infer.mcmc_basic import mh_accept_util
from sigvisa.infer.propose_hough import hough_location_proposal
from sigvisa.infer.propose_lstsqr import overpropose_new_locations
from sigvisa.infer.correlations.event_proposal import correlation_location_proposal, sample_corr_kwargs

from sigvisa.infer.event_birthdeath import ev_birth_executor,ev_death_executor, sample_hough_kwargs, prior_location_proposal

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

    chosen_eids = None
    lp = -np.inf
    if len(eids) >= n_events:
        combinations = scipy.misc.comb(n, n_events)

        lp = -np.log(combinations)
        if fix_result is None:
            chosen_eids = []
            for i in range(n_events):
                eid = np.random.choice(eids)
                eids.remove(eid)
                chosen_eids.append(eid)

            chosen_eids = sorted(chosen_eids)
        else:
            lp = lp if np.all([eid in eids for eid in fix_result]) else -np.inf

    #if np.isinf(lp):
    #    import pdb; pdb.set_trace()

    if fix_result is None:
        return chosen_eids, lp
    else:
        return lp

def sample_repropose_args_hough(sg):
    hough_kwargs = sample_hough_kwargs(sg)
    def hlp(sg, **kwargs):
        kwargs.update(hough_kwargs)
        return hough_location_proposal(sg, **kwargs)

    dumb_birth = np.random.rand() < 0.5 
    birth_args = {"proposal_type": "dumb" if dumb_birth else "mh",
                  "use_correlation": False,
                  "repropose_uatemplates": False}
    flp = prior_location_proposal if dumb_birth else hlp

    dumb_death = np.random.rand() < 0.5 
    death_args = {"birth_type": "dumb" if dumb_death else "mh",
                  "use_correlation": False,
                  "repropose_uatemplates": False}
    rlp = prior_location_proposal if dumb_death else hlp

    return flp, rlp, birth_args, death_args

def repropose_event_move_hough(sg, **kwargs):

    flp, rlp, birth_args, death_args = sample_repropose_args_hough(sg)
    return swap_events_move(sg, n_events=1, 
                            forward_location_proposal=flp, 
                            reverse_location_proposal=rlp, 
                            birth_args=birth_args,
                            death_args=death_args,
                            **kwargs)


def repropose_event_move_lstsqr(sg, **kwargs):
    return swap_events_move(sg, n_events=1, location_proposal=overpropose_new_locations, **kwargs)


def sample_repropose_args_corr(sg):
    corr_kwargs = sample_corr_kwargs(sg)
    def clp(sg, fix_result=None, **clp_kwargs):
        clp_kwargs.update(corr_kwargs)
        return correlation_location_proposal(sg, fix_result=fix_result, **clp_kwargs)

    dumb_birth = np.random.rand() < 0.5 
    birth_args = {"proposal_type": "dumb" if dumb_birth else "mh",
                  "use_correlation": not dumb_birth,
                  "repropose_uatemplates": not dumb_birth}
    flp = prior_location_proposal if dumb_birth else clp
            

    dumb_death = np.random.rand() < 0.5 
    death_args = {"birth_type": "dumb" if dumb_death else "mh",
                  "use_correlation": not dumb_death,
                  "repropose_uatemplates": not dumb_death}
    rlp = prior_location_proposal if dumb_death else clp
    return flp, rlp, birth_args, death_args
    

def repropose_event_move_corr(sg, **kwargs):

    """
TODO (copied from email)
another issue: event mbs get reset to 3.0 in the proposal. this means the probs are technically incorrect since I can't ever reverse to recover a non-3.0 event. should either keep the current mb (which requires a notion of event identity when swapping multiple events), or propose new mb from some distribution?
    """

    flp, rlp, birth_args, death_args = sample_repropose_args_corr(sg)
    return swap_events_move(sg, n_events=1, 
                            forward_location_proposal=flp, 
                            reverse_location_proposal=rlp, 
                            birth_args=birth_args,
                            death_args=death_args, 
                            **kwargs)

def swap_threeway_lstsqr(sg, **kwargs):
    return swap_events_move(sg, n_events=3, location_proposal=overpropose_new_locations, **kwargs)

def swap_threeway_hough(sg, **kwargs):
    return swap_events_move(sg, n_events=3, location_proposal=hough_location_proposal, **kwargs)


def swap_events_move(sg, n_events=2, log_to_run_dir=None, return_probs=False, **kwargs):

    eids, lp_swap = sample_events_to_swap(sg, n_events=n_events)
    if eids is None:
        return False

    old_evs = dict([(eid, sg.get_event(eid)) for eid in eids])

    r = rebirth_events_helper(sg, eids, **kwargs)
    if r == False:
        return False

    lp_new, lp_old, log_qforward, log_qbackward, revert_move = r
    log_qforward += lp_swap

    lp_swap_reverse = sample_events_to_swap(sg, fix_result=eids, n_events=n_events)
    log_qbackward += lp_swap_reverse

    new_evs = dict([(eid, sg.get_event(eid)) for eid in eids])

    if log_to_run_dir is not None:
        log_file = os.path.join(log_to_run_dir, "swap_proposals.txt")
        with open(log_file, 'a') as f:
            f.write("swap lp %.2f reverse %.2f\n" % (lp_swap, lp_swap_reverse))
            f.write("kwargs %s\n" % kwargs)
            for eid in eids:
                f.write(" swapping eid %d\n" % eid)
                old_ev = old_evs[eid]
                f.write("  old %s\n" % str(old_ev))
                new_ev = new_evs[eid]
                f.write("  new %s\n" % str(new_ev))
            f.write("lp_new %.1f lp_old %.1f log_qbackward %.1f log_qforward %.1f ratio %.1f\n\n" % (lp_new, lp_old, log_qbackward, log_qforward, lp_new+log_qbackward - (lp_old+log_qforward)))
            

    if return_probs:
        return lp_old, lp_new, log_qforward, log_qbackward, revert_move
    else:
        return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=None, revert_move=revert_move)


def swap_events_move_hough(*args, **kwargs):
    kwargs['forward_location_proposal']=hough_location_proposal
    kwargs['reverse_location_proposal']=hough_location_proposal
    return swap_events_move(*args, **kwargs)

def swap_events_move_lstsqr(*args, **kwargs):
    kwargs['location_proposal']=overpropose_new_locations
    return swap_events_move(*args, **kwargs)

def rebirth_events_helper(sg, eids, 
                          forward_location_proposal=None, 
                          reverse_location_proposal=None, 
                          birth_args = None, 
                          death_args = None,
                          birth_eids = None,
                          inference_step=-1):
    """
    Given a list of events, propose killing them all and rebirthing
    new events from the resulting uatemplates.

    dumb_birth_forward asks whether we should propose templates for the new event 
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

    The same validity constraints apply to forward_location_proposal and reverse_location_proposal.

    """

    if birth_args is None:
        birth_args = {"proposal_type": "dumb"}
    if death_args is None:
        death_args = {"birth_type": "dumb"}


    old_evs = [sg.get_event(eid) for eid in eids]

    lp_old = sg.current_log_p()
    log_qforward = 0
    log_qbackward = 0
    revert_moves = []

    seeds = [np.random.choice(1000) for eid in eids]
    if birth_eids is not None and len(birth_eids) > len(eids):
        seeds = [np.random.choice(1000) for eid in birth_eids]

    lp_old = None

    for i, eid in enumerate(eids):
        lp = lambda *args, **kwargs : reverse_location_proposal(*args, proposal_dist_seed=seeds[i], **kwargs)

        r = ev_death_executor(sg, force_kill_eid=eid, location_proposal=lp, **death_args)
        if r is None:
            revert_moves.reverse()
            for r in revert_moves:
                r()
            return False

        lp_intermediate, lpo, lqf_old, lqb_old, redeath_old, rebirth_old, proposal_extra = r
        if lp_old is None:
            lp_old = lpo


        if np.isinf(lqb_old) and lqb_old < 0:
            revert_moves.reverse()
            for r in revert_modes:
                r()
            sg.logger.info("short-circuit rejection of event rebirth because killing eid %d returned probs %.1f %.1f %.1f %.1f" % (eid, lp_intermediate, lpo, lqf_old, lqb_old))
            return False

        redeath_old()
        # leave the graph in the dead state, so the new death proposals are with respect to
        # this one already having happened...

        print "repropose death %d lqf %f lqb %f" % (eid, lqf_old, lqb_old)
        log_qforward += lqf_old
        log_qbackward += lqb_old
        revert_moves.append(rebirth_old)

    if birth_eids is None:
        birth_eids = eids

    for i, eid in enumerate(birth_eids):
        lp = lambda *args, **kwargs : forward_location_proposal(*args, proposal_dist_seed=seeds[i], **kwargs)
        r = ev_birth_executor(sg, location_proposal=lp, force_eid=eid, **birth_args)
        if r is None:
            revert_moves.reverse()
            for r in revert_moves:
                r()
            return False

        lp_new, lp_int, lqf_new, lqb_new, rebirth_new, redeath_new, proposal_extra = r


        if np.isinf(lqb_new) and lqb_new < 0:
            revert_moves.reverse()
            for r in revert_modes:
                r()
            sg.logger.info("short-circuit rejection of event rebirth because birthing eid %d returned probs %.1f %.1f %.1f %.1f" % (eid, lp_new, lp_int, lqf_new, lqb_new))
            return False

        
        rebirth_new()

        print "repropose birth %d lqf %f lqb %f" % (eid, lqf_new, lqb_new)
        log_qforward += lqf_new
        log_qbackward += lqb_new
        revert_moves.append(redeath_new)

    revert_moves.reverse()
    def revert_move():
        for r in revert_moves:
            r()

    return lp_new, lp_old, log_qforward, log_qbackward, revert_move

def repropose_lsqr_new(sg, eid, **kwargs):

    def cache_location_proposal():
        old_ev = sg.get_event(eid)
        init_z = np.array((old_ev.lon, old_ev.lat, old_ev.depth, old_ev.time))

        z, C = ev_lstsqr_dist(sg, eid, init_z=init_z)
        rv = scipy.stats.multivariate_normal(z, C)

        def loc_prop(sg, fix_result=None, **kwargs):
            if fix_result is not None:
                new_ev = fix_result
                proposed_vals = np.array((new_ev.lon, new_ev.lat, new_ev.depth, new_ev.time))
            else:
                proposed_vals = rv.rvs(1)


            lon, lat, depth, time = proposed_vals

            # this definitely breaks Gaussianity, we should be explicitly truncating the distribution
            if depth > 700:
                depth = 700
            elif depth < 0:
                depth = 0

            proposal_lp = np.logaddexp(rv.logpdf(proposed_vals), -50)

            # this breaks Gaussianity, technically we should be using a
            # circular (von Mises?) distribution. but hopefully it doesn't
            # matter.
            lon, lat = wrap_lonlat(lon, lat)

            new_ev = Event(lon=lon, lat=lat, depth=depth, time=time, mb=old_ev.mb)

            if fix_result is not None:
                return proposal_lp
            else:
                return new_ev, proposal_lp, (z, C)

        return loc_prop

    return swap_events_move(sg, n_events=1, location_proposal=cache_location_proposal(), **kwargs)
