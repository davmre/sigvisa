import numpy as np
import sys
import os
import traceback
import pickle
import copy
import itertools

from sigvisa import Sigvisa

from sigvisa.infer.mcmc_basic import mh_accept_util
from sigvisa.infer.event_swap import rebirth_events_helper, sample_repropose_args_hough, sample_repropose_args_corr
from sigvisa.utils.counter import Counter
from sigvisa.utils.geog import dist_km

def event_merge_dist(sg):
    evs = [(eid, sg.get_event(eid)) for eid in sg.evnodes.keys() if eid not in sg.fixed_events]

    if len(evs) < 2:
        return None

    merge_probs = Counter()
    for (eid1, ev1), (eid2, ev2) in itertools.combinations(evs, 2):
        surface_dist = dist_km((ev1.lon, ev1.lat), (ev2.lon, ev2.lat))
        time_dist = np.abs(ev1.time - ev2.time)
        dist = time_dist * 10.0 + surface_dist

        # give some small probability to all merge pairs
        dist = min(dist, 5000.0)
        weight = 1.0/dist
        merge_probs[(min(eid1, eid2), max(eid1, eid2))] = weight
    merge_probs.normalize()
    return merge_probs

def sample_events_to_merge(sg, fix_result=None):
    merge_probs = event_merge_dist(sg)
    if merge_probs is None:
        return None, None, 0.0

    if fix_result is not None:
        merge_eid1, merge_eid2 = sorted(fix_result)
    else:
        merge_eid1, merge_eid2 = merge_probs.sample()

    lp = np.log(merge_probs[(merge_eid1, merge_eid2)])
    return merge_eid1, merge_eid2, lp

def sample_event_to_split(sg, fix_result=None):
    eids = [eid for eid in sg.evnodes.keys() if eid not in sg.fixed_events]

    if len(eids)  == 0:
        return None, 0.0

    if fix_result is not None:
        eid = fix_result
    else:
        eid = np.random.choice(eids)
    lp = -np.log(len(eids))
    
    return eid, lp

def ev_merge_move_hough(sg, **kwargs):

    flp, rlp, birth_args, death_args = sample_repropose_args_hough(sg)

    return merge_helper_newstyle(sg, 
                                 forward_location_proposal=flp, 
                                 reverse_location_proposal=rlp, 
                                 birth_args=birth_args,
                                 death_args=death_args,
                                 **kwargs)

def ev_merge_move_corr(sg, **kwargs):

    flp, rlp, birth_args, death_args = sample_repropose_args_corr(sg)

    return merge_helper_newstyle(sg, 
                                 forward_location_proposal=flp, 
                                 reverse_location_proposal=rlp, 
                                 birth_args=birth_args,
                                 death_args=death_args,
                                 **kwargs)

def ev_split_move_hough(sg, **kwargs):

    flp, rlp, birth_args, death_args = sample_repropose_args_hough(sg)

    return split_helper_newstyle(sg, 
                                 forward_location_proposal=flp, 
                                 reverse_location_proposal=rlp, 
                                 birth_args=birth_args,
                                 death_args=death_args,
                                 **kwargs)

def ev_split_move_corr(sg, **kwargs):

    flp, rlp, birth_args, death_args = sample_repropose_args_corr(sg)

    return split_helper_newstyle(sg, 
                                 forward_location_proposal=flp, 
                                 reverse_location_proposal=rlp, 
                                 birth_args=birth_args,
                                 death_args=death_args,
                                 **kwargs)


def merge_helper_newstyle(sg, 
                          log_to_run_dir=None,
                          return_probs=False, 
                          fix_merge_eids=None,
                          **kwargs):

    eid1, eid2, lp_merge = sample_events_to_merge(sg, fix_result=fix_merge_eids)

    if eid1 is None:
        return False

    old_evs = [sg.get_event(eid1), sg.get_event(eid2)]
    eids = (eid1, eid2)
    merge_eid = sg.next_eid
    lp_new, lp_old, log_qforward, log_qbackward, revert_move = rebirth_events_helper(sg, eids, 
                                                                                     birth_eids=(merge_eid,), 
                                                                                     **kwargs)
    log_qforward += lp_merge

    _, lp_split = sample_event_to_split(sg, fix_result=merge_eid)
    log_qbackward += lp_split
    new_ev = sg.get_event(merge_eid)

    if log_to_run_dir is not None:
        log_file = os.path.join(log_to_run_dir, "merge_proposals.txt")
        
        with open(log_file, 'a') as f:
            f.write("merging %d %d to %d, lp %.2f reverse %.2f\n" % (eid1, eid2, merge_eid, lp_merge, lp_split))
            f.write("kwargs %s\n" % kwargs)
            f.write("old eids %d %d\n" % (eid1, eid2))
            for old_ev in old_evs:
                f.write("  %s\n" % str(old_ev))

            f.write("new eid %d\n" % (merge_eid))
            f.write("  %s\n" % str(new_ev))
            f.write("lp_new %.1f lp_old %.1f log_qbackward %.1f log_qforward %.1f ratio %.1f\n\n" % (lp_new, lp_old, log_qbackward, log_qforward, lp_new+log_qbackward - (lp_old+log_qforward)))
            
                    

    if return_probs:
        return lp_old, lp_new, log_qforward, log_qbackward, revert_move
    else:
        return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=None, revert_move=revert_move)

def split_helper_newstyle(sg, 
                          log_to_run_dir=None, 
                          return_probs=False, 
                          fix_split_eid=None,
                          **kwargs):

    eid, lp_split = sample_event_to_split(sg, fix_result=fix_split_eid)
    if eid is None:
        return False


    old_ev = sg.get_event(eid)

    split_eids = (sg.next_eid, sg.next_eid+1)
    lp_new, lp_old, log_qforward, log_qbackward, revert_move = rebirth_events_helper(sg, (eid,), 
                                                                                     birth_eids=split_eids, 
                                                                                     **kwargs)
    log_qforward += lp_split
    new_evs = [sg.get_event(new_eid) for new_eid in split_eids]

    _, _, lp_merge = sample_events_to_merge(sg, fix_result=split_eids)
    log_qbackward += lp_merge

    if log_to_run_dir is not None:
        log_file = os.path.join(log_to_run_dir, "split_proposals.txt")
        with open(log_file, 'a') as f:
            f.write("splitting %d to %d %d, lp %.2f reverse %.2f\n" % (eid, split_eids[0], split_eids[1], 
                                                                       lp_split, lp_merge))
            f.write("kwargs %s\n" % kwargs)

            f.write("old eid %d\n" % (eid))
            f.write("  %s\n" % str(old_ev))

            f.write("new eids %d %d\n" % split_eids)
            for new_ev in new_evs:
                f.write("  %s\n" % str(new_ev))

            f.write("lp_new %.1f lp_old %.1f log_qbackward %.1f log_qforward %.1f ratio %.1f\n\n" % (lp_new, lp_old, log_qbackward, log_qforward, lp_new+log_qbackward - (lp_old+log_qforward)))

    if return_probs:
        return lp_old, lp_new, log_qforward, log_qbackward, revert_move
    else:
        return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=None, revert_move=revert_move)
