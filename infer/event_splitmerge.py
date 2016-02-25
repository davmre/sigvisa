import numpy as np
import sys
import os
import traceback
import pickle
import copy
import itertools

from sigvisa import Sigvisa


from sigvisa.infer.event_birthdeath import ev_bare_birth_move, ev_bare_death_move, ev_template_birth_helper, ev_template_death_helper
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
        return None, 0.0

    if fix_result is not None:
        merge_eid1, merge_eid2 = sorted(fix_result)
    else:
        merge_eid1, merge_eid2 = merge_probs.sample()

    lp = np.log(merge_probs[(merge_eid1, merge_eid2)])
    return merge_eid1, merge_eid2, lp

def sample_event_to_split(sg, fix_result=None):
    eids = [eid for eid in sg.evnodes.keys() if eid not in sg.fixed_events]

    if len(eids) is None:
        return None, 0.0

    if fix_result is not None:
        eid = fix_result
    else:
        eid = np.random.choice(eids)
    lp = -np.log(len(eids))
    
    return eid, lp

def noop():
    pass

def merge_events_helper(sg, location_proposal, 
                        proposal_includes_mb=True,
                        use_correlation=False,
                        birth_proposal_type="mh",
                        debug_info={},
                        force_eids=None,
                        force_new_ev=None,
                        fix_result=None):

    log_qforward = 0.0
    merge_record = {}

    fr_eids = fix_result["eids"] if fix_result is not None else None
    if fr_eids is None:
        fr_eids = force_eids
    if fr_eids is not None and (fr_eids[0] < 0 or fr_eids[1] < 0):
        # if the location proposal in the split move failed, then one
        # or both events were not created and so don't now need to be
        # killed.
        log_qforward = -np.inf

        replicate_untmpls1 = noop
        replicate_death1 = noop

        replicate_untmpls2 = noop
        replicate_death2 = noop

    else:
        eid1, eid2, merge_lp = sample_events_to_merge(sg, fix_result=fr_eids)
        print "merging eids", eid1, eid2
        if eid1 is None:
            # not enough events for a merge move
            return None, None, None
        
        merge_record["merged_ev1"] = (sg.get_event(eid1), eid1) 
        merge_record["merged_ev2"]= (sg.get_event(eid2), eid2) 
        log_qforward += merge_lp

        # remove both merged events
        fr_eid1 = fix_result["split_tmpls1"] if fix_result is not None else None
        lqf1, replicate_untmpls1, death_records1 = ev_template_death_helper(sg, eid1, fix_result=fr_eid1)
        merge_record["merged_tmpls1"] = death_records1
        log_qforward += lqf1
        replicate_death1, ev1 = ev_bare_death_move(sg, eid1)

        fr_eid2 = fix_result["split_tmpls2"] if fix_result is not None else None
        lqf2, replicate_untmpls2, death_records2 = ev_template_death_helper(sg, eid2, fix_result=fr_eid2)
        merge_record["merged_tmpls2"] = death_records2
        log_qforward += lqf2
        replicate_death2, ev2 = ev_bare_death_move(sg, eid2)
        sg._topo_sort()

    lp_baseline = sg.current_log_p()
    print "baseline lp", lp_baseline

    fr_ev = fix_result["split_ev"] if fix_result is not None else None
    if force_new_ev is not None:
        fr_ev = (force_new_ev, None)
    lq_loc, replicate_birth, eid, extra = ev_bare_birth_move(sg, location_proposal, debug_info=debug_info, fix_result=fr_ev)
    if lq_loc is None:
        # hack: if location proposal fails, just don't birth an
        # event so that the "merge" becomes a double death move (which
        # will presumably fail). This means we have to be careful to
        # ensure that the split move effectively reverses the effects
        # of the merge-so-far.
        merge_record["merged_eid"] = -1
        replicate_birth = noop
        replicate_tmpls = noop
    else:
        merge_record["merged_eid"] = eid
        log_qforward += lq_loc

        
        fr_birth = fix_result["split_tmpls"] if fix_result is not None else None
        lqf, replicate_tmpls, birth_records = \
                    ev_template_birth_helper(sg, eid, 
                                             associate_using_mb=proposal_includes_mb, 
                                             use_correlation=use_correlation,
                                             proposal_type=birth_proposal_type,
                                             debug_info=debug_info,
                                             fix_result=fr_birth)
        log_qforward += lqf
        sg._topo_sort()
        merge_record["merged_tmpls"] = birth_records

    def replicate_merge():
        replicate_untmpls1()
        replicate_untmpls2()
        replicate_death1()
        replicate_death2()
        sg._topo_sort()
        replicate_birth()
        replicate_tmpls()
        sg._topo_sort()

    print "merge lqf", "choice", merge_lp, "death1", lqf1, "death2", lqf2, "loc", lq_loc, "tmpls", lqf
    print "merge lqf totals", merge_lp + lqf1 + lqf2 + lq_loc + lqf, log_qforward

    return log_qforward, replicate_merge, merge_record

def split_events_helper(sg, location_proposal,                         
                        proposal_includes_mb=True,
                        use_correlation=False,
                        debug_info = {},
                        birth_proposal_type="mh",
                        flip_ordering=None,
                        fix_result=None):

    log_qforward = 0.0
    split_record = {}

    fr_eid = fix_result["merged_eid"] if fix_result is not None else None
    if fr_eid is not None and fr_eid < 0:
        # if we're reversing a merge whose location proposal failed,
        # then there is no event to delete
        log_qforward = -np.inf
    else:
        eid, split_lp = sample_event_to_split(sg, fix_result=fr_eid)
        if eid is None:
            # no events to split
            return None, None, None
        log_qforward += split_lp

        # remove the event to be split
        fr_birth = fix_result["merged_tmpls"] if fix_result is not None else None
        lqf, replicate_untmpls, death_records = ev_template_death_helper(sg, eid, fix_result=fr_birth)
        split_record["split_tmpls"] = death_records
        log_qforward += lqf

        replicate_death, ev = ev_bare_death_move(sg, eid)
        split_record["split_ev"] = (ev, eid)
        sg._topo_sort()

    lp_baseline = sg.current_log_p()
    print "baseline lp", lp_baseline

    # birth the first of the resulting events
    debug_info["ev1"] = {}
    debug_info["ev2"] = {}
    fr_ev1 = fix_result["merged_ev1"] if fix_result is not None else None
    fr_ev2 = fix_result["merged_ev2"] if fix_result is not None else None
    lq_loc1, replicate_birth1, eid1, extra1 = ev_bare_birth_move(sg, location_proposal, debug_info=debug_info["ev1"], fix_result=fr_ev1)
    lq_loc2, replicate_birth2, eid2, extra2 = ev_bare_birth_move(sg, location_proposal, debug_info=debug_info["ev2"], fix_result=fr_ev2)
    if lq_loc1 is None or lq_loc2 is None:
        # if the location proposal fails,
        # don't birth anything
        if eid1 is not None:
            sg.remove_event(eid1)
        if eid2 is not None:
            sg.remove_event(eid2)
        eid1, eid2 = -1, -1

        replicate_birth1 = noop
        replicate_tmpls1 = noop

        replicate_birth2 = noop
        replicate_tmpls2 = noop
    else:
        log_qforward += lq_loc1
        log_qforward += lq_loc2

        # warning: haven't thought through the correctness of this
        # in the context of the overall move. the "right" thing is 
        # to sum over both orderings
        if flip_ordering is None:
            flip_ordering = np.random.choice((True, False))
        fr_tmpls1 = fix_result["merged_tmpls1"] if fix_result is not None else None
        fr_tmpls2 = fix_result["merged_tmpls2"] if fix_result is not None else None
        if flip_ordering:
            birth_eid1, birth_eid2 = eid2, eid1
            fr_tmpls1, fr_tmpls2 = fr_tmpls2, fr_tmpls1
        else:
            birth_eid1, birth_eid2 = eid1, eid2

        log_qforward += np.log(0.5)

        lqf1, replicate_tmpls1, birth_records1 = \
                        ev_template_birth_helper(sg, birth_eid1, 
                                                 associate_using_mb=proposal_includes_mb, 
                                                 use_correlation=use_correlation,
                                                 proposal_type=birth_proposal_type,
                                                 debug_info=debug_info["ev1"],
                                                 fix_result=fr_tmpls1)
        log_qforward += lqf1

        lqf2, replicate_tmpls2, birth_records2 = \
                        ev_template_birth_helper(sg, birth_eid2,
                                                 associate_using_mb=proposal_includes_mb, 
                                                 use_correlation=use_correlation,
                                                 proposal_type=birth_proposal_type,
                                                 debug_info=debug_info["ev2"],
                                                 fix_result=fr_tmpls2)
        log_qforward += lqf2

        if flip_ordering:
            # "tmpls1" always refers to eid1, not birth_eid1
            split_record["split_tmpls1"] = birth_records2
            split_record["split_tmpls2"] = birth_records1
        else:
            split_record["split_tmpls1"] = birth_records1
            split_record["split_tmpls2"] = birth_records2


        print "split lps", split_lp, lqf, lq_loc1, lq_loc2, lqf1, lqf2, 
        print "total", log_qforward, split_lp+ lqf+ lq_loc1+ lq_loc2+ lqf1+ lqf2
    split_record["eids"] = (eid1, eid2)
    sg._topo_sort()

    def replicate_split():
        replicate_untmpls()
        replicate_death()
        sg._topo_sort()
        replicate_birth1()
        replicate_tmpls1()
        replicate_birth2()
        replicate_tmpls2()
        sg._topo_sort()

    return log_qforward, replicate_split, split_record
    
