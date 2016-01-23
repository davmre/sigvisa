import numpy as np
import itertools
import logging
import copy

from sigvisa.utils.counter import Counter
from sigvisa.utils.geog import dist_km
from sigvisa.models.distributions import Beta


def propose_env_tmvals(raw_tmvals, reverse=False):
    # hack: hardcode a proposal distribution to avoid bookkeeping.
    # this should really be the learned prior for each station/phase.
    model = Beta(alpha=6.3043781426475194, beta=1.4500748320708032)

    tmvals = copy.copy(raw_tmvals)
    try:
        if reverse:
            mult_wiggle_std = raw_tmvals["mult_wiggle_std"]
            del tmvals["mult_wiggle_std"]
        else:
            mult_wiggle_std = model.sample()
            tmvals["mult_wiggle_std"] = mult_wiggle_std    
    except Exception as e:
        print e
        import pdb; pdb.set_trace()

    return tmvals, model.log_p(mult_wiggle_std)

def propose_ev_tmvals(tmvals, reverse=False):
    r = {}
    lp = 0.0
    for eid, eid_dict in tmvals.items():
        r[eid] = {}
        for key, sta_dict in eid_dict.items():
            r[eid][key] = {}
            for phase, phase_dict in sta_dict.items():
                r[eid][key][phase], lp_local = propose_env_tmvals(phase_dict, reverse=reverse)
                lp += lp_local
    return r, lp

def crossover_events_helper(sg1, sg2, raw1, raw2, get_swapped):

    evs1 = sg1.get_event_locs()
    evs2 = sg2.get_event_locs()

    swapped_from_sg1, swapped_from_sg2 = get_swapped(evs1, evs2)
    if swapped_from_sg1 is None:
        return False

    # get templates for each event we'll be swapping
    sg1_orig_tmvals = {}
    sg2_orig_tmvals = {}
    for eid in swapped_from_sg1:
        sg1_orig_tmvals[eid] = sg1.get_event_templates(eid)
    for eid in swapped_from_sg2:
        sg2_orig_tmvals[eid] = sg2.get_event_templates(eid)

    # match raw-signals templates with env-signal templates if needed
    if raw1 and not raw2:
        sg1_swappable_tmvals, log_qforward = propose_ev_tmvals(sg1_orig_tmvals)
        sg2_swappable_tmvals, log_qbackward = propose_ev_tmvals(sg2_orig_tmvals, reverse=True)
    elif raw2 and not raw1:
        sg1_swappable_tmvals, log_qbackward = propose_ev_tmvals(sg1_orig_tmvals, reverse=True)
        sg2_swappable_tmvals, log_qforward = propose_ev_tmvals(sg2_orig_tmvals)
    else:
        sg1_swappable_tmvals = sg1_orig_tmvals
        sg2_swappable_tmvals = sg2_orig_tmvals
        log_qforward = 0.0
        log_qbackward = 0.0

    sg1_lp_old = sg1.current_log_p()
    sg2_lp_old = sg2.current_log_p()
    lp_old = sg1_lp_old + sg2_lp_old

    #bd1_old = sg1.current_log_p_breakdown()
    #bd2_old = sg2.current_log_p_breakdown()

    reverse_ids1 = {}
    reverse_ids2 = {}
    for eid in swapped_from_sg1:
        sg1.kill_event(eid)
        reverse_ids1[eid] = sg2.birth_event(evs1[eid], sg1_swappable_tmvals[eid])
    for eid in swapped_from_sg2:
        sg2.kill_event(eid)
        reverse_ids2[eid] = sg1.birth_event(evs2[eid], sg2_swappable_tmvals[eid])

    def reverse_move():
        for eid in swapped_from_sg1:
            sg2.kill_event(reverse_ids1[eid])
            sg1.birth_event(evs1[eid], sg1_orig_tmvals[eid], force_id=eid)
        for eid in swapped_from_sg2:
            sg1.kill_event(reverse_ids2[eid])
            sg2.birth_event(evs2[eid], sg2_orig_tmvals[eid], force_id=eid)

    sg1_lp_new = sg1.current_log_p()
    sg2_lp_new = sg2.current_log_p()
    lp_new = sg1_lp_new + sg2_lp_new

    #bd1_new = sg1.current_log_p_breakdown()
    #bd2_new = sg2.current_log_p_breakdown()

    ratio = (lp_new + log_qbackward) - (lp_old + log_qforward)


    u = np.random.rand()
    if np.log(u) < ratio:
        accept = True
    else:
        accept = False
        reverse_move()

    logging.info("event swap includes %d events, lp_new  %.1f (%.1f/%.1f) lp_old %.1f (%.1f/%.1f)  ratio %.1f" % (len(swapped_from_sg1) + len(swapped_from_sg2), lp_new, sg1_lp_new, sg2_lp_new, lp_old, sg1_lp_old, sg2_lp_old, ratio))

    return accept
        
def swap_events_move(sg1, sg2, raw1, raw2):

    def propose_pairwise_swap(evs1, evs2):
        if len(evs1) == 0 or len(evs2) == 0:
            return None, None

        eid1 = np.random.choice(evs1.keys())
        eid2 = np.random.choice(evs2.keys())

        swapped_from_sg1 = (eid1,)
        swapped_from_sg2 = (eid2,)

        return swapped_from_sg1, swapped_from_sg2

    return crossover_events_helper(sg1, sg2, raw1, raw2, propose_pairwise_swap)

def crossover_event_region_move(sg1, sg2, raw1, raw2,                      
                                crossover_radius_km = 200,
                                crossover_radius_s = 200):

    def evs_in_region(evs, center_dict):
        r = set()
        for eid, evdict in evs.items():
            d = dist_km((evdict["lon"], evdict["lat"]), (center_dict["lon"], center_dict["lat"]))
            t = np.abs(evdict["time"] - center_dict["time"])
            if d < crossover_radius_km and t < crossover_radius_s:
                r.add(eid)
        return r


    def propose_crossover_region(evs1, evs2):
        # choose an event at random from the union of the two chains
        # then swap *all* events within a region around that event



        # proposal is symmetric since we're only 
        # swapping locations

        if len(evs1) == 0 and len(evs2) == 0:
            return None, None

        unified_locs = evs1.values() + evs2.values()
        region_center = np.random.choice(unified_locs)

        swapped_from_sg1 = evs_in_region(evs1, region_center)
        swapped_from_sg2 = evs_in_region(evs2, region_center)
        return swapped_from_sg1, swapped_from_sg2

    return crossover_events_helper(sg1, sg2, raw1, raw2, propose_crossover_region)


def crossover_uatemplates(sg1, sg2, scb, raw1, raw2, 
                          crossover_period_s=50.0, 
                          crossover_period_pre_s=10.0):
    # scb = (sta, band, chan) tuple
    # raw1/2: whether each sg uses raw signals
    
    # choose a template at random from the overall set of templates
    # then, starting at the peak time for that template, exchange all templates 
    # that peak within the next crossover_period seconds
    
    # otherwise, exchange the set of templates peaking with that period

    def peak_times(sg_templates):
        return dict([(tmid, t["arrival_time"] + np.exp(t["peak_offset"])) for (tmid, t) in sg_templates.items()])
    
    def propose_swap_set(sg1_templates, sg2_templates, fix_result=None):
        sg1_peaks = peak_times(sg1_templates)
        sg2_peaks = peak_times(sg2_templates)
        
        overall_peaks = sg1_peaks.values() + sg2_peaks.values()

        if fix_result is not None:
            swap_start = fix_result
        else:
            swap_start = np.random.choice(overall_peaks) - crossover_period_pre_s

        sg1_templates_swappable = {}
        sg2_templates_swappable = {}

        for (tmid, peak_time) in sg1_peaks.items():
            if (swap_start <= peak_time < swap_start + crossover_period_s):
                sg1_templates_swappable[tmid] = sg1_templates[tmid]

        for (tmid, peak_time) in sg2_peaks.items():
            if (swap_start <= peak_time < swap_start + crossover_period_s):
                sg2_templates_swappable[tmid] = sg2_templates[tmid]
        
        return sg1_templates_swappable, sg2_templates_swappable, swap_start

    def kill_templates(sg, t):
        for tmid in t.keys():
            sg.kill_template(tmid)

    def birth_templates(sg, t, force_ids=None):
        reverse_ids = {}
        for tmid in t.keys():
            force_id = None
            if force_ids is not None:
                force_id = force_ids[tmid]
            new_tmid = sg.birth_template(scb, t[tmid], force_id=force_id)
            reverse_ids[new_tmid] = tmid
        return reverse_ids

    def match_raw_params(sg1_templates, sg2_templates, raw1, raw2):

        sg1_templates_proposed = {}
        sg2_templates_proposed = {}
        log_qforward = 0.0
        log_qbackward = 0.0
        if raw1 and not raw2:
            # propose msw for sg1 templates (going to sg2), and
            # delete msw from sg2 templates (going to sg1)
            for tmid in sg1_templates.keys():
                sg1_templates_proposed[tmid], lqf = propose_env_tmvals(sg1_templates[tmid])    
                log_qforward += lqf
            for tmid in sg2_templates.keys():
                sg2_templates_proposed[tmid], lqb = propose_env_tmvals(sg2_templates[tmid], 
                                                                       reverse=True)
                log_qbackward += lqb

        elif raw2 and not raw1:
            # propose msw for sg1 templates (going to sg2), and
            # delete msw from sg2 templates (going to sg1)
            for tmid in sg2_templates.keys():
                sg2_templates_proposed[tmid], lqf = propose_env_tmvals(sg2_templates[tmid])    
                log_qforward += lqf
            for tmid in sg1_templates.keys():
                sg1_templates_proposed[tmid], lqb = propose_env_tmvals(sg1_templates[tmid], 
                                                                       reverse=True)
                log_qbackward += lqb
        else:
            sg1_templates_proposed = sg1_templates
            sg2_templates_proposed = sg2_templates

        return sg1_templates_proposed, sg2_templates_proposed, log_qforward, log_qbackward

    # nested dicts mapping uatemplate ids to param:value dicts
    sg1_templates = sg1.uatemplates_at_scb(scb)
    sg2_templates = sg2.uatemplates_at_scb(scb)

    if len(sg1_templates) == 0 and len(sg2_templates) == 0:
        print "null return", sg1_templates, sg2_templates
        return False

    sg1_logp_old = sg1.logp_at_scb(scb)
    sg2_logp_old = sg2.logp_at_scb(scb)
    lp_old = sg1_logp_old + sg2_logp_old

    sg1_templates_swappable, sg2_templates_swappable, swap_start = propose_swap_set(sg1_templates, sg2_templates, fix_result=None)

    sg1_templates_proposed, sg2_templates_proposed, log_qforward, log_qbackward = match_raw_params(sg1_templates_swappable, sg2_templates_swappable, raw1, raw2)

    swap_id = np.random.randint(1000000)

    #sg1.dump_img_scb(scb, "crossover_%d_pre1" % swap_id)
    #sg2.dump_img_scb(scb, "crossover_%d_pre2" % swap_id)

    kill_templates(sg1, sg1_templates_proposed)
    kill_templates(sg2, sg2_templates_proposed)
    reverse1 = birth_templates(sg2, sg1_templates_proposed)
    reverse2 = birth_templates(sg1, sg2_templates_proposed)


    #sg1.dump_img_scb(scb, "crossover_%d_post1" % swap_id)
    #sg2.dump_img_scb(scb, "crossover_%d_post2" % swap_id)

    sg1_logp_new = sg1.logp_at_scb(scb)
    sg2_logp_new = sg2.logp_at_scb(scb)
    lp_new = sg1_logp_new + sg2_logp_new

    def reverse_move():
        kill_templates(sg1, reverse2)
        kill_templates(sg2, reverse1)
        birth_templates(sg1, sg1_templates_swappable)
        birth_templates(sg2, sg2_templates_swappable)

    ratio = (lp_new + log_qbackward) - (lp_old + log_qforward)
    u = np.random.rand()
    if np.log(u) < ratio:
        accept = True
    else:
        accept = False
        reverse_move()

    logging.info("swap from time %.1f includes %d templates, lp_new  %.1f (%.1f/%.1f) lp_old %.1f (%.1f/%.1f)  ratio %.1f" % (swap_start, len(reverse1) + len(reverse2), lp_new, sg1_logp_new, sg2_logp_new, lp_old, sg1_logp_old, sg2_logp_old, ratio))

    # debugging code to verify that the reverse move 
    # returns to the original state
    """
    reverse_move()
    sg1_logp_old2 = sg1.logp_at_scb(scb)
    sg2_logp_old2 = sg2.logp_at_scb(scb)
    lp_old2 = sg1_logp_old2 + sg2_logp_old2
    assert( np.abs(lp_old2 - lp_old) < 1e-6)
    accept = False
    """

    return accept


