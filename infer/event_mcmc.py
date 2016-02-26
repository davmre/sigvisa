
import numpy as np
import sys
import os
import traceback
import pickle
import copy

import numdifftools as nd
from collections import defaultdict
from optparse import OptionParser
from sigvisa.database.signal_data import *
from sigvisa.database.dataset import *
import itertools

from sigvisa.models.ttime import tt_predict, tt_predict_grad
from sigvisa.graph.sigvisa_graph import SigvisaGraph, get_param_model_id
from sigvisa.learn.train_param_common import load_modelid
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_segments
from sigvisa.infer.event_birthdeath import ev_sta_template_death_helper, ev_sta_template_birth_helper
from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.infer.mcmc_basic import mh_accept_util
from sigvisa.infer.propose_lstsqr import ev_lstsqr_dist
from sigvisa.infer.template_mcmc import preprocess_signal_for_sampling, improve_offset_move, indep_peak_move, get_sorted_arrivals, relevant_nodes_hack
from sigvisa.graph.graph_utils import create_key, parse_key
from sigvisa.models.distributions import TruncatedGaussian, Bernoulli
from sigvisa.plotting.plot import savefig, plot_with_fit
from matplotlib.figure import Figure
from sigvisa.utils.geog import wrap_lonlat

from sigvisa.graph.graph_utils import get_parent_value
from sigvisa.source.event import Event
import sigvisa.source.brune_source as brune

from scipy.optimize import leastsq



def phases_changed(sg, eid, ev):
    # given a proposed event, return dictionaries mapping sites to
    # - new_phases: phases that are newly invalid at the new location so must be killed if they currently exist
    # - invalid_phases: phases that are newly valid at the new location, so may be birthed. (and must be birthed with some probability in order to achieve detailed balance


    old_ev = sg.get_event(eid)
    new_phases = dict()
    invalid_phases = dict()
    for site in sg.site_elements.keys():
        old_site_phases = sg.predict_phases_site(ev=old_ev, site=site)
        new_site_phases = sg.predict_phases_site(ev=ev, site=site)
        new_phases[site] = new_site_phases - old_site_phases
        invalid_phases[site] = old_site_phases - new_site_phases

    return new_phases, invalid_phases

def propose_phasejump_deaths(sg, eid, invalid_phases):
    ev = sg.get_event(eid)
    death_phases = {}
    for site, stas in sg.site_elements.items():
        current_phases = sg.ev_arriving_phases(eid, site=site)        
        death_phases[site] = current_phases & invalid_phases[site]
    return death_phases

def propose_phasejump_births(sg, eid, new_phases, fix_result=None):
    ev = sg.get_event(eid)
    birth_phases = {}
    log_qforward = 0.0
    for site, stas in sg.site_elements.items():
        birth_phases[site] = set()
        for phase in new_phases[site]:
            pm = sg.get_phase_existence_model(phase)

            fr_phase = None
            if fix_result is not None:
                fr_phase = phase in fix_result[site]
            birth, lqf = pm.sample(ev=ev, site=site, fix_result=fr_phase)

            if birth:
                birth_phases[site].add(phase)
            log_qforward += lqf

    return birth_phases, log_qforward




def cheap_decouple_proposal(sg, wn, eid, sta_fixable_nodes, sta_relevant_nodes):

    # if no arrivals will be preserved at this station,
    # it's meaningless whether we decouple or not
    if len(sta_fixable_nodes) == 0:
        return False

    noise_std = np.sqrt(wn.nm_env.marginal_variance())    
    max_height = np.max([n.get_value() for n in sta_fixable_nodes if "coda_height" in n.label])

    score = np.exp(max_height )/noise_std
    if score < 0.5: 
        # if the max template is smaller than half a noise std, it can't be telling us much
        return True
    elif score > 6:
        # if the max template is bigger than six noise stds, it must
        # be affecting the signal (might be able to get away with less
        # here, but being cautious for now...)  our default now should
        # be to *not* decouple, so we keep the signal explanation
        # constant. 



        # the only exception is if the new
        # event location will violate a param constraint, making the
        # resulting state extraordinarily unlikely.
        # in this case we defer to the expensive proposal.
        residual_nodes = [n for n in sta_relevant_nodes if "tt_residual" in n.label ]
        penalty = np.sum([n.param_truncation_penalty(n.label, n.get_value(), coarse_tts=n.hack_coarse_tts is not None) for n in residual_nodes if n.hack_param_constraint])
        if penalty < -100:
            return None
        else:
            return False
    else:
        # in the intermediate cases, we'll do a more expensive proposal using the actual
        # signal probabilities
        return None

def set_event_proposing_decoupling(sg, eid, new_ev, invalid_phases, 
                                   decouple_templates="adaptive", fix_result=None):

    # to "decouple" means that when an evtime changes, the atime also
    # changes (i.e., tt_residual remains constant). I'm not sure why
    # past-me chose this terminology; it almost seems backwards. but
    # I'm sticking with it now.

    evdict = new_ev.to_dict()
    evnodes = sg.evnodes[eid]

    log_qforward = 0.0
    decouple_record = dict()
    replicate_fns = []

    # record the values of the nodes we're going to preserve, so
    # we can re-set them later.
    fixed_vals = dict()
    for n in sg.extended_evnodes[eid]:
        try:
            fixed_vals[n.label] = n.get_value()
        except KeyError:
            # ignore the evnodes themselves
            continue

    # set the new event values
    def set_event():
        for k in evdict.keys():
            evnodes[k].set_local_value(evdict[k], key=k, force_deterministic_consistency=False)

    set_event()
    replicate_fns.append(set_event)

    for site in sg.site_elements.keys():
        for sta in sg.site_elements[site]:
            
            wns = sg.station_waves[sta]
            wnlabel = wns[0].label

            sta_fixable_nodes = []
            sta_relevant_nodes = []
            for n in sg.extended_evnodes[eid]:
                try:
                    (eid2, phase, sta2, chan, band, param) = parse_key(n.label)
                except ValueError:
                    # ignore the ev nodes
                    continue
                if not (sta2 == sta or sta2 == site): 
                    continue
                elif phase in invalid_phases[site]:
                    continue

                if n.deterministic():
                    sta_fixable_nodes.append(n)
                else:
                    sta_relevant_nodes.append(n)

            def sta_logp():
                lp = np.sum([n.log_p() for n in sta_relevant_nodes])
                lp += np.sum([wn.log_p() for wn in wns])
                return lp

            # for some reason Python doesn't include the sta nodes inside the
            # closure for these functions unless we explicitly list them as arguments. 
            # (more precisely, it *does* include them, but it ends up pointing to the
            #  nodes of the last station rather than the one for this iteration of the 
            #  station loop...)
            def do_decouple(sta_fixable_nodes=sta_fixable_nodes,sta_relevant_nodes=sta_relevant_nodes):
                for n in sta_relevant_nodes:
                    if n.get_value() != fixed_vals[n.label]:
                        n.set_value(fixed_vals[n.label])
                for n in sta_fixable_nodes:
                    n.parent_predict()

            def do_preserve(sta_fixable_nodes=sta_fixable_nodes):  
                for n in sta_fixable_nodes:
                    n.set_value(fixed_vals[n.label])                    


            do_preserve()
            currently_decoupled=False
            if decouple_templates=="adaptive":
                wn = wns[0]
                decouple_heuristic = cheap_decouple_proposal(sg, wn, eid, sta_fixable_nodes, sta_relevant_nodes)
                if decouple_heuristic is not None:
                    p_decoupled = 1.0 - 1e-6 if decouple_heuristic else 1e-6
                else:
                    lp_preserved = sta_logp()
                    do_decouple()
                    currently_decoupled=True

                    lp_decoupled = sta_logp()
                    lpmax = max(lp_preserved, lp_decoupled)
                    lp_preserved = max(lp_preserved - lpmax, -10)
                    lp_decoupled = max(lp_decoupled - lpmax, -10)
                    p_decoupled = np.exp(lp_decoupled) / (np.exp(lp_decoupled) + np.exp(lp_preserved))

                dist = Bernoulli(p_decoupled)
                if fix_result is not None:
                    decouple = fix_result[wnlabel]["decoupled"]
                else:
                    decouple = dist.sample()

                lqf = dist.log_p(decouple)
                log_qforward += lqf

                # print "decouple %s at %s with p_decoupled %.3f lqf %.3f heuristic %s" % (decouple, wn.sta, p_decoupled, lqf, decouple_heuristic)
            else:
                decouple = decouple_templates
                
            decouple_record[wnlabel] = decouple

            if decouple:
                if not currently_decoupled:
                    do_decouple()
                replicate_fns.append(do_decouple)
            else:
                # if we did the expensive proposal, and then 
                # decided to preserve, then we need to 
                # reset the preserved state
                if currently_decoupled:
                    do_preserve()
                replicate_fns.append(do_preserve)

    def replicate_move():
        for fn in replicate_fns:
            fn()

    return log_qforward, replicate_move, decouple_record


def ev_phasejump_helper(sg, eid, new_ev, params_changed, 
                        decouple_templates=False,
                        birth_type="mh", fix_result=None):
    # assume: we are proposing to update eid to new_ev, but have not 
    # yet done so in the graph. 
    # this method:
    #   - sets the new event in the graph
    #   - proposes deassociation or death for all current phases that are impossible at the new    #     location, and modifies the graph
    #   - proposes to birth or associate all phases that are newly
    #     necessary at the new location.
    # The end result is the graph modified to the new state, and a
    # proposal probability for this state.
    
    log_qforward = 0.0
    replicate_fns = []
    phasejump_record = {}

    # figure out which phases we need to add and remove
    new_phases, invalid_phases = phases_changed(sg, eid, new_ev)
    
    # set the event to the new position, and either let the arrival times and coda_heights shift, or don't
    lqf, rset, decouple_record = set_event_proposing_decoupling(sg, eid, new_ev, invalid_phases, fix_result=fix_result, decouple_templates=decouple_templates)
    replicate_fns.append(rset)
    log_qforward += lqf
    for wnlabel in decouple_record.keys():
        if wnlabel not in phasejump_record:
            phasejump_record[wnlabel] = {}
        phasejump_record[wnlabel]["decoupled"] = decouple_record[wnlabel]

    fr_births = None
    if fix_result is not None:
        fr_births = fix_result["births"]
    birth_phases, lqf = propose_phasejump_births(sg, eid, new_phases, fix_result=fr_births)
    death_phases = propose_phasejump_deaths(sg, eid, invalid_phases)
    log_qforward += lqf
    # the phases we just killed will be reborn on the reverse move
    phasejump_record["births"] = death_phases

    # do all deaths before births, so that we eliminate invalid phases
    # from the graph before attempting to run birth proposals.
    for site in death_phases.keys():
        killed = death_phases[site]
        if len(killed) == 0:
            continue
        for sta in sg.site_elements[site]:
            for wn in sg.station_waves[sta]:
                if wn.label not in phasejump_record:
                    phasejump_record[wn.label] = {}
                fr_sta_death = None
                if fix_result is not None:
                    fr_sta_death=fix_result[wn.label]['death']

                # propose a death solution
                if len(killed) == 0:
                    death_record = None
                else:
                    lqf, re_death, death_record = ev_sta_template_death_helper(sg, wn, eid, phases=killed, 
                                                                               fix_result=fr_sta_death) 
                    log_qforward += lqf

                    replicate_fns.append(re_death)
                # we store the death record as the future 
                # fixed_result of the reverse birth move
                phasejump_record[wn.label]['birth'] = death_record
                

    for site in birth_phases.keys():
        birthed = birth_phases[site]
        if len(birthed) == 0:
            continue
        for sta in sg.site_elements[site]:
            for wn in sg.station_waves[sta]:
                if wn.label not in phasejump_record:
                    phasejump_record[wn.label] = {}
                fr_sta_birth = None
                if fix_result is not None:
                    fr_sta_birth=fix_result[wn.label]['birth']

                # propose a birth solution
                if len(birthed) == 0:
                    birth_record = None
                else:
                    lqf, re_birth, birth_record = ev_sta_template_birth_helper(sg, wn, eid, site_phases=birthed, fix_result=fr_sta_birth, proposal_type=birth_type)
                    log_qforward += lqf
                    replicate_fns.append(re_birth)
                phasejump_record[wn.label]['death'] = birth_record


    sg._topo_sort()
    replicate_fns.append(lambda : sg._topo_sort())



    def replicate_move():
        for fn in replicate_fns:
            fn()

    return log_qforward, replicate_move, phasejump_record



def phasejump_reduced_logp(sg, eid, phasejump_record):
    # compute the logp of only the relevant nodes for this move,
    # based on the record of where we added/removed phases

    # if we're using GP wiggle models, we need to compute logps at every station
    if sg.wiggle_model_type != "dummy":
        return sg.current_log_p()


    lp = 0.0

    # always compute the ev prior, and
    # always compute all params at all stations, since
    #  - if we're not using GPs, this is cheap (though maybe not necessary)
    #  - if we're using GPs, this is necessary, since any parent-conditional 
    #    distribution can be changed by a change in ev location. 
    lp += np.sum([n.log_p() for n in sg.extended_evnodes[eid] if not n.deterministic()])


    # always compute phase existence model at all stations because I'm lazy and this should be cheap
    lp += sg.phase_existence_lp()

    # at every station where we've changed the set of arriving phases,
    # compute the signal lp, and also include lps for all uatemplates
    # (since we might have associated or disassociated templates)


    for wnlabel in phasejump_record.keys():
        if wnlabel == "births":
            continue
        # include any wns where our signal explanation will have changed.
        # if the only record is that we didn't decouple, we can skip this wn.
        if len(phasejump_record[wnlabel]) == 1 and not phasejump_record[wnlabel]['decoupled']:
            continue

        wn = sg.all_nodes[wnlabel]
        lp += wn.log_p()

        tmids = [-eid for (eid, phase) in wn.arrivals() if phase=="UA"]
        lp += sg.ntemplates_sta_log_p(wn, n=len(tmids))
        for tmid in tmids:
            uanodes = sg.uatemplates[tmid]
            lp += np.sum([n.log_p() for n in uanodes.values()])

    return lp

def ev_move_full(sg, ev_node, std, params, eid=None,
                 decouple_templates="auto",
                 proposer=None,
                 forward_type=None,
                 reverse_type=None,
                 logger=None, return_debug=False):
    def update_ev(old_ev, params, new_v):
        new_ev = copy.copy(old_ev)
        try:
            i = params.index("lon")
            new_ev.lon = new_v[i]
        except ValueError:
            pass
        try:
            i = params.index("lat")
            new_ev.lat = new_v[i]
        except ValueError:
            pass
        try:
            i = params.index("depth")
            new_ev.depth = new_v[i]
        except ValueError:
            pass
        try:
            i = params.index("time")
            new_ev.time = new_v[i]
        except ValueError:
            pass
        try:
            i = params.index("mb")
            new_ev.mb = new_v[i]
        except ValueError:
            pass
        return new_ev

    def propose_ev(ev_node, eid, params):
        d = len(params)
        # get the current values of the params we're updating
        current_v = np.zeros((d,))
        for i in range(d):
            current_v[i] = ev_node.get_local_value(params[i])

        log_qforward = 0.0
        log_qbackward = 0.0

        # propose a new set of param values
        if params[0] == "depth":
            current_v = float(current_v)
            rv1 = TruncatedGaussian(current_v, std=std, a=0.0, b=700.0)
            new_v = rv1.sample()
            rv2 = TruncatedGaussian(new_v, std=std, a=0.0, b=700.0)
            log_qforward += rv1.log_p(new_v)
            log_qbackward += rv2.log_p(current_v)
            new_v = np.array((new_v,))
        else:
            gsample = np.random.normal(0, std, d)
            move = gsample * std
            new_v = current_v + move

        if "lon" in params:
            new_v[0], new_v[1] = wrap_lonlat(new_v[0], new_v[1])

        old_ev = sg.get_event(eid)
        new_ev = update_ev(old_ev, params, new_v)
        
        return old_ev, new_ev, log_qforward, log_qbackward

            
    if eid is None:
        eid = int(ev_node.label.split(';')[0])

    if proposer is not None:
        old_ev, new_ev, log_qforward, log_qbackward = proposer(sg, eid, params)
    else:
        old_ev, new_ev, log_qforward, log_qbackward = propose_ev(ev_node, eid, params)

    if forward_type is None:
        dumb_forward = np.random.rand() < 0.2
        forward_type = "dumb" if dumb_forward else "mh" 
    if reverse_type is None:
        dumb_backward = np.random.rand() < 0.2
        reverse_type = "dumb" if dumb_backward else "mh" 

    if decouple_templates=="auto":        
        adaptive_decouple = ("lon" in params or "time" in params or "depth" in params)
        decouple_templates = "adaptive" if adaptive_decouple else False

    # set the event, and propose values for new and removed phases, as needed
    lqf, replicate_forward, phasejump_record = ev_phasejump_helper(sg, eid, new_ev, params, 
                                                                   decouple_templates=decouple_templates,
                                                                   birth_type=forward_type)
    log_qforward += lqf

    #lp_new = sg.current_log_p()
    lp_new_quick = phasejump_reduced_logp(sg, eid, phasejump_record)
    
    # set the old event and propose values 
    lqb, _, phasejump_reverse_record = ev_phasejump_helper(sg, eid, old_ev, params, 
                                                           birth_type=reverse_type,
                                                           decouple_templates=decouple_templates,
                                                           fix_result=phasejump_record)
    log_qbackward += lqb

    #lp_old = sg.current_log_p()
    lp_old_quick = phasejump_reduced_logp(sg, eid, phasejump_reverse_record)

    #diff1 = lp_new - lp_old
    #diff2 = lp_new_quick - lp_old_quick
    #assert(np.abs(diff2 - diff1) < 1e-1)
    
    accepted = mh_accept_util(lp_old_quick, lp_new_quick, log_qforward, log_qbackward, accept_move=replicate_forward)

    if logger is not None:
                            
        phasejump_reverse_record["forward_type"] = forward_type
        phasejump_reverse_record["reverse_type"] = reverse_type

        logger.log_event_move(sg, eid, old_ev, new_ev, 
                              phasejump_reverse_record,
                              lp_old_quick, lp_new_quick, 
                              log_qforward, log_qbackward,
                              accepted=accepted)

    if return_debug:
        return lp_old_quick, lp_new_quick, log_qforward, log_qbackward, replicate_forward, phasejump_reverse_record
    else:
        return accepted
    
def ev_lsqr_move(sg, eid, 
                 forward_type=None, 
                 reverse_type=None,
                 decouple_templates="adaptive",
                 logger=None,
                 return_debug=False):


    old_ev, new_ev, log_qforward, z, C = propose_event_lsqr(sg, eid)

    if forward_type is None:
        dumb_forward = np.random.rand() < 0.4
        forward_type = "dumb" if dumb_forward else "smart"
    if reverse_type is None:
        dumb_backward = np.random.rand() < 0.4
        reverse_type = "dumb" if dumb_backward else "smart"

    params = ["lon", "lat", "depth", "time"]

    # set the event, and propose values for new and removed phases, as needed
    lqf, replicate_forward, phasejump_record = ev_phasejump_helper(sg, eid, new_ev, params, 
                                                                   decouple_templates=decouple_templates,
                                                                   birth_type=forward_type)
    log_qforward += lqf

    #lp_new = sg.current_log_p()
    lp_new_quick = phasejump_reduced_logp(sg, eid, phasejump_record)
    

    _, _, log_qbackward, _, _ = propose_event_lsqr(sg, eid, fix_result=old_ev)

    # set the old event and propose values 
    lqb, _, phasejump_reverse_record = ev_phasejump_helper(sg, eid, old_ev, params, 
                                                           birth_type=reverse_type,
                                                           decouple_templates=decouple_templates,
                                                           fix_result=phasejump_record)
    log_qbackward += lqb

    #lp_old = sg.current_log_p()
    lp_old_quick = phasejump_reduced_logp(sg, eid, phasejump_reverse_record)

    #diff1 = lp_new - lp_old
    #diff2 = lp_new_quick - lp_old_quick
    #assert(np.abs(diff2 - diff1) < 1e-1)
    
    accepted = mh_accept_util(lp_old_quick, lp_new_quick, log_qforward, log_qbackward, accept_move=replicate_forward)

    if logger is not None:
                            
        phasejump_reverse_record["forward_type"] = forward_type
        phasejump_reverse_record["reverse_type"] = reverse_type

        logger.log_event_move(sg, eid, old_ev, new_ev, 
                              ("lsqr", " ".join(["%.2f" %x for x in z]), C, phasejump_reverse_record),
                              lp_old_quick, lp_new_quick, 
                              log_qforward, log_qbackward,
                              accepted=accepted)

    if return_debug:
        return lp_old_quick, lp_new_quick, log_qforward, log_qbackward, replicate_forward, phasejump_reverse_record
    else:
        return accepted


def ev_lonlat_density(frame=None, fname="ev_viz.png"):

    d = np.load("ev_vals.npz")
    latlons = d['evloc']
    lonlats = np.array([(a,b) for (b,a) in latlons])

    if frame is not None:
        if frame > len(lonlats):
            raise ValueError("no more frames!")
        lonlats_plot = lonlats[:frame]
    else:
        lonlats_plot = lonlats

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from sigvisa.plotting.event_heatmap import EventHeatmap

    f = Figure((11,8))
    ax = f.add_subplot(111)
    hm = EventHeatmap(f=None, autobounds=lonlats, autobounds_quantile=0.9995, calc=False)
    hm.init_bmap(axes=ax)
    hm.plot_earth(y_fontsize=16, x_fontsize=16)

    ev = get_event(evid=5393637)

    baseline_alpha = 0.008
    alpha_fade_time = 500
    if frame is not None:
        alpha = np.ones((frame,)) * baseline_alpha
        t = min(frame,alpha_fade_time)
        alpha[-t:] = np.linspace(baseline_alpha, 0.2, alpha_fade_time)[-t:]
    else:
        alpha = baseline_alpha

    #hm.plot_locations(X, marker=".", ms=6, mfc="red", mec="none", mew=0, alpha=0.2)
    scplot = hm.plot_locations(lonlats_plot, marker=".", ms=8, mfc="red", mew=0, mec="none", alpha=alpha)
    hm.plot_locations(np.array(((ev.lon, ev.lat),)), marker="x", ms=5, mfc="blue", mec="blue", mew=3, alpha=1.0)
    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig(fname, bbox_inches="tight", dpi=300, transparent=True, )

def ev_lonlat_frames():
    for i in range(40, 10000, 40):
        ev_lonlat_density(frame=i, fname='ev_viz_step%06d.png' % i)


def propose_event_lsqr(sg, eid, fix_result=None):

    # breaks reversibility to initialize at current location
    # but I don't super care
    old_ev = sg.get_event(eid)
    init_z = np.array((old_ev.lon, old_ev.lat, old_ev.depth, old_ev.time))

    try:
        z, C = ev_lstsqr_dist(sg, eid, init_z=init_z)
    except Exception as e:
        sg.logger.warning("exception proposing lsqr location for eid %d: %s" % (eid, str(e)))
        z = init_z
        C = np.eye(4)

    try:
        rv = scipy.stats.multivariate_normal(z, C)
    except:
        C = np.eye(4) * 50.0
        rv = scipy.stats.multivariate_normal(z, C)

    if fix_result is None:
        proposed_vals = rv.rvs(1)
        lon, lat, depth, time = proposed_vals
    else:
        lon, lat, depth, time = fix_result.lon, fix_result.lat, fix_result.depth, fix_result.time
        proposed_vals = np.array((lon, lat, depth, time))

    proposal_lp = rv.logpdf(proposed_vals)

    # hack: pretend we propose from a mix of the given Gaussian and a
    # "dumb" uniform-ish distribution, in order to keep the reverse
    # logps reasonable.
    proposal_lp = np.logaddexp(proposal_lp, -10)

    # this breaks Gaussianity, technically we should be using a
    # circular (von Mises?) distribution. but hopefully it doesn't
    # matter.
    lon, lat = wrap_lonlat(lon, lat)

    # this definitely breaks Gaussianity, we should be explicitly truncating the distribution
    if depth > 700:
        depth = 700
    elif depth < 0:
        depth = 0

    new_ev = copy.copy(old_ev)
    new_ev.lon = lon
    new_ev.lat = lat
    new_ev.depth=depth
    new_ev.time=time

    return old_ev, new_ev, proposal_lp, z, C


def sample_uniform_pair_to_swap(sg, wn, adjacency_decay=0.8):
    sorted_arrs = get_sorted_arrivals(wn)
    n = len(sorted_arrs)

    # if we sample adjacency=1, swap an adjacent pair
    # adjacency=2 => swap a pair separated by another template
    # etc.
    adjacency = np.random.geometric(adjacency_decay)
    adjacency_prob = adjacency_decay * (1-adjacency_decay)**(adjacency-1)
    if adjacency > n-1:
        return None, None, 1.0

    # then just propose a pair to swap uniformly at random
    first_idx = np.random.choice(np.arange(n-adjacency))
    second_idx = first_idx + adjacency
    choice_prob = 1.0/(n-adjacency)
    return sorted_arrs[first_idx], sorted_arrs[second_idx], adjacency_prob*choice_prob



def swap_params(t1nodes, t2nodes):
    for (p, (k1, n1)) in t1nodes.items():
        if p=="amp_transfer" or p=="tt_residual":
            continue
        k2, n2 = t2nodes[p]
        v1 = n1.get_value(key=k1)
        v2 = n2.get_value(key=k2)
        n1.set_value(key=k1, value=v2)
        n2.set_value(key=k2, value=v1)

        if p == "arrival_time":
            atime1, atime2 = v1, v2
    return atime1, atime2

def swap_association_move(sg, wn, repropose_events=False, debug_probs=False, stas=None):

    # sample from all pairs of adjacent templates
    arr1, arr2, pair_prob = sample_uniform_pair_to_swap(sg, wn)
    if arr1 is None:
        return False

    # don't bother "swapping" uatemplates
    if arr1[2]=="UA" and arr2[2] == "UA":
        return False


    # get all relevant nodes for the arrivals we sampled
    t1nodes = sg.get_template_nodes(eid=arr1[1], phase=arr1[2], sta=wn.sta, band=wn.band, chan=wn.chan)
    t2nodes = sg.get_template_nodes(eid=arr2[1], phase=arr2[2], sta=wn.sta, band=wn.band, chan=wn.chan)
    rn = set(relevant_nodes_hack(t1nodes) + relevant_nodes_hack(t2nodes))

    # if we're using GP wiggle models, swapping templates will affect the wiggle prior
    if len(wn.wavelet_param_models) > 0:
        rn.add(wn)

    if repropose_events:
        if arr1[1] > 0:
            evnodes = set([n for n in sg.extended_evnodes[arr1[1]] if not n.deterministic()])
            rn = rn.union(evnodes)
        if arr2[1] != arr1[1] and arr2[1] > 0:
            evnodes = [n for n in sg.extended_evnodes[arr2[1]] if not n.deterministic()]
            rn = rn.union(evnodes)

    if debug_probs:
        lp_old_full = sg.current_log_p()
        lps_old = dict([(n.label, n.log_p()) for n in sg.all_nodes.values() if not n.deterministic()])
    lp_old = sg.joint_logprob_keys(rn)
    #lp_old_full = sg.current_log_p()
    #lp_old_bd = sg.current_log_p_breakdown()

    # swap proposal is symmetric, but we still need to track
    # probabilities of the event proposals.
    log_qforward = 0
    log_qbackward = 0

    if repropose_events:
        if arr1[1] > 0:
            log_qbackward += propose_event_lsqr_prob(sg, eid=arr1[1], stas=stas)
        if (arr2[1] != arr1[1]) and arr2[1] > 0:
            log_qbackward += propose_event_lsqr_prob(sg, eid=arr2[1], stas=stas)

    # switch their parameters
    atime1, atime2 = swap_params(t1nodes, t2nodes)

    revert_fns = []

    lp_old_delta1 = 0
    lp_old_delta2 = 0
    lp_new_delta1 = 0
    lp_new_delta2 = 0
    node_lps1 = None
    node_lps2 = None
    if repropose_events:
        if arr1[1] > 0:
            proposal_lp, old_lp, revert_move, jump_required, node_lps1 = propose_event_lsqr(sg, eid=arr1[1], stas=stas)
            log_qforward += proposal_lp
            revert_fns.append(revert_move)
            lp_old_delta1 = node_lps1.update_lp_old(sg, rn)
            lp_new_delta1 = node_lps1.update_lp_new(sg, rn)
            node_lps1.update_relevant_nodes_for_lpnew(rn)
        if (arr2[1] != arr1[1]) and arr2[1] > 0:
            proposal_lp, old_lp, revert_move, jump_required, node_lps2 = propose_event_lsqr(sg, eid=arr2[1], stas=stas)
            log_qforward += proposal_lp
            revert_fns.append(revert_move)
            lp_old_delta2 = node_lps2.update_lp_old(sg, rn)
            lp_new_delta2 = node_lps2.update_lp_new(sg, rn)
            node_lps2.update_relevant_nodes_for_lpnew(rn)

    lp_new = sg.joint_logprob_keys(rn)
    #lp_new_full = sg.current_log_p()
    #lp_new_bd = sg.current_log_p_breakdown()
    lp_new += lp_new_delta1 + lp_new_delta2
    lp_old += lp_old_delta1 + lp_old_delta2

    if debug_probs:
        lp_new_full = sg.current_log_p()
        lps_new = dict([(n.label, n.log_p()) for n in sg.all_nodes.values() if not n.deterministic()])


        if node_lps1 is not None:
            print "updates from ev1 proposal"
            node_lps1.dump_debug_info()
        if node_lps2 is not None:
            print "updates from ev2 proposal"
            node_lps2.dump_debug_info()
        print "actual changes:"
        node_lps1.dump_detected_changes(lps_old, lps_new, rn)


        assert( np.abs( (lp_new-lp_old) - (lp_new_full-lp_old_full) ) < 1e-8)

    u = np.random.rand()

    def revert_all():
        for fn in revert_fns:
            fn()
        atime1, atime2 = swap_params(t1nodes, t2nodes)

    if (lp_new + log_qbackward) - (lp_old + log_qforward) > np.log(u):
        return True
    else:
        revert_all()
        return False



def ev_source_type_move(sg, eid, logger=None, **kwargs):
    evnode = sg.evnodes[eid]["natural_source"]
    # propose a new source type while holding coda heights constant
    coda_heights = [(n, n.get_value()) for n in sg.extended_evnodes[eid] if "coda_height" in n.label]

    def set_source(is_natural):
        evnode.set_value(is_natural)
        for height_node, fixed_height in coda_heights:
            height_node.set_value(fixed_height)

    old_value = evnode.get_value()
    proposed_value = not old_value

    amp_transfers = [n for n in sg.extended_evnodes[eid] if "amp_transfer" in n.label]
    relevant_nodes = [evnode,] + amp_transfers
    lp_old = sg.joint_logprob_keys(relevant_nodes)

    set_source(proposed_value)
    lp_new = sg.joint_logprob_keys(relevant_nodes)

    def revert_move():
        set_source(old_value)

    return mh_accept_util(lp_old, lp_new, revert_move=revert_move)
