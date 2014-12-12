import numpy as np
import copy
import sys
import traceback
import pdb
import pickle
import os

from sigvisa import Sigvisa
from sigvisa.graph.array_node import lldlld_X
from sigvisa.graph.graph_utils import extract_sta_node, create_key, get_parent_value, parse_key
from sigvisa.graph.sigvisa_graph import get_param_model_id, dummyPriorModel
from sigvisa.infer.propose_hough import hough_location_proposal, visualize_hough_array
from sigvisa.infer.propose_lstsqr import overpropose_new_locations
from sigvisa.infer.propose_mb import propose_mb

from sigvisa.infer.template_mcmc import get_signal_based_amplitude_distribution, propose_wiggles_from_signal, wiggle_proposal_lprob_from_signal, get_signal_diff_positive_part, sample_peak_time_from_signal, merge_distribution, peak_log_p
from sigvisa.infer.mcmc_basic import mh_accept_util
from sigvisa.learn.train_param_common import load_modelid
from sigvisa.models.ttime import tt_residual, tt_predict
from sigvisa.models.templates.coda_height import amp_transfer
from sigvisa.utils.counter import Counter
from sigvisa.utils.fileutils import mkdir_p
from sigvisa.source.event import get_event


hough_options = {'bin_width_deg':1.0, 'time_tick_s': 10.0, 'smoothbins': True}

def set_hough_options(ho):
    global hough_options
    hough_options = ho

def unass_template_logprob(sg, sta, template_dict, ignore_mb=False):
    """

    return the log prob of a set of template parameters, under the
    model of unassociated templates at a station sta.

    """

    # HACK
    assert(len(sg.station_waves[sta]) == 1)
    wn = sg.station_waves[sta][0]

    tg = sg.template_generator(phase="UA")
    wg = sg.wiggle_generator(phase="UA", srate=wn.srate)


    lp = 0.0
    lp += -np.log(float(wn.npts)/wn.srate) # arrival time
    for param in tg.params():
        if ignore_mb and param=="coda_height": continue
        model = tg.unassociated_model(param, nm=wn.nm)
        lp += model.log_p(template_dict[param])
    for param in wg.params():
        model = wg.unassociated_model(param, nm=wn.nm)
        lp += model.log_p(template_dict[param])
    return lp

def param_logprob(sg, site, sta, ev, phase, chan, band, param, val, basisid=None, wiggle=False):

    """

    return the log probability for an individual template parameter,
    as generated by an event phase arrival, WITHOUT interfering with
    the graph.

    """

    model_type = sg._tm_type(param, site=site, wiggle_param=wiggle)
    if model_type == "dummy":
        return 0.0
    if model_type == "dummyPrior":
        model = dummyPriorModel(param)
        return model.log_p(x=val)

    s = Sigvisa()
    if s.is_array_station(site) and sg.arrays_joint:
        modelid = get_param_model_id(runid=sg.runid, sta=site,
                                     phase=phase, model_type=model_type,
                                     param=param, template_shape=sg.template_shape,
                                     chan=chan, band=band, basisid=basisid)
        cond = lldlld_X(ev, sta)
    else:
        modelid = get_param_model_id(runid=sg.runid, sta=sta,
                                     phase=phase, model_type=model_type,
                                     param=param, template_shape=sg.template_shape,
                                     chan=chan, band=band, basisid=basisid)
        cond = ev

    model = load_modelid(modelid)
    return model.log_p(x = val, cond = cond)

def ev_phase_template_logprob(sg, sta, eid, phase, template_dict, verbose=False, ignore_mb=False):

    """

    return log p(template params in template_dict) under the distribution generated by a phase arrival from event eid at station sta.

    """

    ev = sg.get_event(eid)

    s = Sigvisa()
    site = s.get_array_site(sta)

    # HACK
    assert(len(sg.station_waves[sta]) == 1)
    wn = sg.station_waves[sta][0]
    wg = sg.wiggle_generator(phase=phase, srate=wn.srate)
    wiggle_params = set(wg.params())

    if 'tt_residual' not in template_dict and 'arrival_time' in template_dict:
        template_dict['tt_residual'] = tt_residual(ev, sta, template_dict['arrival_time'], phase=phase)

    # TODO: implement multiple bands/chans
    assert (len(list(sg.site_bands[site])) == 1)
    band = list(sg.site_bands[site])[0]

    if 'amp_transfer' not in template_dict and "coda_height" in template_dict:
        template_dict['amp_transfer'] = amp_transfer(ev, band, phase, template_dict['coda_height'])
    # note if coda height is not specified, we'll ignore amp params
    # entirely: this is used by the create-new-template proposer
    # (in phase_template_proposal_logp etc)

    assert (len(list(sg.site_chans[site])) == 1)
    chan = list(sg.site_chans[site])[0]

    lp = 0
    for (param, val) in template_dict.items():
        if param in ('arrival_time', 'coda_height'): continue
        if ignore_mb and param == 'amp_transfer': continue
        if param in wiggle_params:
            basisid = wg.basisid
            wiggle = True
        else:
            basisid = None
            wiggle=False
        lp_param = param_logprob(sg, site, sta, ev, phase, chan, band, param, val, basisid=basisid, wiggle=wiggle)
        if verbose:
            print "%s lp %s=%.2f is %.2f" % (sta, param, val, lp_param)
        lp += lp_param

    return lp


def template_association_logodds(sg, sta, tmid, eid, phase, ignore_mb=False):

    tmnodes = sg.uatemplates[tmid]
    param_values = dict([(k, n.get_value()) for (k,n) in tmnodes.items()])

    lp_unass = unass_template_logprob(sg, sta, param_values, ignore_mb=ignore_mb)
    lp_ev = ev_phase_template_logprob(sg, sta, eid, phase, param_values, ignore_mb=ignore_mb)


    logodds = lp_ev - lp_unass
    #print "%s %d logodds %f" % (sta, tmid, logodds)
    return logodds


def template_association_distribution(sg, sta, eid, phase, ignore_mb=False):
    """
    Returns a counter with normalized probabilities for associating
    any existing unassociated templates at station sta with a given
    phase of event eid. Probability of no association is given by c[None].

    """
    s = Sigvisa()
    site = s.get_array_site(sta)

    assert (len(list(sg.site_bands[site])) == 1)
    band = list(sg.site_bands[site])[0]
    assert (len(list(sg.site_chans[site])) == 1)
    chan = list(sg.site_chans[site])[0]

    c = Counter()
    for tmid in sg.uatemplate_ids[(sta,chan,band)]:
        c[tmid] += np.exp(template_association_logodds(sg, sta, tmid, eid,
                                                       phase, ignore_mb=ignore_mb))

    # if there are no unassociated templates, there's nothing to sample.
    n_u = len(sg.uatemplate_ids[(sta, chan, band)])
    if n_u == 0:
        c[None] = 1.0
        return c

    c[None] = np.exp(sg.ntemplates_sta_log_p(sta, n=n_u) - sg.ntemplates_sta_log_p(sta, n=n_u-1))

    c.normalize()

    # smooth probabilities slightly, so we don't get proposals that
    # are impossible to reverse
    nkeys = len(c.keys())
    for k in c.keys():
        c[k] += 1e-4/nkeys
    c.normalize()

    return c

def sample_template_to_associate(sg, sta, eid, phase, ignore_mb=False):
    """
    Propose associating an unassociate template at sta with the
    (eid,phase) arrival, with probability proportional to the odds
    ratio p_{E,P}(T)/p_U(T). Alternately propose creating a new
    template, with probability proportional to p(N_U = n_U)/p(N_U =
    n_U - 1).

    Return:
    tmid: the unassociated template id proposed for
          association. (value of None indicates proposing a creation
          move)
    assoc_logprob: log probability of the proposal

    """


    c = template_association_distribution(sg, sta, eid, phase, ignore_mb=ignore_mb)
    tmid = c.sample()
    assoc_logprob = np.log(c[tmid])

    return tmid, assoc_logprob

def associate_template(sg, sta, tmid, eid, phase, create_phase_arrival=False, node_lps=None):
    """

    Transform the graph to associate the template tmid with the arrival of eid/phase at sta.

    """

    tmnodes = sg.uatemplates[tmid]

    s = Sigvisa()
    site = s.get_array_site(sta)

    assert (len(list(sg.site_bands[site])) == 1)
    band = list(sg.site_bands[site])[0]
    assert (len(list(sg.site_chans[site])) == 1)
    chan = list(sg.site_chans[site])[0]
    values = dict([(k, n.get_value()) for (k, n) in tmnodes.items()])
    phase_created = False
    if create_phase_arrival and phase not in sg.ev_arriving_phases(eid, sta=sta):
        tg = sg.template_generator(phase)
        wg = sg.wiggle_generator(phase, sg.base_srate)
        sg.add_event_site_phase(tg, wg, site, phase, sg.evnodes[eid])
        phase_created=True

    if node_lps is not None:
        if phase_created:
            node_lps.register_new_phase_pre(sg, site, phase, eid)
        else:
            node_lps.register_phase_changed_oldvals(sg, site, phase, eid, wn_invariant=True)

    # if a newly birthed event, it already has a phase arrival that just needs to be set
    sg.set_template(eid, sta, phase, band, chan, values)

    if node_lps is not None:
        if phase_created:
            node_lps.register_new_phase_post(sg, site, phase, eid)
        else:
            node_lps.register_phase_changed_newvals(sg, site, phase, eid, wn_invariant=True)
        node_lps.register_remove_uatemplate(sg, tmid, wn_invariant=True)
    sg.destroy_unassociated_template(tmnodes, nosort=True)
    return

def unassociate_template(sg, sta, eid, phase, tmid=None, remove_event_phase=False, node_lps=None):

    s = Sigvisa()
    site = s.get_array_site(sta)

    assert (len(list(sg.site_bands[site])) == 1)
    band = list(sg.site_bands[site])[0]
    assert (len(list(sg.site_chans[site])) == 1)
    chan = list(sg.site_chans[site])[0]
    ev_tmvals = sg.get_arrival_vals(eid, sta, phase, band, chan)

    wave_node = sg.station_waves[sta][0]
    atime = ev_tmvals['arrival_time']
    tmnodes = sg.create_unassociated_template(wave_node, atime, wiggles=True, nosort=True,
                                           tmid=tmid, initial_vals=ev_tmvals)
    tmid = tmnodes.values()[0].tmid
    if node_lps is not None:
        node_lps.register_new_uatemplate(sg, tmid)

    if remove_event_phase:
        # if we're just unassociating this phase (not deleting the
        # whole event), we need to delete the event phase arrival.
        if node_lps is not None:
            node_lps.register_phase_removed_pre(sg, site, phase, eid, wn_invariant=True)
        sg.delete_event_phase(eid, sta, phase)

    return tmid

def deassociation_logprob(sg, sta, eid, phase, deletion_prob=False, min_logprob=-6):

    # return prob of deassociating (or of deleting, if deletion_prob=True).

    s = Sigvisa()
    site = s.get_array_site(sta)


    assert (len(list(sg.site_bands[site])) == 1)
    band = list(sg.site_bands[site])[0]
    assert (len(list(sg.site_chans[site])) == 1)
    chan = list(sg.site_chans[site])[0]
    ev_tmvals = sg.get_arrival_vals(eid, sta, phase, band, chan)

    unass_lp = unass_template_logprob(sg, sta, ev_tmvals)

    n_u = len(sg.uatemplate_ids[(sta,chan,band)])
    ntemplates_ratio_log = sg.ntemplates_sta_log_p(sta, n=n_u+1) - sg.ntemplates_sta_log_p(sta, n=n_u)


    deassociation_ratio_log = unass_lp + ntemplates_ratio_log

    wave_node = sg.station_waves[sta][0]
    signal_lp_with_template = wave_node.log_p()
    arrivals = copy.copy(wave_node.arrivals())
    arrivals.remove((eid, phase))
    signal_lp_without_template = wave_node.log_p(arrivals=arrivals)
    deletion_ratio_log = signal_lp_without_template - signal_lp_with_template

    log_normalizer = np.logaddexp(deassociation_ratio_log, deletion_ratio_log)

    # smooth the probabilities so we always give at least some
    # probability to each option (needed in order for reverse proposal
    # probabilities to be reasonable)
    adj = min_logprob + log_normalizer
    deassociation_ratio_log = np.logaddexp(deassociation_ratio_log, adj)
    deletion_ratio_log = np.logaddexp(deletion_ratio_log, adj)
    log_normalizer = np.logaddexp(log_normalizer, np.log(2) + adj)

    if deletion_prob:
        return deletion_ratio_log - log_normalizer
    else:
        return deassociation_ratio_log - log_normalizer

def sample_deassociation_proposal(sg, sta, eid, phase):
    lp = deassociation_logprob(sg, sta, eid, phase)
    u = np.random.rand()
    deassociate = u < np.exp(lp)
    deassociate_lp = lp if deassociate else np.log(1-np.exp(lp))
    return deassociate, deassociate_lp

def smart_peak_time_proposal(sg, wn, tmvals, eid, phase, pred_atime, fix_result=None):
    # instead of sampling arrival time from the prior, sample
    # from the product of the prior with unexplained signal mass
    ptime = np.exp(tmvals['peak_offset'])

    pred_peak_time = pred_atime + np.exp(tmvals['peak_offset'])

    arrivals = wn.arrivals()
    other_arrivals = [a for a in arrivals if a != (eid, phase)]
    signal_diff_pos = get_signal_diff_positive_part(wn, other_arrivals)
    t = np.linspace(wn.st, wn.et, wn.npts)

    # consider using a vague travel-time prior to
    # acknowldege the possibility that the event is not currently
    # in the correct location
    #tt_spread = np.random.choice((2.0, 10.0, 30.0, 80.0))
    tt_spread = 4.0

    peak_prior = np.exp(-np.abs(t - pred_atime)/tt_spread)
    peak_cdf = merge_distribution(signal_diff_pos, peak_prior)


    if not fix_result:
        peak_time, peak_lp = sample_peak_time_from_signal(peak_cdf, wn.st, wn.srate, return_lp=True)
        proposed_atime = peak_time - np.exp(tmvals['peak_offset'])
        proposed_tt_residual = proposed_atime - pred_atime
        tmvals["tt_residual"] = proposed_tt_residual
        tmvals["arrival_time"] = proposed_atime
        return peak_lp
    else:
        peak_time = tmvals["arrival_time"] + np.exp(tmvals["peak_offset"])
        peak_lp = peak_log_p(peak_cdf, wn.st, wn.srate, peak_time)
        return peak_lp

def propose_phase_template(sg, sta, eid, phase, tmvals=None, smart_peak_time=True, fix_result=False, ev=None):
    # sample a set of params for a phase template from an appropriate distribution (as described above).
    # return as an array.

    s = Sigvisa()
    site = s.get_array_site(sta)

    assert (len(list(sg.site_bands[site])) == 1)
    band = list(sg.site_bands[site])[0]
    assert (len(list(sg.site_chans[site])) == 1)
    chan = list(sg.site_chans[site])[0]

    # we assume that add_event already sampled all the params parent-conditionally
    if tmvals is None:
        tmvals = sg.get_template_vals(eid, sta, phase, band, chan)

    if ev is None:
        ev = sg.get_event(eid)
    pred_atime = ev.time + tt_predict(ev, sta, phase)
    wn = sg.get_wave_node_by_atime(sta, band, chan, pred_atime, allow_out_of_bounds=True)
    lp = 0
    if smart_peak_time:
        peak_lp = smart_peak_time_proposal(sg, wn, tmvals, eid, phase, pred_atime, fix_result=fix_result)
        lp += peak_lp
        proposed_tt_residual = tmvals["tt_residual"]
        proposed_atime = tmvals["arrival_time"]

    amp_dist = get_signal_based_amplitude_distribution(sg, wn, tmvals, exclude_arr=(eid, phase))

    if 'amp_transfer' in tmvals:
        del tmvals['amp_transfer']

    if smart_peak_time:
        del tmvals["tt_residual"]
        del tmvals["arrival_time"]

    if amp_dist is not None:

        if fix_result:
            amplitude = tmvals["coda_height"]
        else:
            amplitude = amp_dist.sample()
        del tmvals['coda_height']

        # compute log-prob of non-amplitude parameters
        param_lp = ev_phase_template_logprob(sg, sta, eid, phase, tmvals)
        lp += param_lp

        tmvals['coda_height'] = amplitude
        lp += amp_dist.log_p(amplitude)




    else:
        lp += ev_phase_template_logprob(sg, sta, eid, phase, tmvals)

    if smart_peak_time:
        tmvals["tt_residual"] = proposed_tt_residual
        tmvals["arrival_time"] = proposed_atime

    if np.isnan(np.array(tmvals.values(), dtype=float)).any():
        raise ValueError()

    # wiggles!
    wg = sg.wiggle_generator(phase=phase, srate=wn.srate)
    wnodes = sg.get_wiggle_nodes(eid, sta, phase, band, chan)
    if fix_result:
        wiggle_proposal_lp = wiggle_proposal_lprob_from_signal(eid, phase, wn, wg, wvals=tmvals)
    else:
        wiggle_proposal_lp = propose_wiggles_from_signal(eid, phase, wn, wg, wnodes)
        wvals = sg.get_wiggle_vals(eid, sta, phase, band, chan)
        tmvals.update(wvals)
    lp += wiggle_proposal_lp

    if fix_result:
        return lp
    else:
        return tmvals, lp

#########################################################################################

def death_proposal_log_ratio(sg, eid):

    lp_unass = 0
    lp_ev = 0

    ev = sg.get_event(eid)
    eid = ev.eid

    for (site, elements) in sg.site_elements.items():
        assert (len(list(sg.site_bands[site])) == 1)
        assert (len(list(sg.site_chans[site])) == 1)
        for sta in elements:
            for phase in sg.ev_arriving_phases(eid, sta):
                for chan in sg.site_chans[site]:
                    for band in sg.site_bands[site]:
                        tmvals = sg.get_arrival_vals(eid, sta, phase, band, chan)

                        lp_unass_tmpl = unass_template_logprob(sg, sta, tmvals)
                        lp_ev_tmpl = ev_phase_template_logprob(sg, sta, eid, phase, tmvals)
                        lp_unass += lp_unass_tmpl
                        lp_ev += lp_ev_tmpl

    return lp_unass - lp_ev

def death_proposal_distribution(sg):
    c = Counter()
    for eid in sg.evnodes.keys():
        c[eid] = death_proposal_log_ratio(sg, eid)

    c.normalize_from_logs()

    # with probability ~.1, just sample an event uniformly.
    # this way all events have some possibility to die.
    for k in c.keys():
        c[k] += .1/len(c)
    c.normalize()

    return c

def sample_death_proposal(sg):
    c = death_proposal_distribution(sg)
    if len(c) == 0:
        return None, 1.0
    eid = c.sample()
    return eid, np.log(c[eid])

def death_proposal_logprob(sg, eid):
    c = death_proposal_distribution(sg)
    if len(c) == 0:
        return 1.0
    return np.log(c[eid])


def ev_death_helper(sg, eid, associate_using_mb=True):

    ev = sg.get_event(eid)

    next_uatemplateid = sg.next_uatemplateid

    move_logprob = 0
    reverse_logprob = 0

    forward_fns = []
    inverse_fns = []
    inverse_fns.append(lambda : sg.add_event(ev, eid=eid))

    tmids = []
    tmid_i = 0

    deassociations = []
    # loop over phase arrivals at each station and propose either
    # associating an existing unass. template with the new event, or
    # creating a new template.
    # don't modify the graph, but generate a list of functions
    # to execute the forward and reverse moves
    for elements in sg.site_elements.values():
        for sta in elements:

            s = Sigvisa()
            site = s.get_array_site(sta)
            assert (len(list(sg.site_bands[site])) == 1)
            band = list(sg.site_bands[site])[0]
            assert (len(list(sg.site_chans[site])) == 1)
            chan = list(sg.site_chans[site])[0]

            for phase in sg.ev_arriving_phases(eid, sta):
                deassociate, deassociate_logprob = sample_deassociation_proposal(sg, sta, eid, phase)
                deassociations.append((sta, phase, deassociate, tmid_i))
                if deassociate:
                    # deassociation will produce a new uatemplated
                    # with incrementing tmid. We keep track of this
                    # tmid (kind of a hack) to ensure that we
                    # reassociate the same template if the move gets
                    # rejected.
                    forward_fns.append(lambda sta=sta,phase=phase: tmids.append(unassociate_template(sg, sta, eid, phase)))
                    inverse_fns.append(lambda sta=sta,phase=phase,tmid_i=tmid_i: associate_template(sg, sta, tmids[tmid_i], eid, phase))
                    tmid_i += 1
                    print "proposing to deassociate at %s (lp %.1f)" % (sta, deassociate_logprob)

                else:
                    template_param_array = sg.get_arrival_vals(eid, sta, phase, band, chan)
                    inverse_fns.append(lambda sta=sta,phase=phase,band=band,chan=chan,template_param_array=template_param_array : sg.set_template(eid,sta, phase, band, chan, template_param_array))
                    tmp = propose_phase_template(sg, sta, eid, phase, template_param_array, fix_result=True)
                    reverse_logprob += tmp
                    print "proposing to delete at %s (lp %f, reverse %f)"% (sta, deassociate_logprob, tmp)

                move_logprob += deassociate_logprob

    # order of operations:
    # first, deassociate the templates we need to deassociate
    # second, calculate probabilities of re-associating them to the event (while it's still around)
    # finally, kill the event
    for fn in forward_fns:
        fn()

    for (sta, phase, deassociate, tmid_i) in deassociations:
        c = template_association_distribution(sg, sta, eid, phase, ignore_mb=not associate_using_mb)
        if deassociate:
            tmid = tmids[tmid_i]
            tmp = np.log(c[tmid])
            reverse_logprob += tmp
            print "reverse deassociation %s %d %s lp %f" % (sta, eid, phase, tmp)
        else:
            tmp = np.log(c[None])
            reverse_logprob += tmp
            print "reverse deletion %s %d %s lp %f" % (sta, eid, phase, tmp)



    sg.remove_event(eid)


    def revert_move():
        for fn in inverse_fns:
            fn()
        sg._topo_sort()
        sg.next_uatemplateid = next_uatemplateid

    return move_logprob, reverse_logprob, revert_move

def ev_death_helper_full(sg, eid, location_proposal):
    ev = sg.get_event(eid)
    lqb, reset_coda_heights = propose_mb(sg, eid, fix_result=ev.mb)
    sg.evnodes[eid]["mb"].set_value(4.0)
    reset_coda_heights()

    log_qforward, log_qbackward, revert_move = ev_death_helper(sg, eid, associate_using_mb=False)
    lp_loc = location_proposal(sg, fix_result=ev)
    print "death helper", lqb, log_qbackward, lp_loc

    def revert():
        revert_move()
        sg.evnodes[eid]["mb"].set_value(ev.mb)
        reset_coda_heights()

    return log_qforward, lqb + log_qbackward + lp_loc, revert

def ev_death_move_abstract(sg, location_proposal, log_to_run_dir=None):
    eid, eid_logprob = sample_death_proposal(sg)
    if eid is None:
        return False
    move_logprob = eid_logprob
    n_current_events = len(sg.evnodes)
    reverse_logprob = -np.log(n_current_events) # this accounts for the different "positions" we can birth an event into

    lp_old = sg.current_log_p()
    log_qforward, log_qbackward, revert_move = ev_death_helper_full(sg, eid, location_proposal)
    lp_new = sg.current_log_p()

    log_qforward += move_logprob
    log_qbackward += reverse_logprob

    print "death move acceptance", (lp_new + log_qbackward) - (lp_old+log_qforward), "from", lp_old, lp_new, log_qbackward, log_qforward

    return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=None, revert_move=revert_move)

def ev_death_move_hough(sg, **kwargs):
    return ev_death_move_abstract(sg, hough_location_proposal, **kwargs)

def ev_death_move_lstsqr(sg, **kwargs):
    return ev_death_move_abstract(sg, overpropose_new_locations, **kwargs)

##########################################################################################

def ev_birth_helper(sg, proposed_ev, associate_using_mb=True, eid=None):

    forward_fns = []
    inverse_fns = []
    associations = []

    # add an event, WITH all its template nodes initialized to parent-sampled values.
    # we need to replace these values before computing any signal-based probabilities.
    # luckily,
    evnodes = sg.add_event(proposed_ev, sample_templates=True, eid=eid)
    eid = evnodes['mb'].eid

    # loop over phase arrivals at each station and propose either
    # associating an existing unass. template with the new event, or
    # creating a new template.
    # don't modify the graph, but generate a list of functions
    # to execute the forward and reverse moves
    log_qforward = 0
    for site,elements in sg.site_elements.items():
        site_phases = sg.predict_phases_site(proposed_ev, site=site)
        for sta in elements:

            s = Sigvisa()
            site = s.get_array_site(sta)
            assert (len(list(sg.site_bands[site])) == 1)
            band = list(sg.site_bands[site])[0]
            assert (len(list(sg.site_chans[site])) == 1)
            chan = list(sg.site_chans[site])[0]

            for phase in site_phases:
                tmid, assoc_logprob = sample_template_to_associate(sg, sta, eid, phase, ignore_mb=not associate_using_mb)
                if tmid is not None:
                    forward_fns.append(lambda sta=sta,phase=phase,tmid=tmid: associate_template(sg, sta, tmid, eid, phase))
                    inverse_fns.append(lambda sta=sta,phase=phase,tmid=tmid: unassociate_template(sg, sta, eid, phase, tmid=tmid))
                    associations.append((sta, phase, True))
                    print "proposing to associate template %d at %s,%s with assoc lp %.1f" % (tmid, sta, phase, assoc_logprob)
                    tmpl_lp  = 0.0
                else:
                    template_param_array, tmpl_lp = propose_phase_template(sg, sta, eid, phase)

                    if np.isnan(np.array(template_param_array.values(), dtype=float)).any():
                        raise ValueError()
                    forward_fns.append(lambda sta=sta,phase=phase,band=band,chan=chan,template_param_array=template_param_array : sg.set_template(eid,sta, phase, band, chan, template_param_array))
                    #inverse_fns.append(lambda : delete_template(sg, sta, eid, phase))
                    associations.append((sta, phase, False))
                    print "proposing to birth new phase %s,%s with assoc lp %.1f tmpl lp %f" % (sta, phase, assoc_logprob, tmpl_lp)



                sta_phase_logprob = assoc_logprob + tmpl_lp
                log_qforward += sta_phase_logprob


    inverse_fns.append(lambda : sg.remove_event(eid))

    # execute all the forward moves
    for fn in forward_fns:
        fn()
    sg._topo_sort()

    # compute log probability of the reverse move.
    # we have to do this in a separate loop so that
    # we can execute all the forward moves first.
    log_qbackward = 0
    for (sta, phase, associated) in associations:
        lp = deassociation_logprob(sg, sta, eid, phase, deletion_prob=not associated)
        log_qbackward += lp
        #print "deassociation logprob %.2f for %s" % (lp, (sta, phase, associated))

    def revert_move():
        for fn in inverse_fns:
            fn()

    return log_qforward, log_qbackward, revert_move, eid, associations

def ev_birth_helper_full(sg, location_proposal, eid=None):
    # propose a new ev location
    ev, lp_loc, extra = location_proposal(sg)

    # propose its associations
    log_qforward, log_qbackward, revert_move, eid, associations = ev_birth_helper(sg, ev, associate_using_mb=False, eid=eid)

    # propose its magnitude
    lqf = propose_mb(sg, eid)

    print "birth helper", lqf, log_qforward, lp_loc

    return lp_loc + log_qforward + lqf, log_qbackward, revert_move, (extra, eid, associations)

def ev_birth_move_abstract(sg, location_proposal, revert_action=None, accept_action=None):

    n_current_events = len(sg.evnodes)
    birth_position_lp = -np.log(n_current_events+1) # we imagine there are n+1 "positions" we can birth an event into

    lp_old = sg.current_log_p()
    log_qforward, log_qbackward, revert_move, proposal_extra = ev_birth_helper_full(sg, location_proposal)
    lp_new = sg.current_log_p()

    (extra, eid, associations) = proposal_extra

    log_qforward += birth_position_lp
    log_qbackward += death_proposal_logprob(sg, eid)

    def revert():
        if revert_action is not None:
            revert_action(proposal_extra, lp_old, lp_new, log_qforward, log_qbackward)
        revert_move()

    def accept():
        if accept_action is not None:
            accept_action(proposal_extra, lp_old, lp_new, log_qforward, log_qbackward)

    print "birth move acceptance", (lp_new + log_qbackward) - (lp_old+log_qforward), "from", lp_old, lp_new, log_qbackward, log_qforward

    return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=accept, revert_move=revert)

def ev_birth_move_hough(sg, log_to_run_dir=None, **kwargs):

    def revert_action(proposal_extra, lp_old, lp_new, log_qforward, log_qbackward):
        hough_array, eid, associations = proposal_extra
        if np.random.rand() < 0.1:
            sites = sg.site_elements.keys()
            print "saving hough array picture...",
            visualize_hough_array(hough_array, sites, os.path.join(log_to_run_dir, 'last_hough.png'))
            print "done"

    def accept_action(proposal_extra, lp_old, lp_new, log_qforward, log_qbackward):
        hough_array, eid, associations = proposal_extra
        if log_to_run_dir is not None:
            log_event_birth(sg, hough_array, log_to_run_dir, eid, associations)
        else:
            raise Exception("why are we not logging?")

    return ev_birth_move_abstract(sg, location_proposal=hough_location_proposal, revert_action=revert_action, accept_action=accept_action, **kwargs)

def ev_birth_move_lstsqr(sg, log_to_run_dir=None, **kwargs):

    def log_action(proposal_extra, lp_old, lp_new, log_qforward, log_qbackward):
        refined_proposals, eid, associations = proposal_extra
        log_file = os.path.join(log_to_run_dir, "lsqr_proposals.txt")

        # proposed event should be the most recently created
        proposed_ev = sg.get_event(np.max(sg.evnodes.keys()))

        with open(log_file, 'a') as f:
            f.write("proposed ev: %s\n" % proposed_ev)
            f.write("acceptance lp %.2f (lp_old %.2f lp_new %.2f log_qforward %.2f log_qbackward %.2f)\n" % (lp_new +log_qbackward - (lp_old + log_qforward), lp_old, lp_new, log_qforward, log_qbackward))
            for abserror, z, C in refined_proposals:
                f.write("  %.2f lon %.2f lat %.2f depth %.2f time %.2f\n" % (abserror, z[0], z[1], z[2], z[3]))
            f.write("\n")

    return ev_birth_move_abstract(sg, location_proposal=overpropose_new_locations, accept_action=log_action, revert_action=log_action, **kwargs)

##############################################################

def log_event_birth(sg, hough_array, run_dir, eid, associations):

    log_dir = os.path.join(run_dir, "ev_%05d" % eid)
    mkdir_p(log_dir)

    # save post-birth signals and general state
    sg.debug_dump(dump_path=log_dir, pickle_graph=False)

    with open(os.path.join(log_dir, 'associations.txt'), 'w') as f:
        for (sta, phase, associated) in associations:
            f.write('%s %s %s\n' % (sta, phase, associated))

    # save Hough transform
    sites = sg.site_elements.keys()
    print "visualizing hough array...",
    visualize_hough_array(hough_array, sites, os.path.join(log_dir, 'hough.png'))
    print "done"
