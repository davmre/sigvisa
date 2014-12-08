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

from sigvisa.infer.run_mcmc import run_open_world_MH
from sigvisa.infer.mcmc_logger import MCMCLogger
from sigvisa.infer.event_birthdeath import ev_birth_helper,ev_death_helper

from sigvisa.infer.template_mcmc import *
from sigvisa.plotting.plot import plot_with_fit, plot_with_fit_shapes, plot_pred_atimes
from sigvisa.signals.common import Waveform
from sigvisa.utils.fileutils import clear_directory, mkdir_p

from sigvisa.experiments.one_station_templates_test import sample_template

def ev_mb_posterior_laplace(sg, eid):

    mb_node = sg.evnodes[eid]["mb"]
    amp_nodes = [n for n in sg.extended_evnodes[eid] if "amp_transfer" in n.label]
    ch_nodes = [nn for n in amp_nodes for nn in n.children if "coda_height" in nn.label]
    coda_heights = [nn.get_value() for nn in ch_nodes]
    reset_coda_heights = lambda : [nn.set_value(ch) for (ch, nn) in zip(coda_heights, ch_nodes)]

    orig_mb = mb_node.get_value()

    def amp_transfer_nlp(x):
        mb = x[0]
        mb_node.set_value(mb)
        reset_coda_heights()
        return -np.sum([n.log_p() for n in amp_nodes])


    x0 = [4.0,]
    r = scipy.optimize.minimize(amp_transfer_nlp, x0)
    mb_node.set_value(orig_mb)
    reset_coda_heights()

    #prec = nd.Hessian(amp_transfer_nlp).hessian(r.x)
    #var = 1.0/prec[0,0]
    return r.x[0], r.hess_inv[0,0], reset_coda_heights


def ev_mb_posterior_laplace_functional(sg, ev, targets, amps):

    s = Sigvisa()
    models = []
    for (sta, phase, chan, band) in targets:
        site = s.get_array_site(sta)
        ampt_model_type = sg._tm_type(param="amp_transfer", site=site, wiggle_param=False)
        modelid = get_param_model_id(runid=sg.runid, sta=sta,
                                     phase=phase, model_type=ampt_model_type,
                                     param="amp_transfer", template_shape=sg.template_shape,
                                     chan=chan, band=band)
        model = load_modelid(modelid)
        models.append(model)

    def amp_transfer_nlp(x):
        mb = x[0]
        lp = 0
        for amp, model, (sta, phase, chan, band) in zip(amps, models, targets):
            source_amp = brune.source_logamp(mb, phase=phase, band=band)
            lp += model.log_p(amp - source_amp, ev)
        return -lp

    x0 = [4.0,]
    r = scipy.optimize.minimize(amp_transfer_nlp, x0)
    prec = nd.Hessian(amp_transfer_nlp).hessian(r.x)

    var = 1.0/prec[0,0]

    return r.x[0], var


def propose_new_locations(sg, old_evs, n_stations=4, p_uniform_component=1e-6):
    # pick n stations at random

    # iterate over all choices of one uatemplate per station
    # for each choice, compute the lsqr location and total tt residual
    # this gives us 8 (say) gaussian distributions. we either take the
    # distribution with the lowest tt residual, or we use a mixture of
    # them according to their tt residual probabilities. the latter might
    # be more awkward, but we've already got all the gaussians so it's not
    # obviously infeasible

    n_events = len(old_evs)

    have_uatemplates = [k for (k, tmids) in sg.uatemplate_ids.items() if len(tmids)>=n_events]
    station_idxs = np.random.choice(len(have_uatemplates), size=n_stations, replace=False)
    stations = [have_uatemplates[i] for i in station_idxs]
    tmids = [sg.uatemplate_ids[k] for k in stations]

    c = Counter()
    dists = []
    tmid_sets = []
    print "proposing associations at stations", [sta for (sta, chan, band) in stations]
    for i, choices in enumerate(itertools.product(*tmids)):
        targets = []
        stas = []
        atimes = []
        for k, (sta, chan, band) in enumerate(stations):
            targets.append((sta, "P"))
            stas.append(sta)
            tmid = choices[k]
            atimes.append(sg.get_template_vals(-tmid, sta, "UA", band, chan)['arrival_time'])
        z, C, sqerror, abserror = ev_lstsqr_dist(sg, stas=stas, atimes=np.array(atimes),targets=targets, return_full=True)
        dists.append((z,C))
        tmid_sets.append(choices)
        c[i] = -abserror
        print "choice", choices, "gives z", z, "abserror", abserror


    old_vals = [np.array((old_ev.lon, old_ev.lat, old_ev.depth, old_ev.time)) for old_ev in old_evs]


    uniform_scaling = np.array((360, 180, 700, sg.end_time - sg.event_start_time), dtype=float)
    new_evs = []
    proposal_lps = []
    reverse_lps = []
    associations = []
    for i, old_ev in enumerate(old_evs):
        max_dist = c.argMax()
        z, C = dists[max_dist]

        print "associating", tmid_sets[max_dist]
        associations.append(tmid_sets[max_dist])
        if i < n_events-1:

            for j, tmid in enumerate(tmid_sets[max_dist]):
                keys = c.keys()
                for k in keys:
                    if tmid_sets[k][j]==tmid:
                        print "deleting", tmid_sets[k]
                        del c[k]

        rv = scipy.stats.multivariate_normal(z, C)

        uniform_proposal=np.random.rand() < p_uniform_component
        if uniform_proposal:
            uniform_shift = np.array((-180, -90, 0, sg.event_start_time), dtype=float)
            proposed_vals = np.random.rand(4) * uniform_scaling + uniform_shift
        else:
            proposed_vals = rv.rvs(1)

        lon, lat, depth, time = proposed_vals

        logpdf = lambda v : np.logaddexp(np.log(1-p_uniform_component) + rv.logpdf(v), np.log(p_uniform_component) + np.sum(-np.log(uniform_scaling)))

        proposal_lp = logpdf(proposed_vals)
        reverse_lp = logpdf(old_vals[i])
        proposal_lps.append(proposal_lp)
        reverse_lps.append(reverse_lp)

        lon, lat, depth, time = proposed_vals

        # this breaks Gaussianity, technically we should be using a
        # circular (von Mises?) distribution. but hopefully it doesn't
        # matter.
        lon, lat = wrap_lonlat(lon, lat)

        # this definitely breaks Gaussianity, we should be explicitly truncating the distribution
        if depth > 699:
            depth = 699
        elif depth < 0:
            depth = 0

        new_ev = Event(lon=lon, lat=lat, depth=depth, time=time, mb=4.0)
        new_evs.append(new_ev)

    return new_evs, proposal_lps, reverse_lps, associations, stations

def propose_new_magnitudes(sg, new_evs, associations, stations):
    def get_amp(tmid, sta, chan, band):
        n = get_parent_value(eid=-tmid, sta=sta, phase="UA", param_name='coda_height', chan=chan, band=band,
                             parent_values=sg.nodes_by_key, return_key=False)
        return n.get_value()

    mb_forward_lps = []
    mb_backward_lps = []
    targets = [(sta, "P", chan, band) for (sta, chan, band) in stations]
    for i, ev in enumerate(new_evs):
        amps = []
        for j, (sta, chan, band) in enumerate(stations):
            tmid = associations[i][j]
            amps.append(get_amp(tmid, sta, chan, band))

        z, v = ev_mb_posterior_laplace(sg, ev, targets=targets, amps=amps)
        rv = scipy.stats.norm(loc=z, scale=np.sqrt(v))
        mb = rv.rvs(1)[0]
        log_qforward = rv.logpdf(mb)
        log_qbackward = rv.logpdf(ev.mb)
        mb_forward_lps.append(log_qforward)
        mb_backward_lps.append(log_qbackward)

        print "ev", i, "magnitude dist", z, v, "proposing", mb, ev.mb, log_qforward, log_qbackward

        ev.mb = mb
    return mb_forward_lps, mb_backward_lps


def propose_new_events(sg, old_evs, propose_mb=True, **kwargs):
    new_evs, proposal_lps, reverse_lps, associations, stations = propose_new_locations(sg, old_evs, **kwargs)
    if propose_mb:
        mb_forward_lps, mb_backward_lps = propose_new_magnitudes(sg, new_evs, associations, stations)
        proposal_lps.extend(mb_forward_lps)
        reverse_lps.extend(mb_backward_lps)
    return new_evs, proposal_lps, reverse_lps

def sample_events_to_swap(sg, fix_result=None):
    # fix_results is a 2-tuple of eids
    # if set, return lp
    # else, return eid1, eid2, lp

    eids = sorted(sg.evnodes.keys())
    n = len(eids)

    lp = np.log(2.0/(n * (n-1)))

    if fix_result is None:
        eid1 = np.random.choice(eids)
        eids.remove(eid1)
        eid2 = np.random.choice(eids)
        if eid1 > eid2:
            eid1, eid2 = eid2, eid1
        return eid1, eid2, lp
    else:
        eid1, eid2 = fix_result
        if eid1 in eids and eid2 in eids:
            return lp
        else:
            return -np.inf

def swap_events_move(sg):
    eid1, eid2, lp_swap = sample_events_to_swap(sg)
    lp_new, lp_old, log_qforward, log_qbackward, revert_move = rebirth_events_helper(sg, (eid1, eid2))
    log_qforward += lp_swap
    log_qbackward += sample_events_to_swap(sg, fix_result=(eid1, eid2))

    return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, accept_move=None, revert_move=revert_move)

def propose_mb(sg, eid, fix_result=None):
    z, v, reset_coda_heights = ev_mb_posterior_laplace(sg, eid)
    rv = scipy.stats.norm(z, scale=np.sqrt(v))

    if fix_result is not None:
        print "reverse dist", z, v, "old mb", fix_result, "lp", rv.logpdf(fix_result)
        return rv.logpdf(fix_result)
    else:
        new_mb=rv.rvs(1)[0]
        print "eid", eid, "proposing mb", new_mb, "from dist", z, np.sqrt(v)
        sg.evnodes[eid]["mb"].set_value(new_mb)
        reset_coda_heights()

        return rv.logpdf(new_mb)

def rebirth_events_helper(sg, eids):
    """
    Given a list of events, propose killing them all and rebirthing
    new events from the resulting uatemplates.
    """

    old_evs = [sg.get_event(eid) for eid in eids]

    lp_old = sg.current_log_p()
    log_qforward = 0
    log_qbackward = 0
    revert_moves = []
    for i, eid in enumerate(eids):
        lqb = propose_mb(sg, eid, old_evs[i].mb)
        log_qbackward += lqb

        lqf, lqb, r = ev_death_helper(sg, eid, associate_using_mb=False)
        log_qforward += lqf
        log_qbackward += lqb
        revert_moves.append(r)

    new_evs, event_lps, reverse_event_lps = propose_new_events(sg, old_evs, n_stations=5, propose_mb=False)
    for ev in new_evs:
        print "proposing new ev", ev
    log_qforward += np.sum(event_lps)
    log_qbackward += np.sum(reverse_event_lps)

    for i, eid in enumerate(eids):
        lqf, lqb, r =ev_birth_helper(sg, new_evs[i], eid=eid, associate_using_mb=False)
        log_qforward += lqf
        log_qbackward += lqb
        revert_moves.append(r)

        lqf = propose_mb(sg, eid)
        log_qforward += lqf

    lp_new = sg.current_log_p()

    revert_moves.reverse()
    def revert_move():
        for r in revert_moves:
            r()

    return lp_new, lp_old, log_qforward, log_qbackward, revert_move
