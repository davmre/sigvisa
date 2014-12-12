import numpy as np
import scipy.stats
import scipy
import sys
import os
import traceback
import pickle
import copy
import time

from sigvisa import Sigvisa

from sigvisa.models.ttime import tt_predict, tt_predict_grad
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.source.event import Event
from sigvisa.utils.geog import wrap_lonlat


import numdifftools as nd



#################################################################################


def ev_lstsqr_dist(sg, eid=None, stas=None, residual_var=2.0, atimes=None, targets=None, return_full=False, init_z=None, n_restarts=10, no_cov=False):
    """
    we start with a bunch of arrival times. take these as gospel.
    we can do this for *all* phases if we like. or just choose a phase, or a subset of stations, or whatever.
    now we want to find the location that minimizes travel-time residuals.
    to do this, we can
    """

    def get_atime(eid, sta, phase):
        n = get_parent_value(eid=eid, sta=sta, phase=phase, param_name='arrival_time', chan=None, band=None,
                             parent_values=sg.nodes_by_key, return_key=False)
        return n.get_value()

    if stas is None:
        stas = sg.station_waves.keys()

    if targets is None:
        assert(eid is not None)
        targets = []
        for sta in sorted(stas):
            phases = sg.ev_arriving_phases(eid, sta)
            for phase in phases:
                targets.append((sta, phase))

    if atimes is None:
        assert(eid is not None)
        atimes = np.array([get_atime(eid, sta, phase) for (sta, phase) in targets])

    # assume any station with a tt residual greater than 30s is a bad association,
    # which we don't want to use.

    # so that the optimization is well conditioned, we
    # rescale lon/lat/depth so that incrementing those
    # variables has approximately the same effect as a
    # 1s increment in origin time
    # 1 degree ~= 100 km ~= 20s of traveltime
    # 1km of depth ~= 0.2s of traveltime
    scaling = np.array((20.0, 20.0, 0.2, 1.0))
    p0 = np.array((0.0, 0.0, 0.0, np.min(atimes)-500))

    # assume a tt residual stddev of 5.0
    # TODO: use the actual values learned from data
    sigma = np.ones((len(targets)+1,)) * residual_var

    def tt_residuals_jac(x):
        lon, lat, depth, origin_time = x / scaling + p0

        #with open('evals.txt', 'a') as f:
        #    f.write(str(x)+"\n")
        pred_tts = np.zeros(len(targets))
        jac = np.zeros((4, len(targets)+1))

        if (~np.isfinite(x)).any():
            return np.ones(x.shape) * np.nan, np.ones(jac.shape) * np.nan


        for i, (sta, phase) in enumerate(targets):
            try:
                tt, grad = tt_predict_grad(lon, lat, depth, origin_time, sta, phase=phase)
                pred_tts[i] = tt
                jac[:, i] = -grad / scaling
            except ValueError as e:
                pred_tts[i] = 1e10
        residuals = atimes-(origin_time + pred_tts)

        # give a smooth warning to the optimizer that we're
        # about to exceed the depth limit
        if depth < 700:
            depth_penalty = np.exp(1.0/(700-depth)) - 1
            jac[2, len(targets)] = ( (depth_penalty + 1) / (depth-700)**2 ) / scaling[2]
        else:
            depth_penalty = 1e40
        rr = np.concatenate([residuals, (   depth_penalty,  )])
        return rr / sigma, jac/sigma

    class TTResidualCache(object):
        def __init__(self):
            self.x = np.ones((4,))

        def ttr(self, x):
            if not (x == self.x).all():
                self.x[:] = x
                self.cached_ttr, self.cached_jac = tt_residuals_jac(x)
            return self.cached_ttr

        def jac(self, x):
            if not (x == self.x).all():
                self.x[:] = x
                self.cached_ttr, self.cached_jac = tt_residuals_jac(x)
            return self.cached_jac

    ttrcache = TTResidualCache()

    def sqerror(x):
        return .5 * np.sum(ttrcache.ttr(x)**2)

    def sqerror_grad(x):
        ttr, jac = tt_residuals_jac(x)
        return np.sum(ttr * jac, axis=1)

    def unscaled_sqerror(x):
        xs = (x - p0)*scaling
        return sqerror(xs)

    def jac(x):
        return approx_gradient(sqerror, x0, 1e-4)

    def abserror(x):
        return np.sum(np.abs(ttrcache.ttr(x)))

    def abserror_grad(x):
        ttr, jac = tt_residuals_jac(x)
        signs = ((ttr > 0) - .5) * 2
        return np.sum(jac * signs, axis=1)

    x0 = -p0 * scaling
    x0[3] = -p0[3] + np.min(atimes) - 100

    if init_z is not None:
        z0 = init_z
        x0 = (z0-p0) * scaling
        x1, cov_x, info_dict, mesg, ier = scipy.optimize.leastsq(ttrcache.ttr, x0, Dfun=ttrcache.jac, full_output=True, col_deriv=True)
        best_x = x1
        best_cov= cov_x
    else:
        # random restarts
        min_atime = np.min(atimes)
        min_sqr = np.inf
        best_x = None
        best_cov = None
        for i in range(n_restarts):
            init_depth = 0.0 if np.random.rand() < 0.8 else 400.0
            init_time = min_atime - np.random.rand() * 1000
            init_lon = np.random.rand()*360.0 - 180.0
            init_lat = np.random.rand()*180.0 - 90.0
            z0 = np.array((init_lon, init_lat, init_depth, init_time))
            x0 = (z0 - p0) * scaling
            x1, cov_x, info_dict, mesg, ier = scipy.optimize.leastsq(ttrcache.ttr, x0, Dfun=ttrcache.jac, full_output=True, col_deriv=True)
            sqr = sqerror(x1)
            if sqr < min_sqr:
                best_x = x1
                best_cov = cov_x
                min_sqr = sqr

    x1 = best_x
    cov_x = best_cov

    scaling_mat = np.outer(scaling, scaling)

    if cov_x is None and not no_cov:
        try:
            r = scipy.optimize.minimize(sqerror, x1, jac=sqerror_grad)
            x2 = r.x
            H = nd.Hessian(sqerror).hessian(x2)
            chol = np.linalg.cholesky(H)
            cx = np.linalg.inv(H)
            cov_x = cx
            x1 = x2
        except np.linalg.LinAlgError as e:
            print "linalg error", e
            cov_x = np.diag(scaling**2)

    z = x1 /scaling + p0

    if cov_x is not None:
        cov_x /= scaling_mat

    if return_full:
        return z, cov_x, sqerror(x1), abserror(x1)
    else:
        return z, cov_x


#####################################################################################


def propose_event_from_gaussian(sg, z, C, p_uniform_component, old_ev=None):
    # given a Gaussian mean and covariance on (lon, lat, depth, time),
    # sample an event to propose. since Gaussian have thin tails, also
    # include a tiny uniform component so that the probability of a reverse
    # proposal is never worse than some reasonable bound.

    rv = scipy.stats.multivariate_normal(z, C)

    uniform_scaling = np.array((360, 180, 700, sg.end_time - sg.event_start_time), dtype=float)
    logpdf = lambda v : np.logaddexp(np.log(1-p_uniform_component) + rv.logpdf(v), np.log(p_uniform_component) + np.sum(-np.log(uniform_scaling)))

    if old_ev is not None:
        old_vals = np.array((old_ev.lon, old_ev.lat, old_ev.depth, old_ev.time))
        return logpdf(old_vals)
    else:
        uniform_proposal=np.random.rand() < p_uniform_component
        if uniform_proposal:
            uniform_shift = np.array((-180, -90, 0, sg.event_start_time), dtype=float)
            proposed_vals = np.random.rand(4) * uniform_scaling + uniform_shift
        else:
            proposed_vals = rv.rvs(1)

        lon, lat, depth, time = proposed_vals
        proposal_lp = logpdf(proposed_vals)
        # this breaks Gaussianity, technically we should be using a
        # circular (von Mises?) distribution. but hopefully it doesn't
        # matter too much.
        lon, lat = wrap_lonlat(lon, lat)

        # this definitely breaks Gaussianity, we should be explicitly truncating the distribution.
        # depending on posterior variance this might actually matter, a substantial portion of the
        # Gaussian mass could be outside of these bounds...
        if depth > 699:
            depth = 699
        elif depth < 0:
            depth = 0

        new_ev = Event(lon=lon, lat=lat, depth=depth, time=time, mb=4.0)
        return new_ev, proposal_lp


def sample_stations(sg, n_stations, min_uatemplates):
    # sample a set of stations, all of which have at least
    # min_uatemplates unassociated templates present.  return a list
    # of (sta, chan, band) tuples, and a list of lists of tmids.

    # choose a subset of stations to propose from
    have_uatemplates = [k for (k, tmids) in sg.uatemplate_ids.items() if len(tmids)>=min_uatemplates]
    if len(have_uatemplates) < n_stations:
        station_idxs =np.arange(len(have_uatemplates))
    else:
        station_idxs = np.random.choice(len(have_uatemplates), size=n_stations, replace=False)
    stations = [have_uatemplates[i] for i in station_idxs]
    tmids = [sg.uatemplate_ids[k] for k in stations]
    return stations, tmids

def eval_associations(sg, stations, choices, **kwargs):
    # given a list of stations and a corresponding list of tmids,
    # solve for the event parameters implied by this particular association.
    targets = []
    stas = []
    atimes = []
    for k, (sta, chan, band) in enumerate(stations):
        targets.append((sta, "P"))
        tmid = choices[k]
        atimes.append(sg.get_template_vals(-tmid, sta, "UA", band, chan)['arrival_time'])
    z, C, sqerror, abserror = ev_lstsqr_dist(sg, stas=stas, atimes=np.array(atimes),
                                             targets=targets, return_full=True, **kwargs)
    return z, C, sqerror, abserror

def bruteforce_locations(sg, stations, tmids):
    # given a list of stations, and a corresponding list containing lists of unassociated tmids at each station,
    # propose an event corresponding to each possible choice of tmids to associate
    hypotheses = []
    for i, choices in enumerate(itertools.product(*tmids)):
        z,C, sqerror, abserror = eval_associations(sg, stations, choices)
        hypotheses.append((abserror, z, C, choices))
    return hypotheses

def tt_residual(sg, z_loc, sta, band, chan, tmid):
    atime = sg.get_template_vals(-tmid, sta, "UA", band, chan)['arrival_time']
    ev = Event(lon=z_loc[0], lat=z_loc[1], depth=z_loc[2], time=z_loc[3])
    try:
        pred_at = tt_predict(ev, sta, 'P') + z_loc[3]
        residual = np.abs(atime - pred_at)
    except ValueError:
        residual = np.inf

    return residual



def overpropose_new_locations(sg, n_locations=40, n_eval_pairs=5, n_refine=5, n_stations=10, min_uatemplates=1, p_uniform_component=1e-4, fix_result=None, proposal_dist_seed=None):
    #

    def sample_associated_triple(sg, valid_stas=None):
        # sample three stations
        station_idxs = np.random.choice(len(valid_stas), size=3, replace=False)
        stations = [have_uatemplates[i] for i in station_idxs]
        tmids = [np.random.choice(list(sg.uatemplate_ids[k])) for k in stations]

        for (sta, chan, band) in stations:
            if len(sg.station_waves[sta]) > 1:
                raise ValueError("brute-force proposal from template triples only makes sense when all templates are from about the same time period, but station %s has multiple wave nodes! so the proposal will probably not be very good; please fix to be smarter about only sampling template triplets within an appropriate time period")

        return stations, tmids

    if proposal_dist_seed is not None:
        np.random.seed(proposal_dist_seed)

    t0 = time.time()

    # sample random associations
    proposals = []
    have_uatemplates = [k for (k, tmids) in sg.uatemplate_ids.items() if len(tmids)>=min_uatemplates]
    for i in range(n_locations):
        stations, tmids = sample_associated_triple(sg, have_uatemplates)
        z, _, sqerror, abserror = eval_associations(sg, stations, tmids, no_cov=True, n_restarts=2)

        # with three stations, a nonzero-error solution means the optimization didn't converge
        if abserror < 1:
            proposals.append((z, stations, tmids))
    t1 = time.time()

    # choose several random evaluation sets
    eval_pairs = []
    for i in range(n_eval_pairs):
        station_idxs = np.random.choice(len(have_uatemplates), size=2, replace=False)
        eval_pairs.append([have_uatemplates[i] for i in station_idxs])
    t2 = time.time()

    # evaluate each proposed location by computing the tt residuals
    # under a greedy association at each of the evaluation pairs of
    # stations.  the score of the proposal is the *minimum* over the
    # evaluation pairs: the idea is that maybe not all stations have
    # detections for this event, but as long as there's one pair of
    # stations that both detect the event, we should get a low
    # residual.
    scored_proposals = []
    for (z, stations, tmids) in proposals:
        best_residual = np.inf
        best_added_stas = None
        best_added_tmids = None
        for a1, a2 in eval_pairs:
            if a1 in stations or a2 in stations: continue
            sta1, chan1, band1 = a1
            sta2, chan2, band2 = a2
            tmids1 = list(sg.uatemplate_ids[a1])
            tmids2 = list(sg.uatemplate_ids[a2])

            residuals1 = [tt_residual(sg, z, sta1, band1, chan1, tmid) for tmid in tmids1]
            residuals2 = [tt_residual(sg, z, sta2, band2, chan2, tmid) for tmid in tmids2]
            residual = np.min(residuals1) + np.min(residuals2)
            if residual < best_residual:
                best_residual = residual
                tmid1 = tmids1[np.argmin(residuals1)]
                tmid2 = tmids2[np.argmin(residuals2)]
                best_added_stas = [a1, a2]
                best_added_tmids = [tmid1, tmid2]

        print "proposing", z, "from", [(s,t) for (s,c,b),t in zip(stations, tmids)], "score", best_residual
        if best_residual < np.inf:
            full_stations = stations + best_added_stas
            full_tmids = tmids + best_added_tmids
            scored_proposals.append((best_residual, z, full_stations, full_tmids))

    t3 = time.time()

    # refine the highest-scoring proposals using all five stations
    refined_proposals = []
    for residual, z, stations, tmids in sorted(scored_proposals, key = lambda s : s[0])[:n_refine]:
        print "refining", z, "with residual", residual
        z, C, sqerror, abserror = eval_associations(sg, stations, tmids, init_z=z)
        print "refined to", z, "with abserror", abserror
        refined_proposals.append((abserror, z, C))

    if len(refined_proposals) > 0:
        best_abs, z, C = sorted(refined_proposals)[0]
    else:
        z = np.zeros((4,))
        z[3] = sg.event_start_time
        C = np.eye(4) * 100

    t4 = time.time()

    print "times: proposals %.2f eval set choice %.2f scoring %.2f refinement %.2f" % (t1-t0, t2-t1, t3-t2, t4-t3)

    if fix_result is not None:
        lp = propose_event_from_gaussian(sg, z, C, p_uniform_component=p_uniform_component, old_ev=fix_result)
        return lp
    else:
        ev, lp = propose_event_from_gaussian(sg, z, C, p_uniform_component=p_uniform_component, old_ev=fix_result)
        return ev, lp, refined_proposals

def propose_new_location_greedy(sg, n_stations=5, min_uatemplates=1, p_uniform_component=1e-6, fix_result=None, beam_size=2, proposal_dist_seed=None):


    def update_location_for_new_station(z_loc, stas, choices, new_sta, tmids_sta):
        sta, chan, band = new_sta
        tmids_sta = list(tmids_sta)
        tt_residuals = [tt_residual(z_loc, sta, band, chan, tmid) for tmid in tmids_sta]
        assoc_tmid = tmids_sta[np.argmin(tt_residuals)]
        choices  = choices + (assoc_tmid,)
        z, C, sqerror, abserror = eval_associations(stas + [new_sta,], choices, init_z=z_loc)
        return z, C, abserror, choices

    if proposal_dist_seed is not None:
        np.random.seed(proposal_dist_seed)

    stations, tmids = sample_stations(sg, n_stations=n_stations, min_uatemplates=min_uatemplates)

    hypotheses = bruteforce_locations(sg, stations[:3], tmids[:3])

    # for each new station
    for i in range(3, len(stations)):
        new_hypotheses = []
        for abserror, z, C, choices  in hypotheses:
            z, C, abserror, choices = update_location_for_new_station(sg, z, stations[:i], choices, stations[i], tmids[i])
            new_hypotheses.append((abserror, z, C, choices))
            print "updated error", abserror, "for hypothesis", z, "with stations", [sta for (sta, chan, band) in stations[:i+1]]
        hypotheses = sorted(new_hypotheses)[:beam_size]

    abserror, z, C, choices = hypotheses[0]
    return propose_event_from_gaussian(sg, z, C, p_uniform_component=p_uniform_component, old_ev=fix_result)
