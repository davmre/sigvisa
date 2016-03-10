import numpy as np
import time

from sigvisa.source.event import get_event
from sigvisa.signals.io import load_event_station_chan
from sigvisa.models.noise.noise_util import get_noise_model
from sigvisa.database.dataset import DET_TIME_COL
from sigvisa.infer.correlations.weighted_event_posterior import hack_ev_time_posterior_with_weight, compute_atime_posteriors
from sigvisa.infer.correlations.historical_signal_library import get_historical_signals
from sigvisa.models.distributions import Gaussian,TruncatedGaussian
from sigvisa.source.event import Event

import cPickle as pickle


def get_atime_posteriors(sg, phase, ar_advantage=False):
    try:
        sg._cached_atime_posteriors
    except:
        sg._cached_atime_posteriors = {}

    key = (phase, ar_advantage)

    if key not in sg._cached_atime_posteriors:
        proposals = get_historical_signals(sg, phase)
        sg._cached_atime_posteriors[key] = compute_atime_posteriors(sg, proposals, phase, use_ar=ar_advantage)
    return sg._cached_atime_posteriors[key]


def reweight_uniform_top(weights, n=20):

    if len(weights) <= n:
        cutoff = -np.inf
    else:
        # give equal weight to the top n candidates, and zero to everyone else
        cutoff = sorted(weights)[-n]

    weights = np.asarray(weights)
    new_weights = np.ones(weights.shape)
    new_weights[weights < cutoff] = 0.0
    new_weights /= np.sum(new_weights)
    return new_weights

def reweight_temper_exp(weights, temper):

    weights = np.asarray(weights)

    new_weights = weights * temper
    new_weights -= np.max(new_weights)
    new_weights = np.exp(new_weights)
    new_weights /= np.sum(new_weights)
    return new_weights

def reweight_uniform_all(weights, uniform_prob=0.05):
    total_weight = np.sum(weights)
    added_uniform_weight = total_weight / (1./uniform_prob - 1.)

    new_weights = weights + added_uniform_weight / len(weights)
    new_weights /= np.sum(new_weights)
    return new_weights
    
def sample_corr_kwargs(sg):
    stas = sg.station_waves.keys()
    sta = np.random.choice(stas)

    phases=["P", "Lg", "S", "Pg"]
    phase = np.random.choice(phases)

    corr_kwargs = {"stas": (sta,), "phase": phase}
    return corr_kwargs

def correlation_location_proposal(sg, fix_result=None, 
                                  proposal_dist_seed=None, 
                                  temper=1.0, phase="P", 
                                  n_proposals=1,
                                  ar_advantage=False,
                                  stas=None):
    # in general we want to generate a list of locations, taus, and signals.
    # this could mean taking random offsets from the historical events, and computing taus with a GP. (formally speaking this would be valid, even as a 'random choice of proposal' over an uncountable space of proposals, since we can just think of incorporating the Gaussian sample as an auxiliary variable).
    #But for right now I'll just use the trivial function that proposes at the historical locations. This means we generate a list of [ev, p(ev|m), tau, signals].

    atime_lls = get_atime_posteriors(sg, phase, ar_advantage=ar_advantage)

    with open("db_cache/atime_lls.pkl", "wb") as f:
        pickle.dump(atime_lls, f)
        

    if len(atime_lls) == 0:
        if fix_result is None:
            return None, -np.inf, ()
        else:
            return -np.inf

    global_srate = 1.0 # granularity at which to make atime proposals, NOT
                       # tied to the srate of any particular signal
    if sg.inference_region is not None:
        global_stime = sg.inference_region.stime
        N = int((sg.inference_region.etime - sg.inference_region.stime)*global_srate)
    else:
        global_stime = sg.event_start_time
        N = int((sg.event_end_time - global_stime)*global_srate)


    t0 = time.time()
    proposal_otime_likelihoods = [hack_ev_time_posterior_with_weight(sg, x, sta_lls, stas=stas,
                                                                     N=N,
                                                                     global_stime = global_stime,
                                                                     global_srate = global_srate,
                                                                     temper=temper)
                                  for (x, sta_lls) in atime_lls]
        


    t1 = time.time()
    sg.logger.info("computed all otime posteriors in %.1f" % (t1-t0))

    proposal_weights = []
    proposal_otime_posteriors = []
    for ll in proposal_otime_likelihoods:
        C = np.max(ll)
        posterior = np.exp(ll-C)
        Z = np.sum(posterior)
        posterior /= Z
        logZ = np.log(Z) + C
        proposal_weights.append(logZ)
        proposal_otime_posteriors.append(posterior)

    proposal_weights = reweight_uniform_all(reweight_uniform_top(proposal_weights, n=15), uniform_prob=0.05)

    
    """
    proposal_weights = np.array(proposal_weights)
    proposal_weights /= 10
    proposal_weights -= np.max(proposal_weights)
    proposal_weights = np.exp(proposal_weights)
    proposal_weights += 0.05/len(atime_lls)
                               # regularize the proposal distribution:
                               # the effect of this is that roughly 5%
                               # of the time we will propose from a
                               # uniform distribution over all
                               # candidates.            
    proposal_weights /= np.sum(proposal_weights)
    """
    n = len(proposal_weights)

    # TODO: more intelligent choice of proposal stddev
    # 1deg ~= 100km
    # so stddev of 5km ~= 0.05deg
    proposal_width_deg = 0.1
    proposal_height_km = 5.0
    if fix_result is None:

        all_proposals = []
        for i in range(n_proposals):
            kernel = np.random.choice(xrange(n), p=proposal_weights)
            otime_dist = proposal_otime_posteriors[kernel]        
            xx, _ = atime_lls[kernel]
            lon, lat, depth = xx[0,0], xx[0,1], xx[0,2]

            londist = Gaussian(lon, proposal_width_deg)
            latdist = Gaussian(lat, proposal_width_deg)
            depthdist = TruncatedGaussian(depth, proposal_height_km, a=0)

            plon, plat, pdepth = (londist.sample(), latdist.sample(), depthdist.sample())
            ptime = sample_time_from_pdf(otime_dist, global_stime, srate=global_srate)
            all_proposals.append((plon, plat, pdepth, ptime))
    else:
        fev = fix_result
        plon, plat, pdepth, ptime = fev.lon, fev.lat, fev.depth, fev.time

    log_qforward = -np.inf

    # compute proposal probability under the explicit mixture
    # distribution of Gaussians centered on historical event.  this
    # will need to be optimized, eventually. but as long as we're
    # explicitly computing signal correlations for and atime
    # posteriors for each historical event, it's not much extra work
    # to compute logp under the resulting distribution.
    for i in range(len(atime_lls)):
        x, _ = atime_lls[i]
        lon, lat, depth = x[0,0], x[0,1], x[0,2]

        otime_dist = proposal_otime_posteriors[i]
        londist = Gaussian(lon, proposal_width_deg)
        latdist = Gaussian(lat, proposal_width_deg)
        depthdist = TruncatedGaussian(depth, proposal_height_km, a=0)
        
        lw = np.log(proposal_weights[i])
        lw += londist.log_p(plon)
        lw += latdist.log_p(plat)
        lw += depthdist.log_p(pdepth)
        lw += logp_from_time_pdf(otime_dist, global_stime, ptime, srate=global_srate)
        log_qforward = np.logaddexp(log_qforward, lw)

    if fix_result is None:

        if n_proposals > 1:
            return np.array(all_proposals)

        proposed_ev = Event(lon=plon, lat=plat, depth=pdepth, time=ptime, mb=4.0)
        return proposed_ev, log_qforward, (atime_lls, proposal_weights, kernel, proposal_otime_posteriors, xx)
    else:
        return log_qforward



def sample_time_from_pdf(pdf, stime, srate):
    n = len(pdf)
    idx = np.random.choice(xrange(n), p=pdf)
    return stime + (idx + np.random.rand())/float(srate)

def logp_from_time_pdf(pdf, stime, x, srate):
    idx = int((x-stime) * srate)
    if idx < 0 or idx > len(pdf):
        return -np.inf
    else:
        return np.log(pdf[idx])
