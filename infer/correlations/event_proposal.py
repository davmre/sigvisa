import numpy as np

from sigvisa.source.event import get_event
from sigvisa.signals.io import load_event_station_chan
from sigvisa.models.noise.noise_util import get_noise_model
from sigvisa.database.dataset import DET_TIME_COL
from sigvisa.infer.correlations.weighted_event_posterior import ev_time_posterior_with_weight
from sigvisa.infer.correlations.historical_signal_library import get_historical_signals
from sigvisa.models.distributions import Gaussian,TruncatedGaussian
from sigvisa.source.event import Event




def correlation_location_proposal(sg, fix_result=None, proposal_dist_seed=None, temper=1.0, stas=None):
    # in general we want to generate a list of locations, taus, and signals.
    # this could mean taking random offsets from the historical events, and computing taus with a GP. (formally speaking this would be valid, even as a 'random choice of proposal' over an uncountable space of proposals, since we can just think of incorporating the Gaussian sample as an auxiliary variable).
    #But for right now I'll just use the trivial function that proposes at the historical locations. This means we generate a list of [ev, p(ev|m), tau, signals].
    proposals = get_historical_signals(sg, "P")

    if len(proposals) == 0:
        if fix_result is None:
            return None, -np.inf, ()
        else:
            return -np.inf

    global_stime = sg.event_start_time
    N = sg.event_end_time - global_stime
    proposal_otime_likelihoods = [ev_time_posterior_with_weight(sg, x, signals,  
                                                                stas=stas,
                                                                N=N, temper=temper,
                                                                global_stime = global_stime) 
                                  for (x, signals) in proposals]
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


    proposal_weights = np.array(proposal_weights)
    proposal_weights /= 10
    proposal_weights -= np.max(proposal_weights)
    proposal_weights = np.exp(proposal_weights)
    proposal_weights += 0.05/len(proposals)
                               # regularize the proposal distribution:
                               # the effect of this is that roughly 5%
                               # of the time we will propose from a
                               # uniform distribution over all
                               # candidates.
    proposal_weights /= np.sum(proposal_weights)
    n = len(proposal_weights)

    # TODO: more intelligent choice of proposal stddev
    # 1deg ~= 100km
    # so stddev of 5km ~= 0.05deg

    if fix_result is None:
        kernel = np.random.choice(xrange(n), p=proposal_weights)
        otime_dist = proposal_otime_posteriors[kernel]        
        xx, signals = proposals[kernel]
        lon, lat, depth = xx[0,0], xx[0,1], xx[0,2]

        londist = Gaussian(lon, 0.02)
        latdist = Gaussian(lat, 0.02)
        depthdist = TruncatedGaussian(depth, 10.0, a=0)

        plon, plat, pdepth = (londist.sample(), latdist.sample(), depthdist.sample())
        ptime = sample_time_from_pdf(otime_dist, global_stime, srate=1.0)
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
    for i in range(len(proposals)):
        x, signals = proposals[i]
        lon, lat, depth = x[0,0], x[0,1], x[0,2]

        otime_dist = proposal_otime_posteriors[i]
        londist = Gaussian(lon, 0.02)
        latdist = Gaussian(lat, 0.02)
        depthdist = TruncatedGaussian(depth, 10.0, a=0)
        
        lw = np.log(proposal_weights[i])
        lw += londist.log_p(plon)
        lw += latdist.log_p(plat)
        lw += depthdist.log_p(pdepth)
        lw += logp_from_time_pdf(otime_dist, global_stime, ptime, srate=1.0)
        log_qforward = np.logaddexp(log_qforward, lw)

    if fix_result is None:
        proposed_ev = Event(lon=plon, lat=plat, depth=pdepth, time=ptime, mb=4.0)
        return proposed_ev, log_qforward, (proposals, proposal_weights, proposal_otime_posteriors, xx)
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
