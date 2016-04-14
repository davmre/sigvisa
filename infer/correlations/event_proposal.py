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
from sigvisa.utils.array import index_to_time, time_to_index

from sigvisa import Sigvisa
from sigvisa.plotting.event_heatmap import EventHeatmap
from sigvisa.plotting.plot import savefig
from matplotlib.figure import Figure
from matplotlib.cm import get_cmap

import cPickle as pickle


def get_atime_posteriors(sg, ar_advantage=False):
    try:
        sg._cached_atime_posteriors
    except:
        proposals = get_historical_signals(sg)
        sg._cached_atime_posteriors = compute_atime_posteriors(sg, proposals, use_ar=ar_advantage)
    return sg._cached_atime_posteriors


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

def sample_time_from_pdf(pdf, stime, srate):
    n = len(pdf)
    idx = np.random.choice(xrange(n), p=pdf)
    return stime + (idx + np.random.rand())/float(srate)

def logp_from_time_pdf(pdf, stime, x, srate):
    idx = time_to_index(x, stime, srate)
    if idx < 0 or idx > len(pdf):
        return -np.inf
    else:
        return np.log(pdf[idx])

def propose_from_otime_posteriors(training_xs, proposal_weights, proposal_otime_posteriors, 
                                  global_stime, global_srate=1.0,
                                  n_proposals=1, proposal_dist_seed=None, fix_result=None):

    if proposal_dist_seed is not None:
        np.random.seed(proposal_dist_seed)

    n = len(proposal_weights)
    proposal_width_deg = 0.03
    proposal_height_km = 5.0
    if fix_result is None:

        all_proposals = []
        for i in range(n_proposals):
            kernel = np.random.choice(xrange(n), p=proposal_weights)
            otime_dist = proposal_otime_posteriors[kernel]        
            xx = training_xs[kernel]
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
    for i in range(n):
        x = training_xs[i]
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

        proposed_ev = Event(lon=plon, lat=plat, depth=pdepth, time=ptime, mb=3.0)
        return proposed_ev, log_qforward, (proposal_weights, kernel, proposal_otime_posteriors, training_xs)
    else:
        return log_qforward


def compute_proposal_distribution(sg, stas, phases, 
                                  origin_stime, origin_etime,
                                  global_srate = 1.0,
                                  ar_advantage=False, temper=1.0):

    # get the giant (hopefully cached) structure containing the (bayesian) cross-correlations
    # of all phases of all training events against all currently loaded wns
    atime_lls = get_atime_posteriors(sg, ar_advantage=ar_advantage)
    if len(atime_lls) == 0:
        return None, None, None


    # then for each training event location, and each timestep,
    # (with proposal timesteps considered at 1hz, not tied to
    # the resolution of the signals under consideration),
    # compute a proxy likelihood for all currently loaded signals
    # under the hypothesis that an event exists at the same
    # origin location and the hypothesized origin time.

    N = time_to_index(origin_etime, origin_stime, global_srate)
    proposal_origin_time_posteriors = []
    proposal_weights = []
    for (x, sta_lls) in atime_lls:
        posterior, weight = hack_ev_time_posterior_with_weight(sg, x, sta_lls, 
                                                               stas=stas,
                                                               phases=phases,
                                                               N=N,
                                                               global_stime = origin_stime,
                                                               global_srate = global_srate,
                                                               temper=temper)
        proposal_origin_time_posteriors.append(posterior)

        proposal_weights.append(weight)

    training_xs = [x for (x, sta_lls) in atime_lls]        
    return training_xs, proposal_weights, proposal_origin_time_posteriors

def sample_corr_kwargs(sg):

    all_stations = sg.station_waves.keys()

    sta_choices = [tuple(all_stations),]
    sta_probs = [1.0,]
    if "PD31" in all_stations:
        sta_choices.append(("PD31",))
        sta_probs.append(0.5)
    if "NVAR" in all_stations:
        sta_choices.append(("NVAR",))
        sta_probs.append(0.5)
    sta_probs = np.array(sta_probs) / np.sum(sta_probs)
    stas_idx = np.random.choice(np.arange(len(sta_choices)), p=sta_probs)
    stas = sta_choices[stas_idx]

    phase_choices = ("Lg", ("Pg", "Lg"), ("Pn", "Lg", "Sn", "Pg"))
    phases_idx = np.random.choice(np.arange(len(phase_choices)))
    phases = phase_choices[phases_idx]

    corr_kwargs = {"stas": stas, "phases": phases}
    
    # time restrictions
    stime = None
    etime = None
    if sg.inference_region is not None:
        stime = sg.inference_region.stime
        etime = sg.inference_region.etime

        restricted_length = 1200.0
        restricted_windows = int(np.ceil((etime - stime) /restricted_length))
        restricted_length = (etime-stime) / restricted_windows
        restricted_stime = np.linspace(stime, etime, restricted_windows)
        restricted_etime = restricted_stime + restricted_length
        intervals = zip(restricted_stime, restricted_etime)
        if np.random.rand() < 0.3:
            idx = np.random.choice(np.arange(len(intervals)))
            interval = intervals[idx]
            corr_kwargs["origin_stime"] = interval[0]
            corr_kwargs["origin_etime"] = interval[1]

    return corr_kwargs


def plot_proposal_weights(training_xs, proposal_weights, stas, true_x=None, ax=None, fname=None):

    training_xs = np.squeeze(np.array(training_xs))

    weights = np.array(proposal_weights)
    #scaled_weights = (weights - np.min(weights)) / weights.ptp()
    scaled_weights = weights

    s = Sigvisa()
    hm = EventHeatmap(f=None, calc=False, autobounds=training_xs, autobounds_quantile=1.0)


    if ax is None:
        fig = Figure(figsize=(15, 15))
        ax = fig.add_subplot(111)

    sta_locations = [s.earthmodel.site_info(n, 0)[0:2] for n in stas]
    hm.init_bmap(axes=ax)
    hm.plot_locations(sta_locations, labels=stas,
                        marker="^", ms=20, mec="none", mew=0,
                        alpha=1.0, mfc="blue")
    
    
    colors = get_cmap("Reds")(scaled_weights)
    if true_x is not None:
        hm.plot_locations(true_x, marker="*", ms=20, mfc="green", alpha=0.5)
    
    hm.bmap.scatter(training_xs[:, 0], training_xs[:, 1], marker="o", s=100 * scaled_weights, alpha=0.5, c=colors)
    ax.axis("off")

    if fname is not None:
        savefig(fname, fig, dpi=300, bbox_inches='tight')


    return fig, ax

def correlation_location_proposal(sg, fix_result=None, 
                                  proposal_dist_seed=None, 
                                  temper=1.0, phases=("P",), 
                                  n_proposals=1,
                                  ar_advantage=False,
                                  origin_stime=None,
                                  origin_etime=None,
                                  stas=None):
    # in general we want to generate a list of locations, taus, and signals.
    # this could mean taking random offsets from the historical events, and computing taus with a GP. (formally speaking this would be valid, even as a 'random choice of proposal' over an uncountable space of proposals, since we can just think of incorporating the Gaussian sample as an auxiliary variable).
    #But for right now I'll just use the trivial function that proposes at the historical locations. This means we generate a list of [ev, p(ev|m), tau, signals].

    if origin_stime is None:
        if sg.inference_region is not None:
            origin_stime = sg.inference_region.stime
            origin_etime = sg.inference_region.etime
        else:
            origin_stime = sg.event_start_time
            origin_etime = sg.event_end_time

    training_xs, proposal_weights, proposal_otime_posteriors = \
        compute_proposal_distribution(sg, stas, phases, 
                                      ar_advantage=ar_advantage, 
                                      origin_stime=origin_stime,
                                      origin_etime=origin_etime,
                                      temper=temper)

    if proposal_weights is None:
        if fix_result is not None:
            return -np.inf
        else:
            return None, -np.inf, None
    
    #proposal_weights = reweight_uniform_top(proposal_weights, n=10)
    proposal_weights = reweight_temper_exp(proposal_weights, temper=0.1)
    proposal_weights = reweight_uniform_all(proposal_weights, uniform_prob=0.05)

    
    return propose_from_otime_posteriors(training_xs, proposal_weights, 
                                         proposal_otime_posteriors, 
                                         global_stime=origin_stime,
                                         proposal_dist_seed=proposal_dist_seed,
                                         n_proposals=n_proposals, 
                                         fix_result=fix_result)


