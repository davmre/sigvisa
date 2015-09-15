import numpy as np

from sigvisa.source.event import get_event
from sigvisa.signals.io import load_event_station_chan
from sigvisa.models.noise.noise_util import get_noise_model
from sigvisa.database.dataset import DET_TIME_COL
from sigvisa.infer.correlations.weighted_event_posterior import ev_time_posterior_with_weight
from sigvisa.models.distributions import Gaussian,TruncatedGaussian
from sigvisa.source.event import Event


def extract_template(wave, atime, len_s):
    #nm, nmid, _ = get_noise_model(waveform=wave, model_type="ar", order=0, return_details=True)
    #sigma2_A = nm.em.std**2
    aidx = int((atime - wave['stime'])*wave['srate'])
    n = int(len_s * wave['srate'])
    A = wave.data[aidx:aidx+n]
    return A


def extract_template_from_leb(sta, evid, chan="auto", band="freq_2.0_4.5", len_s=10.0, srate=10.0):
    w =  load_event_station_chan(evid, sta, chan, evtype="leb")
    w = w.filter("%s;hz_%f" % (band, srate))
    atime =  w['event_arrivals'][0,DET_TIME_COL]
    A = extract_template(w, atime, len_s)
    return A, w["chan"]


def generate_historical_db(stas, evids, bands, chans,len_s=10.0, srate=10.0):
    # return a list of tuples (ev, dict[(sta, chan, band)]: (c_tau, kappa_tau_1m))
    # where (c_tau, kappa_tau_1m) are defined as in the code, as 'intermediate' values
    # that will allow us to compute c_tau and kappa_tau_1m once tau is known (at runtime). 

    history = []
    for evid in evids:
        ev = get_event(evid=evid)
        signals = dict()
        for sta in stas:
            for chan in chans:
                for band in bands:
                    try:
                        A, actual_chan = extract_template_from_leb(sta, evid, chan, band, len_s, srate=srate)
                    except Exception as e:
                        print e
                        continue

                    n = len(A)

                    # if historical signal data for this event is partially missing or corrupted, don't bother
                    if isinstance(A, np.ma.masked_array):
                        if A.mask is not None and isinstance(A.mask, np.ndarray) and A.mask.any():
                            continue
                        A = A.data
                    c = A/np.linalg.norm(A)
                    #c_tau = alpha_hat/(alpha_hat**2+sigma2_A) * A
                    #kappa_tau2_1m = alpha_hat**2 / (alpha_hat**2 + sigma2_A)
                    signals[(sta, actual_chan, band)] = c    
        history.append((ev, signals))
    return history

def load_historical_waveforms():
    try:
        with open("db_cache/signal_history.pkl", 'rb') as f:
            historical_db = pickle.load(f)
    except IOError:
        historical_db = generate_historical_db(stas, evids, ["freq_0.8_4.5"], [""])
    return historical_db


def correlation_location_proposal(sg, fix_result=None, proposal_dist_seed=None, history=None, temper=1.0, stas=None):
    if history is None:
        history = load_historical_waveforms()
    len_s = 10.0

    # in general we want to generate a list of locations, taus, and signals.
    # this could mean taking random offsets from the historical events, and computing taus with a GP. (formally speaking this would be valid, even as a 'random choice of proposal' over an uncountable space of proposals, since we can just think of incorporating the Gaussian sample as an auxiliary variable).
    #But for right now I'll just use the trivial function that proposes at the historical locations. This means we generate a list of [ev, p(ev|m), tau, signals].
    def gen_proposals_trivial(history):
        tau = 0.95
        return [(ev, 1.0, tau, signals) for (ev, signals) in history]
    proposals = gen_proposals_trivial(history)

    global_stime = sg.event_start_time
    N = sg.event_end_time - global_stime
    proposal_otime_likelihoods = [ev_time_posterior_with_weight(sg, ev, signals, tau, 
                                                                stas=stas,
                                                                N=N, temper=temper,
                                                                global_stime = global_stime,
                                                                len_s=len_s) for (ev, p, tau, signals) in proposals]
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
    if fix_result is None:
        kernel = np.random.choice(xrange(n), p=proposal_weights)
    else:
        # TODO: kernel should be provided (sampled elsewhere from a
        # uniuform). But it might not be. Which means we need to
        # sample it from a uniform. And this probability should go
        # inside of a log_qbackward. But that's not part of the
        # interface of this function. So maybe we can just subtract it
        # from log_qforward. But this is hacky and I just need to be
        # sure I'm consistent about it, since if I'm doing this hack
        # for death moves, I need to do the reverse of the hack for
        # birth moves.
        assert(False)
    log_qforward = np.log(proposal_weights[kernel])
    otime_dist = proposal_otime_posteriors[kernel]
    ev, pm, tau, signals = proposals[kernel]
    log_qforward += np.log(pm)

    # TODO: more intelligent choice of proposal stddev
    # 1deg ~= 100km
    # so stddev of 5km ~= 0.05deg
    londist = Gaussian(ev.lon, 0.05)
    latdist = Gaussian(ev.lat, 0.05)
    depthdist = TruncatedGaussian(ev.depth, 10.0, a=0)
    if fix_result is None:
        plon, plat, pdepth = (londist.sample(), latdist.sample(), depthdist.sample())
        ptime = sample_time_from_pdf(otime_dist, global_stime, srate=1.0)
    else:
        fev = fixed_result
        plon, plat, pdepth, ptime = fev.lon, fev.lat, fev.depth, fev.time
    log_qforward += londist.log_p(plon)
    log_qforward += latdist.log_p(plat)
    log_qforward += depthdist.log_p(pdepth)
    log_qforward += logp_from_time_pdf(otime_dist, global_stime, ptime, srate=1.0)

    if fix_result is None:
        proposed_ev = Event(lon=plon, lat=plat, depth=pdepth, time=ptime, mb=4.0)
        return proposed_ev, log_qforward, (proposal_weights, proposal_otime_posteriors)
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
