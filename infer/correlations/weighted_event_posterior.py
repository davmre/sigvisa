import numpy as np

import time

import scipy.weave as weave
from scipy.weave import converters

from sigvisa import Sigvisa
from sigvisa.infer.correlations.ar_correlation_model import estimate_ar, ar_advantage, iid_advantage
from sigvisa.utils.array import index_to_time, time_to_index



def compute_atime_posteriors(sg, proposals, 
                             global_srate=1.0, 
                             use_ar=False,
                             raw_data=False,
                             event_idx=None):
    """
    compute the bayesian cross-correlation (logodds of signal under an AR noise model) 
    for all signals in the historical library, against all signals in the current SG.
    
    This is quite expensive so should in general be run only once, and the results cached. 
    """

    atime_lls = []
    i = 0
    for idx, (x, signals) in enumerate(proposals):

        if event_idx is not None and event_idx != idx: 
            continue

        sta_lls = dict()
        for (sta, chan, band, phase), c in signals.items():
            wns = sg.station_waves[sta]
            if len(wns) == 0: 
                continue
            elif len(wns) > 1:
                raise Exception("haven't worked out correlation proposals with multiple wns from same station")
            wn = wns[0]

            if raw_data:
                sdata = wn.get_value().data.copy()
                sdata[np.isnan(sdata)] = 0.0
            else:
                sdata = wn.unexplained_kalman()

            if use_ar:
                lls = ar_advantage(sdata, c, wn.nm)
            else:
                normed_sdata = sdata / wn.nm_env.c #np.std(sdata)
                lls = np.sqrt(iid_advantage(normed_sdata, c)) # sqrt for laplacian noise, essentially

            tt_array, tt_mean = build_ttr_model_array(sg, x, sta, wn.srate, phase=phase)
            
            origin_ll, origin_stime = atime_likelihood_to_origin_likelihood(lls, wn.st, wn.srate, tt_mean, tt_array, global_srate)

            signal_scale = wn.nm_env.c
            sta_lls[(wn.label, phase)] = origin_ll, origin_stime, signal_scale

            sg.logger.info("computed advantage for %s %s %s" % (x, wn.label, phase))
            i += 1
        atime_lls.append((x, sta_lls))


    return atime_lls


def build_ttr_model_array(sg, x, sta, srate, K=None, phase="P"):
    s = Sigvisa()
    phaseid = s.phaseids[phase]
    lon, lat, depth = x[0,0], x[0,1], x[0,2]
    
    tt_mean = s.sigmodel.mean_travel_time(lon, lat, depth, sg.event_start_time, sta, phaseid - 1)

    model, modelid = sg.get_model("tt_residual", sta, phase)
    pred_ttr  = float(model.predict(x))
    tt_mean += pred_ttr

    if K is None:
        # also hardcoded in the wn_origin_posterior hack...
        K = int(15*srate)

    ttrs = np.linspace(-K/float(srate), K/float(srate), 2*K+1)
    ll_array = np.array([model.log_p(ttr + pred_ttr, cond=x, include_obs=True) for ttr in ttrs]).flatten()
    ttr_model = np.exp(ll_array)
    ttr_model = np.where(ttr_model <= 0, 1e-300, ttr_model)
    return ttr_model, tt_mean

def atime_likelihood_to_origin_likelihood(ll, ll_stime, srate, mean_tt, ttr_model, out_srate):
    # given:
    # ll: array of length N-n, giving p(signal| X, atime=t) at a particular station,
    # ll_stime: arrival time corresponding to ll[0]
    # ll_srate: sampling rate in Hz of ll
    # mean_tt: mean travel time 
    # ttr_model: symmetric array of length 2*K+1, for arbitrary K, giving ttr_model[i] = p(travel time residual in second = (i-K)/srate) 
    #
    # returns:
    # origin_stime: start of origin time distribution
    # origin_ll: log likelihood of signal given origin times



    # we have to leave logspace to do the convolution,
    # but this loses tons of precision. so we
    # redo the calculation separately at 
    # multiple "volume" levels and combine the results
    # to attempt to retain the full dynamic range
    # of log-probabilities.
    # note this exploits the fact that these logodds are nonnegative. 
    llmax = np.max(ll)
    nlevels = int(np.ceil(llmax/500.))+1
    origin_ll_prev = None

    for level in np.linspace(0, llmax, nlevels):
        ll_exp = np.exp(ll-level)
        r = np.convolve(ll_exp, ttr_model, "full")
        rr = integrate_downsample(r, srate, out_srate)
        origin_ll = np.log(rr) + level

        if origin_ll_prev is None:
            origin_ll = np.where(origin_ll < 0, 0.0, origin_ll )
        else:
            origin_ll = np.where(origin_ll < 0, origin_ll_prev, origin_ll )

        origin_ll_prev = origin_ll

    if not np.isfinite(np.max(origin_ll)):
        import pdb; pdb.set_trace()


    K = (len(ttr_model)-1)/2
    origin_stime = ll_stime - mean_tt - float(K)/srate

    if np.isinf(np.min(origin_ll)):
        import pdb; pdb.set_trace()


    return origin_ll, origin_stime



def wn_origin_posterior(sg, wn, cached_for_wn, 
                        temper = 1, corr_s=20.0,
                        wn_env_cache=None,
                        global_srate=1.0):

    """
    update the cached origin-time posterior to reflect currently 
    hypothesized events, by zeroing out the origin times that would
    generate the templates observed from those events.
    """

    if wn_env_cache is None:
        # the trivial cache
        wn_env_cache = {}

    # padding added by the traveltime convolution
    # todo: don't hardcode this hack
    tt_K = 15
    tt_s = tt_K*global_srate

    origin_ll, origin_stime, signal_scale = cached_for_wn
    new_lls = origin_ll.copy()

    if wn.label not in wn_env_cache:
        ev_arrivals = [(eid, phase) for (eid, phase) in wn.arrivals() if phase!="UA"]
        wn_env_cache[wn.label] = (wn.assem_env(ev_arrivals, srate=global_srate) + signal_scale) / signal_scale
    pred_env = wn_env_cache[wn.label]

    margin = int(corr_s * global_srate)
    i1 = margin/2
    i2 = margin-i1
    new_lls[tt_K:-tt_K] /= pred_env[i1:-i2]


    # corresponds to a precomputed mean traveltime for this phase and ev location
    timeshift = wn.st - (origin_stime + tt_s)
    for (eid, phase) in wn.arrivals():
        if phase=="UA": continue
        v, tg = wn.get_template_params_for_arrival(eid, phase)
        atime = v["arrival_time"]
        otime = atime - timeshift
        ot_idx = time_to_index(otime, origin_stime, global_srate)
        
        corr_idx = int(corr_s * global_srate)

        sidx = ot_idx - corr_idx - tt_K
        eidx = ot_idx + corr_idx 

        sidx = max(sidx, 0)
        eidx = min(eidx, len(new_lls))
        if sidx > len(new_lls) or eidx < 0:
            continue
        new_lls[sidx:eidx] = 1.0 # todo figure out a correct default value

    new_lls /= temper
    return new_lls, origin_stime
    
    
def hack_ev_time_posterior_with_weight(sg, sta_lls, global_stime, N, global_srate, 
                                       phases=None, stas=None, temper=1.0):
    """
    Given a training event x, and a set of precomputed atime log-likelihoods
     (sta_lls: dict mapping (wn_label, phase) to the bayesian cross correlation 
        (arrival time likelihood) for each wn based on the GP predictions for
        each arriving phase of this training event), 
    do the following:
     - add the hack correction which zeros out the atime likelihood for time periods
       where we already have an event arrival, to avoid reproposing existing events.
     - convert the atime likelihood into an origin time likelihood, by convolving 
       with the traveltime model, and downsampling to the global srate
     - add up the origin time log-likelihoods across the specified set of stations and
       phases to get a combined log-likelihood on the origin time for an event in this
       training location. 
     - normalize the origin-time log-likelihood to get a posterior on origin time, along
       with the normalizing constant which gives the marginal likelihood of an event at
       this training location. 
    """

    global_ll = np.zeros((N,))
    wn_env_cache = {}
    for (wn_label, phase), cached_for_wn in sta_lls.items():
        wn = sg.all_nodes[wn_label]

        if stas is not None and wn.sta not in stas: continue
        if phases is not None and phase not in phases: continue

        t0 = time.time()
        origin_ll, origin_stime = wn_origin_posterior(sg, wn=wn, cached_for_wn=cached_for_wn, 
                                                      global_srate=global_srate, 
                                                      wn_env_cache = wn_env_cache,
                                                      temper=temper)
        if not np.isfinite(np.max(origin_ll)):
            import pdb; pdb.set_trace()

        t1 = time.time()
        global_offset = int((origin_stime - global_stime)*global_srate)
        align_sum(global_ll, 
                  origin_ll, 
                  global_offset)

        if np.isinf(np.max(global_ll)):
            import pdb; pdb.set_trace()


        t2 = time.time()

    C = np.max(global_ll)
    posterior = np.exp(global_ll-C)
    Z = np.sum(posterior)
    posterior /= Z
    logZ = np.log(Z) + C

    if not np.isfinite(logZ):
        import pdb; pdb.set_trace()


    return posterior, logZ

def integrate_downsample(A, srate1, srate2):
    sratio = srate1/srate2
    assert(np.abs(sratio - int(sratio)) < 1e-10)
    sratio = int(sratio)

    n1 = len(A)
    n2 = n1/sratio
    B = np.empty((n2,))
    code = """
    for(int i=0; i < n2; ++i) {
       double tmp = 0;
       for (int j=0; j < sratio; ++j) {
         tmp += A(sratio*i+j);
       }
       B(i) = tmp/sratio;
    }
    """
    weave.inline(code,[ 'n2', 'sratio', 'A','B'],type_converters = converters.blitz,
                 verbose=2,compiler='gcc',)
    return B
    
def align_sum(A, B, offset, default=0.0):
    nA = len(A)
    nB = len(B)

    end_offset = offset+nB
    
    if offset > 0:
        # B starts after A
        A_start = offset
        B_start = 0
    else:
        # B starts before A
        A_start = 0
        B_start = -offset

    if end_offset < nA:
        # B ends before A
        A_end = end_offset
        B_end = nB
    else:
        A_end = nA
        B_end = B_start + (A_end-A_start)

    if A_end < 0 or A_start >= nA:
        A += default
    else:
        A[:A_start] += default
        A[A_start:A_end] += B[B_start:B_end]
        A[A_end:] += default
    
    
