import numpy as np

import time

import scipy.weave as weave
from scipy.weave import converters

from sigvisa import Sigvisa
from sigvisa.infer.correlations.ar_correlation_model import estimate_ar, ar_advantage, iid_advantage




def compute_atime_posteriors(sg, proposals, phase, use_ar=False):
    """
    compute the bayesian cross-correlation (logodds of signal under an AR noise model) 
    for all signals in the historical library, against all signals in the current SG.
    
    This is quite expensive so should in general be run only once, and the results cached. 
    """

    origin_lls = []
    i = 0
    for (x, signals) in proposals:
        sta_lls = dict()
        for sta in sg.station_waves.keys():
            for wn in sg.station_waves[sta]:
                try:
                    c = signals[(wn.sta, wn.chan, wn.band)]
                except KeyError:
                    continue
                sdata = wn.unexplained_kalman()
                if use_ar:
                    lls = ar_advantage(sdata, c, wn.nm)
                else:
                    normed_sdata = sdata / np.std(sdata)
                    lls = iid_advantage(normed_sdata, c)

                tt_array, tt_mean = build_ttr_model_array(sg, x, wn.sta, wn.srate, phase=phase)

                sta_lls[wn.label] = (lls, tt_array, tt_mean)

                sg.logger.info("computed advantage for %s %s" % (x, wn.label))
                i += 1
        origin_lls.append((x, sta_lls))

        

        #if i > 10:
        #    break

    return origin_lls


def build_ttr_model_array(sg, x, sta, srate, K=None, phase="P"):
    s = Sigvisa()
    phaseid = s.phaseids[phase]
    lon, lat, depth = x[0,0], x[0,1], x[0,2]
    
    tt_mean = s.sigmodel.mean_travel_time(lon, lat, depth, sg.event_start_time, sta, phaseid - 1)

    model, modelid = sg.get_model("tt_residual", sta, phase)
    pred_ttr  = float(model.predict(x))
    tt_mean += pred_ttr

    if K is None:
        K = int(15*srate)

    ttrs = np.linspace(-K/float(srate), K/float(srate), 2*K+1)
    ll_array = np.array([model.log_p(ttr + pred_ttr, cond=x, include_obs=True) for ttr in ttrs]).flatten()
    return np.exp(ll_array), tt_mean

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

    llmax = np.max(ll)
    ll_exp = np.exp(ll-llmax)
    r = np.convolve(ll_exp, ttr_model, "full")

    rr = integrate_downsample(r, srate, out_srate)

    origin_ll = np.log(rr) + llmax

    K = (len(ttr_model)-1)/2
    origin_stime = ll_stime - mean_tt - float(K)/srate
    return origin_ll, origin_stime


def update_lls_for_wn(wn, atime_lls, pre_s=3.0, temper=1.0):
    """
    Take the atime likelihoods computed by ar_advantage, and zero out
    atimes corresponding to currently-hypothesized events. This is a hack in
    substitution for actually rerunning ar_advantage on the new unexplained_kalman signal. 
    """

    new_lls = atime_lls.copy()
    for (eid, phase) in wn.arrivals():
        if phase=="UA": continue
        v, tg = wn.get_template_params_for_arrival(eid, phase)
        atime = v["arrival_time"]
        at_idx = int((atime-wn.st) * wn.srate)
        
        post_idx = int(10.0 * wn.srate)
        post_idx = max(post_idx,  tg.abstract_logenv_length(v, min_logenv=np.log(wn.nm_env.c), srate=wn.srate))

        sidx = at_idx - int(pre_s * wn.srate)
        eidx = at_idx + post_idx
        sidx = max(sidx, 0)
        eidx = min(eidx, len(new_lls))
        if sidx > wn.npts or eidx < 0:
            continue
        new_lls[sidx:eidx] = 0.0

    new_lls /= temper

    return new_lls

def wn_origin_posterior(sg, wn, x, cached_for_wn, out_srate, temper=1):

    atime_lls, tt_array, tt_mean = cached_for_wn

    srate = wn.srate
    ll_stime = wn.st
    hacked_atime_lls = update_lls_for_wn(wn, atime_lls, temper=temper)
    origin_ll, origin_stime = atime_likelihood_to_origin_likelihood(hacked_atime_lls, ll_stime, srate, tt_mean, tt_array, out_srate)


    return origin_ll, origin_stime
    
def hack_ev_time_posterior_with_weight(sg, x, sta_lls, global_stime, N, global_srate, stas=None, temper=1.0):

    global_ll = np.zeros((N,))
    for wn_label, cached_for_wn in sta_lls.items():
        wn = sg.all_nodes[wn_label]

        if stas is not None and wn.sta not in stas: continue
        
        t0 = time.time()
        origin_ll, origin_stime = wn_origin_posterior(sg, wn, x, cached_for_wn, 
                                                      global_srate, temper=temper)
        t1 = time.time()
        global_offset = int((origin_stime - global_stime)*global_srate)
        align_sum(global_ll, 
                  origin_ll, 
                  global_offset)
        t2 = time.time()

    return global_ll

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
    
    
