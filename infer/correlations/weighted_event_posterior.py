import numpy as np
from sigvisa.models.ttime import tt_predict
import scipy.weave as weave
from scipy.weave import converters

from sigvisa.infer.correlations.ar_correlation_model import estimate_ar, ar_advantage

def build_ttr_model_array(sg, ev, sta, srate, K=None, phase="P"):
    tt_mean = tt_predict(ev, sta, phase) 
    model, modelid = sg.get_model("tt_residual", sta, phase)
    pred_ttr  = model.predict(ev)
    tt_mean += pred_ttr

    if K is None:
        K = int(15*srate)

    ttrs = np.linspace(-K/float(srate), K/float(srate), 2*K+1)
    ll_array = np.array([model.log_p(ttr + pred_ttr, cond=ev, include_obs=True) for ttr in ttrs])
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

def wn_origin_posterior(sg, wn, ev, c, out_srate, temper=1):
    sdata = wn.unexplained_kalman()
    unobs_lp = 0.0

    srate = wn.srate
    ll_stime = wn.st
    try:
        tt_array, tt_mean = build_ttr_model_array(sg, ev, wn.sta, srate, phase="P")
    except ValueError:
        # if the phase is impossible, then we get no information about
        # origin time and we should just model the signal under the
        # unobs probability (i.e., as being explained completely by
        # the baseline iid model). 
        return np.array(()), 0.0, unobs_lp


    nm = estimate_ar(sdata)
    lls = ar_advantage(sdata, c, nm)
    if temper != 1:
        lls /= temper

    origin_ll, origin_stime = atime_likelihood_to_origin_likelihood(lls, ll_stime, srate, tt_mean, tt_array, out_srate)
    return origin_ll, origin_stime, unobs_lp
    

def ev_time_posterior_with_weight(sg, ev, signals, tau, global_stime=None, N=None, global_srate=1.0, len_s=10.0, temper=1, stas=None):
    if global_stime  is None:
        global_stime = sg.event_start_time
        N = int((sg.event_end_time - global_stime)*global_srate)
    global_ll = np.zeros((N,))

    if stas is None:
        stas = sg.station_waves.keys()
    for sta in stas:
        for wn in sg.station_waves[sta]:
            try:
                c = signals[(wn.sta, wn.chan, wn.band)]
            except KeyError:
                continue
            #c = c_tau * tau
            #kappa = 1 - tau**2 * kappa_tau2_1m
            origin_ll, origin_stime, unobs_lp = wn_origin_posterior(sg, wn, ev, c, global_srate, temper=temper)

            global_offset = int((origin_stime - global_stime)*global_srate)
            align_sum(global_ll, origin_ll, global_offset, unobs_lp)
            
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
    
def align_sum(A, B, offset, default):
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
    
    
