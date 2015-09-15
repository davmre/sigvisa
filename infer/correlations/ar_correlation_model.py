import numpy as np
import scipy.weave as weave
from scipy.weave import converters

def unmask(S):
    # input:
    # S might be a np array, a masked array with the trivial mask, or a masked array with an actual  mask
    # output: a numpy array, and a boolean array representing the mask
    N = len(S)
    if isinstance(S, np.ma.masked_array):
        m = S.mask
        d = S.data
        try:
            m[0]
        except:
            m = np.isnan(d)
    else:
        d = S
        m = np.isnan(d)
    return d, m


def estimate_ar(S, n_p=10):
    from sigvisa.models.noise.armodel.model import ARModel, ErrorModel
    from sigvisa.models.noise.armodel.learner import ARLearner

    S, mask = unmask(S)
    l = ARLearner(S, sf=10)
    p, std = l.yulewalker(n_p)
    em = ErrorModel(0, std)
    nm = ARModel(p, em, l.c)
    return nm


def ar_advantage(S, c, nm):
    from sigvisa.models.noise.armodel.model import fastar_support

    local_support = """

    double reset_compute_ar(int n_p, blitz::Array<bool, 1> mask, blitz::Array<double, 1> d,
                          double var, blitz::Array<double, 1> p, blitz::Array<double, 2> tmp,
                           int start_idx,
                           int end_idx,
                           blitz::Array<double, 2> K,
                           blitz::Array<double, 1> u,
                           blitz::Array<double, 1> llarray,
                           int return_llarray) {
       for (int kk=0; kk < n_p; ++kk) {
            for (int kj=0; kj < n_p; ++kj) {
            K(kk, kj) = 0;
            tmp(kk, kj) = 0;
            }
            K(kk, kk) = 1e4;
            u(kk) = 0;
        }
        return compute_ar(n_p, mask, d, var, p, tmp, start_idx, end_idx, K, u, llarray, 0);
    }
    """

    arcode = """



    for (int k=0; k < N-n-n_p; ++k) {
    //for (int k=100; k < 500; ++k) {
        double f_a;
        double f_0;
        double g_b_a;
        double g_b;

        // TODO: can compute these outside of the loop if the mask is not changing. 
        for (int i=k-n_p; i < k+n+n_p; ++i) {
           if (i < 0) continue;
           tmpS(i) = 0.0;
        }    
        f_0 = reset_compute_ar(n_p, mask, tmpS, var, p, tmp, k, k+n+n_p, K, u, llarray, 0);

        for (int i=k; i < k+n; ++i) {
            tmpS(i) = c(i-k);
        }
        f_a = reset_compute_ar(n_p, mask, tmpS, var, p, tmp, k, k+n+n_p, K, u, llarray, 0);


        for (int i=k-n_p; i < k; ++i) {
           if (i < 0) continue;
           tmpS(i) = S(i);
        }    
        for (int i=k; i < k+n+n_p; ++i) {
            tmpS(i) = S(i);
        }
        g_b = reset_compute_ar(n_p, mask, tmpS, var, p, tmp, k, k+n+n_p, K, u, llarray, 0);

        for (int i=k; i < k+n; ++i) {
            tmpS(i) = S(i) - c(i-k);
        }
        g_b_a = reset_compute_ar(n_p, mask, tmpS, var, p, tmp, k, k+n+n_p, K, u, llarray, 0);

        double mnum = (g_b + f_a - g_b_a - f_0); // -(a' R^-1 (b-c))
        double mdenom = (2*(f_a - f_0)); // - a' R^-1 a;

        double beta_hat =  mnum / mdenom;

        double lp_delta = .5 * mdenom * beta_hat * beta_hat - mnum * beta_hat;

        result(k) = lp_delta;
    }

    """

    S, mask = unmask(S)
    tmpS = S.copy()
    N = len(S)
    n = len(c)

    p = np.array(nm.params).copy()
    n_p = len(p)
    tmp = np.zeros((n_p, n_p))
    K = np.eye(n_p) * 1.0e4
    u = np.zeros((n_p,))
    var = float(nm.em.std**2)
    llarray = np.zeros((1,), dtype=np.float)
    result = np.zeros((N-n),)

    weave.inline(arcode,['S', 'mask', 'c', 'N', 'n', 'result', 'n_p', 'var', 'p', 'tmpS', 'tmp', 'K', 'u', 'llarray'],type_converters = converters.blitz, verbose=2,compiler='gcc',support_code=fastar_support + local_support)
        
    return result
