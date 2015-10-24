import numpy as np
import scipy.weave as weave
from scipy.weave import converters
import scipy.stats

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

def preprocess_historical(A, sigma2_A, tau):
    alpha_hat = np.sqrt(np.dot(A, A)/n - sigma2_A)
    c = tau*alpha_hat/(alpha_hat**2+sigma2_A) * A
    kappa = alpha_hat**2 * tau**2 / (alpha_hat**2 + sigma2_A)


def likelihood_windowed(S, c, kappa, sigma2_B):
    S, mask = unmask(S)

    N = len(S)
    n = len(c)
    result = np.zeros((N-n,))

    aux="""
    inline double model_lik(double beta, double kappa, double sigma2_B, double norm2B, double Bc, double norm2c, int nvalid) {
        double C = beta*beta*kappa + sigma2_B;
        double d = log(6.28318530718 * C);
        return -.5 * (norm2B - 2*beta*Bc + beta*beta * norm2c) / C - .5*nvalid*d;
    }

    double iid_lp(blitz::Array<double,1> S, blitz::Array<bool,1> mask, int k, int n, double sigma2_B) {
        double lp = 0;
        double c = -.5*log(6.28318530718 * sigma2_B);
        for(int i=k; i < k+n; ++i) {
           if (mask(i)) continue;
           lp += -.5 * (S(i)*S(i)/sigma2_B) + c;
        }
        return lp;
    }

    double model_lp(blitz::Array<double,1> S, blitz::Array<bool,1> mask, 
                    blitz::Array<double,1> c, int k, int n, 
                    double kappa, double sigma2_B) {
        /*
        return the likelihood of the correlation model for a latent signal c of length n, beginning at time k in S.
        That is, we return
        log(S | latent signal starts at time k)
        using a maximum likelihood estimate of the scale parameter for the latent signal. 
        */

        double norm2B = 0;
        double Bc = 0;
        double norm2c = 0;
        int nvalid = 0;
        for(int i=k; i < k+n; ++i) {
           if (mask(i)) continue;
           norm2B += S(i)*S(i);
           Bc += S(i)*c(i-k);
           norm2c += c(i-k)*c(i-k);
           nvalid += 1;
        }    

        double aa = n*kappa*kappa; 
        double ab = kappa*Bc;
        double ac = n*kappa* sigma2_B - kappa *norm2B + sigma2_B * norm2c;
        double ad = -sigma2_B*Bc;

        double d0 = ab*ab - 3*aa*ac;
        double d1 = 2*ab*ab*ab-9*aa*ab*ac+27*aa*aa*ad;
        double inside = d1*d1 - 4*d0*d0*d0;

        double base_lp = iid_lp(S, mask, k, n, sigma2_B);
        double best_lik = base_lp; // corresponding to beta=0
        double lik;
        double best_b = 0;  
        if (inside > 0) {
           // one real root
           // which should therefore be a global maximum
           double C = cbrt((d1 + sqrt(inside))/2.0);
           double b = -1.0/(3*aa) * (ab + C + d0/C);
           if (b > 0) {
              lik = model_lik(b, kappa, sigma2_B, norm2B, Bc, norm2c, nvalid);
              best_lik = lik;
           }
        } else {
           // three real roots
           double _Complex C = cpow((d1 + csqrt(inside))/2.0, 1/3.0);

           double _Complex u1 = (-1+sqrt(3.0)*I)/2.0;
           double _Complex u2 = (-1-sqrt(3.0)*I)/2.0;
           double _Complex u3 = 1;
           double b1 = creal(-1/(3*aa) * (ab + u1*C + d0/(u1*C)));
           double b2 = creal(-1/(3*aa) * (ab + u2*C + d0/(u2*C)));
           double b3 = creal(-1/(3*aa) * (ab + u3*C + d0/(u3*C)));

           if (b1 > 0) {
               lik = model_lik(b1, kappa, sigma2_B, norm2B, Bc, norm2c, nvalid);
               if (lik > best_lik) {
                  best_lik = lik;
                  best_b = b1;
               }
           }
           if (b2 > 0) {
               lik = model_lik(b2, kappa, sigma2_B, norm2B, Bc, norm2c, nvalid);
               if (lik > best_lik) {
                  best_lik = lik;
                  best_b = b2;
               }
           }
           if (b3 > 0) {
               lik = model_lik(b3, kappa, sigma2_B, norm2B, Bc, norm2c, nvalid);
               if (lik > best_lik) {
                  best_lik = lik;
                  best_b = b3;
               }
           }

       }
       return best_lik;

    }


    """


    code="""
    for(int k=0; k < N-n; ++k) {
       result(k) = model_lp(S, mask, c, k, n, kappa, sigma2_B);
    }       
    """
    weave.inline(code,['S', 'mask', 'c', 'N', 'n', 'kappa', 'sigma2_B', 'result',],type_converters = converters.blitz,
                 verbose=2,compiler='gcc', support_code=aux, headers=["<math.h>","<complex.h>"],)
    return result

def arbaseline(S, n):
    # given an array of length N, return an array of length N-n giving, for each k,
    # the likelihood of sdata[:k] and sdata[k+n:N] under an AR model estimated from 
    # all of sdata. (that is, the likelihood excluding the period of length n starting at
    # index k)
    from sigvisa.models.noise.armodel.model import ARModel, ErrorModel
    from sigvisa.models.noise.armodel.learner import ARLearner

    S, mask = unmask(S)
    N = len(S)

    l = ARLearner(S, sf=10)
    p, std = l.yulewalker(n_p)
    em = ErrorModel(0, std)
    nm = ARModel(p, em, l.c)
    llarray = np.zeros(S.shape)
    arll = nm.fastAR_missingData(S, nm.c, nm.em.std, llarray=llarray, mask=mask)
    
    arll_init = np.sum(llarray[:n])

    result = np.zeros((N-n,))
    code="""
    double arll_window = arll_init;
    for(int k=0; k < N-n; ++k) {
       double baselp = arll - arll_window;
       result(k) = baselp;
       arll_window -= llarray(k);
       if (k+n < N) {
          arll_window += llarray(k+n);
       }
    }       
    """
    weave.inline(code,[ 'N', 'n', 'result','llarray', 'arll', 'arll_init'],type_converters = converters.blitz,
                 verbose=2,compiler='gcc',)
    return result, arll

def iidbaseline(S, n, s2=None):
    # given an array of length N, return an array of length N-n giving, for each k,
    # the likelihood of sdata[:k] and sdata[k+n:N] under an AR model estimated from 
    # all of sdata. (that is, the likelihood excluding the period of length n starting at
    # index k)

    if s2 is None:
        s = np.std(S)
    else:
        s = np.sqrt(s2)

    S, mask = unmask(S)

    llarray = scipy.stats.norm(scale=s).logpdf(S)
    llarray[mask] = 0
    ll = float(np.sum(llarray))

    ll_init = float(np.sum(llarray[:n]))

    N = len(S)
    result = np.zeros((N-n,))
    code="""
    double ll_window = ll_init;
    for(int k=0; k < N-n; ++k) {
       double baselp = ll - ll_window;
       result(k) = baselp;
       ll_window -= llarray(k);
       if (k+n < N) {
          ll_window += llarray(k+n);
       }
    }       
    """
    weave.inline(code,[ 'N', 'n', 'result','llarray', 'll', 'll_init'],type_converters = converters.blitz,
                 verbose=2,compiler='gcc',)
    return result, ll
