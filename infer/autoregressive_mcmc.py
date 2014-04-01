import numpy as np

from sigvisa.models.distributions import Gaussian, InvGamma
from sigvisa.models.noise.armodel.model import ARModel, ErrorModel
from sigvisa.infer.latent_arrival_mcmc_c import gibbs_sweep_c
from sigvisa.infer.latent_arrival_mcmc_stupid import gibbs_sweep_python

import scipy.weave as weave
from scipy.weave import converters

def wiggle_param_step(sg, wave_node, tmnodes):
    k_latent, n_latent = tmnodes['latent_arrival']

    noise_std = n_latent.arwm.em.std
    params = n_latent.arwm.params

    proposed_std = float(np.exp(np.log(noise_std) + np.random.normal(0, 0.1, 1)))
    #proposed_params = params + np.random.normal(0, 0.1, len(params))

    lp_old = n_latent.log_p()

    old_wm = n_latent.arwm

    em = ErrorModel(mean=0.0, std=proposed_std)
    n_latent.arwm = ARModel(params=params, c=1.0, em = em, sf=n_latent.srate)

    lp_new = n_latent.log_p()

    u = np.random.uniform()
    if lp_new - lp_old > np.log(u):
        print "updated wiggle params:", proposed_std, params, lp_new-lp_old
        return True
    else:
        n_latent.arwm = old_wm
        return False

def latent_arrival_block_gibbs(sg, wave_node, tmnodes):
    k_latent, n_latent = tmnodes['latent_arrival']

    gibbs_sweep(n_latent)

    return True

def gibbs_sweep(latent, **kwargs):
    return gibbs_sweep_c(latent, **kwargs)

"""
def noise_param_step(wn):
    noise_mean = wn.nm.c
    noise_std = wn.nm.em.std
    params = wn.nm.params

    proposed_mean = float(np.exp(np.log(noise_mean) + np.random.normal(0, 0.1, 1)))
    proposed_std = float(np.exp(np.log(noise_std) + np.random.normal(0, 0.1, 1)))
    #proposed_params = params + np.random.normal(0, 0.1, len(params))

    lp_old = wn.log_p()

    old_nm = wn.nm

    em = ErrorModel(mean=0.0, std=proposed_std)
    wn.nm = ARModel(params=params, c=proposed_mean, em = em, sf=wn.srate)

    lp_new = wn.log_p()

    u = np.random.uniform()
    if lp_new - lp_old > np.log(u):
        print "updated noise params:", proposed_mean, proposed_std, params, lp_new-lp_old
        print "   previous:", noise_mean, noise_std, params
        return True
    else:
        wn.nm = old_nm
        return False
"""


def ar_param_posterior(signal, armodel, prior_mean, prior_cov):
    n = len(signal)
    n_p = len(prior_mean)
    em_var = armodel.em.std**2
    zeroed_signal = signal - armodel.c
    try:
        prior_mean[2]
    except:
        prior_mean = np.ones((n_p,)) * prior_mean
    phi_squared = np.zeros((n_p, n_p))
    params_hat = np.zeros((n_p,))
    for i in range(n_p):
        offset_i_val = np.dot(zeroed_signal[n_p-1:n-1], zeroed_signal[n_p-i-1: n-i-1])
        params_hat[i] = np.dot(zeroed_signal[n_p:n], zeroed_signal[n_p-1-i: n-1-i])
        for j in range(n_p-i):
            phi_squared[j, j+i] = offset_i_val
            phi_squared[j+i, j] = offset_i_val
    prior_precision = np.linalg.inv(prior_cov)
    C = np.linalg.inv(prior_precision * np.eye(n_p) + 1.0/em_var * phi_squared)
    c = np.dot(C, (np.dot(prior_precision, prior_mean) + params_hat/em_var))
    return c, C

def ar_var_posterior(signal, arm, prior_alpha, prior_beta):
    n = len(signal)
    n_p = arm.p
    zeroed_signal = signal - arm.c
    params = np.array(arm.params, copy=True)
    code = """
    double sum_sqerrs = 0;
    for (int i=0; i < n-n_p; ++i) {
        double pred_signal = 0;
        for (int j=0; j < n_p; ++j) {
            pred_signal += zeroed_signal(i+j) * params(n_p-j-1);
        }
        double err = zeroed_signal(i+n_p) - pred_signal;
        sum_sqerrs += err * err;
    }
    return_val = sum_sqerrs;
    """
    sum_sqerrs = weave.inline(code, ['n', 'zeroed_signal', 'n_p', 'params'],
                     type_converters=converters.blitz,
                     compiler='gcc')

    posterior_alpha = prior_alpha + (n-n_p)/2.0
    posterior_beta = sum_sqerrs/2.0
    return posterior_alpha, posterior_beta

def ar_mean_posterior(signal, arm, prior_mu, prior_sigma2):
    n = len(signal)
    n_p = arm.p
    params = np.array(arm.params, copy=True)
    mu = prior_mu
    sigma2 = prior_sigma2
    em_var  = arm.em.std**2
    sum_alpha_m1 = np.sum(params)-1
    w = float(sum_alpha_m1*sum_alpha_m1/em_var)
    p = float(sum_alpha_m1/em_var)

    returns = np.zeros((2,))
    code = """
    for (int i=n_p; i < n; ++i) {
        double err = -signal(i);
        for (int j = 0; j < n_p; ++j) {
            err += signal(i-n_p+j) * params(n_p-1-j);
        }
        double sigma2_new = sigma2 / (1.0 + sigma2*w);
        mu = sigma2_new * (mu/sigma2 + p * err);
        sigma2 = sigma2_new;
    }
    returns(0) = mu;
    returns(1) = sigma2;
    """
    weave.inline(code, ['n', 'n_p', 'params', 'signal',
                                     'w', 'p', 'mu',
                                     'sigma2', 'returns'],
                     type_converters=converters.blitz,
                     compiler='gcc')
    mu, sigma2 = returns
    return mu, sigma2

def latent_wiggle_param_gibbs(sg, wave_node, tmnodes):
    k_latent, n_latent = tmnodes['latent_arrival']

    gibbs_update_wiggle_params(n_latent)

    return True


def gibbs_update_wiggle_params(n_latent):
    prior_alpha = 1000.0
    prior_beta = 1000.0

    wiggle, shape, rw = n_latent.get_signal_components()

    posterior_alpha, posterior_beta = ar_var_posterior(wiggle, n_latent.arwm, prior_alpha, prior_beta)
    print posterior_alpha, posterior_beta
    new_var = InvGamma(posterior_alpha, posterior_beta).sample()
    print "resampled std", np.sqrt(new_var), ", previous", n_latent.arwm.em.std
    n_latent.arwm.em.std = float(np.sqrt(new_var))

    # Gibbs sample new AR params. Here the prior is a multivariate
    # Gaussian, truncated to the stationary region. We enforce this
    # through rejection sampling: sample new params from the Gaussian
    # posterior (i.e., the posterior that we'd get from a
    # non-truncated Gaussian prior) repeatedly until we happen to
    # sample a valid set of stationary params.
    prior_param_mean = np.zeros((3,))
    prior_param_cov = np.eye(3) * 0.01
    stationary = False
    max_tries = 10
    tries = 0
    orig_params = np.copy(n_latent.arwm.params)
    while not stationary and tries < max_tries:
        param_mean, param_cov = ar_param_posterior(wiggle, n_latent.arwm, prior_param_mean, prior_param_cov)
        new_params = np.random.multivariate_normal(mean=param_mean, cov=param_cov, size=1).flatten()
        n_latent.arwm.params = new_params
        n_latent.arwm.p = len(new_params)
        stationary = n_latent.arwm.stationary()
        tries += 1
    if not stationary:
        print "WARNING: Gibbs sampling new AR coefficients failed at %s; each of %d samples returned nonstationary params. Reverting to old params." % (n_latent.label, tries)
        n_latent.arwm.params = orig_params
    else:
        print "resample params", new_params, "previous", orig_params
