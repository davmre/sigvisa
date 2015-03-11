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

def ar_param_posterior(signal, signal_vars, armodel, prior_mean, prior_cov):
    # warning: this method has accumulated hacks, I don't think I have a good
    # probabilistic story for what it's doing, though hopefully one exists.

    n = len(signal)
    n_p = len(prior_mean)
    em_var = armodel.em.std**2
    # HACK: rescale each signal by the observation noise stddev, ignoring correlations
    zeroed_signal = (signal - armodel.c)/np.sqrt(signal_vars + em_var)
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
    C = np.linalg.inv(prior_precision + phi_squared)
    c = np.dot(C, (np.dot(prior_precision, prior_mean) + params_hat))
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
    posterior_beta = prior_beta + sum_sqerrs/2.0
    return posterior_alpha, posterior_beta

def ar_mean_posterior(signal, signal_vars, arm, prior_mu, prior_sigma2):
    n = len(signal)
    n_p = arm.p
    params = np.array(arm.params, copy=True)
    mu = prior_mu
    sigma2 = prior_sigma2
    em_var  = float(arm.em.std**2)
    sum_alpha_m1 = float(np.sum(params)-1)

    returns = np.zeros((2,))
    code = """
    for (int i=n_p; i < n; ++i) {
        double err = -signal(i);
        double obs_v = em_var + signal_vars(i);
        double w = sum_alpha_m1*sum_alpha_m1/obs_v;
        double p = sum_alpha_m1/obs_v;

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
    weave.inline(code, ['n', 'n_p', 'params', 'signal', 'signal_vars', 'em_var',
                                     'sum_alpha_m1', 'mu',
                                     'sigma2', 'returns'],
                     type_converters=converters.blitz,
                     compiler='gcc')
    mu, sigma2 = returns
    return mu, sigma2


def sample_ar_params_from_truncated_gaussian(param_mean, param_cov, arm, max_tries=10):
    stationary = False
    tries = 0
    while not stationary and tries < max_tries:
        new_params = np.random.multivariate_normal(mean=param_mean, cov=param_cov, size=1).flatten()
        arm.params = new_params
        stationary = arm.stationary()
        tries += 1
    if not stationary:
        raise Exception("Gibbs sampling new AR coefficients failed")
    return new_params


def sample_posterior_armodel_from_signal(signal_mean, signal_var, arm_prior, arm_old=None):
    arm = arm_prior.copy()

    # sample a new process mean
    prior_mu = arm.c
    prior_sigma2 = prior_mu/100.0
    posterior_mu, posterior_s2 =  ar_mean_posterior(signal_mean, signal_var, arm, prior_mu, prior_sigma2)
    c_dist = Gaussian(posterior_mu, std=np.sqrt(posterior_s2))
    new_c = c_dist.sample()
    arm.c = float(new_c)


    # sample a new noise variance (IGNORING signal obs variance because I
    # haven't worked out how to handle it...)
    prior_alpha = 100
    prior_mean = arm.em.std**2
    prior_beta = prior_mean * (prior_alpha-1)
    posterior_alpha, posterior_beta = ar_var_posterior(signal_mean, arm, prior_alpha, prior_beta)
    var_dist = InvGamma(posterior_alpha, posterior_beta)
    new_var = var_dist.sample()
    arm.em.std = float(np.sqrt(new_var))

    # sample new params
    prior_param_mean = arm.params
    prior_param_cov = np.eye(3) * 0.01
    param_mean, param_cov = ar_param_posterior(signal_mean, signal_var, arm, prior_param_mean, prior_param_cov)
    new_params = sample_ar_params_from_truncated_gaussian(param_mean, param_cov, arm)
    arm.params = new_params


    #lp_qforward =
    #lp_qbackward = 0.0

    return arm

def arnoise_gibbs_move(sg, wave_node):

    means = wave_node.signal_component_means()
    noise_mean = means['noise']
    noise_var = wave_node.signal_component_means(return_stds_instead=True)['noise']**2

    arm = sample_posterior_armodel_from_signal(noise_mean, noise_var, wave_node.prior_nm)

    # TODO: should really do a MH step here because the noise model we sample
    # isn't really a Gibbs move.
    # (though need to be careful: the prior on noise models is not currently
    # part of the graph logp. really the noise model should be its owave_node
    # separate node...)

    #lp1 = wave_node.log_p()
    wave_node.set_noise_model(arm)
    wave_node.cached_logp = None
    #lp2 = wave_node.log_p()

    return True
