import numpy as np

from sigvisa.models.distributions import Gaussian, InvGamma, MultiGaussian, TruncatedGaussian
from sigvisa.models.noise.armodel.model import ARModel, ErrorModel
from sigvisa.infer.mcmc_basic import mh_accept_util

import scipy.weave as weave
from scipy.weave import converters


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

"""
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

def sample_from_posterior(nm, c_dist, var_dist, param_mean, param_cov):
    c = c_dist.sample()
    v = var_dist.sample()
    nm.c = c
    nm.em.std = np.sqrt(v)
    new_params = sample_ar_params_from_truncated_gaussian(param_mean, param_cov, nm)
    nm.params = new_params
    return nm

def posterior_logp(nm, c_dist, var_dist, param_mean, param_cov):
    lp = c_dist.log_p(nm.c)
    lp += var_dist.log_p(nm.em.std**2)

    # this is not quite right because of truncation
    lp += MultiGaussian(param_mean, param_cov).log_p(nm.params)
    return lp

def posterior_armodel_from_signal(signal_mean, signal_var, nm_node):
    arm = nm_node.get_value().copy()

    n_p = len(arm.params)

    # sample a new process mean
    prior_mu, prior_sigma2 = nm_node.prior_mean_dist.mean, nm_node.prior_mean_dist.var
    posterior_mu, posterior_s2 =  ar_mean_posterior(signal_mean, signal_var, arm, prior_mu, prior_sigma2)
    c_dist = Gaussian(posterior_mu, std=np.sqrt(posterior_s2))


    # sample a new noise variance (IGNORING signal obs variance because I
    # haven't worked out how to handle it...)
    prior_alpha, prior_beta = nm_node.prior_var_dist.alpha, nm_node.prior_var_dist.beta
    posterior_alpha, posterior_beta = ar_var_posterior(signal_mean, arm, prior_alpha, prior_beta)
    var_dist = InvGamma(posterior_alpha, posterior_beta)

    # sample new params
    prior_param_mean, prior_param_cov = nm_node.prior_param_dist.mean, nm_node.prior_param_dist.cov
    param_mean, param_cov = ar_param_posterior(signal_mean, signal_var, arm, prior_param_mean, prior_param_cov)

    return c_dist, var_dist, param_mean, param_cov
"""

def arnoise_mean_rw_move(sg, wn, std=None):
    if std is None:
        std = np.sqrt(wn.nm_node.prior_mean_dist.variance())/4.0

    nm1 = wn.nm_node.get_value()
    oldvals = (nm1)
    relevant_nodes = (wn.nm_node, wn)
    lp_old =  sg.joint_logprob_keys(relevant_nodes)

    if wn.is_env:
        proposal_dist = TruncatedGaussian(nm1.c, std=std, a=0.0)
        proposed = proposal_dist.sample()
        log_qforward = proposal_dist.log_p(proposed)

        reverse_dist = TruncatedGaussian(proposed, std=std, a=0.0)
        log_qbackward = reverse_dist.log_p(nm1.c)
    else:
        proposed = nm1.c + np.random.randn() * std
        log_qforward = 0.0
        log_qbackward = 0.0

    nm2 = nm1.copy()
    nm2.c = proposed
    wn.nm_node.set_value(nm2)
    lp_new =  sg.joint_logprob_keys(relevant_nodes)

    def revert():
        wn.nm_node.set_value(nm1)


    v = mh_accept_util(lp_old, lp_new, log_qforward=log_qforward, log_qbackward=log_qbackward, revert_move=revert)
    #if "TX01" in wn.label:
    #    if v:
    #        print "TX01 accepted move from", nm1.c, "to", nm2.c
    #    else:
    #        print "TX01 rejected move from", nm1.c, "to", nm2.c

    return v

def arnoise_std_rw_move(sg, wn, std=None):
    if std is None:
        std = np.sqrt(wn.nm_node.prior_var_dist.predict())/10.0

    nm1 = wn.nm_node.get_value()
    oldvals = (nm1)
    relevant_nodes = (wn.nm_node, wn)
    lp_old = sg.joint_logprob_keys(relevant_nodes)


    proposal_dist = TruncatedGaussian(nm1.em.std, std=std, a=0.0)
    proposed = proposal_dist.sample()
    log_qforward = proposal_dist.log_p(proposed)

    reverse_dist = TruncatedGaussian(proposed, std=std, a=0.0)
    log_qbackward = reverse_dist.log_p(nm1.em.std)

    nm2 = nm1.copy()
    nm2.em.std = proposed

    wn.nm_node.set_value(nm2)
    lp_new = sg.joint_logprob_keys(relevant_nodes)

    def revert():
        wn.nm_node.set_value(nm1)

    return mh_accept_util(lp_old, lp_new, log_qforward=log_qforward, log_qbackward=log_qbackward, revert_move=revert)


def arnoise_params_rw_move(sg, wn, std=0.01):
    if std is None:
        std = np.array(wn.nm_node.prior_nm.params) / 5.0
    n_p = len(wn.nm_node.prior_nm.params)

    nm1 = wn.nm_node.get_value()
    oldvals = (nm1)
    relevant_nodes = (wn.nm_node, wn)
    lp_old =  sg.joint_logprob_keys(relevant_nodes)

    nm2 = nm1.copy()
    nm2.params += np.random.randn(n_p) * std

    wn.nm_node.set_value(nm2)
    lp_new =  sg.joint_logprob_keys(relevant_nodes)

    def revert():
        wn.nm_node.set_value(nm1)

    return mh_accept_util(lp_old, lp_new, revert_move=revert)


"""
def arnoise_mean_move(sg, wn):
    means = wn.signal_component_means()
    noise_mean = means['noise']
    noise_var = wn.signal_component_means(return_stds_instead=True)['noise']**2

    nm1 = wn.nm_node.get_value()
    oldvals = (nm1)
    relevant_nodes = (wn.nm_node, wn)
    lp_old = np.sum([n.log_p() for n in relevant_nodes])

    prior_mu, prior_sigma2 = nm_node.prior_mean_dist.mean, nm_node.prior_mean_dist.var
    posterior_mu, posterior_s2 =  ar_mean_posterior(signal_mean, signal_var, arm, prior_mu, prior_sigma2)
    c_dist = Gaussian(posterior_mu, std=np.sqrt(posterior_s2))
    nm2 = nm1.copy()
    nm2.c = c_dist.sample()
    log_qforward = c_dist.log_p(nm2)

    wn.nm_node.set_value(nm2)
    lp_new = np.sum([n.log_p() for n in relevant_nodes])

    means = wn.signal_component_means()
    noise_mean = means['noise']
    noise_var = wn.signal_component_means(return_stds_instead=True)['noise']**2
    c_dist2, var_dist2, param_mean2, param_cov2 = posterior_armodel_from_signal(noise_mean, noise_var, wn.nm_node)
    log_qbackward = posterior_logp(nm1, c_dist2, var_dist2, param_mean2, param_cov2)

    def revert():
        wn.nm_node.set_value(nm1)

    return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, revert_move=revert)

def arnoise_mh_move(sg, wn):
    means = wn.signal_component_means()
    noise_mean = means['noise']
    noise_var = wn.signal_component_means(return_stds_instead=True)['noise']**2

    nm1 = wn.nm_node.get_value()
    oldvals = (nm1)
    relevant_nodes = (wn.nm_node, wn)
    lp_old = np.sum([n.log_p() for n in relevant_nodes])

    c_dist, var_dist, param_mean, param_cov = posterior_armodel_from_signal(noise_mean, noise_var, wn.nm_node)
    nm2 = sample_from_posterior(nm1.copy(), c_dist, var_dist, param_mean, param_cov)
    log_qforward = posterior_logp(nm2, c_dist, var_dist, param_mean, param_cov)

    wn.nm_node.set_value(nm2)
    lp_new = np.sum([n.log_p() for n in relevant_nodes])

    means = wn.signal_component_means()
    noise_mean = means['noise']
    noise_var = wn.signal_component_means(return_stds_instead=True)['noise']**2
    c_dist2, var_dist2, param_mean2, param_cov2 = posterior_armodel_from_signal(noise_mean, noise_var, wn.nm_node)
    log_qbackward = posterior_logp(nm1, c_dist2, var_dist2, param_mean2, param_cov2)

    def revert():
        wn.nm_node.set_value(nm1)

    import pdb; pdb.set_trace()
    return mh_accept_util(lp_old, lp_new, log_qforward, log_qbackward, revert_move=revert)
"""

"""
def arnoise_gibbs_move(sg, wn):

    means = wn.signal_component_means()
    noise_mean = means['noise']
    noise_var = wn.signal_component_means(return_stds_instead=True)['noise']**2
    
    c_dist, var_dist, param_mean, param_cov = posterior_armodel_from_signal(noise_mean, noise_var, wn.nm_node)

    #lp1 = wave_node.log_p()
    wn.set_noise_model(arm)
    wn.cached_logp = None
    #lp2 = wave_node.log_p()

    return True
"""
