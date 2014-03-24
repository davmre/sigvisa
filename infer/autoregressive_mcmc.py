import numpy as np

from sigvisa.models.noise.armodel.model import ARModel, ErrorModel
from sigvisa.infer.latent_arrival_mcmc_c import gibbs_sweep_c
from sigvisa.infer.latent_arrival_mcmc_stupid import gibbs_sweep_python

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
