from sigvisa.models.wiggles.wavelets import construct_wavelet_basis, parse_wavelet_basis_str, construct_implicit_basis, implicit_basis_to_C_format

import autograd.numpy as np
from autograd import grad

import scipy.optimize
import os
from sigvisa import Sigvisa

def implicit_to_explicit(start_times, end_times, identities, prototypes, N):
    m = len(start_times)
    basis = []
    for i in range(m):
        st = max(start_times[i], 0)
        et = min(end_times[i], N)
        npts= et-st
        offset = st-start_times[i]

        # this is kinda awkward but makes autograd happy
        thisbasis = np.concatenate([np.zeros((st,)), prototypes[identities[i]][offset:offset+npts], np.zeros((N-et))])

        basis.append(thisbasis)
    basis = np.array(basis)
    return basis

def optimize_for_uniform_variance(start_times, end_times, identities, prototypes, N, target_var=1.0, stationary=True):
    """
    Given a wavelet basis, find the minimal perturbation such that
    - the marginal signal variances are uniform over time
    - (if stationary=True) the local covariances between signal elements are uniform over time

    THIS IS A HACK because truncated Daubechies bases (i.e., taking the full basis and
    lopping off the most time-localized components) do not have uniform marginal variances,
    and that has weird effects on the probability model, incentivizing alignments that put
    the moments of highest marginal variance on top of larger signal perturbations.
    """

    n_prototypes = len(prototypes)
    plens = [len(p) for p in prototypes]
    pstarts = np.concatenate([np.array((0,)), np.cumsum(plens)[:-1]])
    def unflatten(x):
        scale = x[-1]
        prototypes = [x[pstarts[i]:pstarts[i]+plens[i]]*scale for i in range(n_prototypes)]
        return prototypes

    def flatten(prototypes, scale):
        return np.concatenate([np.concatenate(prototypes), np.array((scale,))])

    x0 = flatten(prototypes, 1.0)

    def obj(x):
        prototypes = unflatten(x)
        basis = implicit_to_explicit(start_times, end_times, identities, prototypes, N)

        m = np.dot(basis.T, basis)
        sqr = np.sum((np.diag(m)-1.0)**2)+ 0.1*np.sum((x[:-1]-x0[:-1])**2)
        if stationary:
            v1 = m[2, 0:5]
            v2 = m[3, 1:6]
            v3 = m[4, 2:7]
            v4 = m[5, 3:8]
            sqr += np.sum((v1-v2)**2)+ np.sum((v4-v3)**2)

        return sqr

    gradobj = grad(obj)
    r = scipy.optimize.minimize(lambda x : (obj(x), gradobj(x)), x0, jac=True)
    prototypes_new = unflatten(r.x)

    return prototypes_new

def uvar_wavelet_basis(srate, wavelet_str):
    wavelet_family, wavelet_resolution_hz, decomp_levels, wiggle_len_s = parse_wavelet_basis_str(wavelet_str)

    opt=False
    stationary=False
    if wavelet_family.endswith("uvars"):
        wf = wavelet_family[:-5]
        opt=True
        stationary=True
    elif wavelet_family.endswith("uvar"):
        wf = wavelet_family[:-4]
        opt=True
    else:
        wf = wavelet_family

    ws = "%s_%.2f_%d_%.2f" % (wf, wavelet_resolution_hz, decomp_levels, wiggle_len_s)
    start_times, end_times, identities, prototypes, N = construct_implicit_basis(srate, ws)

    start_times, end_times, identities, M, N = implicit_basis_to_C_format((start_times, end_times, identities, prototypes, N))

    # get an M matrix representating a basis with marginal signal variance 1.0
    if opt:
        s = Sigvisa()
        prototype_cache = os.path.join(s.homedir, "cache", "wavelets", "%.1f_%s%s.npy" % (srate, ws, "_sta" if stationary else ""))
        if os.path.exists(prototype_cache):
            M = np.load(prototype_cache)
        else:
            pnew = optimize_for_uniform_variance(start_times, end_times, identities, prototypes, N, stationary=stationary)
            start_times, end_times, identities, M, N = implicit_basis_to_C_format((start_times, end_times, identities, pnew, N))
            np.save(prototype_cache, M)
    else:
        basis = construct_wavelet_basis(srate, ws)
        mean_std = np.mean(np.sqrt(np.diag(np.dot(basis.T, basis))))
        M /= mean_std

    return start_times, end_times, identities, M, N
