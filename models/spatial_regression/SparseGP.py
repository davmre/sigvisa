import os
import time
import numpy as np
import collections
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scikits.sparse.cholmod
import sklearn.preprocessing
import hashlib
import types
import marshal

from sigvisa.treegp.gp import GP, GPCov
from sigvisa.models.spatial_regression.baseline_models import ParamModel
from sigvisa.models.distributions import InvGamma, LogNormal



default_other_params = (2.0, InvGamma(beta=5.0, alpha=.1),
        GPCov([3.4,], [ 100.0, 40.0], dfn_str="lld",
              wfn_priors=[InvGamma(beta=5.0, alpha=.5),],
              dfn_priors =[LogNormal(mu=5, sigma=1.0), LogNormal(mu=3, sigma=1.0)]))

default_amp_params = (.1, InvGamma(beta=.1, alpha=1),
        GPCov([.1,], [ 100.0, 40.0], dfn_str="lld",
              wfn_priors=[InvGamma(beta=.1, alpha=1.0),],
              dfn_priors =[LogNormal(mu=5, sigma=1.0), LogNormal(mu=3, sigma=1.0)]))

default_phase_params = (1.0, InvGamma(beta=1.0, alpha=1),
        GPCov([1.0,], [ 100.0, 40.0], dfn_str="lld",
              wfn_priors=[InvGamma(beta=1.0, alpha=1.0),],
              dfn_priors =[LogNormal(mu=5, sigma=1.0), LogNormal(mu=3, sigma=1.0)]))

# wavelets should have almost no obs noise because that's already implicit in the kalman
# filter posterior uncertainty
default_wavelet_params = (.0001, InvGamma(beta=0.001, alpha=10.0),
        GPCov([0.265,], [ 11.0, 5.0], dfn_str="lld",
              wfn_priors=[InvGamma(beta=1.0, alpha=3.0),],
              wfn_str="compact2",
              dfn_priors =[LogNormal(mu=3, sigma=1.0), LogNormal(mu=3, sigma=1.0)]))


start_params_lld = {"coda_decay": default_other_params,
                    "peak_decay": default_other_params,
                    "amp_transfer": default_other_params,
                    "peak_offset": default_other_params,
                    "tt_residual": default_other_params,
                    "amp": default_amp_params,
                    "phase": default_phase_params,
                    "db4_2.0_3_30": default_wavelet_params,
                    "db4_2.0_3_20.0": default_wavelet_params,
                    }



default_other_params_lldlld = (2.0, InvGamma(beta=5.0, alpha=.1),
                               GPCov([3.4,], [ 100.0, 40.0, 100.0, 40.0], dfn_str="lldlld",
              wfn_priors=[InvGamma(beta=5.0, alpha=.5),],
              dfn_priors =[LogNormal(mu=5, sigma=1.0), LogNormal(mu=3, sigma=1.0),
                           LogNormal(mu=5, sigma=1.0), LogNormal(mu=3, sigma=1.0)]))

default_amp_params_lldlld = (.1, InvGamma(beta=.1, alpha=1),
                             GPCov([.1,], [ 100.0, 40.0, 100.0, 40.0], dfn_str="lldlld",
              wfn_priors=[InvGamma(beta=.1, alpha=1.0),],
              dfn_priors =[LogNormal(mu=5, sigma=1.0), LogNormal(mu=3, sigma=1.0),
                           LogNormal(mu=5, sigma=1.0), LogNormal(mu=3, sigma=1.0)]))

default_phase_params_lldlld = (1.0, InvGamma(beta=1.0, alpha=1),
                               GPCov([1.0,], [ 100.0, 40.0, 100.0, 40.0], dfn_str="lldlld",
              wfn_priors=[InvGamma(beta=1.0, alpha=1.0),],
              dfn_priors =[LogNormal(mu=5, sigma=1.0), LogNormal(mu=3, sigma=1.0),
                           LogNormal(mu=5, sigma=1.0), LogNormal(mu=3, sigma=1.0)]))


start_params_lldlld = {"coda_decay": default_other_params_lldlld,
                    "amp_transfer": default_other_params_lldlld,
                    "peak_offset": default_other_params_lldlld,
                    "tt_residual": default_other_params_lldlld,
                    "amp": default_amp_params_lldlld,
                    "phase": default_phase_params_lldlld,
                    }



start_params = {"lld": start_params_lld,
                "lldlld": start_params_lldlld,
                }

X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI = range(5)


def marshal_fn(f):
    if f.func_closure is not None:
        raise ValueError("function has non-empty closure %s, cannot marshal!" % repr(f.func_closure))
    s = marshal.dumps(f.func_code)
    return s

def unmarshal_fn(dumped_code):
    f_code = marshal.loads(dumped_code)
    f = types.FunctionType(f_code, globals())
    return f

X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI = range(5)


def extract_hyperparams(dfn_str, wfn_str, hyperparams, train_std=None):
    if dfn_str == "lld" and (wfn_str == "se" or wfn_str=="matern32" or wfn_str.startswith("compact")):
        if len(hyperparams) == 4:
            (noise_var, signal_var, ll_scale, d_scale) = hyperparams
        elif len(hyperparams) == 3:
            (noise_var, signal_var, ll_scale) = hyperparams
            d_scale = 5.0
        elif len(hyperparams) == 2:
            (noise_ratio, ll_scale) = hyperparams
            d_scale = 5.0
            noise_var = noise_ratio * train_std**2
            signal_var = (1-noise_ratio) * train_std**2
        elif len(hyperparams) == 1:
            ll_scale = hyperparams[0]
            d_scale = 5.0
            noise_var = train_std**2 / 2.0
            signal_var = train_std**2 / 2.0

        dfn_params = np.array((ll_scale, d_scale), dtype=np.float)
        wfn_params = np.array((signal_var,), copy=True, dtype=np.float)
    elif dfn_str == "euclidean" and (wfn_str == "se" or wfn_str=="matern32" or wfn_str.startswith("compact")):
        noise_var = hyperparams[0]
        wfn_params = np.array((hyperparams[1],), copy=True, dtype=np.float)
        dfn_params = np.array((hyperparams[2:]), dtype=np.float)
    elif dfn_str == "lldlld" and (wfn_str == "se" or wfn_str=="matern32"):
        noise_var = hyperparams[0]
        wfn_params = np.array((hyperparams[1],), copy=True, dtype=np.float)
        dfn_params = np.array((hyperparams[2:]), dtype=np.float)
    return noise_var, dfn_params, wfn_params

def sparse_kernel_from_tree(tree, X, sparse_threshold, identical, noise_var):
    max_distance = np.sqrt(-np.log(sparse_threshold)) # assuming a SE kernel
    n = len(X)
    t0 = time.time()
    entries = tree.sparse_training_kernel_matrix(X, max_distance)
    spK = scipy.sparse.coo_matrix((entries[:,2], (entries[:,0], entries[:,1])), shape=(n,n), dtype=float)
    t1 = time.time()
    print "sparse kernel", t1-t0

    if identical:
        spK = spK + noise_var * scipy.sparse.eye(spK.shape[0])
    spK = spK + 1e-8 * scipy.sparse.eye(spK.shape[0])
    return spK.tocsc()


def prior_sample(X, hyperparams, dfn_str, wfn_str, sparse_threshold=1e-20, return_kernel=False):
    n = X.shape[0]
    noise_var, dfn_params, wfn_params = extract_hyperparams(dfn_str, wfn_str, hyperparams)
    predict_tree = VectorTree(X, 1, dfn_str, dfn_params, wfn_str, wfn_params)

    spK = sparse_kernel_from_tree(predict_tree, X, sparse_threshold, True, noise_var)
    factor = scikits.sparse.cholmod.cholesky(spK)
    L = factor.L()
    P = factor.P()
    Pinv = np.argsort(P)
    z = np.random.randn(n)
    y = np.array((L * z)[Pinv]).reshape((-1,))
    if return_kernel:
        return y, spK
    else:
        return y

class SparseGP(GP, ParamModel):

    def __init__(self, *args, **kwargs):
        ParamModel.__init__(self, **kwargs)
        if 'sta' in kwargs:
            del kwargs['sta']
        GP.__init__(self, *args, **kwargs)

    def standardize_input_array(self, c, **kwargs):
        return ParamModel.standardize_input_array(self, c, **kwargs)

    def pack_npz(self):
        d = super(SparseGP, self).pack_npz()
        d['base_str'] = ParamModel.__repr_base_params__(self)
        return d

    def unpack_npz(self, npzfile):
        super(SparseGP, self).unpack_npz(npzfile)
        if len(str(npzfile['base_str'])) > 0:
            ParamModel.__unrepr_base_params__(self, str(npzfile['base_str']))
