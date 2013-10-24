import os
import sys
import numpy as np
import scipy.io
import time

from sigvisa.models.spatial_regression.SparseGP import SparseGP, sparsegp_nll_ngrad
from sigvisa.learn.train_param_common import subsample_data
from sigvisa.infer.optimize.optim_utils import minimize, construct_optim_params

def test_predict(sdir, sgp=None):
    sgp = SparseGP(fname=os.path.join(sdir, "trained.gp"), build_tree=False) if sgp is None else sgp
    testX = np.load(os.path.join(sdir, "testX.npy"))
    testy = np.load(os.path.join(sdir, "testy.npy"))
    pred_y = sgp.predict_naive(testX)
    r = pred_y - testy

    meanTest = np.mean(testy)
    varTest = np.var(testy)
    mse = np.mean(r **2)
    smse = mse/(varTest+meanTest**2)
    mean_ad = np.mean(np.abs(r))
    median_ad = np.median(np.abs(r))

    with open(os.path.join(sdir, "accuracy.txt"), "w") as f:
        f.write("mse: %f\n" % mse)
        f.write("smse: %f\n" % smse)
        f.write("mean_ad: %f\n" % mean_ad)
        f.write("median_ad: %f\n" % median_ad)

def learn_hyperparams(sdir, X, y, hyperparams, dfn_str="euclidean", wfn_str="se", k=500, **kwargs):

    sX, sy = subsample_data(X=X, y=y, k=k)
    print "learning hyperparams on", len(sy), "examples"
    kwargs['build_tree'] = False
    llgrad = lambda p : sparsegp_nll_ngrad(X=sX, y=sy, hyperparams=p, dfn_str=dfn_str, wfn_str=wfn_str, **kwargs)

    bounds = [(1e-20,None),] * len(hyperparams)
    optim_params = construct_optim_params("'disp': True, 'normalize': False, 'bfgs_factor': 1e10")
    params, ll = minimize(f=llgrad, x0=hyperparams, optim_params=optim_params, fprime="grad_included", bounds=bounds)
    return params


def train_realdata_gp(sdir, X, y, hyperparams, dfn_str="euclidean", wfn_str="se", **kwargs):

    hyperparams = np.array(hyperparams, copy=True, dtype=float, order="C")

    sgp = SparseGP(X=X, y=y, hyperparams=hyperparams, dfn_str=dfn_str, wfn_str=wfn_str, build_tree=False, sparse_threshold=1e-8, **kwargs)
    sgp.save_trained_model(os.path.join(sdir, "trained.gp"))
    np.save("K.npy", sgp.K)
    return sgp
