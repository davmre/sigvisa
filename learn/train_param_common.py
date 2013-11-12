
from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *
from sigvisa.database import db

import time
import sys
import os

from sigvisa import Sigvisa
import numpy as np
import scipy.linalg
import hashlib

from sigvisa.models.spatial_regression.SparseGP import SparseGP, start_params
from sigvisa.sparsegp.gp import GP, optimize_gp_hyperparams

import sigvisa.models.spatial_regression.baseline_models as baseline_models
import sigvisa.infer.optimize.optim_utils as optim_utils


from functools32 import lru_cache
from sigvisa.learn.grad_ascend import *

X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI = range(5)

def insert_model(dbconn, fitting_runid, param, site, chan, band, phase, model_type, model_fname, training_set_fname, training_ll, require_human_approved, max_acost, n_evids, min_amp, elapsed, template_shape=None, wiggle_basisid=None, optim_method=None, hyperparams=None):
    return execute_and_return_id(dbconn, "insert into sigvisa_param_model (fitting_runid, template_shape, wiggle_basisid, param, site, chan, band, phase, model_type, model_fname, training_set_fname, n_evids, training_ll, timestamp, require_human_approved, max_acost, min_amp, elapsed, optim_method, hyperparams) values (:fr,:ts,:wbid,:param,:site,:chan,:band,:phase,:mt,:mf,:tf, :ne, :tll,:timestamp, :require_human_approved, :max_acost, :min_amp, :elapsed, :optim_method, :hyperparams)", "modelid", fr=fitting_runid, ts=template_shape, wbid=wiggle_basisid, param=param, site=site, chan=chan, band=band, phase=phase, mt=model_type, mf=model_fname, tf=training_set_fname, tll=training_ll, timestamp=time.time(), require_human_approved='t' if require_human_approved else 'f', max_acost=max_acost if np.isfinite(max_acost) else 99999999999999, ne=n_evids, min_amp=min_amp, elapsed=elapsed, optim_method=optim_method, hyperparams=hyperparams)

def model_params(model, model_type):
    if model_type.startswith('gplocal'):
        d = dict()
        d['kernel'] =model.hyperparams
        d['mean'] = model.beta_bar
        d['covar'] = np.dot(model.invc.T, model.invc)
        r = repr(d)
        if len(r) > 4000:
            del d['covar']
            r = repr(d)
        return r
    if model_type.startswith('gp'):
        return repr(model.hyperparams)
    elif model_type.startswith('param'):
        d = dict()
        d['mean'] = model.mean
        d['covar'] = np.dot(model.sqrt_covar.T, model.sqrt_covar)
        r = repr(d)
        if len(r) > 4000:
            del d['covar']
            r = repr(d)
        return r
    else:
        return None

def learn_model(X, y, model_type, sta, target=None, optim_params=None, gp_build_tree=True, array=False ):
    if model_type.startswith("gplocal"):
        s = model_type.split('+')
        kernel_str = s[1]
        basisfn_str = s[2]
        model = learn_gp(X=X, y=y, sta=sta,
                         basisfn_str=basisfn_str,
                         kernel_str=kernel_str,
                         target=target, build_tree=gp_build_tree,
                         optim_params=optim_params)
    elif model_type.startswith("gp_"):
        kernel_str = model_type[3:]
        model = learn_gp(X=X, y=y, sta=sta,
                         kernel_str=kernel_str,
                         target=target, build_tree=gp_build_tree,
                         optim_params=optim_params)
    elif model_type == "constant_gaussian":
        model = learn_constant_gaussian(sta=sta, X=X, y=y)
    elif model_type == "linear_distance":
        model = learn_linear(sta=sta, X=X, y=y)
    elif model_type.startswith('param_dist'):
        basisfn_str = model_type[6:]
        model = learn_parametric(X=X, y=y, sta=sta, basisfn_str=basisfn_str)
    else:
        raise Exception("invalid model type %s" % (model_type))
    return model

def basisfns_from_str(basisfn_str, param_var=1000):
    def distance_poly_basisfns(order):
        basisfn_strs = ["lambda x : " + ("1" if d==0 else "(x[3]/100.0)**%d" % d)   for d in range(order+1)]
        return [eval(s) for s in basisfn_strs]

    basisfns = []
    if basisfn_str.startswith('dist'):
        dist_order = int(basisfn_str[4:])
        basisfns += distance_poly_basisfns(dist_order)

    k = len(basisfns)
    b = np.zeros((k,))
    B = np.eye(k) * param_var
#    B[0,0] = (1000000)**2 # be very accomodating in estimating the constant term
    B[0,0] = (10)**2 # don't let the constant term get too big: note the constant term is the value at distance=0

    return basisfns, b, B

def learn_parametric(sta, X, y, basisfn_str, param_var=10000, optimize_marginal_ll=False):
    basisfns, b, B = basisfns_from_str(basisfn_str, param_var=param_var)
    k = len(basisfns)

    H = np.array([[f(x) for f in basisfns] for x in X], dtype=float)


    def nllgrad(std):
        try:
            lbm = baseline_models.LinearBasisModel(X=X, y=y, basisfns=basisfns, param_mean=b, param_covar=B, noise_std=std, H=H, compute_ll=True)
        except scipy.linalg.LinAlgError:
            print " warning: lin alg error for std %f" % std
            return (np.inf, np.array((-1.0,)))
        return (-lbm.ll, -lbm.ll_deriv)

    if optimize_marginal_ll:
        x0 = 10
        result = scipy.optimize.minimize(fun=nllgrad, x0=x0, jac=True,
                                         tol=.0001, method='L-BFGS-B', bounds=((1e-3, None),))
        opt_std = result['x']
        print "got opt std", opt_std
    else:
        std = 10
        lbm = baseline_models.LinearBasisModel(X=X, y=y, basisfns=basisfns, param_mean=b, param_covar=B, noise_std=std, H=H, compute_ll=False, sta=sta)
        p = lbm.predict(X)
        r = y-p
        opt_std = np.std(r)

    return baseline_models.LinearBasisModel(X=X, y=y, basisfns=basisfns, param_mean=b, param_covar=B, noise_std=opt_std, H=H, compute_ll=True, sta=sta)


def subsample_data(X, y, k=250):
    # subsample for efficient hyperparam learning
    n = len(y)
    if n > k:
        np.random.seed(0)
        perm = np.random.permutation(n)[0:k]
        sX = X[perm, :]
        sy = y[perm, :]
    else:
        sX = X
        sy = y
    return sX, sy

def build_starting_hparams(kernel_str, target):
    noise_var, noise_prior, cov_main = start_params[kernel_str][target]
    return noise_var, noise_prior, cov_main, None

def learn_gp(sta, X, y, kernel_str, basisfn_str=None, noise_var=None, noise_prior=None, cov_main=None, cov_fic=None, target=None, optimize=True, optim_params=None, param_var=100000, build_tree=True, array=False, basisfns=None, b=None, B=None, k=500, bounds=None, **kwargs):

    if basisfn_str:
        basisfns, b, B = basisfns_from_str(basisfn_str, param_var=param_var)
    elif basisfns is None:
        basisfns = ()
        b = np.array(())
        B = np.array(((),))

    try:
        st = target.split('_')
        float(st[1])
        target = st[0]
    except (ValueError, AttributeError):
        pass

    if cov_main is None:
        noise_var, noise_prior, cov_main, cov_fic = build_starting_hparams(kernel_str, target)

    if optimize:
        if k is not None:
            sX, sy = subsample_data(X=X, y=y, k=k)
        else:
            sX, sy = X, y
        print "learning hyperparams on", len(sy), "examples"
        nllgrad, x0, build_gp, covs_from_vector = optimize_gp_hyperparams(X=sX, y=sy, basisfns=basisfns, param_mean=b, param_cov=B, build_tree=False, noise_var=noise_var, noise_prior=noise_prior, cov_main=cov_main, cov_fic=cov_fic, **kwargs)

        bounds = [(1e-20,None),] * len(params) if bounds is None else bounds
        if array:
            params, ll = grad_ascend(nllgrad, precision=0.01, step=0.001, initial_guess=[100, 100, 1, 1, 1, 1])
        else:
            params, ll = optim_utils.minimize(f=nllgrad, x0=x0, optim_params=optim_params, fprime="grad_included", bounds=bounds)
        print "got params", params, "giving ll", ll
        noise_var, cov_mean, cov_fic = covs_from_vector(params)

    if len(y) > 10000:
        X, y = subsample_data(X=X, y=y, k=10000)

    gp = SparseGP(X=X, y=y, basisfns=basisfns, param_mean=b, param_cov=B, noise_var=noise_var, cov_main=cov_main, cov_fic=cov_fic, sta=sta, compute_ll=True, build_tree=build_tree,  **kwargs)
    return gp


def learn_linear(X, y, sta, optim_params=None):
    return baseline_models.LinearModel(X=X, y=y, sta=sta)


def learn_constant_gaussian(X, y, sta, optim_params=None):
    return baseline_models.ConstGaussianModel(X=X, y=y, sta=sta)

def load_modelid(modelid, **kwargs):
    s = Sigvisa()
    cursor = s.dbconn.cursor()
    cursor.execute("select model_fname, model_type from sigvisa_param_model where modelid=%d" % modelid)
    fname, model_type = cursor.fetchone()
    cursor.close()
    return load_model(fname=os.path.join(os.getenv("SIGVISA_HOME"), fname), model_type=model_type, **kwargs)

@lru_cache(maxsize=None)
def load_model(fname, model_type, gpmodel_build_trees=True):
    if model_type.startswith("gp"):
        model = SparseGP(fname=fname, build_tree=gpmodel_build_trees)
    elif model_type.startswith("param"):
        model = baseline_models.LinearBasisModel(fname=fname)
    elif model_type == "constant_gaussian":
        model = baseline_models.ConstGaussianModel(fname=fname)
    elif model_type == "linear_distance":
        model = baseline_models.LinearModel(fname=fname)
    else:
        raise Exception("invalid model type %s" % (model_type))
    return model


def analyze_model_fname(fname):
    d = dict()

    fname, d['filename'] = os.path.split(fname)
    d['evidhash'], d['model_type'] = d['filename'].split('.')[-2:]
    fname, d['band'] = os.path.split(fname)
    fname, d['chan'] = os.path.split(fname)
    fname, d['phase'] = os.path.split(fname)
    fname, d['sta'] = os.path.split(fname)
    fname, d['target'] = os.path.split(fname)
    fname, d['model_name'] = os.path.split(fname)
    fname, d['run_iter'] = os.path.split(fname)
    fname, d['run_name'] = os.path.split(fname)
    fname, d['prefix'] = os.path.split(fname)

    if d['model_name'].startswith("basis_"):
        d['basisid'] = int(d['model_name'].split('_')[1])

    return d


def get_model_fname(run_name, run_iter, sta, chan, band, phase, target, model_type, evids, basisid=None, model_name="paired_exp", prefix=os.path.join("parameters", "runs"), unique=False):
    if basisid is None:
        path_components = [prefix, run_name, "iter_%02d" % run_iter, model_name, target, sta, phase, chan, band]
    else:
        path_components = [prefix, run_name, "iter_%02d" % run_iter, 'basis_%d' % basisid, target, sta, phase, chan, band]

    path = os.path.join(*path_components)

    ensure_dir_exists(path)

    evidhash = hashlib.sha1(repr(evids)).hexdigest()[0:8]

    if unique:
        uniq_hash = hashlib.sha1(repr(time.time())).hexdigest()[0:8]
        fname = ".".join([evidhash, uniq_hash, model_type])
    else:
        fname = ".".join([evidhash, model_type])
    return os.path.join(path, fname)
