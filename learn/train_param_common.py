
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

import cPickle as pickle

from sigvisa.models.spatial_regression.SparseGP import SparseGP, start_params
from sigvisa.models.spatial_regression.local_gp_ensemble import LocalGPEnsemble, optimize_localgp_hyperparams, load_lgp_ensemble
from sigvisa.models.distributions import LogNormal, Gaussian, Laplacian
from sigvisa.treegp.gp import GP, optimize_gp_hyperparams

import sigvisa.models.spatial_regression.baseline_models as baseline_models
from sigvisa.models.spatial_regression.linear_basis import LinearBasisModel
from sigvisa.treegp.features import ortho_poly_fit, featurizer_from_string, recover_featurizer
import sigvisa.infer.optimize.optim_utils as optim_utils


from functools32 import lru_cache
from sigvisa.learn.grad_ascend import *

X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI = range(5)

def insert_model(dbconn, fitting_runid, param, site, chan, band, phase, model_type, model_fname, training_set_fname, training_ll, require_human_approved, max_acost, n_evids, min_amp, elapsed, template_shape=None, wiggle_basisid=None, optim_method=None, hyperparams=None, shrinkage=None, shrinkage_iter=0):
    return execute_and_return_id(dbconn, "insert into sigvisa_param_model (fitting_runid, template_shape, wiggle_basisid, param, site, chan, band, phase, model_type, model_fname, training_set_fname, n_evids, training_ll, timestamp, require_human_approved, max_acost, min_amp, elapsed, optim_method, hyperparams, shrinkage, shrinkage_iter) values (:fr,:ts,:wbid,:param,:site,:chan,:band,:phase,:mt,:mf,:tf, :ne, :tll,:timestamp, :require_human_approved, :max_acost, :min_amp, :elapsed, :optim_method, :hyperparams, :shrinkage, :shrinkage_iter)", "modelid", fr=fitting_runid, ts=template_shape, wbid=wiggle_basisid, param=param, site=site, chan=chan, band=band, phase=phase, mt=model_type, mf=model_fname, tf=training_set_fname, tll=training_ll, timestamp=time.time(), require_human_approved='t' if require_human_approved else 'f', max_acost=max_acost if np.isfinite(max_acost) else 99999999999999, ne=n_evids, min_amp=min_amp, elapsed=elapsed, optim_method=optim_method, hyperparams=hyperparams, shrinkage=shrinkage, shrinkage_iter=shrinkage_iter)

def model_params(model, model_type):
    if model_type.startswith('gpparam'):
        d = dict()
        d['cov_main'] =model.cov_main
        d['cov_fic'] =model.cov_fic
        d['noise_var'] =model.noise_var
        d['mean'] = model.beta_bar
        d['covar'] = np.dot(model.invc.T, model.invc)
        r = repr(d)
        if len(r) > 4000:
            del d['covar']
            r = repr(d)
        return r
    if model_type.startswith('gplocal'):
        d = {}
        try:
            d['mean'] = model.param_mean()
            d['covar'] = model.param_covariance()
        except:
            pass
        
        noises = np.array([lgp.noise_var for lgp in model.local_gps])
        svars = np.array([lgp.cov_main.wfn_params[0] for lgp in model.local_gps])
        lscales = np.array([lgp.cov_main.dfn_params[0] for lgp in model.local_gps])

        d["noise_min"]=np.min(noises)
        d["noise_mean"]=np.mean(noises)
        d["noise_max"]=np.max(noises)

        d["svar_min"]=np.min(svars)
        d["svar_mean"]=np.mean(svars)
        d["svar_max"]=np.max(svars)

        d["lscale_min"]=np.min(lscales)
        d["lscale_mean"]=np.mean(lscales)
        d["lscale_max"]=np.max(lscales)

        return repr(d)
    elif model_type.startswith('gp'):
        return str([model.noise_var, model.cov_main, model.cov_fic])
    elif model_type.startswith('param'):
        d = dict()
        d['mean'] = model.mean
        d['covar'] = np.dot(model.sqrt_covar.T, model.sqrt_covar)
        d['noise_var'] = model.noise_var
        r = repr(d)
        if len(r) > 4000:
            del d['covar']
            r = repr(d)
        return r
    else:
        return None

def learn_model(X, y, model_type, sta, yvars=None, target=None, optim_params=None, gp_build_tree=True, k=500, bounds=None, optimize=True, cluster_centers_fname=None, remove_outliers=False, **kwargs):
    if model_type.startswith("gpparam"):
        s = model_type.split('+')
        if "param_var" in kwargs:
            del kwargs['param_var']
        kernel_str = s[1]
        basisfn_str = s[2]
        model = learn_gp(X=X, y=y, y_obs_variances=yvars, sta=sta,
                         basisfn_str=basisfn_str,
                         kernel_str=kernel_str,
                         target=target, build_tree=gp_build_tree,
                         optim_params=optim_params, k=k,
                         bounds=bounds, optimize=optimize, **kwargs)
    if model_type.startswith("gplocal"):
        s = model_type.split('+')

        kernel_str = s[1]
        basisfn_str = None if s[2]=="none" else s[2]
        model = learn_localgps(X=X, y=y, y_obs_variances=yvars, sta=sta,
                               basisfn_str=basisfn_str,
                               kernel_str=kernel_str,
                               target=target, 
                               optim_params=optim_params, 
                               cluster_centers_fname=cluster_centers_fname, 
                               **kwargs)
    elif model_type.startswith("gp_"):
        kernel_str = model_type[3:]
        if "param_var" in kwargs:
            del kwargs['param_var']
        model = learn_gp(X=X, y=y, y_obs_variances=yvars, sta=sta,
                         kernel_str=kernel_str,
                         target=target, build_tree=gp_build_tree,
                         optim_params=optim_params, k=k,
                         bounds=bounds, optimize=optimize, **kwargs)
    elif model_type == "constant_gaussian":
        model = learn_constant_gaussian(sta=sta, X=X, y=y, yvars=yvars, **kwargs)
    elif model_type == "constant_laplacian":
        noise_prior = LogNormal(0.5, 1.0)
        if remove_outliers:
            model = iterative_fit_laplace_outliers(X=X, ymeans=y, yvars=yvars, sta=sta,
                                                   noise_prior=noise_prior, **kwargs)
        else:
            model = learn_constant_laplacian(sta=sta, X=X, y=y, yvars=yvars, 
                                             noise_prior=noise_prior, **kwargs)

    elif model_type == "constant_beta":
        model = learn_constant_beta(sta=sta, X=X, y=y, yvars=yvars, **kwargs)
    elif model_type.startswith('param_'):
        basisfn_str = model_type[6:]

        generic_noise_prior = LogNormal(0.5, 0.5)        
        if target=="amp_transfer":
            noise_prior = LogNormal(-1, 0.3)
        elif "decay" in target:
            noise_prior = LogNormal(-2, 0.3)
        else:
            noise_prior = generic_noise_prior


        if remove_outliers:
            model = iterative_fit_parametric_outliers(X=X, ymeans=y, yvars=yvars, sta=sta,
                                                      optimize_marginal_ll=optimize,
                                                      noise_prior = noise_prior,
                                                      basisfn_str=basisfn_str, **kwargs)
        else:
            model = learn_parametric(X=X, y=y, yvars=yvars, sta=sta,
                                     optimize_marginal_ll=optimize,
                                     noise_prior = noise_prior,
                                     basisfn_str=basisfn_str, **kwargs)
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

def pre_featurizer(basisfn_str):
    # make sure we use the same set of orthogonal polynomials for
    # each station model, so we can compare their parameters.
    if basisfn_str.startswith("poly"):
        degree = int(basisfn_str[4:])
        fakeX = np.reshape(np.linspace(0, 10000, 100), (-1, 1))
        Z, norm2, alpha = ortho_poly_fit(fakeX, degree)
        featurizer_recovery = {'norm2': norm2, 'alpha': alpha, 'extract_dim': 3}
        extract_dim=None
    elif basisfn_str == "linear_mb":
        basisfn_str = "mlinear"
        featurizer_recovery = {'means': np.array((3.9,)), 'scales': np.array((.5,)), 'extract_dim': (4,)}
        extract_dim=None
    elif basisfn_str == "linear_distmb":
        basisfn_str = "mlinear"
        featurizer_recovery = {'means': np.array((3500,3.9,)), 'scales': np.array((1000,.5,)), 'extract_dim': (3,4)}
        extract_dim=None
    elif basisfn_str == "bias":
        basisfn_str = "mlinear"
        featurizer_recovery = {'means': np.array(()), 'scales': np.array(()), 'extract_dim': ()}
        extract_dim = None
    else:
        featurizer_recovery = None
        extract_dim=3
    return basisfn_str, featurizer_recovery, extract_dim


def iterative_fit_laplace_outliers(X, ymeans, yvars, **kwargs):

    def score_laplace_points(cl, X, ymeans, yvars):
        uniform_lp = -np.log(np.max(ymeans) - np.min(ymeans))

        lp1s = []
        for (x, ym, yv) in zip(X, ymeans, yvars):
            x = x.reshape((1, -1))
            lp1 = cl.log_p(ym)
            lp1s.append(lp1)

        lp1s = np.array(lp1s).flatten()
        diffs = lp1s-uniform_lp
        return diffs
    
    idx_good, = np.where(yvars < 9.0)
        
    cl = learn_constant_laplacian(X[idx_good], ymeans[idx_good], **kwargs)
    
    idx_good_old = idx_good
    for i in range(10):  
        diffs = score_laplace_points(cl, X, ymeans, yvars)

        idx_good, = np.where( (diffs > -3) * (yvars < 9.0) )
        idx_bad, = np.where(diffs <= -3)
        if len(idx_good) == len(idx_good_old) and (idx_good==idx_good_old).all():
            break

        #plt.figure()
        #sns.distplot(ymeans[idx_good])
        #sns.distplot(ymeans[idx_bad])
        
        cl = learn_constant_laplacian(X[idx_good], ymeans[idx_good], **kwargs)
        
        #x = np.linspace(np.min(ymeans), np.max(ymeans))
        #lps = [cl.log_p(xx) for xx in x]
        #plt.plot(x, np.exp(lps))
        
        idx_good_old = idx_good
    return cl

def iterative_fit_parametric_outliers(X, ymeans, yvars, **kwargs):

    def score_all_points(lbm, X, ymeans, yvars):
        baseline = Gaussian(np.mean(ymeans), std=np.std(ymeans))
        lp1s = []
        lp2s = []
        for (x, ym, yv) in zip(X, ymeans, yvars):
            x = x.reshape((1, -1))
            m = lbm.predict(x)
            v = lbm.variance(x, include_obs=True)
            residual = Gaussian(m, std=np.sqrt(v+yv))
            lp1 = residual.log_p(ym)
            lp2 = baseline.log_p(ym)
            lp1s.append(lp1)
            lp2s.append(lp2)

        lp1s = np.array(lp1s).flatten()
        lp2s = np.array(lp2s).flatten()
        diffs = lp1s-lp2s
        return diffs

    lbm = learn_parametric(X=X, y=ymeans, yvars=yvars, **kwargs)
    idx_bad_old = []
    for i in range(10):  
        diffs = score_all_points(lbm, X, ymeans, yvars)

        idx_good, = np.where(diffs > -2.5)
        idx_bad, = np.where(diffs <= -2.5)
        if len(idx_bad) == len(idx_bad_old) and (idx_bad==idx_bad_old).all():
            break

        #plt.figure()
        #plt.scatter(dists[idx_good], ymeans[idx_good], color="blue", alpha=0.4)
        #plt.scatter(dists[idx_bad], ymeans[idx_bad], color="red", alpha=0.4)

        yvar_delta = np.zeros(yvars.shape)
        yvar_delta[idx_bad] = 100.0
        lbm = learn_parametric(X=X, y=ymeans, yvars=yvars+yvar_delta, **kwargs)
        idx_bad_old = idx_bad
    return lbm

def learn_parametric(sta, X, y, basisfn_str, noise_prior=None, optimize_marginal_ll=True, **kwargs):

    basisfn_str, featurizer_recovery, extract_dim = pre_featurizer(basisfn_str)

    def nllgrad(log_std):
        std = np.exp(log_std)
        var = std**2

        if var < 1e-100:
            return (np.inf, np.array((-1.0,)))

        try:
            lbm = LinearBasisModel(X=X, y=y, basis=basisfn_str, noise_std=std, compute_ll=True, extract_dim=extract_dim, featurizer_recovery=featurizer_recovery, **kwargs)
        except scipy.linalg.LinAlgError:
            print " warning: lin alg error for std %f" % std
            return (np.inf, np.array((-1.0,)))
        ll = lbm.ll
        ll_deriv = lbm.ll_deriv * std # df/d_exp(logstd) * d_exp(logstd)/d_logstd
        
        if noise_prior is not None:
            ll += noise_prior.log_p(var)
            prior_deriv_var = noise_prior.deriv_log_p(var)
            prior_deriv_std = prior_deriv_var * 2 * std # df/dvar * dvar/dstd
            prior_deriv_logstd = prior_deriv_std * std
            ll_deriv += prior_deriv_logstd

        return (-ll, -ll_deriv)


    if optimize_marginal_ll:
        x0 = 0.0
        result = scipy.optimize.minimize(fun=nllgrad, x0=x0, jac=True,
                                         tol=.0001)
        logstd = result['x']
        std = np.exp(logstd)

        print "got opt std", std
    else:
        std = 1.0
        lbm = LinearBasisModel(X=X, y=y, basis=basisfn_str, noise_std=std, compute_ll=False, sta=sta, extract_dim=3, featurizer_recovery=featurizer_recovery, **kwargs)
        p = lbm.predict(X)
        r = y-p
        std = np.std(r)
        if std == 0:
            std = 1.0

    return LinearBasisModel(X=X, y=y, basis=basisfn_str, noise_std=std, compute_ll=True, sta=sta, extract_dim=3, featurizer_recovery=featurizer_recovery, **kwargs)


def subsample_data(X, y, yvars=None, k=250):
    # subsample for efficient hyperparam learning
    n = len(y)
    if n > k:
        np.random.seed(0)
        perm = np.random.permutation(n)[0:k]
        sX = X[perm]
        sy = y[perm]
        syvars = yvars[perm] if yvars is not None else yvars
    else:
        sX = X
        sy = y
        syvars = yvars

    
    return sX, sy, yvars

def build_starting_hparams(kernel_str, target):
    noise_var, noise_prior, cov_main = start_params[kernel_str][target]
    return noise_var, noise_prior, cov_main, None

def learn_gp(sta, X, y, y_obs_variances, kernel_str, basisfn_str=None, noise_var=None, noise_prior=None, cov_main=None, cov_fic=None, target=None, optimize=True, optim_params=None, build_tree=True, array=False, k=500, bounds=None, max_n=10000, **kwargs):

    try:
        st = target.split('_')
        int(st[-1])
        target = "_".join(st[:-1])
    except (ValueError, AttributeError):
        pass


    distance_idx = None
    if kernel_str == "lld":
        distance_idx = 3
    elif kernel_str == "lldlld":
        distance_idx = 6
    if basisfn_str is not None:
        basisfn_str, featurizer_recovery, extract_dim = pre_featurizer(basisfn_str)
        kwargs["basis"] = basisfn_str
        kwargs["extract_dim"] = extract_dim
        kwargs["featurizer_recovery"] = featurizer_recovery
    else:
        featurizer_recovery = None
        extract_dim = None

    if cov_main is None:
        noise_var, noise_prior, cov_main, cov_fic = build_starting_hparams(kernel_str, target)

    if optimize:
        if k is not None:
            sX, sy, syvars = subsample_data(X=X, y=y, yvars=y_obs_variances ,k=k)
        else:
            sX, sy = X, y
            syvars = y_obs_variances
        print "learning hyperparams on", len(sy), "examples"
        nllgrad, x0, bounds, build_gp, covs_from_vector = optimize_gp_hyperparams(X=sX, y=sy, y_obs_variances=syvars, build_tree=False, noise_var=noise_var, noise_prior=noise_prior, cov_main=cov_main, cov_fic=cov_fic, **kwargs)

        #bounds = [(1e-2,50.0,),(1e-2, 50.0)] + [(1e-2,1000.0),] * (len(x0) -2) if bounds is None else bounds
        params, ll = optim_utils.minimize(f=nllgrad, x0=x0, optim_params=optim_params, fprime="grad_included", bounds=bounds)
        print "got params", params, "giving ll", ll
        noise_var, cov_main, cov_fic = covs_from_vector(params)

    if len(y) > max_n:
        X, y, y_obs_variances = subsample_data(X=X, y=y, yvars=y_obs_variances, k=max_n)


    gp = SparseGP(X=X, y=y, y_obs_variances=y_obs_variances, noise_var=noise_var, cov_main=cov_main, cov_fic=cov_fic, sta=sta, compute_ll=True, build_tree=build_tree,  **kwargs)
    return gp


def learn_localgps(sta, X, y, y_obs_variances, kernel_str, basisfn_str=None, 
                   noise_prior=None, cov_main=None, target=None, optim_params=None, 
                   cluster_centers_fname=None, param_var=100, **kwargs):

    try:
        st = target.split('_')
        int(st[-1])
        target = "_".join(st[:-1])
    except (ValueError, AttributeError):
        pass

    if basisfn_str is not None:
        basisfn_str, featurizer_recovery, extract_dim = pre_featurizer(basisfn_str)

        kwargs["basis"] = basisfn_str
        kwargs["extract_dim"] = extract_dim
        kwargs["featurizer_recovery"] = featurizer_recovery

        if featurizer_recovery is None:
            if basisfn_str is not None:
                H, _, _ = featurizer_from_string(X[:1], basisfn_str, extract_dim=extract_dim, transpose=True)
        else:
            f, _ = recover_featurizer(basisfn_str, featurizer_recovery, transpose=True)
            H = f(X[:1])
        nparams = H.shape[0]

        kwargs["prior_mean"] = np.zeros((nparams,))
        kwargs["prior_cov"] = np.eye(nparams)*param_var

    else:
        featurizer_recovery = None
        extract_dim = None

    cluster_centers = np.loadtxt(cluster_centers_fname).reshape((-1, 3))

    if cov_main is None:
        noise_var, noise_prior, cov_main, cov_fic = build_starting_hparams(kernel_str, target)


    nllgrad, x0, build_gp, covs_from_vector = optimize_localgp_hyperparams(X=X, y=y, 
                y_obs_variances=y_obs_variances, noise_prior=noise_prior, 
                cov_main=cov_main, cluster_centers=cluster_centers, 
                **kwargs)

    params, ll = optim_utils.minimize(f=nllgrad, x0=x0, optim_params=optim_params, fprime="grad_included")
    print "got params", params, "giving ll", ll
    cluster_covs, cluster_noise_vars = covs_from_vector(np.exp(params))

    gp = LocalGPEnsemble(X=X, y=y, y_obs_variances=y_obs_variances, 
                         cluster_centers=cluster_centers,
                         cluster_noise_vars=cluster_noise_vars,
                         cluster_covs=cluster_covs,
                         sta=sta,  **kwargs)
    return gp



def learn_linear(X, y, sta, optim_params=None):
    return baseline_models.LinearModel(X=X, y=y, sta=sta)


def learn_constant_gaussian(X, y, sta, optimize_marginal_ll=True, optim_params=None, noise_prior=None, loc_prior=None, **kwargs):

    # technically, as long as the loc prior is Gaussian and noise (variance) prior is InvGamma,
    # we have conjugate priors and should be able to find the posterior in closed form.
    # but doing the optimization is easy, and I'm lazy.
    def nll(x):
        mean, var = x
        if var <= 0:
            return np.float('inf')

        g = baseline_models.ConstGaussianModel(X=X, y=y, mean=mean, std=np.sqrt(var))
        ll = g.ll

        if noise_prior is not None:
            ll += noise_prior.log_p(var)
        if loc_prior is not None:
            ll += loc_prior.log_p(mean)
        return -ll
    if optimize_marginal_ll:
        x0 = [0, 1]
        result = scipy.optimize.fmin(nll, x0=x0)
        mean, var = result
        std = np.sqrt(var)
        print "got optimal gaussian", mean,std
    else:
        mean=None
        std = None

    return baseline_models.ConstGaussianModel(X=X, y=y, sta=sta, mean=mean, std=std)

def learn_constant_laplacian(X, y, sta, optimize_marginal_ll=True, optim_params=None, noise_prior=None, loc_prior=None, **kwargs):

    def nll(x):
        center, scale_sq = x
        if scale_sq <= 0:
            return np.float('inf')
        g = baseline_models.ConstLaplacianModel(X=X, y=y, center=center, scale=np.sqrt(scale_sq))
        ll = g.ll
        if noise_prior is not None:
            ll += noise_prior.log_p(scale_sq)
        if loc_prior is not None:
            ll += loc_prior.log_p(center)
        return -ll

    if optimize_marginal_ll:
        x0 = [0, 1]
        result = scipy.optimize.fmin(nll, x0=x0)
        center, scale_sq = result
        scale = np.sqrt(scale_sq)
        print "got optimal laplacian", center, scale
    else:
        center=None
        scale = None

    return baseline_models.ConstLaplacianModel(X=X, y=y, sta=sta, center=center, scale=scale)

def learn_constant_beta(X, y, sta, **kwargs):
    return baseline_models.ConstBetaModel(X=X, y=y, sta=sta)


modelid_cache = {}
def load_modelid(modelid, memoize=True, **kwargs):
    s = Sigvisa()
    fname, model_type = s.sql("select model_fname, model_type from sigvisa_param_model where modelid=%d" % modelid)[0]

    if modelid not in modelid_cache:
        model = load_model_notmemoized(fname=os.path.join(os.getenv("SIGVISA_HOME"), fname), model_type=model_type, **kwargs)
        model.modelid = modelid
        modelid_cache[modelid] = model

    return modelid_cache[modelid]

def load_modelid_evids(modelid):
    s = Sigvisa()
    fname, = s.sql("select training_set_fname from sigvisa_param_model where modelid=%d" % modelid)[0]
    evids = np.loadtxt(os.path.join(s.homedir, fname), dtype=int)
    return evids

@lru_cache(maxsize=2048)
def load_model(*args, **kwargs):
    return load_model_notmemoized(*args, **kwargs)

def load_model_notmemoized(fname, model_type, gpmodel_build_trees=False):
    if fname.startswith("parameters"):
        fname = os.path.join(Sigvisa().homedir, fname)
    
    if model_type.startswith("gplocal"):
        model = load_lgp_ensemble(fname)
        
    elif model_type.startswith("gp"):
        model = SparseGP(fname=fname, build_tree=gpmodel_build_trees)
    elif model_type.startswith("param"):
        model = LinearBasisModel(fname=fname)
    elif model_type == "constant_gaussian":
        model = baseline_models.ConstGaussianModel(fname=fname)
    elif model_type == "constant_laplacian":
        model = baseline_models.ConstLaplacianModel(fname=fname)
    elif model_type == "constant_beta":
        model = baseline_models.ConstBetaModel(fname=fname)
    else:
        try:
            with open(fname, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            raise Exception("unrecognized model type %s" % (model_type))
    return model


def analyze_model_fname(fname):
    d = dict()

    fname, d['filename'] = os.path.split(fname)
    d['evidhash'], d['model_type'], d['shrinkage_iter'] = d['filename'].split('.')[-3:]
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


def get_model_fname(run_name, run_iter, sta, chan, band, phase, target, model_type, evids, basisid=None, shrinkage_iter=0, model_name="paired_exp", prefix=os.path.join("parameters", "runs"), unique=False):
    if basisid is None:
        path_components = [prefix, run_name, "iter_%02d" % run_iter, model_name, target, sta, phase, chan, band]
    else:
        path_components = [prefix, run_name, "iter_%02d" % run_iter, 'basis_%d' % basisid, target, sta, phase, chan, band]

    path = os.path.join(*path_components)

    ensure_dir_exists(path)

    evidhash = hashlib.sha1(repr(evids)).hexdigest()[0:8]

    if unique:
        uniq_hash = hashlib.sha1(repr(time.time())).hexdigest()[0:8]
        fname = ".".join([evidhash, uniq_hash, model_type, str(shrinkage_iter)])
    else:
        fname = ".".join([evidhash, model_type, str(shrinkage_iter)])
    return os.path.join(path, fname)
