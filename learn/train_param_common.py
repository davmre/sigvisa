
from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *
from sigvisa.database import db

import sys
import os
import pickle

import sigvisa.utils.geog
import obspy.signal.util
from sigvisa import *
import numpy as np
import scipy.linalg
import hashlib

from sigvisa.models.spatial_regression.SpatialGP import distfns, SpatialGP, start_params, gp_extract_features
import sigvisa.models.spatial_regression.baseline_models as baseline_models
import sigvisa.gpr.learn
from sigvisa.gpr.distributions import InvGamma, LogNormal

X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI = range(5)




def insert_model(dbconn, fitting_runid, param, site, chan, band, phase, model_type, model_fname, training_set_fname, training_ll, require_human_approved, max_acost, n_evids, min_amp, elapsed, template_shape=None, wiggle_basisid=None):
    return execute_and_return_id(dbconn, "insert into sigvisa_param_model (fitting_runid, template_shape, wiggle_basisid, param, site, chan, band, phase, model_type, model_fname, training_set_fname, n_evids, training_ll, timestamp, require_human_approved, max_acost, min_amp, elapsed) values (:fr,:ts,:wbid,:param,:site,:chan,:band,:phase,:mt,:mf,:tf, :ne, :tll,:timestamp, :require_human_approved, :max_acost, :min_amp, :elapsed)", "modelid", fr=fitting_runid, ts=template_shape, wbid=wiggle_basisid, param=param, site=site, chan=chan, band=band, phase=phase, mt=model_type, mf=model_fname, tf=training_set_fname, tll=training_ll, timestamp=time.time(), require_human_approved='t' if require_human_approved else 'f', max_acost=max_acost if np.isfinite(max_acost) else 99999999999999, ne=n_evids, min_amp=min_amp, elapsed=elapsed)


def learn_model(X, y, model_type, sta, target=None):
    if model_type.startswith("gp"):
        distfn = model_type[3:]
        params = start_params[distfn][target]
        model = learn_gp(X=X, y=y, sta=sta, distfn=distfn, params=params)
    elif model_type == "constant_gaussian":
        model = learn_constant_gaussian(sta=sta, X=X, y=y)
    elif model_type == "linear_distance":
        model = learn_linear(sta=sta, X=X, y=y)
    else:
        raise Exception("invalid model type %s" % (model_type))
    return model


def learn_gp(sta, X, y, distfn, params, optimize=True):

    X = gp_extract_features(X, distfn)

    if optimize:
        priors = [None for p in params]

        # subsample for efficient hyperparam learning
        n = len(y)
        if n > 250:
            np.random.seed(0)
            perm = np.random.permutation(n)[0:250]
            sX = X[perm, :]
            sy = y[perm, :]
        else:
            sX = X
            sy = y
        print "learning hyperparams on", len(sy), "examples"

        params, ll = gpr.learn.learn_hyperparams(
            sX, sy, kernel="distfn", start_kernel_params=params, kernel_priors=priors, kernel_extra=distfns[distfn])

        print "got params", params, "giving ll", ll

    gp = SpatialGP(X=X, y=y, sta=sta, distfn_str=distfn, kernel_params=params)
    return gp


def learn_linear(X, y, sta):
    return baseline_models.LinearModel(X=X, y=y, sta=sta)


def learn_constant_gaussian(X, y, sta):
    return baseline_models.ConstGaussianModel(X=X, y=y, sta=sta)


def load_model(fname, model_type):
    if model_type.startswith("gp"):
        model = SpatialGP(fname=fname)
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


def get_model_fname(run_name, run_iter, sta, chan, band, phase, target, model_type, evids, basisid=None, model_name="paired_exp", prefix=os.path.join("parameters", "runs")):
    if basisid is None:
        path_components = [prefix, run_name, "iter_%02d" % run_iter, model_name, target, sta, phase, chan, band]
    else:
        path_components = [prefix, run_name, "iter_%02d" % run_iter, 'basis_%d' % basisid, target, sta, phase, chan, band]

    path = os.path.join(*path_components)

    ensure_dir_exists(path)

    evidhash = hashlib.sha1(repr(evids)).hexdigest()[0:8]
    fname = ".".join([evidhash, model_type])
    return os.path.join(path, fname)