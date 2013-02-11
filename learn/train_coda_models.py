
from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *
from sigvisa.database import db

import matplotlib.pyplot as plt


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
import gpr.learn
from gpr.distributions import InvGamma, LogNormal
from optparse import OptionParser

X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI = range(5)


class NoDataException(Exception):
    pass


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

    return d


def get_model_fname(run_name, run_iter, sta, chan, band, phase, target, model_type, evids, model_name="paired_exp", prefix=os.path.join("parameters", "runs")):
    path_components = [prefix, run_name, "iter_%02d" % run_iter, "paired_exp", target, sta, phase, chan, band]
    path = os.path.join(*path_components)

    ensure_dir_exists(path)

    evidhash = hashlib.sha1(repr(evids)).hexdigest()[0:8]
    fname = ".".join([evidhash, model_type])
    return os.path.join(path, fname)


def get_training_data(run_name, run_iter, sta, chan, band, phases, target, require_human_approved=False, max_acost=200, min_amp=-10, **kwargs):
    s = Sigvisa()
    cursor = s.dbconn.cursor()

    runid = get_fitting_runid(cursor, run_name, run_iter, create_if_new=False)

    print "loading %s fit data... " % (phases),
    fit_data = load_shape_data(cursor, chan=chan, band=band, sta=sta, runids=[runid, ], phases=phases,
                               require_human_approved=require_human_approved, max_acost=max_acost, min_amp=min_amp, **kwargs)
    print str(fit_data.shape[0]) + " entries loaded"

    try:
        if target == "coda_decay":
            y = fit_data[:, FIT_CODA_DECAY]
        elif target == "amp_transfer":
            y = fit_data[:, FIT_AMP_TRANSFER]
        elif target == "coda_height":
            y = fit_data[:, FIT_CODA_HEIGHT]
        elif target == "peak_offset":
            y = fit_data[:, FIT_PEAK_DELAY]
        else:
            raise KeyError("invalid target param %s" % (target))
    except IndexError as e:
        raise NoDataException()

    X = fit_data[:, [FIT_LON, FIT_LAT, FIT_DEPTH, FIT_DISTANCE, FIT_AZIMUTH]]

    evids = fit_data[:, FIT_EVID]

    return X, y, evids


def main():
    parser = OptionParser()

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    parser.add_option(
        "-s", "--sites", dest="sites", default=None, type="str", help="comma-separated list of sites for which to train models")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="run_name")
    parser.add_option("--run_iter", dest="run_iter", default="latest", type="str", help="run iteration (latest)")
    parser.add_option("-c", "--channel", dest="chan", default="BHZ", type="str", help="name of channel to examine (BHZ)")
    parser.add_option(
        "-n", "--band", dest="band", default="freq_2.0_3.0", type="str", help="name of band to examine (freq_2.0_3.0)")
    parser.add_option("-p", "--phases", dest="phases", default=",".join(s.phases), type="str",
                      help="comma-separated list of phases for which to train models)")
    parser.add_option("-t", "--targets", dest="targets", default="coda_decay,amp_transfer,peak_offset", type="str",
                      help="comma-separated list of target parameter names (coda_decay,amp_transfer,peak_offset)")
    parser.add_option("--template_shape", dest="template_shape", default="paired_exp", type="str", help="")
    parser.add_option(
        "-m", "--model_type", dest="model_type", default="gp_dad_log", type="str", help="type of model to train (gp_dad_log)")
    parser.add_option("--require_human_approved", dest="require_human_approved", default=False, action="store_true",
                      help="only train on human-approved good fits")
    parser.add_option(
        "--max_acost", dest="max_acost", default=200, type=float, help="maximum fitting cost of fits in training set (200)")
    parser.add_option("--min_amp", dest="min_amp", default=1, type=float,
                      help="only consider fits above the given amplitude (does not apply to amp_transfer fits)")
    parser.add_option("--min_amp_for_at", dest="min_amp_for_at", default=-5, type=float,
                      help="only consider fits above the given amplitude (for amp_transfer fits)")

    (options, args) = parser.parse_args()

    sites = options.sites.split(',')
    chan = options.chan
    phases = options.phases.split(',')
    targets = options.targets.split(',')
    model_type = options.model_type
    band = options.band

    run_name = options.run_name
    if options.run_iter == "latest":
        iters = read_fitting_run_iterations(cursor, run_name)
        run_iter = np.max(iters[:, 0])
    else:
        run_iter = int(options.run_iter)

    runid = get_fitting_runid(cursor, run_name, run_iter, create_if_new=False)

    for site in sites:
        for target in targets:

            if target == "amp_transfer":
                min_amp = options.min_amp_for_at
            else:
                min_amp = options.min_amp

            for phase in phases:

                try:
                    X, y, evids = get_training_data(run_name, run_iter, site, chan, band, [phase, ], target,
                                                    require_human_approved=options.require_human_approved, max_acost=options.max_acost, min_amp=min_amp)
                except NoDataException:
                    print "no data for %s %s %s, skipping..." % (site, target, phase)
                    continue

                model_fname = get_model_fname(
                    run_name, run_iter, site, chan, band, phase, target, model_type, evids, model_name=options.template_shape)
                evid_fname = os.path.splitext(model_fname)[0] + '.evids'
                np.savetxt(evid_fname, evids, fmt='%d')

                distfn = model_type[3:]
                model = learn_model(X, y, model_type, target=target, sta=site)

                if np.isnan(model.log_likelihood()):
                    print "error training model for %s %s %s, likelihood is nan! skipping.." % (site, target, phase)
                    continue

                model.save_trained_model(model_fname)
                modelid = insert_model(
                    s.dbconn, fitting_runid=runid, template_shape=options.template_shape, param=target, site=site, chan=chan, band=band, phase=phase, model_type=model_type, model_fname=model_fname, training_set_fname=evid_fname, training_ll=model.log_likelihood(
                    ),
                    require_human_approved=options.require_human_approved, max_acost=options.max_acost, n_evids=len(evids), min_amp=min_amp)
                print "inserted as", modelid, "ll", model.log_likelihood()

if __name__ == "__main__":
    main()
