
from database.dataset import *
from database.signal_data import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sys, os, pickle

import utils.geog
import obspy.signal.util
from sigvisa import *
import numpy as np
import scipy.linalg
import hashlib

from learn.SpatialGP import distfns, SpatialGP, start_params, gp_extract_features
import learn.baseline_models as baseline_models
import gpr.learn
from gpr.distributions import InvGamma, LogNormal


X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI  = range(5)

def learn_model(X, y, model_type, target=None):
    if model_type.startswith("gp"):
        distfn = model_type[3:]
        params = start_params[distfn][target]
        model = learn_gp(X, y, distfn=distfn, params=params)
    elif model_type == "constant":
        model = learn_constant(X, y)
    elif model_type == "linear_distance":
        model = learn_linear(X, y)
    else:
        raise Exception("invalid model type %s" % (model_type))
    return model

def learn_gp(X, y, distfn, params, optimize=True):

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

        params, ll = gpr.learn.learn_hyperparams(sX, sy, kernel= "distfn", start_kernel_params = params, kernel_priors=priors, kernel_extra=distfns[distfn])
            
        print "got params", params , "giving ll", ll

    gp = SpatialGP(X, y, distfn_str=distfn, kernel_params=params)
    return gp

def learn_linear(X, y):
    return baseline_models.LinearModel(X,y)

def learn_constant(X,y):
    return baseline_models.ConstantModel(X,y)


def load_model(fname, model_type):
    if model_type.startswith("gp"):
        model = SpatialGP(fname=fname)
    elif model_type == "constant":
        model = baseline_models.ConstantModel(fname=fname)
    elif model_type == "linear_distance":
        model = baseline_models.LinearModel(fname=fname)
    else:
        raise Exception("invalid model type %s" % (model_type))
    return model

def analyze_model_fname(fname):
    d = dict()

    fname, d['filename'] = os.path.split(fname)
    d['evidhash'], d['model_type'] = d['filename'].split('.')
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

def get_model_fname(run_name, run_iter, sta, chan, band, phase, target, model_type, evids, model_name="paired_exp", prefix="parameters"):
    path_components = [prefix, run_name, "iter_%02d" % run_iter, "paired_exp", target, sta, phase, chan,band]
    path = os.path.join(*path_components)

    ensure_dir_exists(path)

    evidhash = hashlib.sha1(repr(evids)).hexdigest()[0:8]
    fname = ".".join([evidhash, model_type])
    return os.path.join(path, fname)

def get_training_data(run_name, run_iter, sta, chan, band, phases, target):
    s = Sigvisa()

    runid = get_fitting_runid(s.cursor, run_name, run_iter, create_if_new=False)

    print "loading %s fit data... " % (phases),
    fit_data = load_shape_data(s.cursor, chan=chan, band=band, sta=sta, runids=[runid,], phases=phases)
    print str(fit_data.shape[0]) + " entries loaded"

    if target=="decay":
        y = fit_data[:, FIT_CODA_DECAY]
    elif target=="amp_transfer":
        y = np.zeros((fit_data.shape[0],))
        for (i, fit) in enumerate(fit_data):
            phase = s.phasenames[int(fit[FIT_PHASEID])-1]
#            band = sigvisa_c.canonical_band_name(int(fit[FIT_BANDID]))
            ev = Event(evid=int(fit[FIT_EVID]))
            y[i] = fit[FIT_CODA_HEIGHT] - ev.source_logamp(band, phase)
        print y
    elif target=="onset":
        y = fit_data[:, FIT_PEAK_DELAY]
    else:
        raise KeyError("invalid target param %s" % (target))

    X = fit_data[:, [FIT_LON, FIT_LAT, FIT_DEPTH, FIT_DISTANCE, FIT_AZIMUTH]]

    evids = fit_data[:, FIT_EVID]

    return X, y, evids

def main():
    parser = OptionParser()

    s = Sigvisa()

    parser.add_option("-s", "--sites", dest="sites", default=None, type="str", help="comma-separated list of sites for which to train models")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="run_name")
    parser.add_option("--run_iter", dest="run_iter", default="latest", type="str", help="run iteration (latest)")
    parser.add_option("-c", "--channel", dest="chan", default="BHZ", type="str", help="name of channel to examine (BHZ)")
    parser.add_option("-n", "--band", dest="band", default="freq_2.0_3.0", type="str", help="name of band to examine (freq_2.0_3.0)")
    parser.add_option("-p", "--phases", dest="phases", default=",".join(s.phases), type="str", help="comma-separated list of phases for which to train models)")
    parser.add_option("-t", "--targets", dest="targets", default="decay,amp_transfer,onset", type="str", help="comma-separated list of target parameter names (decay,amp_transfer,onset)")
    parser.add_option("-m", "--model_type", dest="model_type", default="gp_dad_log", type="str", help="type of model to train (gp_dad_log)")

    (options, args) = parser.parse_args()

    sites = options.sites.split(',')
    chan = options.chan
    phases = options.phases.split(',')
    targets = options.targets.split(',')
    model_type = options.model_type
    band = options.band

    run_name = options.run_name
    if options.run_iter == "latest":
        iters = read_fitting_run_iterations(s.cursor, run_name)
        run_iter = np.max(iters)
    else:
        run_iter = int(options.run_iter)



    for site in sites:
        for target in targets:
            for phase in phases:

                X, y, evids = get_training_data(run_name, run_iter, site, chan, band, [phase,], target)

                model_fname = get_model_fname(run_name, run_iter, sta, chan, band, phase, target, model_type, evids, model_name="paired_exp")

                distfn = model_type[3:]
                model = learn_model(X, y, model_type, start_params = start_params[distfn][target])

                model.save_trained_model(model_fname)

if __name__ == "__main__":
    main()
