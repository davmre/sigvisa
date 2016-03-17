from sigvisa.database import db
from sigvisa.database.signal_data import get_fitting_runid, read_fitting_run_iterations, NoDataException

import time
import sys
import os
import pickle

from sigvisa import Sigvisa
import numpy as np

from optparse import OptionParser

from sigvisa.models.ttime import tt_predict
from sigvisa.learn.train_param_common import insert_model, learn_model, load_model, get_model_fname, model_params, build_starting_hparams
from sigvisa.learn.train_coda_models import chan_for_site, load_site_data, check_duplicates
from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.models.wiggles.wavelets import construct_full_basis_implicit, wavelet_idx_to_level
from sigvisa.models.spatial_regression.local_gp_ensemble import LocalGPEnsemble, optimize_localgp_hyperparams
import sigvisa.infer.optimize.optim_utils as optim_utils

import scipy.optimize

def wiggle_params_by_level(wiggle_srate, wiggle_family):
    
    start_times, end_times, identities, prototypes, level_sizes, N = construct_full_basis_implicit(wiggle_srate, wiggle_family)

    params_by_level = []
    nlevels = len(level_sizes)
    i = 0
    for level in range(nlevels):
        params_by_level.append(["%s_%d" % (wiggle_family, i+k) for k in range(level_sizes[level])])
        i += level_sizes[level]


    # reverse level order: the param order specifies toplevel
    # coefficients first, but I consider "level 0" to be the lowest
    # level and "level 3" (or whatever) to be the highest level
    params_by_level = params_by_level[::-1]

    return params_by_level

def load_wiggle_data_for_level(level, level_params, runid, site, 
                               chan, band, phase, min_amp, filter_evids=None):

    Xs = []
    ys = []
    yvarss = []
    evidss = []
    for param in level_params:
        try:
            X, y, yvars, evids = load_site_data([site,], target=param, runid=runid, chan=chan, band=band, phases=[phase, ], min_amp=min_amp)

            if filter_evids is not None:
                good_evids = np.array([(evid in filter_evids) for evid in evids])
                X = X[good_evids]
                y = y[good_evids]
                yvars = yvars[good_evids]
                evids = evids[good_evids]



        except NoDataException:
            if param == level_params[0]:
                print "no data for %s %s level %d, skipping..." % (site, phase, level)
                return None, None, None, None
            else:
                raise Exception("wtf? we have data for some wiggle params at a decomposition level but not others...")

        # avoid numerical issues from really uncertain observations of large values
        y[yvars > 1e15] = 0.0


        Xs.append(X)
        ys.append(y)
        yvarss.append(yvars)
        evidss.append(evids)

        if len(evidss) > 1:
            assert (np.all(evids == evidss[0]))

        if len(Xs) > 1:
            assert ( np.all(X == Xs[0]))

    return Xs[0], ys, yvarss, evidss[0]

def train_tied_hparam_models(X, y_list, yvars_list, sta, 
                             wiggle_family,
                             optim_params, param_var,
                             cluster_centers_fname):

    featurizer_recovery = None
    extract_dim = None

    cluster_centers = np.array(np.matrix(np.loadtxt(cluster_centers_fname)))

    kernel_str="lld"
    noise_var, noise_prior, cov_main, cov_fic = build_starting_hparams(kernel_str, wiggle_family)



    nllgrad, x0, build_gps, covs_from_vector = \
        optimize_localgp_hyperparams(X=X, y_list=y_list, yvars_list=yvars_list, 
                                     noise_prior=noise_prior, sta=sta,
                                     force_unit_var=True,
                                     cov_main=cov_main, cluster_centers=cluster_centers)


    #r = scipy.optimize.minimize(nllgrad, x0, jac=True, method="bfgs", options={"disp": True})
    #ll = r.fun
    #params = r.x

    params, ll = optim_utils.minimize(f=nllgrad, x0=x0, optim_params=optim_params, fprime="grad_included")



    print "got params", params, "giving ll", ll

    gps = build_gps(params)
    return gps

def main():
    parser = OptionParser()

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    parser.add_option(
        "-s", "--sites", dest="sites", default=None, type="str", help="comma-separated list of sites for which to train models")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="run_name")
    parser.add_option("--run_iter", dest="run_iter", default="latest", type="str", help="run iteration (latest)")
    parser.add_option("-c", "--channel", dest="chan", default="vertical", type="str", help="name of channel to examine (vertical)")
    parser.add_option(
        "-n", "--band", dest="band", default="freq_0.8_4.5", type="str", help="name of band to examine (freq_2.0_3.0)")
    parser.add_option("-p", "--phases", dest="phases", default=",".join(s.phases), type="str",
                      help="comma-separated list of phases for which to train models)")
    parser.add_option("--min_amp", dest="min_amp", default=-3, type=float,
                      help="only consider fits above the given amplitude (does not apply to amp_transfer fits)")
    parser.add_option("--optim_params", dest="optim_params", default="'method': 'bfgs', 'normalize': True, 'disp': True, 'bfgs_factr': 1e8", type="str", help="fitting param string")
    parser.add_option("--param_var", dest="param_var", default=100.0, type="float",
                      help="variance for the Gaussian prior on global model params (100.0)")
    parser.add_option("--wiggle_family", dest="wiggle_family", default=None, type="str",
                      help="wiggle family for which to trian models")
    parser.add_option("--wiggle_srate", dest="wiggle_srate", default=None, type="float",
                      help="sampling rate of signals for which to train latent wiggle models (only if preset=db...)")
    parser.add_option("--enable_dupes", dest="enable_dupes", default=False, action="store_true",
                      help="train models even if a model of the same type already appears in the DB")
    parser.add_option("--centers", dest="cluster_centers_fname", default=None, type="str",
                      help="file (np.savetxt) with list of cluster centers for local GPs")
    parser.add_option("--filter_evids", dest="filter_evids_fname", default=None, type="str",
                      help="file (np.savetxt) with list of evids to restrict training to")

    (options, args) = parser.parse_args()

    phases = options.phases.split(',')
    band = options.band
    wiggle_family  = options.wiggle_family

    # hardcoded since this is the only sensible model for wiggles...
    model_type="gplocal+lld+none"

    optim_params = construct_optim_params(options.optim_params)

    run_name = options.run_name
    if options.run_iter == "latest":
        iters = read_fitting_run_iterations(cursor, run_name)
        run_iter = np.max(iters[:, 0])
    else:
        run_iter = int(options.run_iter)
    runid = get_fitting_runid(cursor, run_name, run_iter, create_if_new=False)


    # if we're not modeling arrays as joint, replace each array with
    # its default station.

    sites = options.sites.split(",")
    sites = [s.get_default_sta(site) for site in sites]

    sitechans = {}
    for site in sites:
        try:
            sitechans[site] = chan_for_site(site, options.chan)
        except KeyError as e:
            print "skipping site %s:" % site, e
            continue

    params_by_level = wiggle_params_by_level(options.wiggle_srate, wiggle_family)


    if options.filter_evids_fname is not None:
        filter_evids = np.loadtxt(options.filter_evids_fname)
        filter_evids = set([int(evid) for evid in filter_evids])
    else:
        filter_evids = None

    for site in sites:
        chan = sitechans[site]
        for phase in phases:
            for level, level_params in enumerate(params_by_level):

                dupe_modelid = check_duplicates(cursor, model_type, site, 
                                                chan, band, phase, runid, 
                                                target=level_params[0])
                if dupe_modelid is not None and not options.enable_dupes:
                    print "model already trained for %s, %s, %s (modelid %d), skipping..." % (site, "level %d" % level, phase, dupe_modelid)
                    continue

                
                X, ys, yvars, evids = load_wiggle_data_for_level(level, level_params, runid, 
                                                                 site, chan, band, phase, options.min_amp, filter_evids=filter_evids)
                if X is None:
                    continue

                st = time.time()
                models = train_tied_hparam_models(X=X, y_list=ys, yvars_list=yvars, sta=site, 
                                                  wiggle_family = wiggle_family,
                                                  optim_params=optim_params, param_var=options.param_var,
                                                  cluster_centers_fname=options.cluster_centers_fname)
                et = time.time()

                for (model, target) in zip(models, level_params):

                    model_fname = get_model_fname(run_name, run_iter, site, chan, band, phase, target, 
                                                  model_type, evids, model_name="", unique=True)
                    evid_fname = os.path.splitext(os.path.splitext(os.path.splitext(model_fname)[0])[0])[0] + '.evids'
                    np.savetxt(evid_fname, evids, fmt='%d')

                    model.save_trained_model(model_fname)

                    modelid = insert_model(s.dbconn, fitting_runid=runid, param=target, site=site, chan=chan, 
                                           band=band, phase=phase, model_type=model_type, 
                                           model_fname=model_fname, 
                                           training_set_fname=evid_fname, 
                                           training_ll=model.log_likelihood(), 
                                           template_shape="lin_polyexp",
                                           max_acost = 1e20,
                                           require_human_approved=False, 
                                           n_evids=len(evids), 
                                           min_amp=options.min_amp, 
                                           elapsed=(et-st), 
                                           hyperparams = model_params(model, model_type), 
                                           optim_method = repr(optim_params), 
                                           shrinkage=0)
                    print "inserted as", modelid, "for", target


if __name__ == "__main__":
    main()
