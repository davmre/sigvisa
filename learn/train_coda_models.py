
from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *
from sigvisa.database import db

import time
import sys
import os
import pickle

from sigvisa import *
import numpy as np

from optparse import OptionParser

from sigvisa.learn.train_param_common import insert_model, learn_model, load_model, get_model_fname
from sigvisa.infer.optimize.optim_utils import construct_optim_params

def get_shape_training_data(run_name, run_iter, site, chan, band, phases, target, require_human_approved=False, max_acost=200, min_amp=-10, **kwargs):
    s = Sigvisa()
    cursor = s.dbconn.cursor()

    runid = get_fitting_runid(cursor, run_name, run_iter, create_if_new=False)

    try:
        print "loading %s fit data... " % (phases),
        fit_data = load_shape_data(cursor, chan=chan, band=band, site=site, runids=[runid, ], phases=phases,
                                   require_human_approved=require_human_approved, max_acost=max_acost, min_amp=min_amp, **kwargs)
        print str(fit_data.shape[0]) + " entries loaded"
    except:
        raise NoDataException()

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
        "--max_acost", dest="max_acost", default=np.float('inf'), type=float, help="maximum fitting cost of fits in training set (inf)")
    parser.add_option("--min_amp", dest="min_amp", default=1, type=float,
                      help="only consider fits above the given amplitude (does not apply to amp_transfer fits)")
    parser.add_option("--min_amp_for_at", dest="min_amp_for_at", default=-5, type=float,
                      help="only consider fits above the given amplitude (for amp_transfer fits)")
    parser.add_option("--enable_dupes", dest="enable_dupes", default=False, action="store_true",
                      help="train models even if a model of the same type already appears in the DB")
    parser.add_option("--optim_params", dest="optim_params", default="'method': 'bfgs_fastcoord', 'normalize': False, 'disp': True, 'bfgs_factr': 1e10, 'random_inits': 2", type="str", help="fitting param string")

    (options, args) = parser.parse_args()

    sites = options.sites.split(',')
    chan = options.chan
    phases = options.phases.split(',')
    targets = options.targets.split(',')
    model_type = options.model_type
    band = options.band

    optim_params = construct_optim_params(options.optim_params)

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


                # check for duplicate model
                sql_query = "select modelid from sigvisa_param_model where model_type='%s' and site='%s' and chan='%s' and band='%s' and phase='%s' and fitting_runid=%d and param='%s' " % (
                    model_type, site, chan, band, phase, runid, target)
                cursor.execute(sql_query)
                dups = cursor.fetchall()
                if len(dups) > 0 and not options.enable_dupes:
                    print "model already trained for %s, %s, %s (modelid %d), skipping..." % (site, target, phase, dups[0][0])
                    continue



                try:
                    X, y, evids = get_shape_training_data(run_name=run_name, run_iter=run_iter, site=site,
                                                    chan=chan, band=band, phases=[phase, ], target=target,
                                                    require_human_approved=options.require_human_approved,
                                                    max_acost=options.max_acost, min_amp=min_amp)
                except NoDataException:
                    print "no data for %s %s %s, skipping..." % (site, target, phase)
                    continue

                model_fname = get_model_fname(
                    run_name, run_iter, site, chan, band, phase, target, model_type, evids, model_name=options.template_shape, unique=True)
                evid_fname = os.path.splitext(os.path.splitext(model_fname)[0])[0] + '.evids'
                np.savetxt(evid_fname, evids, fmt='%d')

                distfn = model_type[3:]
                st = time.time()
                model = learn_model(X, y, model_type, target=target, sta=site, optim_params=optim_params)

                et = time.time()

                if np.isnan(model.log_likelihood()):
                    print "error training model for %s %s %s, likelihood is nan! skipping.." % (site, target, phase)
                    continue

                model.save_trained_model(model_fname)
                modelid = insert_model(
                    s.dbconn, fitting_runid=runid, template_shape=options.template_shape, param=target, site=site, chan=chan, band=band, phase=phase, model_type=model_type, model_fname=model_fname, training_set_fname=evid_fname, training_ll=model.log_likelihood(
                    ),
                    require_human_approved=options.require_human_approved, max_acost=options.max_acost, n_evids=len(evids), min_amp=min_amp, elapsed=(et-st), hyperparams = repr(model.kernel_params) if model_type.startswith('gp') else None, optim_method = repr(optim_params) if model_type.startswith('gp') else None)
                print "inserted as", modelid, "ll", model.log_likelihood()

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        import sys, traceback, pdb
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
