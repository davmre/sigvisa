
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

from sigvisa.models.ttime import tt_predict
from sigvisa.learn.train_param_common import insert_model, learn_model, load_model, get_model_fname, model_params
from sigvisa.learn.train_global_param_models import retrain_models
from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.models.wiggles import load_wiggle_generator

def get_wiggle_training_data(runid, wg, target_num, array=False, **kwargs):
    s = Sigvisa()
    cursor = s.dbconn.cursor()

    wiggle_data, sta_data = load_wiggle_data(cursor, runids=[runid, ], basisid=wg.basisid, **kwargs)
    print str(wiggle_data.shape[0]) + " entries loaded"

    try:
        y = wiggle_data[:, WIGGLE_PARAM0 + target_num]
    except IndexError as e:
        print "nd2"
        raise NoDataException()

    if array:
        sta_pos = np.empty((0, 3))
        for i in range(len(sta_data)):
            sta_pos = np.append(sta_pos, np.array([list(s.earthmodel.site_info(sta_data[i], wiggle_data[i][FIT_ATIME]))[:3]]), axis = 0)
        X = wiggle_data[:, [FIT_LON, FIT_LAT, FIT_DEPTH]]
        X = np.concatenate((sta_pos, X), axis = 1)
    else:
        X = wiggle_data[:, [FIT_LON, FIT_LAT, FIT_DEPTH, FIT_DISTANCE, FIT_MB]]

    evids = wiggle_data[:, FIT_EVID]

    return X, y, evids


def get_shape_training_data(runid, site, chan, band, phases, target, require_human_approved=False, max_acost=200, min_amp=-10, array=False, HACK_FAKE_POINTS=False, **kwargs):
    s = Sigvisa()


    try:
        print "loading %s fit data... " % (phases),
        cursor = s.dbconn.cursor()
        fit_data, sta_data = load_shape_data(cursor, chan=chan, band=band, site=site, runids=[runid, ], phases=phases, require_human_approved=require_human_approved, max_acost=max_acost, min_amp=min_amp, **kwargs)
        #import pdb; pdb.set_trace()
        cursor.close()
        print str(fit_data.shape[0]) + " entries loaded"
    except:
        raise

    if array:
        sta_pos = np.empty((0, 3))
        for i in range(len(sta_data)):
            sta_pos = np.append(sta_pos, np.array([list(s.earthmodel.site_info(sta_data[i], fit_data[i][FIT_ATIME]))[:3]]), axis = 0)
        X = fit_data[:, [FIT_LON, FIT_LAT, FIT_DEPTH, FIT_DISTANCE, FIT_MB]]
        X = np.concatenate((sta_pos, X), axis = 1)
    else:
        X = fit_data[:, [FIT_LON, FIT_LAT, FIT_DEPTH, FIT_DISTANCE, FIT_MB]]

    try:
        evids = fit_data[:, FIT_EVID]
    except IndexError as e:
        raise NoDataException()


    try:

        if HACK_FAKE_POINTS:
            short_distance_X = np.array([[0.0, 0.0, 0.0, 1, 0.0],])

            ld1 = np.array([[0.0, 0.0, 0.0, 13000, 0.0],])
            ld2 = np.array([[0.0, 0.0, 0.0, 16000, 0.0],])
            ld3 = np.array([[0.0, 0.0, 0.0, 20000, 0.0],])
            long_distance_X = np.vstack([ld1, ld1, ld1, ld2, ld2, ld2, ld3, ld3, ld3])
            new_evids = [0,0,0,0,0,0,0,0,0]

        if target == "tt_residual":
#            import pdb; pdb.set_trace()
            y = np.array(fit_data[:, FIT_ATIME])
            for (i, row) in enumerate(fit_data):
                ev = get_event(evid=row[FIT_EVID])
                pred = tt_predict(ev, site, phaseid=int(row[FIT_PHASEID]))
                y[i] -= (ev.time + pred)
        elif target == "coda_decay":
            y = fit_data[:, FIT_CODA_DECAY]

            if HACK_FAKE_POINTS:
                maxy = np.max(y)
                y = np.concatenate([y, [-.25, -.25, -.25, -.3, -.3, -.3, -.5, -.5, -.5, maxy]])
                X = np.vstack([X, long_distance_X, short_distance_X])
                evids = np.concatenate([evids, new_evids])
        elif target == "amp_transfer":
            y = fit_data[:, FIT_AMP_TRANSFER]

            if HACK_FAKE_POINTS:
                miny = np.min(y)
                maxy = np.max(y)
                y = np.concatenate([y, [miny-.1, miny-.4, miny-.2, miny-.4, miny-.6, miny-.8, miny-1, miny-.8, miny-1.5, maxy+2]])
                X = np.vstack([X, long_distance_X, short_distance_X])
                evids = np.concatenate([evids, new_evids])
        elif target == "peak_offset":
            y = fit_data[:, FIT_PEAK_DELAY]
        else:
            raise KeyError("invalid target param %s" % (target))
    except IndexError as e:
        raise NoDataException()

    return X, y, evids



def chan_for_site(site, options):
    s = Sigvisa()
    if options.chan=="vertical":
        if s.earthmodel.site_info(site, 0.0)[3] == 1:
            chan = s.array_default_channel(site)
        else:
            sta=site
            chan = s.default_vertical_channel[sta]
    else:
        chan = options.chan
    chan = s.canonical_channel_name[chan]
    return chan

def explode_sites(options):
    sites = options.sites.split(',')

    if not options.array_joint:
        s = Sigvisa()
        sites = [s.get_default_sta(site) for site in sites]
    return sites



def load_site_data(elems, wiggles, target, param_num, wg=None, **kwargs):
    X, y, evids = None, None, None
    for array_elem in elems:
        try:
            if wiggles:
                X_part, y_part, evids_part = get_wiggle_training_data(wg=wg, site=array_elem,target_num=param_num, **kwargs)
            else:
                X_part, y_part, evids_part = get_shape_training_data(site=array_elem, target=target, **kwargs)
        except NoDataException as e:
            print e
            continue
        X = np.append(X, X_part, axis = 0) if X is not None else X_part
        y = np.append(y, y_part) if y is not None else y_part
        evids = np.append(evids, evids_part) if evids is not None else evids_part
    if X is None:
        raise NoDataException("no data for target '%s' at any element of %s" % (target, elems))
    return X, y, evids

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
        "-n", "--band", dest="band", default="freq_2.0_3.0", type="str", help="name of band to examine (freq_2.0_3.0)")
    parser.add_option("-p", "--phases", dest="phases", default=",".join(s.phases), type="str",
                      help="comma-separated list of phases for which to train models)")
    parser.add_option("-t", "--targets", dest="targets", default=None, type="str",
                      help="comma-separated list of target parameter names (coda_decay,amp_transfer,peak_offset)")
    parser.add_option("-b", "--basisid", dest="basisid", default=None, type="int", help="basisid (from the sigvisa_wiggle_basis DB table) for which to train wiggle param models")
    parser.add_option("--template_shape", dest="template_shape", default="paired_exp", type="str", help="")
    parser.add_option(
        "-m", "--model_type", dest="model_type", default="gp_lld", type="str", help="type of model to train (gp_lld)")
    parser.add_option("--require_human_approved", dest="require_human_approved", default=False, action="store_true",
                      help="only train on human-approved good fits")
    parser.add_option(
        "--max_acost", dest="max_acost", default=np.float('inf'), type=float, help="maximum fitting cost of fits in training set (inf)")
    parser.add_option("--min_amp", dest="min_amp", default=-3, type=float,
                      help="only consider fits above the given amplitude (does not apply to amp_transfer fits)")
    parser.add_option("--min_amp_for_at", dest="min_amp_for_at", default=-5, type=float,
                      help="only consider fits above the given amplitude (for amp_transfer fits)")
    parser.add_option("--enable_dupes", dest="enable_dupes", default=False, action="store_true",
                      help="train models even if a model of the same type already appears in the DB")
    parser.add_option("--optim_params", dest="optim_params", default="'method': 'bfgs_fastcoord', 'normalize': True, 'disp': True, 'bfgs_factr': 1e10, 'random_inits': 1", type="str", help="fitting param string")
    parser.add_option("--array_joint", dest="array_joint", default=False, action="store_true",
                      help="model array station jointly; don't explode array into individual elements (False)")
    parser.add_option("--subsample", dest="subsample", default=500, type="float",
                      help="use a subset of the data to learn GP hyperparameters more quickly (500)")
    parser.add_option("--bounds", dest="bounds", default=None, type="str",
                      help="comma-separated list of hyperparam bounds low1,high1,low2,high2,... (None)")
    parser.add_option("--param_var", dest="param_var", default=1.0, type="float",
                      help="variance for the Gaussian prior on global model params (1.0)")
    parser.add_option("--slack_var", dest="slack_var", default=1.0, type="float",
                      help="additional variance allowed on top of global model params in the per-station params (1.0)")
    parser.add_option("--fake_points", dest="fake_points", default=False, action="store_true",
                      help="add some fake points at long and short distances to help condition the polynomials (False)")
    parser.add_option("--iterate", dest="iterate", default=0, type="int",
                      help="perform additional retraining iterations to shrink towards a global model (0)")
    parser.add_option("--preset", dest="preset", default=None, type="str",
                      help="use preset models types to build models for all targets in one command. options are 'param' to build parametric models (amp_transfer: param_sin1, tt_residual: constant_laplacian, coda_decay: param_linear_distmb, peak_offset: param_linear_mb) or 'gp' to build nonparametric GP models")



    (options, args) = parser.parse_args()

    phases = options.phases.split(',')
    model_type = options.model_type
    band = options.band

    if options.basisid is None:
        wiggles = False
        basisid = None
        wg = None
        targets = options.targets.split(',') if options.targets is not None else None
    else:
        wiggles = True
        basisid = options.basisid
        wg = load_wiggle_generator(basisid=basisid)
        targets = wg.params()


    optim_params = construct_optim_params(options.optim_params)

    run_name = options.run_name
    if options.run_iter == "latest":
        iters = read_fitting_run_iterations(cursor, run_name)
        run_iter = np.max(iters[:, 0])
    else:
        run_iter = int(options.run_iter)

    allsites = explode_sites(options)

    cursor.close()

    if options.bounds is None:
        bounds = None
    else:
        b = np.array([float(x) for x in options.bounds.split(',')])
        low_bounds = b[0::2]
        high_bounds = b[1::2]
        bounds = zip(low_bounds, high_bounds)

    sitechans = {}
    for site in allsites:
        try:
            sitechans[site] = chan_for_site(site, options)
        except KeyError as e:
            print "skipping site %s:" % site, e
            continue

    model_types = dict([(t, options.model_type) for t in targets]) if targets is not None else None
    iterations = options.iterate
    param_var = options.param_var
    slack_var = options.slack_var
    if options.preset == "param":
        targets = ['amp_transfer', 'tt_residual', 'coda_decay', 'peak_offset'] if targets is None else targets
        model_types = {'amp_transfer': 'param_sin1', 'tt_residual': 'constant_laplacian', 'coda_decay': 'param_linear_distmb', 'peak_offset': 'param_linear_mb'}
        iterations = 5
    if options.preset == "gplocal":
        targets = ['tt_residual', 'coda_decay', 'peak_offset', 'amp_transfer'] if targets is None else targets
        model_types = {'amp_transfer': 'gplocal+lld+sin1', 'tt_residual': 'constant_laplacian', 'coda_decay': 'gplocal+lld+linear_distmb', 'peak_offset': 'gplocal+lld+linear_mb'}
        iterations = 5

    modelids = do_training(run_name, run_iter, allsites, sitechans, band, targets, phases, model_types, optim_params, bounds, options.min_amp_for_at, options.min_amp, options.enable_dupes, options.array_joint, options.require_human_approved, options.max_acost, options.fake_points, options.template_shape, wiggles, wg, basisid, options.subsample, param_var + slack_var)

    for target in targets:
        model_type = model_types[target]
        for i in range(iterations):
            print "training models for iteration %d..." % (i,)
            modelids[target] = retrain_models(modelids[target], model_type, global_var=param_var, station_slack_var=slack_var)

def do_training(run_name, run_iter, allsites, sitechans, band, targets, phases, model_types, optim_params, bounds, min_amp_for_at, min_amp, enable_dupes, array_joint, require_human_approved, max_acost, fake_points, template_shape, wiggles, wg, basisid, subsample, prior_var):
    s = Sigvisa()
    cursor = s.dbconn.cursor()
    runid = get_fitting_runid(cursor, run_name, run_iter, create_if_new=False)
    modelids = defaultdict(list)
    for site in allsites:
        try:
            chan = sitechans[site]
        except KeyError as e:
            continue

        for (param_num, target) in enumerate(targets):
            if target == "amp_transfer":
                min_amp = min_amp_for_at
            else:
                min_amp = min_amp
            model_type = model_types[target]

            for phase in phases:
                if wiggles:
                    basisid_cond = 'and wiggle_basisid=%d' % basisid
                else:
                    basisid_cond = ''

                # check for duplicate model
                sql_query = "select modelid, shrinkage_iter from sigvisa_param_model where model_type='%s' and site='%s' and chan='%s' and band='%s' and phase='%s' and fitting_runid=%d and param='%s' %s" % (
                    model_type, site, chan, band, phase, runid, target, basisid_cond)
                cursor.execute(sql_query)
                dups = cursor.fetchall()
                if len(dups) > 0 and not enable_dupes:
                    modelid = sorted(dups, key = lambda x : -x[1])[0][0]
                    modelids[target].append(modelid)
                    print "model already trained for %s, %s, %s (modelid %d), skipping..." % (site, target, phase, modelid)

                    continue

                if array_joint:
                    elems = s.get_array_elements(site)
                else:
                    elems = [site,]

                try:
                    X, y, evids = load_site_data(elems, wiggles, target=target, param_num=param_num, runid=runid, chan=chan, band=band, phases=[phase, ], require_human_approved=require_human_approved, max_acost=max_acost, min_amp=min_amp, array = array_joint, HACK_FAKE_POINTS=fake_points)
                except NoDataException:
                    print "no data for %s %s %s, skipping..." % (site, target, phase)
                    continue

                if wiggles:
                    model_fname = get_model_fname(run_name, run_iter, site, chan, band, phase, target, model_type, evids, model_name=template_shape, unique=True)
                else:
                    model_fname = get_model_fname(run_name, run_iter, site, chan, band, phase, target, model_type, evids, basisid=basisid, unique=True)
                evid_fname = os.path.splitext(os.path.splitext(os.path.splitext(model_fname)[0])[0])[0] + '.evids'
                Xy_fname = os.path.splitext(os.path.splitext(os.path.splitext(model_fname)[0])[0])[0] + '.Xy.npz'
                np.savetxt(evid_fname, evids, fmt='%d')
                np.savez(Xy_fname, X=X, y=y)

                distfn = model_type[3:]
                st = time.time()
                try:
                    print "training mode for target", target
                    model = learn_model(X, y, model_type, target=target, sta=site, optim_params=optim_params, gp_build_tree=False, k=subsample, bounds=bounds, param_var=prior_var)
                except Exception as e:
                    raise
                    print e
                    continue

                et = time.time()

                if np.isnan(model.log_likelihood()):
                    print "error training model for %s %s %s, likelihood is nan! skipping.." % (site, target, phase)
                    continue

                model.save_trained_model(model_fname)
                wiggle_options = {'wiggle_basisid': basisid,}
                template_options = {'template_shape': template_shape, }
                insert_options = wiggle_options if wiggles else template_options

                shrinkage = {}
                if model_type.startswith("param") or model_type.startswith('gplocal'):
                    n = len(model.param_mean())
                    shrinkage = {'mean': np.zeros((n,)),
                                 'cov': np.eye(n) * (prior_var),}

                modelid = insert_model(s.dbconn, fitting_runid=runid, param=target, site=site, chan=chan, band=band, phase=phase, model_type=model_type, model_fname=model_fname, training_set_fname=evid_fname, training_ll=model.log_likelihood(), require_human_approved=require_human_approved, max_acost=max_acost, n_evids=len(evids), min_amp=min_amp, elapsed=(et-st), hyperparams = model_params(model, model_type), optim_method = repr(optim_params) if model_type.startswith('gp') else None, shrinkage=repr(shrinkage), **insert_options)
                print "inserted as", modelid, "ll", model.log_likelihood()
                modelids[target].append(modelid)

    cursor.close()
    return modelids

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
