
from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *
from sigvisa.database import db

import sys
import os
import pickle
import pdb
import traceback

from sigvisa import *
import numpy as np

from optparse import OptionParser

from sigvisa.learn.train_param_common import insert_model, learn_model, load_model, get_model_fname

from sigvisa.models.wiggles import load_wiggle_generator

def get_wiggle_training_data(run_name, run_iter, wm_node, target, **kwargs):
    s = Sigvisa()
    cursor = s.dbconn.cursor()

    runid = get_fitting_runid(cursor, run_name, run_iter, create_if_new=False)

    wiggle_data = load_wiggle_data(cursor, runids=[runid, ], basisid=wm_node.basisid, **kwargs)
    print str(wiggle_data.shape[0]) + " entries loaded"

    try:
        param_num = wm_node.param_name_to_num(target)
        y = wiggle_data[:, WIGGLE_PARAM0 + param_num]
    except IndexError as e:
        print "nd2"
        raise NoDataException()

    X = wiggle_data[:, [FIT_LON, FIT_LAT, FIT_DEPTH, FIT_DISTANCE, FIT_AZIMUTH]]
    evids = wiggle_data[:, FIT_EVID]

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
    parser.add_option("-b", "--basisid", dest="basisid", default=None, type="int", help="basisid (from the sigvisa_wiggle_basis DB table) for which to train wiggle param models")
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

    (options, args) = parser.parse_args()

    sites = options.sites.split(',')
    chan = options.chan
    phases = options.phases.split(',')
    model_type = options.model_type
    band = options.band

    run_name = options.run_name
    if options.run_iter == "latest":
        iters = read_fitting_run_iterations(cursor, run_name)
        run_iter = np.max(iters[:, 0])
    else:
        run_iter = int(options.run_iter)

    runid = get_fitting_runid(cursor, run_name, run_iter, create_if_new=False)

    wm_node = load_wiggle_node(basisid=options.basisid)

    for site in sites:
        for phase in phases:
            for param_num in range(wm_node.dimension()):
                target = wm_node.param_num_to_name(param_num)

                # check for duplicate model
                sql_query = "select modelid from sigvisa_param_model where model_type='%s' and site='%s' and chan='%s' and band='%s' and phase='%s' and fitting_runid=%d and param='%s' and wiggle_basisid=%d " % (
                    model_type, site, chan, band, phase, runid, target, options.basisid)
                cursor.execute(sql_query)
                dups = cursor.fetchall()
                if len(dups) > 0:
                    print "model already trained for %s, %s, %s (modelid %d), skipping..." % (site, target, phase, dups[0][0])
                    continue

                try:
                    X, y, evids = get_wiggle_training_data(run_name=run_name, run_iter=run_iter, wm_node=wm_node, site=site, chan=chan, band=band, phases=[phase, ], target=target, require_human_approved=options.require_human_approved, max_acost=options.max_acost, min_amp=options.min_amp)
                except NoDataException:
                    raise
                    print "no data for %s %s %s, skipping..." % (site, target, phase)
                    continue


                model_fname = get_model_fname(run_name, run_iter, site, chan, band, phase, target, model_type, evids, basisid=options.basisid)
                evid_fname = os.path.splitext(model_fname)[0] + '.evids'
                np.savetxt(evid_fname, evids, fmt='%d')

                distfn = model_type[3:]
                model = learn_model(X, y, model_type, target=target, sta=site)

                if np.isnan(model.log_likelihood()):
                    print "error training model for %s %s %s, likelihood is nan! skipping.." % (site, target, phase)
                    continue

                model.save_trained_model(model_fname)
                modelid = insert_model(s.dbconn, fitting_runid=runid, wiggle_basisid=options.basisid,
                                       param=target, site=site, chan=chan, band=band, phase=phase,
                                       model_type=model_type, model_fname=model_fname,
                                       training_set_fname=evid_fname, training_ll=model.log_likelihood(),
                                       require_human_approved=options.require_human_approved,
                                       max_acost=options.max_acost, n_evids=len(evids),
                                       min_amp=options.min_amp)
                print "inserted as", modelid, "ll", model.log_likelihood()

if __name__ == "__main__":
    from bdb import BdbQuit

    try:
        main()
    except KeyboardInterrupt:
        raise
    except BdbQuit:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
