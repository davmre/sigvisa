import os, errno, sys, time, traceback
import numpy as np, scipy

from database.dataset import *
from database.signal_data import *
from database import db
from signals.template_models.load_by_name import load_template_model

import utils.geog
import obspy.signal.util

from optparse import OptionParser

from sigvisa import Sigvisa
from signals.io import *
from learn.optimize import minimize_matrix

def construct_optim_params(optim_param_str):

    def copy_dict_entries(keys, src, dest):
        if len(keys) == 0: keys = src.keys()
        for key in keys:
            dest[key] = src[key]

    defaults = {
        "method": "bfgscoord",
        "fix_first_cols": 1,
        "normalize": True,
        'disp': False,
        "eps": 1e-4, # increment for approximate gradient evaluation

        "bfgscoord_iters": 5,
        "bfgs_factr": 10, # used by bfgscoord and bfgs
        "xtol": 0.01, # used by simplex
        "ftol": 0.01, # used by simplex, tnc
        "grad_stopping_eps": 1e-4,
        }
    overrides = eval("{%s}" % optim_param_str)

    optim_params = {}
    optim_params['method'] = overrides['method'] if 'method' in overrides else defaults['method']
    copy_dict_entries(["fix_first_cols", "normalize", "disp", "eps"], src=defaults, dest=optim_params)
    method = optim_params['method']

    # load appropriate defaults for each method
    if method == "bfgscoord":
        copy_dict_entries(["bfgscoord_iters", "bfgs_factr"], src=defaults, dest=optim_params)
    elif method == "bfgs":
        copy_dict_entries(["bfgs_factr",], src=defaults, dest=optim_params)
    elif method == "tnc":
        copy_dict_entries(["ftol",], src=defaults, dest=optim_params)
    elif method == "simplex":
        copy_dict_entries(["ftol", "xtol",], src=defaults, dest=optim_params)
    elif method == "grad":
        copy_dict_entries(["grad_stopping_eps",], src=defaults, dest=optim_params)

    copy_dict_entries([], src=overrides, dest=optim_params)
    return optim_params

def fit_template(wave, ev, tm, optim_params, init_run_name=None, init_iteration=None, iid=False, hz=None):
    """
    Return the template parameters which best fit the given waveform.
    """

    s = Sigvisa()
    cursor = s.dbconn.cursor()
    sta = wave['sta']
    chan = wave['chan']
    band = wave['band']

    print "fitting template for", sta, chan, band

    best_params = None

    # initialize the search using the outcome of a previous run
    if init_run_name is not None:
        start_param_vals, phaseids_loaded, fit_cost = load_template_params(cursor, ev.evid, sta, chan, band, run_name=init_run_name, iteration=init_iteration)
        phases = [s.phasenames[phaseid-1] for phaseid in phaseids_loaded]

        arriving_phases = s.arriving_phases(ev, wave['sta'])
        assert(phases == arriving_phases)

    # or if this is the first run, initialize heuristically
    else:
        print "getting heuristic start params..."
        (phases, start_param_vals) = tm.heuristic_starting_params(wave)
        (phases, start_param_vals) = filter_and_sort_template_params(phases, start_param_vals, s.phases)
        print "done"

    if hz is not None:
        wave_to_fit = wave.filter("hz_%.1f" % hz)
    else:
        wave_to_fit = wave


    if iid:
        wave_to_fit = wave_to_fit.filter("smooth")
        f = lambda vals: -tm.log_likelihood((phases, vals), ev, sta, chan, band) - tm.waveform_log_likelihood_iid(wave_to_fit, (phases, vals))
    else:
        f = lambda vals: -tm.log_likelihood((phases, vals), ev, sta, chan, band) - tm.waveform_log_likelihood(wave_to_fit, (phases, vals))

    low_bounds = None
    high_bounds = None
    if optim_params['method'] != "simplex":
        atimes = start_param_vals[:, 0]
        low_bounds = tm.low_bounds(phases, default_atimes=atimes)
        high_bounds = tm.high_bounds(phases, default_atimes=atimes)


    print "minimizing matrix", start_param_vals
    best_param_vals, best_cost = minimize_matrix(f, start_param_vals, low_bounds=low_bounds, high_bounds=high_bounds, optim_params=optim_params)
    print "done", best_param_vals, best_cost

    range = np.log(np.max(wave_to_fit.data)) - np.log(np.min(wave_to_fit.data))
    acost = best_cost / (range * wave_to_fit['npts']) * 1000


    return (phases, best_param_vals), acost

def fit_event_wave(event, sta, chan, band, tm, output_run_name, output_iteration, init_run_name=None, init_iteration=None, plot=False, optim_params=None, iid=False, fit_hz=5):
    """
    Find the best-fitting template parameters for each band/channel of
    a particular event at a particular station. Store the template
    parameters in the database.
    """

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    wave = load_event_station_chan(event.evid, sta, chan, cursor=cursor).filter("%s;env" % band)

    # DO THE FITTING
    method = optim_params['method']
    if method == "load":
        fit_params, fit_cost, fitid = load_template_params(cursor, event.evid, chan, band, init_run_name, siteid)
        if fit_params is None:
            raise Exception("no params in database for evid %d siteid %d runid %d chan %s band %s, skipping" % (evid, siteid, init_run_name, chan, band))
    else:
        st = time.time()
        fit_params, acost = fit_template(wave, ev=event, tm=tm, optim_params=optim_params, iid=iid, init_run_name=init_run_name, init_iteration=init_iteration, hz=fit_hz)
        et = time.time()
        fitid = store_template_params(wave, fit_params, optim_param_str=repr(optim_params)[1:-1], iid=iid, acost=acost, run_name=output_run_name, iteration=output_iteration, elapsed=et-st, hz=fit_hz)
        s.dbconn.commit()
    return fitid


def main():
    parser = OptionParser()

    parser.add_option("-s", "--sta", dest="sta", default=None, type="str", help="name of station for which to fit templates")
    parser.add_option("--optim_params", dest="optim_params", default="", type="str", help="fitting param string")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="run name")
    parser.add_option("-i", "--run_iteration", dest="run_iteration", default=None, type="int", help="run iteration (default is to use the next iteration)")
    parser.add_option("-e", "--evid", dest="evid", default=None, type="int", help="event ID")
    parser.add_option("--orid", dest="orid", default=None, type="int", help="origin ID")
    parser.add_option("--init_run_name", dest="init_run_name", default=None, type="str", help="initialize template fitting with results from this run name")
    parser.add_option("--init_run_iteration", dest="init_run_iteration", default=None, type="int", help="initialize template fitting with results from this run iteration (default: most recent)")
    parser.add_option("-p", "--plot", dest="plot", default=False, action="store_true", help="save plots")
    parser.add_option("--template_shape", dest = "template_shape", default="paired_exp", type="str", help="template model type to fit parameters under (paired_exp)")
    parser.add_option("--template_model", dest = "template_model", default="gp_dad", type="str", help="")
    parser.add_option("--band", dest = "band", default="freq_2.0_3.0", type="str", help="")
    parser.add_option("--chan", dest = "chan", default="BHZ", type="str", help="")


    (options, args) = parser.parse_args()

    s = Sigvisa()
    cursor = s.dbconn.cursor()


    if options.run_name is None or options.run_iteration is None:
        raise Exception("must specify run name and iteration!")

    if options.run_iteration == 1:
        iid=True
        fix_first_cols = 2
    elif options.run_iteration == 2:
        iid=False
        fix_first_cols = 1
    else:
        iid=False
        fix_first_cols = 0
    if not iid:
        raise Exception("non-iid fits not currently implemented!")
#        raise Exception("need to specify wiggle model for non-iid fits!")

    if options.init_run_name is None:
        tm = load_template_model(template_shape = options.template_shape, model_type="dummy")
    else:
        tm = load_template_model(template_shape = options.template_shape, run_name=options.init_run_name, run_iter=options.init_run_iteration, model_type=options.template_model)


    if not (options.evid is None and options.orid is None):
        ev = get_event(evid=options.evid, orid=options.orid)
    else:
        raise Exception("Must specify event id (evid) or origin id (orid) to fit.")

    try:
        fitid = fit_event_wave(event=ev, sta=options.sta, band=options.band, chan=options.chan, tm = tm, output_run_name=options.run_name, output_iteration=options.run_iteration, init_run_name=options.init_run_name, init_iteration=options.init_run_iteration, plot=options.plot, optim_params=construct_optim_params(options.optim_params), iid=iid)
    except KeyboardInterrupt:
        s.dbconn.commit()
        raise
    except:
        s.dbconn.commit()
        print traceback.format_exc()

    print "fit id %d completed successfully." % fitid

if __name__ == "__main__":
    main()
