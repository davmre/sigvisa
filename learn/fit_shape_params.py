import os, errno, sys, time, traceback
import numpy as np, scipy

from database.dataset import *
from database.signal_data import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plot
import utils.geog
import obspy.signal.util

from optparse import OptionParser

from sigvisa import *
from signals.io import *
from plotting.plot_coda_decays import *
from learn.train_wiggles import *
from learn.optimize import minimize_matrix
from signals.noise_model import *


def fit_template(wave, ev, tm, pp, method="bfgs", wiggles=None, init_run_name=None, init_iteration=None, optimize_arrival_times=False, iid=False):
    """
    Return the template parameters which best fit the given waveform.
    """

    s = tm.sigvisa
    sta = wave['sta']
    chan = wave['chan']
    band = wave['band']

    print "fitting template for", sta, chan, band

    if wiggles is not None:
        load_wiggle_models(s.cursor, s.sigmodel, wiggles)
    best_params = None

    # initialize the search using the outcome of a previous run
    if init_run_name is not None:
        start_param_vals, phaseids_loaded, fit_cost = load_template_params(ev.evid, sta, chan, band, run_name=init_run_name, iteration=init_iteration)
        phases = [s.phasenames[phaseid-1] for phaseid in phaseids_loaded]

        arriving_phases = s.arriving_phases(ev, wave['sta'])
        assert(phases == arriving_phases)

    # or if this is the first run, initialize heuristically
    else:
        print "getting heuristic start params..."
        (phases, start_param_vals) = tm.heuristic_starting_params(wave)
        print "done"

    if iid:
        smooth_wave = wave.filter("smooth")
        f = lambda vals: -tm.log_likelihood((phases, vals), ev, sta, chan, band) - tm.waveform_log_likelihood_iid(smooth_wave, (phases, vals))

    else:
        f = lambda vals: -tm.log_likelihood((phases, vals), ev, sta, chan, band) - tm.waveform_log_likelihood(wave, (phases, vals))

    low_bounds = None
    high_bounds = None
    if method == "bfgs" or method == "tnc":
        low_bounds = tm.low_bounds()
        high_bounds = tm.high_bounds()

    if pp is not None:
        plot_waveform_with_pred(pp, wave, tm, (phases, start_param_vals), title = "start (cost %f, evid %s)" % (f(start_param_vals), ev.evid), logscale=True)

    print "minimizing matrix", start_param_vals
    best_param_vals, best_cost = minimize_matrix(f, start_param_vals, low_bounds=low_bounds, high_bounds=high_bounds, method=method, fix_first_col=(not optimize_arrival_times))
    print "done", best_param_vals, best_cost

    if pp is not None:
        plot_waveform_with_pred(pp, wave, tm, (phases, best_param_vals), title = "best (cost %f, evid %s)" % (f(best_param_vals), ev.evid), logscale=True)


    return (phases, best_param_vals), best_cost

def fit_event_segment(event, sta, tm, output_run_name, output_iteration, init_run_name=None, init_iteration=None, plot=False, wiggles=None, method="simplex", iid=False, extract_wiggles=True):
    """
    Find the best-fitting template parameters for each band/channel of
    a particular event at a particular station. Store the template
    parameters in the database, and extract wiggles from the saved
    parameters and save them to disk.
    """

    try:
        pp = None
        s = Sigvisa()
        bands = s.bands
        chans = s.chans
        cursor = s.cursor

        base_coda_dir = get_base_dir(sta, output_run_name)
        seg = load_event_station(event.evid, sta, cursor=cursor).with_filter("env")
        for (band_idx, band) in enumerate(bands):
            pdf_dir = ensure_dir_exists(os.path.join(base_coda_dir, band))
            band_seg = seg.with_filter(band)
            for chan in chans:
                if plot:
                    fname = os.path.join(pdf_dir, "%d_%s.pdf" % (event.evid, chan))
                    print "writing to %s..." % (fname,)
                    pp = PdfPages(fname)
                else:
                    pp = None

                wave = band_seg[chan]

                # DO THE FITTING
                if method == "load":
                    fit_params, fit_cost = load_template_params(event.evid, chan, band, init_run_name, siteid)
                    if fit_params is None:
                        print "no params in database for evid %d siteid %d runid %d chan %s band %s, skipping" % (evid, siteid, init_run_name, chan, band)
                        continue
                    set_noise_process(s.sigmodel, tr)
                    fit_cost = fit_cost * time_len
                else:
                    fit_params, fit_cost = fit_template(wave, pp=pp, ev=event, tm=tm, method=method, wiggles=wiggles, iid=iid, init_run_name=init_run_name, init_iteration=init_iteration)
                    if pp is not None:
                        print "wrote plot"

                if extract_wiggles:
                    save_wiggles(wave=wave, tm=tm, run_name=output_run_name, template_params=fit_params)
                if method != "load":
                    store_template_params(wave, fit_params, method_str=method, iid=iid, fit_cost=fit_cost, run_name=output_run_name, iteration=output_iteration)
                s.dbconn.commit()
                if pp is not None:
                    pp.close()

    except:
        if pp is not None:
            pp.close()
        raise

def main():
    parser = OptionParser()

    parser.add_option("-s", "--sta", dest="sta", default=None, type="str", help="name of station for which to fit templates")
    parser.add_option("-m", "--method", dest="method", default="simplex", type="str", help="fitting method (simplex)")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="run name")
    parser.add_option("-i", "--run_iteration", dest="run_iteration", default=None, type="int", help="run iteration (default is to use the next iteration)")
    parser.add_option("-e", "--evid", dest="evid", default=None, type="int", help="event ID")
    parser.add_option("--orid", dest="orid", default=None, type="int", help="origin ID")
    parser.add_option("-w", "--wiggles", dest="wiggles", default=None, type="str", help="filename of wiggle-model params to load")
    parser.add_option("--init_run_name", dest="init_run_name", default=None, type="str", help="initialize template fitting with results from this run name")
    parser.add_option("--init_run_iteration", dest="init_run_iteration", default=None, type="int", help="initialize template fitting with results from this run iteration (default: most recent)")
    parser.add_option("-p", "--plot", dest="plot", default=False, action="store_true", help="save plots")
    parser.add_option("--template_shape", dest = "template_shape", default="paired_exp", type="str", help="template model type to fit parameters under (paired_exp)")
    parser.add_option("--template_model", dest = "template_model", default="gp_dad", type="str", help="")

    (options, args) = parser.parse_args()

    s = Sigvisa()
    cursor = s.cursor


    if options.run_name is None or options.run_iteration is None:
        raise Exception("must specify run name and iteration!")

    if options.run_iteration == 1:
        iid=True
        optimize_arrival_times=False
    elif options.run_iteration == 2:
        iid=False
        optimize_arrival_times=False
    else:
        iid=False
        optimize_arrival_times=True
    if options.wiggles is None and not iid:
        raise Exception("need to specify wiggle model for non-iid fits!")

    if options.init_run_name is None:
        tm = load_template_model(template_shape = options.template_shape, run_name=None, model_type="dummy")
    else:
        tm = load_template_model(template_shape = options.template_shape, run_name=options.init_run_name, model_type=options.template_model)

    if options.start_time is None:
        cursor.execute("select start_time, end_time from dataset where label='training'")
        (st, et) = read_timerange(cursor, "training", hours=None, skip=0)
    else:
        st = options.start_time
        et = options.end_time

    if not (options.evid is None and options.orid is None):
        ev = Event(evid=options.evid, orid=options.orid)
    else:
        raise Exception("Must specify event id (evid) or origin id (orid) to fit.")

    try:
        fit_event_segment(event=ev, sta=options.sta, tm = tm, run_name=options.run_name, iteration=options.run_iteration, init_run_name=options.init_run_name, init_run_iteration=options.init_run_iteration, plot=options.plot, method=options.method, wiggles=options.wiggles)
    except KeyboardInterrupt:
        s.dbconn.commit()
        raise
    except:
        s.dbconn.commit()
        print traceback.format_exc()

if __name__ == "__main__":
    main()






