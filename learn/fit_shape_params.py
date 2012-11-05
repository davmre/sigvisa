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


def fit_template(wave, ev, tm, pp, method="bfgs", wiggles=None, init_run_name=None, optimize_arrival_times=False, iid=False):
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
        start_param_vals, phaseids_loaded, fit_cost = load_template_params(s.cursor, ev.evid, chan, band, init_run_name, wave['siteid'])
        phases = [s.phasenames[phaseid-1] for phaseid in phaseids_loaded]

        arriving_phases = s.arriving_phases(ev, wave['sta'])
        assert(phases == arriving_phases)

    # or if this is the first run, initialize heuristically
    else:
        (phases, start_param_vals) = tm.heuristic_starting_params(wave)


    if iid:
        f = lambda vals: -tm.log_likelihood((phases, vals), ev, sta, chan, band) - tm.waveform_log_likelihood_iid(wave, (phases, vals))
    else:
        f = lambda vals: -tm.log_likelihood((phases, vals), ev, sta, chan, band) - tm.waveform_log_likelihood(wave, (phases, vals))

    low_bounds = None
    high_bounds = None
    if method == "bfgs" or method == "tnc":
        low_bounds = tm.low_bounds()
        high_bounds = tm.high_bounds()

    best_param_vals, best_cost = minimize_matrix(f, start_param_vals, low_bounds=low_bounds, high_bounds=high_bounds, method=method, fix_first_col=(not optimize_arrival_times))

    """
    if wiggles is None or best_params is None:
        # learn from smoothed data w/ iid noise

        if pp is not None:
            plot_channels_with_pred(sigmodel, pp, smoothed, assem_params(best_params), phaseids, None, None, title = "best iid (cost %f, evid %s)" % (best_cost, evid))
            plot_channels_with_pred(sigmodel, pp, env, assem_params(best_params), phaseids, None, None, title = "")
            load_wiggle_models(cursor, sigmodel, wiggles)
            plot_channels_with_pred(sigmodel, pp, smoothed, assem_params(best_params), phaseids, None, None, title = "best iid (cost %f, evid %s)" % (best_cost, evid))

    if wiggles is not None:
        f = lambda params : c_cost(sigmodel, env, phaseids, assem_params(params))
        print "loaded cost is", f(best_params)
#        best_params, best_cost = optimize(f, best_params, bounds, method=method, phaseids= (phaseids if by_phase else None))
        if pp is not None:
            plot_channels_with_pred(sigmodel, pp, env, assem_params(best_params), phaseids, None, None, title = "best (cost %f, evid %s)" % (best_cost, evid))
            plot_channels_with_pred(sigmodel, pp, smoothed, assem_params(best_params), phaseids, None, None, title = "best (cost %f, evid %s)" % (best_cost, evid))

            plot_channels_with_pred(sigmodel, pp, env, assem_params(best_params), phaseids, None, None, title = "best (cost %f, evid %s)" % (best_cost, evid), logscale=False)
            plot_channels_with_pred(sigmodel, pp, smoothed, assem_params(best_params), phaseids, None, None, title = "best (cost %f, evid %s)" % (best_cost, evid), logscale=False)
            """

    return (phases, best_param_vals), best_cost

def fit_event_segment(event, sta, tm, output_run_name, init_run_name=None, plot=False, wiggles=None, method="simplex", iid=False):
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
                    fit_params, fit_cost = fit_template(wave, pp=pp, ev=event, tm=tm, method=method, wiggles=wiggles, iid=iid, init_run_name=init_run_name)
                    if pp is not None:
                        print "wrote plot"

                save_wiggles(wave, run_name=output_run_name, template_params=fit_params)
                if method != "load":
                    store_template_params(wave, event, fit_params, output_run_name, method)
                dbconn.commit()
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
    parser.add_option("-r", "--runid", dest="runid", default=None, type="int", help="runid")
    parser.add_option("-e", "--evid", dest="evid", default=None, type="int", help="event ID")
    parser.add_option("--orid", dest="orid", default=None, type="int", help="origin ID")
    parser.add_option("-w", "--wiggles", dest="wiggles", default=None, type="str", help="filename of wiggle-model params to load (default is to ignore wiggle model and do iid fits)")
    parser.add_option("--init_run_name", dest="init_run_name", default=None, type="int", help="initialize template fitting with results from this runid")
    parser.add_option("-p", "--plot", dest="plot", default=False, action="store_true", help="save plots")
    parser.add_option("--template_shape", dest = "template_shape", default="paired_exp", type="str", help="template model type to fit parameters under (paired_exp)")
    parser.add_option("--template_run_name", dest = "template_run_name", default=None, type="str", help="name of previously trained template model to load (None)")
    parser.add_option("--template_model", dest = "template_model", default="gp_dad", type="str", help="")

    (options, args) = parser.parse_args()

    s = Sigvisa()
    cursor = s.cursor

    iid=True


    if options.template_run_name is None:
        tm = load_template_model(template_shape = options.template_shape, run_name=None, model_type="dummy")
    else:
        tm = load_template_model(template_shape = options.template_shape, run_name=options.template_run_name, model_type=options.template_model)

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
        fit_event_segment(event=ev, sta=options.sta, tm = tm, runid=options.runid, init_run_name=options.init_run_name, plot=options.plot, method=options.method, wiggles=options.wiggles)
    except KeyboardInterrupt:
        s.dbconn.commit()
        raise
    except:
        s.dbconn.commit()
        print traceback.format_exc()

if __name__ == "__main__":
    main()






