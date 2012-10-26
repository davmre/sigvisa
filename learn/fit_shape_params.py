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
from signals.coda_decay_common import *
from signals.io import *
from plotting.plot_coda_decays import *
from learn.train_wiggles import *
from signals.noise_model import *
from signals.template_cost import *



def arrival_peak_offset(trace, window_start_offset, window_end_offset = None):
    srate = trace.stats.sampling_rate

    if window_end_offset is None:
        window_end_offset = window_start_offset + 15

    i = np.floor(window_start_offset*srate)
    j = np.floor(window_end_offset*srate)

    print window_start_offset, window_end_offset, i, j, srate, trace.data.shape

    pt = np.argmax(trace.data[i:j]) / srate
    return (pt +window_start_offset, trace.data[(pt+window_start_offset) * srate ])




def coord_descent(f, x0, converge=0.1, steps=None, maxiters=500):
    ncoords = len(x0)
    x = x0.copy()
    v = f(x)
    for i in range(maxiters):
        incr = 0
        for p in np.random.permutation(ncoords):

            # try taking steps in both directions
            step = steps[p]
            x[p] = x[p] + step
            v1 = f(x)
            x[p] = x[p] - 2*step
            v2 = f(x)
            if v <= v1 and v <= v2:
                x[p] = x[p] + step
                continue

            # continue stepping in the best direction, until there's
            # no more improvement.
            if v1 < v2:
                vold = v1
                x[p] = x[p] + 3 * step
                sign = 1
            else:
                vold = v2
                sign = -1
                x[p] = x[p] - step
            vnew = f(x)
            while vnew <= vold:
                x[p] = x[p] + sign*step
                vold = vnew
                vnew = f(x)

            x[p] = x[p] - sign*step
            incr = np.max([v - vold, incr])
            v = vold
        if incr < converge:
            break
        if i % 10 == 0:
            print "coord iter %d incr %f" % (i, incr)
    return x

def optimize(f, start_params, bounds, method, phaseids=None, maxfun=None):
    if phaseids is not None:
        return optimize_by_phase(f, start_params, bounds, phaseids, method=method,maxfun=maxfun)
    else:
        return minimize(f, start_params, bounds=bounds, method=method, steps=[.1, .1, .005] * (len(start_params)/3), maxfun=maxfun)

def minimize(f, x0, method="bfgs", bounds=None, steps=None, maxfun=None):
    if method=="bfgs":
        x1, best_cost, d = scipy.optimize.fmin_l_bfgs_b(f, x0, approx_grad=1, bounds=bounds, epsilon = 1e-1, factr=1e12, maxfun=maxfun)
    elif method=="tnc":
        x1, nfeval, rc = scipy.optimize.fmin_tnc(f, x0, approx_grad=1, bounds=bounds, maxfun=maxfun)
        x1 = np.array(x1)
    elif method=="simplex":
        x1 = scipy.optimize.fmin(f, x0, maxfun=maxfun, xtol=0.01, ftol=0.01)
    elif method=="anneal":
        x1, jmin, T, feval, iters, accept, retval = scipy.optimize.anneal(f, x0, maxeval=maxfun)
    elif method=="coord":
        x1 = coord_descent(f, x0, steps=steps)
    else:
        raise Exception("unknown optimization method %s" % (method))
    return x1, f(x1)

def optimize_by_phase(f, start_params, bounds, phaseids, method="bfgs", iters=3, maxfun=None):
    nphase_params = len(start_params) / len(phaseids)
    params = start_params.copy()
    for i in range(iters):
        for (pidx, phaseid) in enumerate(phaseids):
            sidx = pidx*nphase_params
            eidx = (pidx+1)*nphase_params
            phase_params = params[sidx:eidx]
            phase_bounds = bounds[sidx:eidx]
            apf = lambda pp : f(np.concatenate([params[:sidx], pp, params[eidx:]]))
            phase_params, c = minimize(apf, phase_params, method=method, bounds=phase_bounds, steps = [.1, .1, .005], maxfun=maxfun)
            print "params", phase_params, "cost", c
            params = np.concatenate([params[:sidx], phase_params, params[eidx:]])
    return params, c


def fit_template(wave, ev, tm, pp, method="bfgs", wiggles=None, init_runid=None):
    """
    Return the template parameters which best fit the given waveform.
    """

    (phases, start_param_vals) = tm.heuristic_starting_params(wave)
    narrs = len(arrs["arrivals"])
    try:
        arr_times = np.reshape(np.array(arrs["arrivals"]), (narrs, -1))
    except:
        import pdb, traceback
        traceback.print_exc()
        pdb.set_trace()

    if fix_peak:
        start_params = remove_peak(start_params)
        bounds = bounds_fp
        assem_params = lambda params: np.hstack([arr_times, restore_peak(np.reshape(params, (narrs, -1)))])
    else:
        assem_params = lambda params: np.hstack([arr_times, np.reshape(params, (narrs, -1))])

    start_params = start_params.flatten()

    print "start params", start_params

    #gen_title = lambda event, fit: "%s evid %d siteid %d mb %f \n dist %f azi %f \n p: %s \n s: %s " % (band, event[EV_EVID_COL], siteid, event[EV_MB_COL], distance, azimuth, fit[0,:],fit[1,:] if fit.shape[0] > 1 else "")

    set_noise_process(sigmodel, env)
    f = lambda params : c_cost(sigmodel, smoothed, phaseids, assem_params(params), iid=True)

    if wiggles is not None:
        load_wiggle_models(cursor, sigmodel, wiggles)
    best_params = None

    # initialize the search using the outcome of a previous run
    if init_runid is not None:
        best_params, phaseids_loaded, fit_cost = load_template_params(cursor, int(evid), env.stats.channel, env.stats.short_band, init_runid, env.stats.siteid)
        best_params = best_params[:, 1:]
        if fix_peak:
            best_params = remove_peak(best_params)
        best_params = best_params.flatten()
        print "loaded"
        print_params(assem_params(best_params))

        if phaseids_loaded != phaseids:
            best_params = None
        elif pp is not None:
            plot_channels_with_pred(sigmodel, pp, smoothed, assem_params(best_params), phaseids, None, None, title = "loaded smoothed iid (cost %f, evid %s)" % (start_cost, evid))

    if wiggles is None or best_params is None:
        # learn from smoothed data w/ iid noise
        best_params, best_cost = optimize(f, start_params, bounds, method=method, phaseids= (phaseids if by_phase else None))
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


    return assem_params(best_params), phaseids, best_cost

def fit_event_segment(event, sta, tm, runid, init_runid=None, plot=False, wiggles=None, method="simplex"):
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

        base_coda_dir = get_base_dir(sta, runid)
        seg = load_event_station(evid, sta, cursor=cursor).with_filter("env")
        for (band_idx, band) in enumerate(bands):
            pdf_dir = get_dir(os.path.join(base_coda_dir, band))
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
                    fit_params, fit_cost = load_template_params(event.evid, chan, band, init_runid, siteid)
                    if fit_params is None:
                        print "no params in database for evid %d siteid %d runid %d chan %s band %s, skipping" % (evid, siteid, init_runid, chan, band)
                        continue
                    set_noise_process(s.sigmodel, tr)
                    fit_cost = fit_cost * time_len
                else:
                    fit_params, fit_cost = fit_template(wave, pp=pp, ev=event, tm=tm, method=method, wiggles=wiggles, init_runid=init_runid)
                    if pp is not None:
                        print "wrote plot"

                save_wiggles(wave, runid=runid, template_params=fit_params)
                if method != "load":
                    store_template_params(wave, event, fit_params, runid, method)
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
    parser.add_option("--init_runid", dest="init_runid", default=None, type="int", help="initialize template fitting with results from this runid")
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
        (st, et) read_timerange(cursor, "training", hours=None, skip=0):
    else:
        st = options.start_time
        et = options.end_time

    if not (options.evid is None and options.orid is None):
        ev = Event(evid=options.evid, orid=options.orid)
    else:
        raise Exception("Must specify event id (evid) or origin id (orid) to fit.")

    try:
        fit_event_segment(event=ev, sta=options.sta, tm = tm, runid=options.runid, init_runid=options.init_runid, plot=options.plot, method=options.method, wiggles=options.wiggles)
    except KeyboardInterrupt:
        s.dbconn.commit()
        raise
    except:
        s.dbconn.commit()
        print traceback.format_exc()
        continue

if __name__ == "__main__":
    main()






