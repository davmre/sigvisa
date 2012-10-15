import os, sys, traceback, pdb
import numpy as np, scipy


from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from optparse import OptionParser
import plot
import learn, sigvisa_util, sigvisa
import signals.SignalPrior
from utils.waveform import *

import utils.geog
import obspy.signal.util

from plotting.heatmap import Heatmap

from utils.draw_earth import draw_events, draw_earth, draw_density
import utils.nonparametric_regression as nr
from signals.coda_decay_common import *
from signals.plot_multi_station_params import *
from signals.train_coda_models import CodaModel
from signals.signal_likelihood import TraceModel
from signals.signal_likelihood import load_gp_params

def plot_residuals(pp, quantity, phaseids, chan, residuals, labels):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    means = [np.mean(np.abs(r)) for r in residuals]
    stds = [np.std(np.abs(r)) for r in residuals]
    locs = [i+1.5 for i in range(len(residuals))]
    rects1 = ax.bar(locs, means, 0.5, color='r', yerr=stds)
    ax.set_ylabel('Residual')
    ax.set_title('%s Residuals (phaseids=%s, chan=%s)' % (quantity, phaseids, chan))
    ax.set_xticks([1.5,2.5,3.5,4.5])
    ax.set_xticklabels( labels )
    pp.savefig()

    for (i, r) in enumerate(residuals):

        fig = plt.figure()
        #    print gp_loc_residuals,
        #    print "mean", mean_residuals
        plt.hist(r, bins=10)
        plt.title('%s %s Residual Distribution (phaseids=%s, chan=%s)' % (quantity, labels[i], phaseids, chan))
        pp.savefig()

def cv_generator(n, k=5):
    data = np.random.permutation(n)
    fold_size = n/k
    folds = [data[i*fold_size:(i+1)*fold_size] for i in range(k)]
    folds[k-1] = data[(k-1)*fold_size:]
    for i in range(k):
        train = np.array(())
        for j in range(k):
            if j != i:
                train = np.concatenate([train, folds[j]])
        test = folds[i]
        yield (train, test)

def cv_external(cursor, fit_data, band_dir, phaseids, chan, target_str, pp = None, lld_params=None, dad_params=None, lldda_sum_params=None, lldda_prod_params=None):

    gp_lld_residuals = []
    gp_dad_residuals = []
    gp_lldda_sum_residuals = []
    gp_lldda_prod_residuals = []
    lin_residuals = []
    mean_residuals = []

    earthmodel = None
    sigmodel = None
    sites = None

    evids = np.array(list(fit_data[:, FIT_EVID]))
    for (train_indices, test_indices) in cv_generator(len(evids)):
        test_evids = evids[test_indices]

        cm = CodaModel(fit_data, band_dir, phaseids, chan, target_str=target_str, ignore_evids = test_evids , earthmodel=earthmodel, sigmodel = sigmodel, sites=sites, lld_params=lld_params, dad_params=dad_params, lldda_sum_params=lldda_sum_params, lldda_prod_params = lldda_prod_params, debug=False)
        earthmodel = cm.earthmodel
        sigmodel = cm.sigmodel
        sites = cm.sites

        for idx in test_indices:
            evid = evids[idx]
            ev = load_event(cursor, evid)
            row = fit_data[idx, :]
            true_output = cm.target_fns[target_str](row)

#            gp_lld_residuals.append(cm.predict(ev, CodaModel.MODEL_TYPE_GP_LLD, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_output)
            gp_dad_residuals.append(cm.predict(ev, CodaModel.MODEL_TYPE_GP_DAD, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_output)
#            gp_lldda_sum_residuals.append(cm.predict(ev, CodaModel.MODEL_TYPE_GP_LLDDA_SUM, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_output)
#            gp_lldda_prod_residuals.append(cm.predict(ev, CodaModel.MODEL_TYPE_GP_LLDDA_PROD, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_output)

            lin_residuals.append(cm.predict(ev, CodaModel.MODEL_TYPE_LINEAR, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_output)
            mean_residuals.append(cm.predict(ev, CodaModel.MODEL_TYPE_GAUSSIAN, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_output)


    print "cross-validated %d events, found residuals for %s:" % (len(evids), target_str)
#    print "mean: gp_lld %f gp_dad %f gp_lldda_sum %f gp_lldda_prod %f lin %f mean %f" % (np.mean(np.abs(gp_lld_residuals)), np.mean(np.abs(gp_dad_residuals)), np.mean(np.abs(gp_lldda_sum_residuals)), np.mean(np.abs(gp_lldda_prod_residuals)), np.mean(np.abs(lin_residuals)), np.mean(np.abs(mean_residuals)))
#    print "median: gp_lld %f gp_dad %f gp_lldda_sum %f gp_lldda_prod %f lin %f median %f" % (np.median(np.abs(gp_lld_residuals)), np.median(np.abs(gp_dad_residuals)), np.median(np.abs(gp_lldda_sum_residuals)),np.median(np.abs(gp_lldda_prod_residuals)), np.median(np.abs(lin_residuals)), np.median(np.abs(mean_residuals)))
    print "mean: gp_dad %f lin %f mean %f" % (np.mean(np.abs(gp_dad_residuals)), np.mean(np.abs(lin_residuals)), np.mean(np.abs(mean_residuals)))
    print "median: gp_dad %f lin %f mean %f" % (np.median(np.abs(gp_dad_residuals)), np.median(np.abs(lin_residuals)), np.median(np.abs(mean_residuals)))

    if pp is not None:
        plot_residuals(pp, target_str, phaseids, chan, [gp_lld_residuals, gp_dad_residuals, gp_lldda_sum_residuals, gp_lldda_prod_residuals, lin_residuals, mean_residuals], labels = ('GP-LLD', 'GP-DAD', 'GP-LLDDA (Sum)', 'GP-LLDDA (Prod)', 'Linear', 'Const'))


def plot_linear(pp, data, b_col, title=""):

    X = data[ : , DISTANCE_COL ]
    y = data[ : , b_col ]

    w = np.linalg.lstsq(np.vstack([X, np.ones((X.shape[0],))]).T, y)[0]

    plt.figure()
    plt.title(title)
    plt.xlabel("distance (km)")
    plt.ylabel("b")
    t = np.vstack([np.linspace(np.min(X)-200, np.max(X)+200, 50), np.ones((50,))]).T
    pred = np.dot(t, w)
    plt.ylim([-0.1, 0])
    plt.plot(t, pred, "k-")
    plt.plot(X, y, 'ro')
    pp.savefig()



def plot_event_location_heat(pp, val, cm, X, sll, model_type, title):

    # plot the conditional likelihood of event locations given the event parameters
    f = lambda lon, lat: np.exp(cm.log_likelihood(val, np.array((lon, lat, 0, 0, 5.0, 0, 0)), model_type))

    heatfile = "logs/%s.heat" % str(hashlib.md5(title).hexdigest())
    try:
        hm = Heatmap(f, fname=heatfile)
    except:
        hm = Heatmap(f, lonbounds=[100, 220], latbounds=[-70, 70], n=20)
        hm.calc(checkpoint=heatfile)
    hm.calc(checkpoint="logs/%s.heat" % str(hashlib.md5(title).hexdigest()))
    hm.plot_density()
    hm.plot_locations((sll,),  marker="x", ms=7, mfc="none", mec="white", mew=2)
    hm.plot_locations(X, marker="o", ms=5, mfc="none", mec="red", mew=2)

    
    maxlon, maxlat, maxval = hm.max()
    hm.plot_locations(((maxlon, maxlat),), marker="*", ms=5, mfc="none", mec="green", mew=2)
    dist = utils.geog.dist_km((maxlon, maxlat), X[0])
    
    plt.title(title + "\n distance: %.2f km" % dist)
    pp.savefig()

    return hm, dist


def plot_events_heat_single(pp, cm, X, sll, model_type, title):

    if model_type == cm.MODEL_TYPE_GP_DAD:
        f = lambda lon, lat: min(cm.predict(np.array((lon, lat, 0, 0, 5.0, 0, 0)), model_type), 0)
    else:
        f = lambda lon, lat: cm.predict(np.array((lon, lat, 0, 0, 5.0, 0, 0)), model_type)

    heatfile = "logs/%s.heat" % str(hashlib.md5(title).hexdigest())
    try:
        hm = Heatmap(f, fname=heatfile)
    except:
        hm = Heatmap(f, lonbounds=[100, 220], latbounds=[-70, 70], n=20)
        hm.calc(checkpoint=heatfile)
    hm.plot_density()
    hm.plot_locations((sll,),  marker="x", ms=7, mfc="none", mec="white", mew=2)
    hm.plot_locations(X, marker=".", ms=2, mfc="none", mec="red", mew=2, alpha=0.6)

    plt.title(title)
    pp.savefig()

def plot_events_heat(pp, fit_data, cm, target_str = ""):

    siteid = fit_data[0, FIT_SITEID]
    X = fit_data[ : , [FIT_LON, FIT_LAT] ]

    cursor = database.db.connect().cursor()
    cursor.execute("SELECT lon, lat from static_siteid where id = %d" % (siteid))
    (slon, slat) = cursor.fetchone()

    model_list = [#("LLD", CodaModel.MODEL_TYPE_GP_LLD, cm.lld_params),
                  ("DAD", CodaModel.MODEL_TYPE_GP_DAD, cm.dad_params),
                  #("LLDDA_SUM", CodaModel.MODEL_TYPE_GP_LLDDA_SUM, cm.lldda_sum_params),
                  #("LLDDA_PROD", CodaModel.MODEL_TYPE_GP_LLDDA_PROD, cm.lldda_prod_params)
                  ]
    for (model_str, model_type, params) in model_list:
        title = target_str + " " + model_str + "\n" + str(cm.lld_params)
        plot_events_heat_single(pp, cm, X, (slon, slat), model_type, title)


def locate_event_from_model(evid, fit_data, dad_params, pp, band_dir, short_band, chan, (phase_label, phaseids), runid, target_str):

    siteid = fit_data[0, FIT_SITEID]
    # need to get event lcoation to pass as X

    try:
        evrow = [r for r in fit_data if r[FIT_EVID]==evid and r[FIT_PHASEID] in phaseids][0]
    except IndexError:
        raise Exception("event %d not found at station %d" % (evid, siteid))
    evll = evrow[2:4]

    cursor = database.db.connect().cursor()
    cursor.execute("SELECT lon, lat from static_siteid where id = %d" % (siteid))
    (slon, slat) = cursor.fetchone()


    # train coda model excluding this event
    cm = CodaModel(fit_data, band_dir, phaseids, chan, target_str=target_str, ignore_evids = [evid,] , dad_params=dad_params, optimize=False, debug=False)

    val = CodaModel.target_fns[target_str](evrow)

    hm, dist = plot_event_location_heat(pp, val, cm, (evll,), (slon, slat), CodaModel.MODEL_TYPE_GP_DAD, "event %d location from %s\nsid %d rid %d band %s chan %s phase %s" % (evid, target_str, siteid, runid, short_band, chan, phase_label))

    return hm, dist

def eval_spatial_model(fit_data, dad_params, pp, band_dir, short_band, chan, (phase_label, phaseids), runid, target_str):


    #            print "evaluating starting hyperparams for", target_str
    #            cv_external(cursor, fit_data, band_dir, phaseids, chan, target_str=target_str, pp = None, dad_params=dad_params[target_str])
    cm = CodaModel(fit_data, band_dir, phaseids, chan, target_str=target_str, ignore_evids = [] , dad_params=dad_params, optimize=False)
    #            print "evaluating learned hyperparams for", target_str
    #            cv_external(cursor, fit_data, band_dir, phaseids, chan, target_str=target_str, pp = None, dad_params=cm.dad_params)
    plot_events_heat(pp, fit_data, cm, target_str)




def main():
    parser = OptionParser()

    parser.add_option("-s", "--siteids", dest="siteids", default=None, type="str", help="comma-separated list of siteids for which to generate plots")
#    parser.add_option("-r", "--runids", dest="runids", default=None, type="str", help="comma-separated list of coda fit runids to examine")
    parser.add_option("-r", "--runid", dest="runid", default=None, type="int", help="coda fit runids to examine")
    parser.add_option("-e", "--evids", dest="evids", default=None, type="str", help="evids of event to locate using its template params")
    parser.add_option("-c", "--channel", dest="chan", default="BHZ", type="str", help="name of channel to examine (BHZ)")
    parser.add_option("-n", "--band", dest="short_band", default="2.00_3.00", type="str", help="name of band to examine (2.00_3.00)")
    parser.add_option("-t", "--targets", dest="targets", default="decay,amp_transfer,onset", type="str", help="comma-separated list of target parameter names (decay,amp_transfer,onset)")

    (options, args) = parser.parse_args()

    cursor, sigmodel, earthmodel, sites, dbconn = sigvisa_util.init_sigmodel()

    density = NestedDict()

    siteids = [int(s) for s in options.siteids.split(',')]
    if options.evids == "all":
        evids="all"
    elif options.evids is None:
        evids = None
    else:
        evids = [int(e) for e in options.evids.split(',')]
    runid=options.runid
    target_strs = [t for t in options.targets.split(',')]

    for siteid in siteids:
        for (phase_label, phaseids) in (('P', P_PHASEIDS),):
            for target_str in target_strs:
                # cache data for faster development
                print "loading %s fit data... " % (phase_label),
                fit_data = load_shape_data(cursor, chan=options.chan, short_band=options.short_band,siteid = siteid, runids=[runid,], phaseids = phaseids)
                print str(fit_data.shape[0]) + " entries loaded"

                if evids == "all":
                    evids = fit_data[:, 0]
                    print "setting evids to", evids

                param_dict = load_gp_params("parameters/gp_hyperparams.txt", "dad")
                dad_params = param_dict[int(siteid)][phase_label]["BHZ"]["2.00_3.00"][target_str]
                print dad_params

                base_coda_dir = get_base_dir(int(siteid), int(runid))
                band_dir = os.path.join(base_coda_dir, options.short_band)

                if options.evids is None:

                    # if no specific event specified, train a model
                    # for each station based on all events from that
                    # station and evaluate/plot its predictions
                    
                    fname = os.path.join(band_dir, "%s_predictions_%s.pdf" % (phase_label, options.chan))
                    pp = PdfPages(fname)
                    print "saving heat map(s) to", fname
                    eval_spatial_model(fit_data, dad_params, pp, band_dir, options.short_band, options.chan, (phase_label, phaseids), runid, target_str)
                    pp.close()
                else:
                    for evid in evids:
                        
                        # if a specific event is specified, train a
                        # model for each station based on all other
                        # events, then try to plot the event location
                        
                        fname = os.path.join(band_dir, "%s_%s_location_%s_%d.pdf" % (phase_label, target_str, options.chan, evid))
                        pp = PdfPages(fname)
                        print "saving heat map(s) to", fname

                        density[evid][siteid][phase_label], dist = locate_event_from_model(evid, fit_data, dad_params, pp, band_dir, options.short_band, options.chan, (phase_label, phaseids), runid, target_str)
                        pp.close()


    if evids is not None:
        for evid in evids:


            fname = "logs/density_%d_%s.pdf" % (evid, target_str)
            pp = PdfPages(fname)

            print "calculating overall density for %d, saved to %s" % (evid, fname)
            od = None


            for sid in density[evid].keys():
                for pl in density[evid][sid].keys():
                    d = density[evid][sid][pl]
                    od = d if od is None else d * od

            priorfn = lambda lon, lat : np.exp(sigmodel.event_location_prior_logprob( (lon + 180) % 360 - 180, lat, 0))
            prior = Heatmap(fn = priorfn, n = od.n, lonbounds=[od.min_lon, od.max_lon],
                            latbounds=[od.min_lat, od.max_lat])
            prior.calc()

            cursor.execute("SELECT lon, lat from static_siteid where id = %d" % (siteid))
            (slon, slat) = cursor.fetchone()
            cursor.execute("SELECT lon, lat from leb_origin where evid = %d" % (evid))
            (evlon, evlat) = cursor.fetchone()

            prior.plot_density()
            prior.plot_locations(((slon,slat),),  marker="x", ms=7, mfc="none", mec="white", mew=2)
            prior.plot_locations(((evlon, evlat),), marker="o", ms=5, mfc="none", mec="red", mew=2)
            maxlon, maxlat, maxval = prior.max()
            prior.plot_locations(((maxlon, maxlat),), marker="*", ms=5, mfc="none", mec="green", mew=2)
            dist = utils.geog.dist_km((maxlon, maxlat), (evlon, evlat))
            title = "%d prior: dist %.2f km" % (evid, dist)
            print title
            plt.title(title)
           
            pp.savefig()
            plt.clf()

            od.init_bmap()
            od.plot_density()
            od.plot_locations(((slon,slat),),  marker="x", ms=7, mfc="none", mec="white", mew=2)
            od.plot_locations(((evlon, evlat),), marker="o", ms=5, mfc="none", mec="red", mew=2)
            maxlon, maxlat, maxval = od.max()
            od.plot_locations(((maxlon, maxlat),), marker="*", ms=5, mfc="none", mec="green", mew=2)
            dist = utils.geog.dist_km((maxlon, maxlat), (evlon, evlat))
            title = "%d likelihood: dist %.2f km" % (evid, dist)
            print title
            plt.title(title)

            pp.savefig()
            plt.clf()

            posterior = prior * od
            posterior.plot_density()
            posterior.plot_locations(((slon,slat),),  marker="x", ms=7, mfc="none", mec="white", mew=2)
            posterior.plot_locations(((evlon, evlat),), marker="o", ms=5, mfc="none", mec="red", mew=2)
            maxlon, maxlat, maxval = posterior.max()
            posterior.plot_locations(((maxlon, maxlat),), marker="*", ms=5, mfc="none", mec="green", mew=2)
            dist = utils.geog.dist_km((maxlon, maxlat), (evlon, evlat))
            title = "%d posterior: dist %.2f km" % (evid, dist)
            print title
            plt.title(title)

            pp.savefig()

            pp.close()

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        raise
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)



def gridsearch_dad(cursor, fit_data, band_dir, phaseids, chan, target_str, pp):
    sigma_n_vals = [.01, .05, .5, 1, 3]
    sigma_f_vals = [.01, .05, .5, 1, 3]
    w_vals = [.5, 1, 2]
    azi_scale_vals = [.0001, ]
    depth_scale_vals = [.0001,]

