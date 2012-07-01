import os, sys, traceback
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

from utils.draw_earth import draw_events, draw_earth, draw_density
import utils.nonparametric_regression as nr
from priors.coda_decay.coda_decay_common import *
from priors.coda_decay.plot_multi_station_params import *
from priors.coda_decay.train_coda_models import CodaModel
from priors.coda_decay.signal_likelihood import TraceModel

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


def plot_events_heat_single(pp, fit_data, cm, X, sll, model_type, title):

    f = lambda lon, lat: cm.predict(np.array((lon, lat, 0, 0, 5.0, 0, 0)), model_type)
    bmap, (aa, bb) = plot_heat(pp, f, lonbounds=[-180, 180], latbounds=[-70, 70], n=40)
    draw_events(bmap, (sll,),  marker="x", ms=50, mfc="none", mec="purple", mew=5)
    draw_events(bmap, X, marker="o", ms=5, mfc="none", mec="yellow", mew=2)
    plt.title(title)
    pp.savefig()



def plot_events_heat(pp, fit_data, cm, target_str = ""):

    siteid = fit_data[0, FIT_SITEID]
    X = fit_data[ : , [FIT_LON, FIT_LAT] ]

    cursor = database.db.connect().cursor()

    cursor.execute("SELECT lon, lat from static_siteid where id = %d" % (siteid))
    (slon, slat) = cursor.fetchone()

    model_list = [("LLD", CodaModel.MODEL_TYPE_GP_LLD, cm.lld_params),
                  ("DAD", CodaModel.MODEL_TYPE_GP_DAD, cm.dad_params),
                  ("LLDDA_SUM", CodaModel.MODEL_TYPE_GP_LLDDA_SUM, cm.lldda_sum_params),
                  ("LLDDA_PROD", CodaModel.MODEL_TYPE_GP_LLDDA_PROD, cm.lldda_prod_params)]
    for (model_str, model_type, params) in model_list:
        title = target_str + " " + model_str + "\n" + str(cm.lld_params)
        plot_events_heat_single(pp, fit_data, cm, X, (slon, slat), model_type, title)


def main():
    parser = OptionParser()

    parser.add_option("-s", "--siteid", dest="siteid", default=None, type="int", help="siteid of station for which to generate plots")
    parser.add_option("-r", "--runid", dest="runid", default=None, type="int", help="runid of coda fits to examine")
    parser.add_option("-c", "--channel", dest="chan", default="BHZ", type="str", help="name of channel to examine (BHZ)")
    parser.add_option("-n", "--band", dest="short_band", default="2.00_3.00", type="str", help="name of band to examine (2.00_3.00)")

    (options, args) = parser.parse_args()

    cursor = db.connect().cursor()

    base_coda_dir = get_base_dir(int(options.siteid), int(options.runid))

    for (phase_label, phaseids) in (('P', P_PHASEIDS),):

        print "loading %s fit data... " % (phase_label),

        # cache data for faster development
        fit_data = load_shape_data(cursor, chan=options.chan, short_band=options.short_band,siteid = options.siteid, runids=[options.runid,], phaseids = phaseids)
        print str(fit_data.shape[0]) + " entries loaded"

        band_dir = os.path.join(base_coda_dir, options.short_band)
        fname = os.path.join(band_dir, "%s_predictions_%s.pdf" % (phase_label, options.chan))
        pp = PdfPages(fname)

        print "saving plots to", fname

        lld_params = {"decay": [.01, .02, 100], "onset": [4.67, 2.3866, 800], "amp_transfer": [.4, .4, 100]}
#        dad_params = {"decay": [.02, .4, 2, .05, 1], "onset": [.02, .4, 2, .05, 1], "amp_transfer": [.02, .4, 2, .05, 1]}
        dad_params = {"decay": [.01, .02, 2, 0.0001, 0.0001], "onset": [2, 5, 2, 0.0001, 0.0001], "amp_transfer": [.3, .8, 2, 0.0001, 0.0001]}

        # sigma_n dist_mag dist_scale azi_mag azi_scale depth_mag depth_scale ll_mag ll_scale
        lldda_sum_params = {"decay": [.01, .05, 1, 0.00001, 20, 0.000001, 1, .05, 300], "onset": [2, 5, 1, 0.00001, 20, 0.000001, 1, 5, 300], "amp_transfer": [.4, 0.00001, 1, 0.00001, 20, 0.00001, 1, .4, 800] }

        # sigma_n sigma_f dist_scale azi_scale depth_scale ll_scale
        lldda_prod_params = {"decay": [.01, .02, 1, 200, 10, 300], "onset": [2, 5, 1, 200, 10, 300], "amp_transfer": [.3, .8, 1, 200, 10, 300] }


        for target_str in ["decay", "amp_transfer", "onset"]:
            print "evaluating starting hyperparams for", target_str
            cv_external(cursor, fit_data, band_dir, phaseids, options.chan, target_str=target_str, pp = None, dad_params=dad_params[target_str])
            cm = CodaModel(fit_data, band_dir, phaseids, options.chan, target_str=target_str, ignore_evids = [] , dad_params=dad_params[target_str], optimize=True)
            print "evaluating learned hyperparams for", target_str
            cv_external(cursor, fit_data, band_dir, phaseids, options.chan, target_str=target_str, pp = None, dad_params=cm.dad_params)
#            plot_events_heat(pp, fit_data, cm, target_str)

        pp.close()


if __name__ == "__main__":
    main()


def gridsearch_dad(cursor, fit_data, band_dir, phaseids, chan, target_str, pp):
    sigma_n_vals = [.01, .05, .5, 1, 3]
    sigma_f_vals = [.01, .05, .5, 1, 3]
    w_vals = [.5, 1, 2]
    azi_scale_vals = [.0001, ]
    depth_scale_vals = [.0001,]

