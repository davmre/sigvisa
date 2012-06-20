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

def cv_external(cursor, fit_data, band_dir, phaseids, chan, pp = None, w = None, sigma_f = None, sigma_n = None):

    gp_loc_decay_residuals = []
    gp_loc_onset_residuals = []
    gp_loc_amp_residuals = []
    gp_ad_decay_residuals = []
    gp_ad_onset_residuals = []
    gp_ad_amp_residuals = []
    lin_decay_residuals = []
    lin_onset_residuals = []
    lin_amp_residuals = []
    mean_decay_residuals = []
    mean_onset_residuals = []
    mean_amp_residuals = []

    earthmodel = None
    sigmodel = None
    sites = None

    gen_decay = None

    evids = np.array(list(set(fit_data[:, FIT_EVID])))
    for (train_indices, test_indices) in cv_generator(len(evids)):
        test_evids = evids[test_indices]
        cm = CodaModel(fit_data, band_dir, phaseids, chan, ignore_evids = test_evids , earthmodel=earthmodel, sigmodel = sigmodel, sites=sites, w = w, sigma_n = sigma_n, sigma_f = sigma_f, debug=False)
        earthmodel = cm.earthmodel
        sigmodel = cm.sigmodel
        sites = cm.sites

        for idx in test_indices:
            evid = evids[idx]
            ev = load_event(cursor, evid)
            row = fit_data[idx, :]
            true_decay = row[FIT_CODA_DECAY]
            true_onset = row[FIT_PEAK_DELAY]
            true_amp = gen_source_amp(row)

            gp_loc_decay_residuals.append(cm.predict_decay(ev, CodaModel.MODEL_TYPE_GP_LOC, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_decay)
            gp_loc_onset_residuals.append(cm.predict_peak_time(ev, CodaModel.MODEL_TYPE_GP_LOC, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_onset)
            gp_loc_amp_residuals.append(cm.predict_peak_amp(ev, CodaModel.MODEL_TYPE_GP_LOC, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_amp)
            gp_ad_decay_residuals.append(cm.predict_decay(ev, CodaModel.MODEL_TYPE_GP_AD, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_decay)
            gp_ad_onset_residuals.append(cm.predict_peak_time(ev, CodaModel.MODEL_TYPE_GP_AD, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_onset)
            gp_ad_amp_residuals.append(cm.predict_peak_amp(ev, CodaModel.MODEL_TYPE_GP_AD, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_amp)
            lin_decay_residuals.append(cm.predict_decay(ev, CodaModel.MODEL_TYPE_LINEAR, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_decay)
            lin_onset_residuals.append(cm.predict_peak_time(ev, CodaModel.MODEL_TYPE_LINEAR, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_onset)
            lin_amp_residuals.append(cm.predict_peak_amp(ev, CodaModel.MODEL_TYPE_LINEAR, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_amp)
            mean_decay_residuals.append(cm.predict_decay(ev, CodaModel.MODEL_TYPE_GAUSSIAN, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_decay)
            mean_onset_residuals.append(cm.predict_peak_time(ev, CodaModel.MODEL_TYPE_GAUSSIAN, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_onset)
            mean_amp_residuals.append(cm.predict_peak_amp(ev, CodaModel.MODEL_TYPE_GAUSSIAN, row[FIT_DISTANCE], row[FIT_AZIMUTH]) - true_amp)
    #        print "cv evid %d / %d" % (evid, len(evids))



    print "cross-validated %d events, found residuals:" % (len(evids))
    print "decay: gpd %f gpt %f lin %f mean %f" % (np.mean(np.abs(gp_loc_decay_residuals)), np.mean(np.abs(gp_ad_decay_residuals)), np.mean(np.abs(lin_decay_residuals)), np.mean(np.abs(mean_decay_residuals)))
    print "onset: gpd %f gpt %f lin %f mean %f" % (np.mean(np.abs(gp_loc_onset_residuals)), np.mean(np.abs(gp_ad_onset_residuals)), np.mean(np.abs(lin_onset_residuals)), np.mean(np.abs(mean_onset_residuals)))
    print "amp: gpd %f gpt %f lin %f mean %f" % (np.mean(np.abs(gp_loc_amp_residuals)), np.mean(np.abs(gp_ad_amp_residuals)), np.mean(np.abs(lin_amp_residuals)), np.mean(np.abs(mean_amp_residuals)))

    if pp is not None:
        plot_residuals(pp, "Decay", phaseids, chan, [gp_loc_decay_residuals, gp_ad_decay_residuals, lin_decay_residuals, mean_decay_residuals], labels = ('GP-LOC', 'GP-AD', 'Linear', 'Const'))
        plot_residuals(pp, "Onset", phaseids, chan, [gp_loc_onset_residuals, gp_ad_onset_residuals, lin_onset_residuals, mean_onset_residuals], labels = ('GP-LOC', 'GP-AD', 'Linear', 'Const'))
        plot_residuals(pp, "Log Amplitude", phaseids, chan, [gp_loc_amp_residuals, gp_ad_amp_residuals, lin_amp_residuals, mean_amp_residuals], labels = ('GP-LOC', 'GP-AD', 'Linear', 'Const'))

#    print "onset: gp %f gpd %f lin %f mean %f" % (np.mean(np.abs(gp_onset_residuals)), np.mean(np.abs(gp_loc_onset_residuals)), np.mean(np.abs(lin_onset_residuals)), np.mean(np.abs(mean_onset_residuals)))
#    print "amp: gp %f gpd %f lin %f mean %f" % (np.mean(np.abs(gp_amp_residuals)), np.mean(np.abs(gp_loc_amp_residuals)), np.mean(np.abs(lin_amp_residuals)), np.mean(np.abs(mean_amp_residuals)))



#    cm = CodaModel(band_data, band_dir, P, chan, ignore_evids = None , earthmodel=earthmodel, sigmodel = sigmodel)
#    return cm

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


def plot_events_heat(pp, fit_data, cm):

    siteid = fit_data[0, FIT_SITEID]
    X = fit_data[ : , [FIT_LON, FIT_LAT] ]

    cursor = database.db.connect().cursor()

    cursor.execute("SELECT lon, lat from static_siteid where id = %d" % (siteid))
    (slon, slat) = cursor.fetchone()


    f = lambda lon, lat: cm.predict_decay(np.array((lon, lat, 0, 0, 5.0, 0, 0)), CodaModel.MODEL_TYPE_GP_LOC)
    bmap, (aa, bb) = plot_heat(pp, f, lonbounds=[-180, 180], latbounds=[-70, 70], n=40)

    draw_events(bmap, ((slon, slat),),  marker="x", ms=50, mfc="none", mec="purple", mew=5)
    draw_events(bmap, X, marker="o", ms=5, mfc="none", mec="yellow", mew=2)
    pp.savefig()

    f = lambda lon, lat: cm.predict_decay(np.array((lon, lat, 0, 0, 5.0, 0, 0)), CodaModel.MODEL_TYPE_GP_AD)
    bmap, (aa, bb) = plot_heat(pp, f, lonbounds=[-180, 180], latbounds=[-90, 90], n=40)

    draw_events(bmap, ((slon, slat),),  marker="x", ms=50, mfc="none", mec="purple", mew=5)
    draw_events(bmap, X, marker="o", ms=5, mfc="none", mec="yellow", mew=2)
    pp.savefig()


def main():
    parser = OptionParser()

    parser.add_option("-s", "--siteid", dest="siteid", default=None, type="int", help="siteid of station for which to generate plots")
    parser.add_option("-r", "--runid", dest="runid", default=None, type="int", help="runid of coda fits to examine")
    parser.add_option("-c", "--channel", dest="chan", default="BHZ", type="str", help="name of channel to examine (BHZ)")
    parser.add_option("-n", "--band", dest="short_band", default="2.00_3.00", type="str", help="name of band to examine (2.00_3.00)")

    (options, args) = parser.parse_args()

    cursor = db.connect().cursor()

    base_coda_dir = get_base_dir(int(options.siteid), int(options.runid))

    p_fit_data = load_shape_data(cursor, chan=options.chan, short_band=options.short_band,siteid = options.siteid, runid=options.runid, phaseids = P_PHASEIDS)
    s_fit_data = load_shape_data(cursor, chan=options.chan, short_band=options.short_band,siteid = options.siteid, runid=options.runid, phaseids = S_PHASEIDS)


    band_dir = os.path.join(base_coda_dir, options.short_band)
    p_fname = os.path.join(band_dir, "p_predictions_%s.pdf" % options.chan)
    pp_p = PdfPages(p_fname)

    s_fname = os.path.join(band_dir, "s_predictions_%s.pdf" % options.chan)
    pp_s = PdfPages(s_fname)
    print "saving plots to", p_fname, s_fname

    w = [1000,1000, 1000]
    sigma_f = [0.01, 1, 1]
    sigma_n = [0.01, 0.5, 0.5]

    cv_external(cursor, p_fit_data, band_dir, P_PHASEIDS, options.chan, pp = pp_p, sigma_f=sigma_f, w=w, sigma_n=sigma_n)
    cm_p = CodaModel(p_fit_data, band_dir, P_PHASEIDS, options.chan, ignore_evids = [] , w = w, sigma_n = sigma_n, sigma_f = sigma_f)
    plot_events_heat(pp_p, p_fit_data, cm_p)
    pp_p.close()

    print "doing s stuff now..."
 #   cv_external(cursor, s_fit_data, band_dir, S_PHASEIDS, options.chan, pp = pp_s, sigma_f=.01, w=100, sigma_n=.01)
    cm_s = CodaModel(s_fit_data, band_dir, S_PHASEIDS, options.chan, ignore_evids = [] , w = w, sigma_n = sigma_n, sigma_f = sigma_f)
    plot_events_heat(pp_s, s_fit_data, cm_s)
    pp_s.close()

if __name__ == "__main__":
    main()








