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
import learn, sigvisa_util
import signals.SignalPrior
from utils.waveform import *
from utils.plot_multi_station_params import *
import utils.geog
import obspy.signal.util

from utils.draw_earth import draw_events, draw_earth, draw_density
import utils.nonparametric_regression as nr
from priors.coda_decay.coda_decay_common import *

from priors.coda_decay.train_coda_models import CodaModel
from priors.coda_decay.signal_likelihood import TraceModel

def cv_residuals_kernel(data, x_indices, y_index, kernel=None):
    kr = []
    linr = []
    n = data.shape[0]
    for i in range(n):
        x0 = data[i, x_indices]
        y0 = data[i, y_index]

        #X = lb[ [j for j in range(n) if j != i], x_indices ]
        X = data[ : , x_indices ]
        X = X[[j for j in range(n) if j != i], :]
        y = data[ [j for j in range(n) if j != i], y_index ]

        kpy0 = nr.kernel_predict(x0, X, y, kernel=kernel)
#        kpy0 = nr.knn_predict(x0, X, y, k = 3)

        kr.append(np.abs(kpy0-y0)[0])

        X =np.hstack([X, np.ones((X.shape[0], 1))])
        # w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        w = np.linalg.lstsq(X, y)[0]

        linpy0 = np.dot(w, np.hstack([x0, [1]]))
        linr.append(np.abs(linpy0-y0)[0])
        # print y0, py0

#    print "knn", kr
    return kr, linr


def cv_residuals_mean(data, y_index):
    r = []
    n = data.shape[0]
    for i in range(n):
        y0 = data[i, y_index]
        #X = lb[ [j for j in range(n) if j != i], x_indices ]
        y = data[ [j for j in range(n) if j != i], y_index ]

        py0 = np.mean(y)

        # print y0, py0
        r.append(np.abs(py0-y0)[0])
#    print "mean", r
    return r

def plot_residuals(pp, quantity, P, vert, gp_residuals, gpd_residuals, gpt_residuals, lin_residuals, mean_residuals):
    gp_residuals = np.asarray(gp_residuals)
    gpd_residuals = np.asarray(gpd_residuals)
    gpt_residuals = np.asarray(gpt_residuals)
    lin_residuals = np.asarray(lin_residuals)
    mean_residuals = np.asarray(mean_residuals)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar([1.5,2.5,3.5,4.5, 5.5], [np.mean(np.abs(gp_residuals)), np.mean(np.abs(gpd_residuals)), np.mean(np.abs(gpt_residuals)), np.mean(np.abs(lin_residuals)), np.mean(np.abs(mean_residuals))], 0.5, color='r', yerr=[np.std(np.abs(gp_residuals)), np.std(np.abs(gpd_residuals)), np.std(np.abs(gpt_residuals)), np.std(np.abs(lin_residuals)), np.std(np.abs(mean_residuals))   ])
    ax.set_ylabel('Residual')
    ax.set_title('%s Residuals (P=%s, vert=%s)' % (quantity, P, vert))
    ax.set_xticks([1.5,2.5,3.5,4.5, 5.5])
    ax.set_xticklabels( ('GP', 'GP+D', 'GP/TT', 'Linear', 'Const') )
    pp.savefig()

    fig = plt.figure()
#    print gpd_residuals,
#    print "mean", mean_residuals
    plt.hist(gpd_residuals, bins=10)
    plt.title('%s GPD Residual Distribution (P=%s, vert=%s)' % (quantity, P, vert))
    pp.savefig()

    fig = plt.figure()
    plt.hist(mean_residuals, bins=10)
    plt.title('%s Baseline Residual Distribution (P=%s, vert=%s)' % (quantity, P, vert))
    pp.savefig()

def cv_external(cursor, band_data, band_idx, band_dir, P, vert, pp = None, w = .001, sigma_f = 500, sigma_n = 0.00001):


    gp_decay_residuals = []
    gp_onset_residuals = []
    gp_amp_residuals = []
    gpd_decay_residuals = []
    gpd_onset_residuals = []
    gpd_amp_residuals = []
    gpt_decay_residuals = []
    gpt_onset_residuals = []
    gpt_amp_residuals = []
    lin_decay_residuals = []
    lin_onset_residuals = []
    lin_amp_residuals = []
    mean_decay_residuals = []
    mean_onset_residuals = []
    mean_amp_residuals = []


    earthmodel = None
    netmodel = None

    gen_decay = None

    evids = list(set(band_data[:, EVID_COL]))
    for evid in evids:
        row = None
        for r in band_data:
            if r[EVID_COL] == evid and r[BANDID_COL] == band_idx:
                row = r
                break

        cm = CodaModel(band_data, band_dir, P, vert, ignore_evids = (int(evid),) , earthmodel=earthmodel, netmodel = netmodel, w = [w, w, w], sigma_n = [sigma_n,sigma_n,sigma_n], sigma_f = [sigma_f,sigma_f,sigma_f])
        earthmodel = cm.earthmodel
        netmodel = cm.netmodel

        if gen_decay is None:
            (b_col, gen_decay, gen_onset, gen_amp) = construct_output_generators(cursor, netmodel, P, vert)

        ev = row_to_ev(cursor, row)
        true_decay = gen_decay(row)
        true_onset = gen_onset(row)
        true_amp = gen_amp(row)

        gp_decay_residuals.append(cm.predict_decay(ev, CodaModel.MODEL_TYPE_GP, row[DISTANCE_COL]) - true_decay)
        gp_onset_residuals.append(cm.predict_peak_time(ev, CodaModel.MODEL_TYPE_GP, row[DISTANCE_COL]) - true_onset)
        gp_amp_residuals.append(cm.predict_peak_amp(ev, CodaModel.MODEL_TYPE_GP, row[DISTANCE_COL]) - true_amp)
        gpd_decay_residuals.append(cm.predict_decay(ev, CodaModel.MODEL_TYPE_GPD, row[DISTANCE_COL]) - true_decay)
        gpd_onset_residuals.append(cm.predict_peak_time(ev, CodaModel.MODEL_TYPE_GPD, row[DISTANCE_COL]) - true_onset)
        gpd_amp_residuals.append(cm.predict_peak_amp(ev, CodaModel.MODEL_TYPE_GPD, row[DISTANCE_COL]) - true_amp)
        gpt_decay_residuals.append(cm.predict_decay(ev, CodaModel.MODEL_TYPE_GPT, row[DISTANCE_COL]) - true_decay)
        gpt_onset_residuals.append(cm.predict_peak_time(ev, CodaModel.MODEL_TYPE_GPT, row[DISTANCE_COL]) - true_onset)
        gpt_amp_residuals.append(cm.predict_peak_amp(ev, CodaModel.MODEL_TYPE_GPT, row[DISTANCE_COL]) - true_amp)
        lin_decay_residuals.append(cm.predict_decay(ev, CodaModel.MODEL_TYPE_LINEAR, row[DISTANCE_COL]) - true_decay)
        lin_onset_residuals.append(cm.predict_peak_time(ev, CodaModel.MODEL_TYPE_LINEAR, row[DISTANCE_COL]) - true_onset)
        lin_amp_residuals.append(cm.predict_peak_amp(ev, CodaModel.MODEL_TYPE_LINEAR, row[DISTANCE_COL]) - true_amp)
        mean_decay_residuals.append(cm.predict_decay(ev, CodaModel.MODEL_TYPE_GAUSSIAN, row[DISTANCE_COL]) - true_decay)
        mean_onset_residuals.append(cm.predict_peak_time(ev, CodaModel.MODEL_TYPE_GAUSSIAN, row[DISTANCE_COL]) - true_onset)
        mean_amp_residuals.append(cm.predict_peak_amp(ev, CodaModel.MODEL_TYPE_GAUSSIAN, row[DISTANCE_COL]) - true_amp)
#        print "cv evid %d / %d" % (evid, len(evids))


    print gp_onset_residuals
    print mean_onset_residuals


    print "cross-validated %d events, found residuals:" % (len(evids))
    print "decay: gp %f gpd %f gpt %f lin %f mean %f" % (np.mean(np.abs(gp_decay_residuals)), np.mean(np.abs(gpd_decay_residuals)), np.mean(np.abs(gpt_decay_residuals)), np.mean(np.abs(lin_decay_residuals)), np.mean(np.abs(mean_decay_residuals)))
    print "onset: gp %f gpd %f gpt %f lin %f mean %f" % (np.mean(np.abs(gp_onset_residuals)), np.mean(np.abs(gpd_onset_residuals)), np.mean(np.abs(gpt_onset_residuals)), np.mean(np.abs(lin_onset_residuals)), np.mean(np.abs(mean_onset_residuals)))
    print "amp: gp %f gpd %f gpt %f lin %f mean %f" % (np.mean(np.abs(gp_amp_residuals)), np.mean(np.abs(gpd_amp_residuals)), np.mean(np.abs(gpt_amp_residuals)), np.mean(np.abs(lin_amp_residuals)), np.mean(np.abs(mean_amp_residuals)))

    if pp is not None:
        plot_residuals(pp, "Decay", P, vert, gp_decay_residuals, gpd_decay_residuals, gpt_decay_residuals, lin_decay_residuals, mean_decay_residuals)
        plot_residuals(pp, "Onset", P, vert, gp_onset_residuals, gpd_onset_residuals, gpt_onset_residuals, lin_onset_residuals, mean_onset_residuals)
        plot_residuals(pp, "Log Amplitude", P, vert, gp_amp_residuals, gpd_amp_residuals, gpt_amp_residuals, lin_amp_residuals, mean_amp_residuals)

#    print "onset: gp %f gpd %f lin %f mean %f" % (np.mean(np.abs(gp_onset_residuals)), np.mean(np.abs(gpd_onset_residuals)), np.mean(np.abs(lin_onset_residuals)), np.mean(np.abs(mean_onset_residuals)))
#    print "amp: gp %f gpd %f lin %f mean %f" % (np.mean(np.abs(gp_amp_residuals)), np.mean(np.abs(gpd_amp_residuals)), np.mean(np.abs(lin_amp_residuals)), np.mean(np.abs(mean_amp_residuals)))



#    cm = CodaModel(band_data, band_dir, P, vert, ignore_evids = None , earthmodel=earthmodel, netmodel = netmodel)
#    return cm


def cross_validate(data, b_col):
#    kresiduals_da = None
#    bestr = np.inf
#    bestw = 0
#    for w in np.linspace(1, 1000, 20):
#        ll_kernel = lambda ( ll1, ll2) : 1/w * np.exp(-1 * utils.geog.dist_km(ll1, ll2)/ w**2)
#        krs, lresiduals_da = cv_residuals_kernel(lb, [LON_COL, LAT_COL], [b_col], kernel=ll_kernel)
#        r = np.mean(krs)
#        print r
#        if r < bestr:
#            bestr = r
#            bestw = w
#            kresiduals_da = krs
#    print "best w", bestw

    w = 30
    ll_kernel = lambda ( ll1, ll2) : np.exp(-1 * utils.geog.dist_km(ll1, ll2)**2/ (w**2))
    #ll_kernel = lambda ( ll1, ll2) : utils.geog.dist_km(ll1, ll2)
    kresiduals_da30, lresiduals_da = cv_residuals_kernel(data, [LON_COL, LAT_COL], [b_col], kernel=ll_kernel)
    w = 5
    ll_kernel = lambda ( ll1, ll2) : np.exp(-1 * utils.geog.dist_km(ll1, ll2)**2 / (w**2))
    #ll_kernel = lambda ( ll1, ll2) : utils.geog.dist_km(ll1, ll2)
    kresiduals_da5, lresiduals_da = cv_residuals_kernel(data, [LON_COL, LAT_COL], [b_col], kernel=ll_kernel)


    kresiduals_a, lresiduals_a = cv_residuals_kernel(data, [AZI_COL], [b_col])
    kresiduals_d, lresiduals_d = cv_residuals_kernel(data, [DISTANCE_COL], [b_col])

    residuals_m = cv_residuals_mean(data, [b_col])

    ll30r = np.mean(kresiduals_da30)
    ll5r = np.mean(kresiduals_da5)
    dr = np.mean(lresiduals_d)
    mr = np.mean(residuals_m)

    return (ll30r, ll5r, dr, mr)


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


def plot_events_heat(pp, data, cm):

    siteid = data[0, SITEID_COL]
    X = data[ : , [LON_COL, LAT_COL] ]

    mins = np.min(X, 0)
    maxs = np.max(X, 0)

#    if maxs[0]-mins[0] > 300:
#        for i in range(X.shape[0]):
#            X[i, 0] = ((X[i,0] + 360) % 360) - 180
#        mins = np.min(X, 0)
#        maxs = np.max(X, 0)

    cursor = database.db.connect().cursor()

    cursor.execute("SELECT lon, lat from static_siteid where id = %d" % (siteid))
    (slon, slat) = cursor.fetchone()

    if maxs[0]-mins[0] > 300:
        print "%f - %f = %f, rotating" % (maxs[0], mins[0], maxs[0] - mins[0])
        new_slon = ((slon + 360) % 360) - 180
        minlon = new_slon
        maxlon = new_slon
        print "starting with [%f %f]..." % (minlon, maxlon)
        for i in range(X.shape[0]):
            minlon = min(minlon, ((X[i, 0] + 360) % 360) - 180)
            maxlon = max(maxlon, ((X[i, 0] + 360) % 360) - 180)

        print "got lon [%f %f], rotating back..." % (minlon, maxlon)

        min_lon = minlon - 180
        max_lon = maxlon - 180
    else:
        min_lon = np.min([slon, mins[0]]) - 3
        max_lon = np.max([slon, maxs[0]]) + 3

    min_lat = np.min([slat, mins[1]]) - 3
    max_lat = np.max([slat, maxs[1]]) + 3

    print "mlon: %f - %f = %f" % (max_lon, min_lon, max_lon - min_lon)
    print "mlat: %f - %f = %f" % (max_lat, min_lat, max_lat - min_lat)

    f = lambda lon, lat: cm.predict_decay(np.array((5.0, lon, lat, -1, 0, 0)), CodaModel.MODEL_TYPE_GP)
    bmap = plot_heat(pp, f, lonbounds=[min_lon-2, max_lon+2], latbounds=[min_lat-2, max_lat+2])

    draw_events(bmap, ((slon, slat),),  marker="x", ms=50, mfc="none", mec="purple", mew=5)
    draw_events(bmap, X, marker="o", ms=5, mfc="none", mec="yellow", mew=2)
    pp.savefig()


def main():
    parser = OptionParser()

    parser.add_option("-s", "--siteid", dest="siteid", default=None, type="int", help="siteid of station for which to generate plots")
    parser.add_option("-r", "--runid", dest="runid", default=None, type="int", help="runid of coda fits to examine")
    parser.add_option("-b", "--basedir", dest="basedir", default=None, type=str, help="")

#    parser.add_option("--scatter", dest="scatter", default=False, action="store_true", help="create scatter plots (False)")
#    parser.add_option("--events", dest="events", default=False, action="store_true", help="(re)creates individual event coda plots (False)")
#    parser.add_option("--merge", dest="merge", default=False, action="store_true", help="merge all available plots for each band (False)")

    (options, args) = parser.parse_args()

    cursor = db.connect().cursor()

    if options.basedir is None:
        siteid = options.siteid
        runid = options.runid

        base_coda_dir = get_base_dir(int(siteid), None, int(runid))
    else:
        base_coda_dir = options.basedir

    fname = os.path.join(base_coda_dir, 'all_data')
    all_data, bands = read_shape_data(fname)

    all_data = add_depth_time(cursor, all_data)

    band_idx = 1
    band = bands[1]
#    for (band_idx, band) in enumerate(bands):
    p_fname = os.path.join(base_coda_dir, band[19:], "p_predictions.pdf")
    pp_p = PdfPages(p_fname)

    s_fname = os.path.join(base_coda_dir, band[19:], "s_predictions.pdf")
    pp_s = PdfPages(s_fname)
    print "saving plots to", p_fname, s_fname

    band_data = extract_band(all_data, band_idx)
    band_dir = os.path.join(base_coda_dir, band[19:])
    print "processing band %s, cleaning %d points..." % (band, band_data.shape[0])
    clean_p_data = clean_points(band_data, P=True, vert=True)
    clean_s_data = clean_points(band_data, P=False, vert=False)

    cv_external(cursor, clean_p_data, band_idx, band_dir, True, True, pp = pp_p, sigma_f=.01, w=100, sigma_n=.01)
    pp_p.close()

    print "doing s stuff now..."
    cv_external(cursor, clean_s_data, band_idx, band_dir, False, False, pp = pp_s, sigma_f=.01, w=100, sigma_n=.01)
 #   cm_s = CodaModel(band_data, band_dir, False, False, sigma_f = [0.01, 1, 1], w = [100, 100, 100], sigma_n = [0.01, 0.01, 0.01])
#    plot_event_heat(pp_s, clean_s_data, cm, 100)
    pp_s.close()

if __name__ == "__main__":
    main()








