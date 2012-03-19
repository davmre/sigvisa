import os, sys, traceback
import numpy as np, scipy


from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plot
import learn, sigvisa_util
import priors.SignalPrior
from utils.waveform import *
from utils.plot_multi_station_params import *
import utils.geog
import obspy.signal.util

from utils.draw_earth import draw_events, draw_earth, draw_density
import utils.nonparametric_regression as nr
from priors.coda_decay.coda_decay_common import *

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


def cross_validate(data):
#    kresiduals_da = None
#    bestr = np.inf
#    bestw = 0
#    for w in np.linspace(1, 1000, 20):
#        ll_kernel = lambda ( ll1, ll2) : 1/w * np.exp(-1 * utils.geog.dist_km(ll1, ll2)/ w**2)
#        krs, lresiduals_da = cv_residuals_kernel(lb, [LON_COL, LAT_COL], [B_COL], kernel=ll_kernel)
#        r = np.mean(krs)
#        print r
#        if r < bestr:
#            bestr = r
#            bestw = w
#            kresiduals_da = krs
#    print "best w", bestw

    w = 40
    ll_kernel = lambda ( ll1, ll2) : np.exp(-1 * utils.geog.dist_km(ll1, ll2)/ (w**2))
    #ll_kernel = lambda ( ll1, ll2) : utils.geog.dist_km(ll1, ll2)
    kresiduals_da50, lresiduals_da = cv_residuals_kernel(data, [LON_COL, LAT_COL], [B_COL], kernel=ll_kernel)
    w = 5
    ll_kernel = lambda ( ll1, ll2) : np.exp(-1 * utils.geog.dist_km(ll1, ll2)/ (w**2))
    #ll_kernel = lambda ( ll1, ll2) : utils.geog.dist_km(ll1, ll2)
    kresiduals_da5, lresiduals_da = cv_residuals_kernel(data, [LON_COL, LAT_COL], [B_COL], kernel=ll_kernel)


    kresiduals_a, lresiduals_a = cv_residuals_kernel(data, [AZI_COL], [B_COL])
    kresiduals_d, lresiduals_d = cv_residuals_kernel(data, [DISTANCE_COL], [B_COL])

    residuals_m = cv_residuals_mean(data, [B_COL])

    ll50r = np.mean(kresiduals_da50)
    ll5r = np.mean(kresiduals_da5)
    dr = np.mean(lresiduals_d)
    mr = np.mean(residuals_m)

    return (ll50r, ll5r, dr, mr)
    

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
    
def plot_heat(pp, data, b_col, w):

    siteid = data[0, SITEID_COL]
    X = data[ : , [LON_COL, LAT_COL] ]
    y = data[ : , b_col ]

    mins = np.min(X, 0)
    maxs = np.max(X, 0)

    cursor = database.db.connect().cursor()

    cursor.execute("SELECT lon, lat from static_siteid where id = %d" % (siteid))
    (slon, slat) = cursor.fetchone()
    print slon, slat


    min_lon = np.min([slon, mins[0]])
    max_lon = np.max([slon, maxs[0]])

    min_lat = np.min([slat, mins[1]])
    max_lat = np.max([slat, maxs[1]])

    bmap = draw_earth("",
                  #"NET-VISA posterior density, NEIC(white), LEB(yellow), "
                  #"SEL3(red), NET-VISA(blue)",
                  projection="mill",
                  resolution="l",
                  llcrnrlon = min_lon, urcrnrlon = max_lon,
                  llcrnrlat = min_lat, urcrnrlat = max_lat,
                  nofillcontinents=True)



    draw_events(bmap, ((slon, slat),),  marker="x", ms=20, mfc="none", mec="black", mew=2)
    draw_events(bmap, X, marker="o", ms=10, mfc="none", mec="yellow", mew=2)

    lon_arr = np.linspace(min_lon, max_lon, 40)
    lat_arr = np.linspace(min_lat, max_lat, 40)

    ll_kernel = lambda ( ll1, ll2) : np.exp(-1 * utils.geog.dist_km(ll1, ll2)/ (w**2))
    out = np.zeros((len(lon_arr), len(lat_arr)))
    for loni, lon in enumerate(lon_arr):
        for lati, lat in enumerate(lat_arr):
            out[loni, lati] = nr.kernel_predict(np.array((lon, lat)), X, y, kernel=ll_kernel)

    print lon_arr.shape, lat_arr.shape, out.shape
    levels = np.linspace(np.min(out), np.max(out), 10)
    draw_density(bmap, lon_arr, lat_arr, out, levels = levels, colorbar=True)

    plt.title("w = %f, miny = %f, maxy = %f" % (w, np.min(y), np.max(y)))
    pp.savefig()

def unfair_para_predict(params_dict, arrival, band, distance):
    evid = int(arrival[5])
    siteid = int(arrival[10])
    phaseid = int(arrival[7])

    phase = "P" if phaseid else "S"
    s = "r" if distance < 1000 else "t"

    key = "%s %d %s %s" % (s, siteid, phase, band[19:])
    v = params_dict[key]
    print "returning [%s] = %f, %f" % (key, v[0], v[1])

    return v

def fair_nonpara_predict(arrival, lb, w):

    evid = int(arrival[5])

    if lb is None:
        return None

    x0 = [arrival[2], arrival[3]]
    X = np.array([ lb[i, [LON_COL, LAT_COL]] for i in range(lb.shape[0]) if lb[i, EVID_COL] != evid ] )
    y = np.array([ lb[i, B_COL] for i in range(lb.shape[0]) if lb[i, EVID_COL] != evid ] )

    if y.shape[0] == 0:
        return None

    ll_kernel = lambda ( ll1, ll2) : np.exp(-1 * utils.geog.dist_km(ll1, ll2)/ (w**2))
    y0 = nr.kernel_predict(x0, X, y, kernel=ll_kernel)
    print "returning", y0
    return y0



def main():

    siteid = sys.argv[1]
    full_label = sys.argv[2]
    runid = sys.argv[3]

    base_coda_dir = get_base_dir(int(siteid), full_label, int(runid))
    fname = os.path.join(base_coda_dir, 'all_data')
    all_data, bands = read_shape_data(fname)

    
    for (band_idx, band) in enumerate(bands):
        p_fname = os.path.join(base_coda_dir, band[19:], "p_predictions.pdf")
        pp_p = PdfPages(p_fname)
        s_fname = os.path.join(base_coda_dir, band[19:], "s_predictions.pdf")
        pp_s = PdfPages(s_fname)
        print "saving plots to", p_fname, s_fname

        band_data = extract_band(all_data, band_idx)
        print "processing band %s, cleaning %d points..." % (band, band_data.shape[0])
        clean_p_data = clean_points(band_data, P=True, vert=True)
        clean_s_data = clean_points(band_data, P=False, vert=False)
        #print "  points remaining = %d" % (clean_data.shape[0])
        (ll50r, ll5r, dr, mr) = cross_validate(clean_p_data)
        plot_linear(pp_p, clean_p_data, title = "%s llr=%0.4f,%0.4f dr=%0.4f mr=%0.4f" % (band[19:], ll5r, ll50r, dr, mr))
        plot_heat(pp_p, clean_p_data, 5)
        plot_heat(pp_p, clean_p_data, 30)
        plot_heat(pp_p, clean_p_data, 50)

        (ll50r, ll5r, dr, mr) = cross_validate(clean_s_data)
        plot_linear(pp_p, clean_s_data, title = "%s llr=%0.4f,%0.4f dr=%0.4f mr=%0.4f" % (band[19:], ll5r, ll50r, dr, mr))
        plot_heat(pp_s, clean_s_data, 5)
        plot_heat(pp_s, clean_s_data, 30)
        plot_heat(pp_s, clean_s_data, 50)


    pp.close()

if __name__ == "__main__":
    main()








