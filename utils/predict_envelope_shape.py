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


def cv_residuals_kernel(lb, x_indices, y_index, kernel=None):
    kr = []
    linr = []
    n = lb.shape[0]
    for i in range(n):
        x0 = lb[i, x_indices]
        y0 = lb[i, y_index]

        #X = lb[ [j for j in range(n) if j != i], x_indices ]
        X = lb[ : , x_indices ]
        X = X[[j for j in range(n) if j != i], :]
        y = lb[ [j for j in range(n) if j != i], y_index ]

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


def cv_residuals_mean(lb, y_index):
    r = []
    n = lb.shape[0]
    for i in range(n):
        y0 = lb[i, y_index]
        #X = lb[ [j for j in range(n) if j != i], x_indices ]
        y = lb[ [j for j in range(n) if j != i], y_index ]

        py0 = np.mean(y)
    
        # print y0, py0
        r.append(np.abs(py0-y0)[0])
#    print "mean", r
    return r


def cross_validate(l, band):
    lb = l[band]

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
    kresiduals_da50, lresiduals_da = cv_residuals_kernel(lb, [LON_COL, LAT_COL], [B_COL], kernel=ll_kernel)
    w = 5
    ll_kernel = lambda ( ll1, ll2) : np.exp(-1 * utils.geog.dist_km(ll1, ll2)/ (w**2))
    #ll_kernel = lambda ( ll1, ll2) : utils.geog.dist_km(ll1, ll2)
    kresiduals_da5, lresiduals_da = cv_residuals_kernel(lb, [LON_COL, LAT_COL], [B_COL], kernel=ll_kernel)


    kresiduals_a, lresiduals_a = cv_residuals_kernel(lb, [AZI_COL], [B_COL])
    kresiduals_d, lresiduals_d = cv_residuals_kernel(lb, [DISTANCE_COL], [B_COL])

    residuals_m = cv_residuals_mean(lb, [B_COL])

    ll50r = np.mean(kresiduals_da50)
    ll5r = np.mean(kresiduals_da5)
    dr = np.mean(lresiduals_d)
    mr = np.mean(residuals_m)

    return (ll50r, ll5r, dr, mr)
    

def plot_linear(pp, l, band, title=""):
    lb = l[band]

    X = lb[ : , DISTANCE_COL ]
    y = lb[ : , B_COL ]

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
    
def plot_heat(pp, l, band, w):
    lb = l[band]

    siteid = lb[0, SITEID_COL]
    X = lb[ : , [LON_COL, LAT_COL] ]
    y = lb[ : , B_COL ]

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

def read_shape_params(fname):
    f = open(fname, 'r')
    params = dict()

    for line in f:
        s = line[:-1].split()
        k = " ".join(s[0:4])
        params[k] = [float(s[4]), float(s[5])]

    return params
    

def read_shape_data(fname):
    f = open(fname, 'r')
    l = dict()

    for line in f:
        if line[0] == 'n':
            
            band = line[:-1]
        
            if band not in l:
                l[band] = None
        else:
            new_row = np.array(  map( lambda x : float(x), (line[:-1]).split()))

            if l[band] is None:
                l[band] = new_row
            else:
                try:
                    l[band] = np.vstack([l[band], new_row])
                except:
                    break
    return l



def main():
    sta_idx = 0
    fname = sys.argv[1]
    label = fname[13:24]
    outfile = 'logs/predict_envelope_%s.pdf' % (label)
    pp = PdfPages(outfile)
    print "saving plots to", outfile
    
    print "opening fname", fname
    l = read_shape_data(fname)

    for band in l.keys():
        print "processing band %s, cleaning %d points..." % (band, l[band].shape[0])
        l[band] = clean_points(l[band])
        print "  points remaining = %d" % (l[band].shape[0])
        (ll50r, ll5r, dr, mr) = cross_validate(l, band)
        plot_linear(pp, l, band, title = "%s llr=%0.4f,%0.4f dr=%0.4f mr=%0.4f" % (band, ll5r, ll50r, dr, mr))
        plot_heat(pp, l, band, 5)
        plot_heat(pp, l, band, 50)

    pp.close()

if __name__ == "__main__":
    main()








