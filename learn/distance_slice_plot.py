import sys, os
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from optparse import OptionParser


from sigvisa import *
import numpy as np
import hashlib

from learn.SpatialGP import distfns, SpatialGP, start_params, gp_extract_features
from learn.train_coda_models import learn_model, load_model, get_model_fname, get_training_data, analyze_model_fname

import utils.geog

from plotting.heatmap import Heatmap


def distance_slice_plot(X, y, model, plotfile, d, event_split=None):
    pp = PdfPages(plotfile)
    print "writing plot to", plotfile


    if event_split is not None:
        special_idx, other_idx = event_split
        X_special = X[special_idx,:]
        y_special = y[special_idx]
        X_all = X
        y_all = y
        X = X[other_idx, :]
        y = y[other_idx]


    distances = X[:, X_DIST]
    azimuths = X[:, X_AZI]
    depths = X[:, X_DEPTH]

    #    if d['target']=="decay":
    #        ylim =[-0.06, 0]
    if False:
        pass
    else:
        sy = sorted(y_all)
        lower = int(.05*len(sy))
        upper = int(.95*len(sy))
        bounds = (sy[lower], sy[upper])
        margin = (bounds[1]-bounds[0])*.1
        ylim=[bounds[0]-margin, bounds[1]+margin]

        ylim = [-.04, 0]

    ylabel = "coda decay (b)" if d['target'] == "decay" else "transfer fn (log-amplitude)"

    for azi in []:
            #    for azi in [0, 90, 180, 270]:

        plt.figure()
        plt.title("azi %d-%d depth 0" % (azi-45, azi+45))
        plt.xlabel("distance (km)")
        plt.ylabel(ylabel)
        ds = np.linspace(1, 10000, 150)

        plt.ylim(ylim)
        plt.xlim([0, 10000])
        pred = np.array([ model.predict(np.array((d, azi, 0))) for d in ds]).flatten()
        plt.plot(ds, pred, "k-")

        std = np.sqrt(np.array([ model.variance(np.array((d, azi, 0)))[0] for d in ds]))

        var_x = np.concatenate((ds, ds[::-1]))
        var_y = np.concatenate((pred + 2*std, (pred - 2*std)[::-1]))
        p = plt.fill(var_x, var_y, edgecolor='w', facecolor='#4343d3', alpha=0.3)

        idx = np.min([np.abs(azimuths - azi), np.abs(azimuths - azi+360)], axis=0) < 45
        idx *= (depths < 10)

        plt.plot(distances[idx], y[idx], 'ro')
        pp.savefig()

    plt.figure()
    plt.xlabel("azimuth (deg)")
    plt.ylabel(ylabel)
    ds = np.linspace(1, 10000, 150)
    plt.ylim(ylim)
    plt.xlim([50, 70])


    quartile_idx = [int(x/100.0 * len(depths)) for x in [0,25,50,75]]
    depth_quartiles = np.asarray(sorted(depths))[ quartile_idx ]

    azi_idx = [int(x/100.0 * len(azimuths)) for x in np.arange(0, 100, 6)]
    azi_quantiles = np.asarray(sorted(azimuths))[ quartile_idx ]


#    for (i, depth) in enumerate(depth_quartiles):
#        for (i, azi) in enumerate(azi_quantiles):
#            pred = np.array([ model.predict(np.array((d, azi, depth))) for d in ds]).flatten()
#    #        plt.plot(ds, pred, "k-")

#            std = np.sqrt(np.array([ model.variance(np.array((d, azi, depth)))[0] for d in ds]))

#            var_x = np.concatenate((ds, ds[::-1]))
#            var_y = np.concatenate((pred + std, (pred - std)[::-1]))
#            p = plt.fill(var_x, var_y, facecolor="#4343d3", alpha=0.03)

    plt.plot(azimuths, y, 'ro')

    if event_split is not None:
        plt.plot(X_special[:, X_AZI], y_special, 'go')
        plt.plot(X_all[:, X_AZI], y_all, 'ko', alpha=0.02)


    pp.savefig()


    pp.close()


def squid_plot(X,y,model, plotfile, d, event_split = None, all_idx=None):
    pp = PdfPages(plotfile)
    print "writing plot to", plotfile

    heatfile = os.path.join(os.path.dirname(plotfile), ".heat_" + hashlib.md5(plotfile).hexdigest()[0:6])

    s = Sigvisa()
    cursor = s.dbconn.cursor()
    cursor.execute("SELECT lon, lat from static_siteid where sta='%s'" % (d['sta']))
    (slon, slat) = cursor.fetchone()

    f = lambda lon, lat: model.predict(np.array([lon, lat, 0, \
                                            utils.geog.dist_km((lon, lat), (slon, slat)), \
                                            utils.geog.azimuth((slon, slat), (lon, lat))]))
    if d['target'] == "decay":
        f_old = f
        f = lambda lon, lat: min(f_old(lon, lat), 0)

    if event_split is not None:
        special_idx, other_idx = event_split
        X_special = X[special_idx,:]
        y_special = y[special_idx]
        X_all = X[all_idx, :]
        idx = X_all[:, X_LON] < 172
        idx *= X_all[:, X_LON] > 100
        X_all = X_all[idx, :]
        y_all = y[all_idx]
        y_all = y_all[idx]
        X = X[other_idx, :]
        y = y[other_idx]
    try:
        hm = Heatmap(f, fname=heatfile)
    except:
        hm = Heatmap(f, autobounds=X_all, n=40, calc=False)
#        hm.calc(checkpoint=heatfile)
    # hm.plot_density()
    hm.plot_locations(((slon,slat),),  marker="x", ms=7, mfc="none", mec="blue", mew=2)
#    hm.plot_locations(X[:, 0:2], marker=".", ms=4, mfc="none", mec="red", mew=4, alpha=0.5)

    if event_split is not None:
#        hm.plot_locations(X_special[:, 0:2], marker=".", ms=4, mfc="none", mec="green", mew=4, alpha=0.7)
        hm.plot_locations(X_all[:, 0:2], marker=".", ms=3, mfc="none", mec="red", mew=2, alpha=0.28)

    pp.savefig()
    pp.close()


X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI  = range(5)
def main():

    parser = OptionParser()

    s = Sigvisa()

    parser.add_option("-m", "--model", dest="model", default=None, type="str", help="filename of model to plot predictions")

    parser.add_option("--distance_slice", dest="distance_slice", default=False, action="store_true", help="distance slice plot")
    parser.add_option("--squid", dest="squid", default=False, action="store_true", help="squid plot")

    (options, args) = parser.parse_args()

    model_fname = options.model
    d = analyze_model_fname(model_fname)
    model_dir = os.path.dirname(model_fname)

    model = load_model(model_fname, d['model_type'])
    print "loaded model from", model_fname

    X = np.loadtxt(os.path.join(model_dir, "X.txt"))
    y = np.loadtxt(os.path.join(model_dir, "y.txt"))
    evids = np.loadtxt(os.path.join(model_dir, "evids.txt"))

    # TEMPROARY HACK
    special_evids = np.loadtxt("CTA_block_3_evids")
    other_evids = np.loadtxt("CTA_block_2_evids")
    all_evids = np.loadtxt("CTA_all_evids.csv")
#    other_evids = np.loadtxt("CTA_all_evids.csv")
    evlist = list(evids)
    special_evid_idx = [evlist.index(evid) for evid in special_evids if evid in evlist]
    other_idx = [evlist.index(evid) for evid in other_evids if evid in evlist]
    all_idx = [evlist.index(evid) for evid in all_evids if evid in evlist]
    #    other_idx = [idx for idx in range(len(evids)) if idx not in special_evid_idx]
    print "special idx", special_evid_idx

    if options.distance_slice:
        plotfile = model_fname + "_distance_slice.pdf"
        distance_slice_plot(X,y,model, plotfile, d,  event_split = (special_evid_idx, other_idx) )

    if options.squid:
        plotfile = model_fname + "_heatmap.pdf"
        squid_plot(X,y,model, plotfile, d,  event_split = (special_evid_idx, other_idx) , all_idx=all_idx)

if __name__ == "__main__":
    try:
        main()
    except:
        import pdb, traceback
        traceback.print_exc()
        pdb.set_trace()
