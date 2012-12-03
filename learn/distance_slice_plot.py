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


def distance_slice_plot(X, y, model, plotfile, d):
    pp = PdfPages(plotfile)
    print "writing plot to", plotfile

    distances = X[:, X_DIST]
    azimuths = X[:, X_AZI]

    for azi in [0, 90, 180, 270]:

        plt.figure()
        plt.title("azi %d-%d depth 0" % (azi-45, azi+45))
        plt.xlabel("distance (km)")
        plt.ylabel("coda decay (b)")
        ds = np.linspace(1, 10000, 150)

        plt.ylim([-0.06, 0])
        plt.xlim([0, 10000])
        pred = np.array([ model.predict(np.array((d, azi, 0))) for d in ds]).flatten()
        plt.plot(ds, pred, "k-")

        std = np.sqrt(np.array([ model.variance(np.array((d, azi, 0)))[0] for d in ds]))

        var_x = np.concatenate((ds, ds[::-1]))
        var_y = np.concatenate((pred + 2*std, (pred - 2*std)[::-1]))
        p = plt.fill(var_x, var_y, edgecolor='w', facecolor='#4343d3', alpha=0.3)

        idx = np.min([np.abs(azimuths - azi), np.abs(azimuths - azi+360)], axis=0) < 45

        plt.plot(distances[idx], y[idx], 'ro')
        pp.savefig()

    plt.figure()
    plt.xlabel("distance (km)")
    plt.ylabel("coda decay (b)")
    ds = np.linspace(1, 10000, 150)
    plt.ylim([-0.06, 0])
    plt.xlim([0, 10000])
#    facecolors=["#d34343", "#43d343", "#4343d3", "#43d3d3"]
    for (i, azi) in enumerate(np.arange(0,360,5)):
        pred = np.array([ model.predict(np.array((d, azi, 0))) for d in ds]).flatten()
#        plt.plot(ds, pred, "k-")

        std = np.sqrt(np.array([ model.variance(np.array((d, azi, 0)))[0] for d in ds]))

        var_x = np.concatenate((ds, ds[::-1]))
        var_y = np.concatenate((pred + std, (pred - std)[::-1]))
        p = plt.fill(var_x, var_y, facecolor="#4343d3", alpha=0.1)
    plt.plot(distances, y, 'ro')
    pp.savefig()


    pp.close()


def squid_plot(X,y,model, plotfile, d):
    pp = PdfPages(plotfile)
    print "writing plot to", plotfile

    heatfile = os.path.join(os.path.dirname(plotfile), ".heat_" + hashlib.md5(plotfile).hexdigest()[0:6])

    s = Sigvisa()
    s.cursor.execute("SELECT lon, lat from static_siteid where sta='%s'" % (d['sta']))
    (slon, slat) = s.cursor.fetchone()

    f = lambda lon, lat: model.predict(np.array([lon, lat, 0, \
                                            utils.geog.dist_km((lon, lat), (slon, slat)), \
                                            utils.geog.azimuth((slon, slat), (lon, lat))]))
    if d['target'] == "decay":
        f_old = f
        f = lambda lon, lat: min(f_old(lon, lat), 0)

    try:
        hm = Heatmap(f, fname=heatfile)
    except:
        hm = Heatmap(f, autobounds=X, n=40)
        hm.calc(checkpoint=heatfile)
    hm.plot_density()
    hm.plot_locations(((slon,slat),),  marker="x", ms=7, mfc="none", mec="white", mew=2)
    hm.plot_locations(X[:, 0:2], marker=".", ms=2, mfc="none", mec="red", mew=2, alpha=0.6)

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

    if options.distance_slice:
        plotfile = model_fname + "_distance_slice.pdf"
        distance_slice_plot(X,y,model, plotfile, d)

    if options.squid:
        plotfile = model_fname + "_heatmap.pdf"
        squid_plot(X,y,model, plotfile, d)

if __name__ == "__main__":
    try:
        main()
    except:
        import pdb, traceback
        traceback.print_exc()
        pdb.set_trace()
