
from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages

import sys, os, pickle

import utils.geog
import obspy.signal.util
import learn, sigvisa_util

import numpy as np
import scipy.linalg
import hashlib

from priors.coda_decay.coda_decay_common import *

from utils.gp_regression import GaussianProcess, optimize_hyperparams
from utils.kernels import InvGamma, LogNormal


def learn_models(band_data, P, vert, earthmodel, gen_target_col, sigma_n, sigma_f, w, pp, label):

    clean_data = clean_points(band_data, P, vert)
    n = clean_data.shape[0]
    Xd = clean_data[:, [LON_COL, LAT_COL, DEPTH_COL]]
    X = clean_data[:, [LON_COL, LAT_COL]]
    y = np.zeros((n,))
    for i in range(n):
        y[i] = gen_target_col(clean_data[i,:])


    distfn = lambda ll1, ll2: utils.geog.dist_km(ll1, ll2)
#    best_params,v = optimize_hyperparams(X, y, kernel="distfn", start_kernel_params=[sigma_n, sigma_f, w], kernel_extra=distfn, kernel_priors = [InvGamma(1.0, 1.0), InvGamma(1.0, 1.0), LogNormal(3, 2.0)])
    best_params = [sigma_n, sigma_f, w]
    print "training GP w/ params", best_params #, "giving ll", v
    gp = GaussianProcess(X, y, kernel="distfn", kernel_params=best_params, kernel_extra=distfn)

    posdef = False
    gpd = None
    gpt = None
    posdef = True

    while not posdef:

        try:
            best_params = [sigma_n, sigma_f, w]

            distfn = lambda ll1, ll2: np.sqrt(utils.geog.dist_km(ll1[0:2], ll2[0:2])**2 + (ll1[2] - ll2[2])**2)
            print "training GPd w/ params", best_params #, "giving ll", v
            gpd = GaussianProcess(Xd, y, kernel="distfn", kernel_params=best_params, kernel_extra=distfn)


            distfn = lambda lld1, lld2: (earthmodel.ArrivalTimeCoord(lld1[0], lld1[1], lld1[2], 0, 0 if P else 4, lld2[0], lld2[1], lld2[2]) + earthmodel.ArrivalTimeCoord(lld2[0], lld2[1], lld2[2], 0, 0 if P else 4, lld1[0], lld1[1], lld1[2]) )/2
            #    best_params,v = optimize_hyperparams(Xd, y, kernel="distfn", start_kernel_params=[sigma_n, sigma_f, w], kernel_extra=distfn, kernel_priors = [InvGamma(1.0, 1.0), InvGamma(1.0, 1.0), LogNormal(3, 2.0)])
            best_params = [sigma_n, sigma_f, w]
            print "training GPt w/ params", best_params #, "giving ll", v
            gpt = GaussianProcess(Xd, y, kernel="distfn", kernel_params=best_params, kernel_extra=distfn, ignore_pos_def_errors=False)


            posdef = True
        except np.linalg.linalg.LinAlgError:
            print "lin alg error, upping sigma_n to %f and tryign again" % (sigma_n)
            sigma_n *= 1.5




    regional_dist = []
    regional_y = []
    tele_dist = []
    tele_y = []
    for i in range(n):
        d = clean_data[i, DISTANCE_COL]
        dy = gen_target_col(clean_data[i,:])
        if d > 1000:
            tele_dist.append(d)
            tele_y.append(dy)
        else:
            regional_dist.append(d)
            regional_y.append(dy)

    regional_model = utils.LinearModel.LinearModel("regional", ["distance"],
                                        [regional_dist,],
                                        regional_y)
    tele_model = utils.LinearModel.LinearModel("tele", ["distance"],
                                        [tele_dist,],
                                        tele_y)


    if pp is not None:
        plt.figure()
        plt.title(label + " linear")
        plt.xlabel("distance (km)")
        plt.ylabel("")
        t = np.linspace(0, 1000, 50)
        pred = [ regional_model[tv] for tv in t ]
        plt.plot(t, pred, "k-")
        t = np.linspace(1000, np.max(tele_dist), 50)
        pred = [ tele_model[tv] for tv in t ]
        plt.plot(t, pred, "k-")
        plt.plot(np.concatenate([regional_dist, tele_dist]), np.concatenate([regional_y, tele_y]), 'ro')
        pp.savefig()

    regional_mean = np.mean(regional_y)
    regional_var = np.var(regional_y)
    tele_mean = np.mean(tele_y)
    tele_var = np.var(tele_y)

    if pp is not None:
        plt.figure()
        plt.title(label + " regional gaussian mean %f sigma")
        n, bins, patches = plt.hist(regional_y, normed=1)
        bincenters = 0.5*(bins[1:]+bins[:-1])
        y = mlab.normpdf( bincenters, regional_mean, np.sqrt(regional_var))
        plt.plot(bincenters, y, 'r--', linewidth=1)
        pp.savefig()

        plt.figure()
        plt.title(label + " tele gaussian")
        n, bins, patches = plt.hist(tele_y, normed=1)
        bincenters = 0.5*(bins[1:]+bins[:-1])
        y = mlab.normpdf( bincenters, tele_mean, np.sqrt(tele_var))
        plt.plot(bincenters, y, 'r--', linewidth=1)
        pp.savefig()


#    print "outputs", y
#    print "learned means ", (regional_mean, regional_var), (tele_mean, tele_var)

    m = dict()
    m["gp"] = gp
    m["gpd"] = gpd
    m["gpt"] = gpt
    m["regional_linear"] = regional_model
    m["tele_linear"] = tele_model
    m["regional_gaussian"] = (regional_mean, regional_var)
    m["tele_gaussian"] = (tele_mean, tele_var)
    return m

class CodaModel:

    MODEL_TYPE_GP, MODEL_TYPE_GPD, MODEL_TYPE_GPT, MODEL_TYPE_LINEAR, MODEL_TYPE_GAUSSIAN = range(5)

    def __init__(self, band_data_with_depth, band_dir, P = True, vert = True, ignore_evids = None, earthmodel = None, netmodel=None, sigma_f = [.001, .5, .5], w = [500, 500, 500], sigma_n = [0.00001, 0.5, 0.5]):

        band_data = band_data_with_depth

        cursor = db.connect().cursor()
        sites = read_sites(cursor)

        # assume that either earthmodel and netmodel are both given, or neither given
        if netmodel is None:
            st  = 1237680000
            et = st + 3600*24
            site_up = read_uptime(cursor, st, et)
            detections, arid2num = read_detections(cursor, st, et, arrival_table="leb_arrival", noarrays=True)
            phasenames, phasetimedef = read_phases(cursor)
            self.earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
            self.netmodel = learn.load_netvisa("parameters", st, et, detections, site_up, sites, phasenames, phasetimedef)
        else:
            self.earthmodel = earthmodel
            self.netmodel = netmodel

        if ignore_evids is not None:
            good_rows = np.array([int(band_data[i, EVID_COL]) not in ignore_evids for i in range(band_data.shape[0])])
            band_data_n = band_data[good_rows, :]
            band_data = band_data_n



        self.siteid = int(band_data[0, SITEID_COL])
        self.slon = sites[self.siteid-1, 0]
        self.slat = sites[self.siteid-1, 1]

        if P:
            outfile = os.path.join(band_dir, 'p_model_fits.pdf')
        else:
            outfile = os.path.join(band_dir, 's_model_fits.pdf')
        pp = None
        if not os.path.exists(outfile):
            pp = PdfPages(outfile)
            print "saving plots to", outfile

        self.decay_models = None
        self.peak_amp_model = None
        self.onset_model = None

        (b_col, gen_decay, gen_onset, gen_amp) = construct_output_generators(cursor, self.netmodel, P, vert)

        # learn decay rate models
        m = learn_models(band_data, P, vert, self.earthmodel, gen_decay, sigma_n[0], sigma_f[0], w[0], pp, "decay P=%s vert=%s" % (P, vert))
        self.decay_models = m

        # learn onset time models
        m = learn_models(band_data, P, vert, self.earthmodel, gen_onset, sigma_n[1], sigma_f[1], w[1], pp, "offset P=%s vert=%s" % (P, vert))
        self.onset_models = m

        # learn peak height models
        m = learn_models(band_data, P, vert, self.earthmodel, gen_amp, sigma_n[2], sigma_f[2], w[2], pp, "amp P=%s vert=%s" % (P, vert))
        self.peak_amp_models = m

        if pp is not None:
            pp.close()

    def predict(self, ev, model_type, model_set, distance = None):

        if distance is None:
            distance = utils.geog.dist_km((ev[EV_LON_COL], ev[EV_LAT_COL]), (self.slon, self.slat))

        if model_type == self.MODEL_TYPE_GP:
#            print "GP predicting at ", (ev[EV_LON_COL], ev[EV_LAT_COL])
            return model_set['gp'].predict((ev[EV_LON_COL], ev[EV_LAT_COL]))
        elif model_type == self.MODEL_TYPE_GPD:
            return model_set['gpd'].predict((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL]))
        elif model_type == self.MODEL_TYPE_GPT:
            return model_set['gpt'].predict((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL]))
        elif model_type == self.MODEL_TYPE_LINEAR:
            if distance < 1000:
                model = model_set['regional_linear']
            else:
                model = model_set['tele_linear']
            return model[distance]
        elif model_type == self.MODEL_TYPE_GAUSSIAN:
            if distance < 1000:
                (mean, var) = model_set['regional_gaussian']
            else:
                (mean, var) = model_set['tele_gaussian']
            return mean


    def predict_decay(self, ev, model_type, distance = None):
        return self.predict(ev, model_type, self.decay_models, distance=distance)

    def predict_peak_time(self, ev, model_type, distance = None):
        return self.predict(ev, model_type, self.onset_models, distance=distance)

    def predict_peak_amp(self, ev, model_type, distance = None):
        return self.predict(ev, model_type, self.peak_amp_models, distance=distance)

    def sample(self, ev, model_type, model_set, distance = None):

        if distance is None:
            distance = utils.geog.dist_km((ev[EV_LON_COL], ev[EV_LAT_COL]), (self.slon, self.slat))

        if model_type == self.MODEL_TYPE_GP:
#            print "GP predicting at ", (ev[EV_LON_COL], ev[EV_LAT_COL])
            return model_set['gp'].sample((ev[EV_LON_COL], ev[EV_LAT_COL]))
        elif model_type == self.MODEL_TYPE_GPD:
            return model_set['gpd'].sample((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL]))
        elif model_type == self.MODEL_TYPE_GPT:
            return model_set['gpt'].sample((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL]))
        elif model_type == self.MODEL_TYPE_LINEAR:
            raise RuntimeError("sampling not yet implemented for linear models")
            if distance < 1000:
                model = model_set['regional_linear']
            else:
                model = model_set['tele_linear']

        elif model_type == self.MODEL_TYPE_GAUSSIAN:
            if distance < 1000:
                (mean, var) = model_set['regional_gaussian']
            else:
                (mean, var) = model_set['tele_gaussian']
            return mean + np.random.randn() * np.sqrt(var)


    def sample_decay(self, ev, model_type, distance = None):
        return self.sample(ev, model_type, self.decay_models, distance=distance)

    def sample_peak_time(self, ev, model_type, distance = None):
        return self.sample(ev, model_type, self.onset_models, distance=distance)

    def sample_peak_amp(self, ev, model_type, distance = None):
        return self.sample(ev, model_type, self.peak_amp_models, distance=distance)

    def log_likelihood(self, val, ev, model_type, model_set, distance = None):

        if distance is None:
            distance = utils.geog.dist_km((ev[EV_LON_COL], ev[EV_LAT_COL]), (self.slon, self.slat))

        if model_type == self.MODEL_TYPE_GP:
#            print "GP predicting at ", (ev[EV_LON_COL], ev[EV_LAT_COL])
            return model_set['gp'].posterior_log_likelihood((ev[EV_LON_COL], ev[EV_LAT_COL]), val)
        elif model_type == self.MODEL_TYPE_GPD:
            return model_set['gpd'].posterior_log_likelihood((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL]), val)
        elif model_type == self.MODEL_TYPE_GPT:
            return model_set['gpt'].posterior_log_likelihood((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL]), val)
        elif model_type == self.MODEL_TYPE_LINEAR:
            raise RuntimeError("log likelihood not yet implemented for linear models")
            if distance < 1000:
                model = model_set['regional_linear']
            else:
                model = model_set['tele_linear']

        elif model_type == self.MODEL_TYPE_GAUSSIAN:
            if distance < 1000:
                (mean, var) = model_set['regional_gaussian']
            else:
                (mean, var) = model_set['tele_gaussian']
            return -.5 *np.log(2*np.pi*var) - .5 *(val-mean)**2 / var


    def log_likelihood_decay(self, val, ev, model_type, distance = None):
        return self.log_likelihood(val, ev, model_type, self.decay_models, distance=distance)

    def log_likelihood_peak_time(self, val, ev, model_type, distance = None):
        return self.log_likelihood(val, ev, model_type, self.onset_models, distance=distance)

    def log_likelihood_peak_amp(self, val, ev, model_type, distance = None):
        return self.log_likelihood(val, ev, model_type, self.peak_amp_models, distance=distance)


def main():
    cm = CodaModel(sys.argv[1], ignore_evids = (5330781,))

if __name__ == "__main__":
    main()
