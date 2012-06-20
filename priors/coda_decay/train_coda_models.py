
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
import learn, sigvisa_util, sigvisa

import numpy as np
import scipy.linalg
import hashlib

from priors.coda_decay.coda_decay_common import *

from utils.gp_regression import GaussianProcess, optimize_hyperparams
from utils.kernels import InvGamma, LogNormal


def learn_models(fit_data, earthmodel, gen_target_col, sigma_n, sigma_f, w, pp, label):

    fit_data = np.reshape(fit_data, (-1, FIT_NUM_COLS))
    n = fit_data.shape[0]

    Xll = fit_data[:, [FIT_LON, FIT_LAT, FIT_DEPTH]]
    Xad = fit_data[:, [FIT_DISTANCE, FIT_AZIMUTH, FIT_DEPTH]]

    y = np.zeros((n,))
    for i in range(n):
        y[i] = gen_target_col(fit_data[i, :])


#    best_params,v = optimize_hyperparams(X, y, kernel="distfn", start_kernel_params=[sigma_n, sigma_f, w], kernel_extra=distfn, kernel_priors = [InvGamma(1.0, 1.0), InvGamma(1.0, 1.0), LogNormal(3, 2.0)])
    gpd = None
    gpt = None

    best_params = [sigma_n, sigma_f, w]
    posdef = False
    while not posdef:
        try:
            distfn = lambda ll1, ll2: np.sqrt(utils.geog.dist_km(ll1[0:2], ll2[0:2])**2 + (ll1[2] - ll2[2])**2)
            print "training location-based GP w/ params", best_params #, "giving ll", v
            gp_loc = GaussianProcess(Xll, y, kernel="distfn", kernel_params=best_params, kernel_extra=distfn)
            posdef=True
        except np.linalg.linalg.LinAlgError:
            best_params[0] *= 1.5
            print "lin alg error, upping sigma_n to %f and tryign again" % (best_params[0])

    best_params = [sigma_n, sigma_f, w]
    posdef = False
    while not posdef:
        try:
            # use gaussian covariance with features given by the cube roots of distance and depth, and azi/20
            distfn = lambda ad1, ad2: np.sqrt( abs(ad1[0]-ad2[0])**(2.0/3) + abs(azi_difference(ad1[1], ad2[1])/20)**2 + abs(ad1[2]-ad2[2])**(2.0/3))
            print "training distance/azimuth-based GP w/ params", best_params #, "giving ll", v
            gp_ad = GaussianProcess(Xad, y, kernel="distfn", kernel_params=best_params, kernel_extra=distfn, ignore_pos_def_errors=False)
            posdef = True
        except np.linalg.linalg.LinAlgError:
            best_params[0] *= 1.5
            print "lin alg error, upping sigma_n to %f and tryign again" % (best_params[0])

    regional_dist = []
    regional_y = []
    tele_dist = []
    tele_y = []
    for i in range(n):
        d = fit_data[i, FIT_DISTANCE]
        dy = gen_target_col(fit_data[i, :])
        if d > 1000:
            tele_dist.append(d)
            tele_y.append(dy)
        else:
            regional_dist.append(d)
            regional_y.append(dy)

    try:
        regional_model = utils.LinearModel.LinearModel("regional", ["distance"],
                                                       [regional_dist,],
                                                       regional_y)
    except ValueError:
        print "regional model failed", regional_dist, regional_y
        regional_model=None
    try:
        tele_model = utils.LinearModel.LinearModel("tele", ["distance"],
                                                   [tele_dist,],
                                                   tele_y)
    except ValueError:
        print "tele model failed", tele_dist, tele_y
        tele_model=None
    

    if pp is not None:
        plt.figure()
        plt.title(label + " linear")
        plt.xlabel("distance (km)")
        plt.ylabel("")
        try:
            t = np.linspace(0, 1000, 50)
            pred = [ regional_model[tv] for tv in t ]
            plt.plot(t, pred, "k-")
        except:
            pass
        try:
            t = np.linspace(1000, np.max(tele_dist), 50)
            pred = [ tele_model[tv] for tv in t ]
            plt.plot(t, pred, "k-")
        except:
            pass
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
    m["gp_loc"] = gp_loc
    m["gp_ad"] = gp_ad
    m["regional_linear"] = regional_model
    m["tele_linear"] = tele_model
    m["regional_gaussian"] = (regional_mean, regional_var)
    m["tele_gaussian"] = (tele_mean, tele_var)
    return m

class CodaModel:

    MODEL_TYPE_GP_LOC, MODEL_TYPE_GP_AD, MODEL_TYPE_LINEAR, MODEL_TYPE_GAUSSIAN = range(4)

    def __init__(self, fit_data, band_dir, phaseids, chan, ignore_evids = None, earthmodel = None, sigmodel=None, sites=None, sigma_f = [.001, .5, .5], w = [500, 500, 500], sigma_n = [0.00001, 0.5, 0.5]):

        # assume that either earthmodel and sigmodel are both given, or neither given
        if sigmodel is None:
            cursor, self.sigmodel, self.earthmodel, self.sites, dbconn = sigvisa_util.init_sigmodel()
        else:
            self.earthmodel = earthmodel
            self.sigmodel = sigmodel
            self.sites=sites

        if ignore_evids is not None:
            good_rows = np.array([int(fit_data[i, FIT_EVID]) not in ignore_evids for i in range(fit_data.shape[0])])
            fit_data_n = fit_data[good_rows, :]
            fit_data = fit_data_n

        self.siteid = int(fit_data[0, FIT_SITEID])
        self.slon = self.sites[self.siteid-1, 0]
        self.slat = self.sites[self.siteid-1, 1]

        outfile = os.path.join(band_dir, "model_fits_%s_%s.pdf" % (":".join([str(p) for p in phaseids]), chan))
        pp = None
        if not os.path.exists(outfile):
            pp = PdfPages(outfile)
            print "saving plots to", outfile

        self.decay_models = None
        self.peak_amp_model = None
        self.onset_model = None

        # learn decay rate models
        m = learn_models(fit_data, self.earthmodel, lambda r: r[FIT_CODA_DECAY], sigma_n[0], sigma_f[0], w[0], pp, "decay phaseids=%s chan=%s" % (phaseids, chan))
        self.decay_models = m

        # learn onset time models
        m = learn_models(fit_data, self.earthmodel, lambda r : r[FIT_PEAK_DELAY], sigma_n[1], sigma_f[1], w[1], pp, "offset phaseids=%s chan=%s" % (phaseids, chan))
        self.onset_models = m

        # learn peak height models
        m = learn_models(fit_data, self.earthmodel, gen_source_amp, sigma_n[2], sigma_f[2], w[2], pp, "amp phaseids=%s chan=%s" % (phaseids, chan))
        self.peak_amp_models = m

        if pp is not None:
            pp.close()

    def predict(self, ev, model_type, model_set, distance = None, azimuth=None):

        if distance is None:
            distance = utils.geog.dist_km((ev[EV_LON_COL], ev[EV_LAT_COL]), (self.slon, self.slat))
        if model_type == self.MODEL_TYPE_GP_LOC:
            return model_set['gp_loc'].predict((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL]))
        elif model_type == self.MODEL_TYPE_GP_AD:
            if azimuth is None:
                azimuth = utils.geog.azimuth((self.slon, self.slat), (ev[EV_LON_COL], ev[EV_LAT_COL]))
            return model_set['gp_ad'].predict((distance, azimuth, ev[EV_DEPTH_COL]))
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

    def predict_decay(self, ev, model_type, distance = None, azimuth=None):
        return self.predict(ev, model_type, self.decay_models, distance=distance, azimuth=azimuth)

    def predict_peak_time(self, ev, model_type, distance = None, azimuth=None):
        return self.predict(ev, model_type, self.onset_models, distance=distance, azimuth=azimuth)

    def predict_peak_amp(self, ev, model_type, distance = None, azimuth=None):
        return self.predict(ev, model_type, self.peak_amp_models, distance=distance, azimuth=azimuth)

    def sample(self, ev, model_type, model_set, distance = None, azimuth=None):

        if distance is None:
            distance = utils.geog.dist_km((ev[EV_LON_COL], ev[EV_LAT_COL]), (self.slon, self.slat))

        if model_type == self.MODEL_TYPE_GP_LOC:
            return model_set['gp_loc'].sample((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL]))
        elif model_type == self.MODEL_TYPE_GP_AD:
            if azimuth is None:
                azimuth = utils.geog.azimuth((self.slon, self.slat), (ev[EV_LON_COL], ev[EV_LAT_COL]))
            return model_set['gp_ad'].sample((distance, azimuth, ev[EV_DEPTH_COL]))
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

    def sample_decay(self, ev, model_type, distance = None, azimuth = None):
        return self.sample(ev, model_type, self.decay_models, distance=distance, azimuth=azimuth)

    def sample_peak_time(self, ev, model_type, distance = None, azimuth = None):
        return self.sample(ev, model_type, self.onset_models, distance=distance, azimuth=azimuth)

    def sample_peak_amp(self, ev, model_type, distance = None, azimuth = None):
        return self.sample(ev, model_type, self.peak_amp_models, distance=distance, azimuth=azimuth)

    def log_likelihood(self, val, ev, model_type, model_set, distance = None, azimuth = None):

        if distance is None:
            distance = utils.geog.dist_km((ev[EV_LON_COL], ev[EV_LAT_COL]), (self.slon, self.slat))

        if model_type == self.MODEL_TYPE_GP_LOC:
            return model_set['gp_loc'].posterior_log_likelihood((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL]), val)
        elif model_type == self.MODEL_TYPE_GP_AD:
            if azimuth is None:
                azimuth = utils.geog.azimuth((self.slon, self.slat), (ev[EV_LON_COL], ev[EV_LAT_COL]))
            return model_set['gp_ad'].posterior_log_likelihood((distance, azimuth, ev[EV_DEPTH_COL]), val)
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


    def log_likelihood_decay(self, val, ev, model_type, distance = None, azimuth = None):
        return self.log_likelihood(val, ev, model_type, self.decay_models, distance=distance, azimuth=azimuth)

    def log_likelihood_peak_time(self, val, ev, model_type, distance = None, azimuth = None):
        return self.log_likelihood(val, ev, model_type, self.onset_models, distance=distance, azimuth=azimuth)

    def log_likelihood_peak_amp(self, val, ev, model_type, distance = None, azimuth = None):
        return self.log_likelihood(val, ev, model_type, self.peak_amp_models, distance=distance, azimuth=azimuth)


def main():
    cm = CodaModel(sys.argv[1], ignore_evids = (5330781,))

if __name__ == "__main__":
    main()
