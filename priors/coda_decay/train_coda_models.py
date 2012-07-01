
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
from priors.coda_decay.source_spectrum import *

from utils.gp_regression import GaussianProcess, optimize_hyperparams
from utils.kernels import InvGamma, LogNormal


def dist_azi_depth_distfn(dad1, dad2, params):
    azi_scale = params[0]
    depth_scale = params[1]
    dist = dad1[0]**(1.0/3)-dad2[0]**(1.0/3)
    avg_dist = (dad1[0]+dad2[0])/2
    azi = utils.geog.degdiff(dad1[1], dad2[1]) * avg_dist**(1.0/3)
    depth = dad1[2]**(1.0/3)- dad2[2]**(1.0/3)

    r = np.sqrt(dist**2 + (azi_scale*azi)**2 + (depth_scale*depth)**2)
    return r

def dist_distfn(lldda1, lldda2, params=None):
    return lldda1[3]**(1.0/3)-lldda2[3]**(1.0/3)

def depth_distfn(lldda1, lldda2, params=None):
    return lldda1[2]**(1.0/3)-lldda2[2]**(1.0/3)

def azi_distfn(lldda1, lldda2, params=None):
    avg_dist = (lldda1[2]+lldda2[2])/2
    return utils.geog.degdiff(lldda1[4], lldda2[4]) * avg_dist**(1.0/3)

def ll_distfn(lldda1, lldda2, params=None):
    return utils.geog.dist_km(lldda1[0:2], lldda2[0:2])

def dist_azi_depth_distfn_deriv(i, dad1, dad2, params):
    azi_scale = params[0]
    depth_scale = params[1]
    dist = dad1[0]**(1.0/3)-dad2[0]**(1.0/3)
    avg_dist = (dad1[0]+dad2[0])/2
    azi = utils.geog.degdiff(dad1[1], dad2[1]) * avg_dist**(1.0/3)
    depth = dad1[2]**(1.0/3)- dad2[2]**(1.0/3)
    r = np.sqrt(dist**2 + (azi_scale*azi)**2 + (depth_scale*depth)**2)

    if i==0: # deriv wrt azi_scale
        deriv = azi_scale * azi**2 / r if r != 0 else 0
    elif i==1: # deriv wrt depth_scale
        deriv = depth_scale * depth**2 / r if r != 0 else 0
    else:
        raise Exception("unknown parameter number %d" % i)

    return deriv

def lon_lat_depth_distfn(lld1, lld2, params=None):
    ll = utils.geog.dist_km(lld1[0:2], lld2[0:2])
    depth = lld1[2] - lld2[2]
    r = np.sqrt(ll**2 + depth**2)
    return r

def learn_models(fit_data, earthmodel, target_fn, lld_params, dad_params, lldda_sum_params, lldda_prod_params, optimize, pp, label):

    fit_data = np.reshape(fit_data, (-1, FIT_NUM_COLS))
    n = fit_data.shape[0]

    X = fit_data[:, [FIT_LON, FIT_LAT, FIT_DEPTH, FIT_DISTANCE, FIT_AZIMUTH]]
    Xll = fit_data[:, [FIT_LON, FIT_LAT, FIT_DEPTH]]
    Xad = fit_data[:, [FIT_DISTANCE, FIT_AZIMUTH, FIT_DEPTH]]

    y = np.zeros((n,))
    for i in range(n):
        try:
            y[i] = target_fn(fit_data[i, :])
        except:
            import pdb, traceback
            traceback.print_exc()
            pdb.set_trace()

    gpd = None
    gpt = None

#    lld_priors = [InvGamma(.2, .5), InvGamma(.2, .5), LogNormal(5.0, 3.0)]
    lld_priors = [None, None, None]
#    dad_priors = [InvGamma(.2, .5), InvGamma(.2, .5), LogNormal(1.0, 3.0), LogNormal(-3.0, 5.0), LogNormal(0.0, 5.0)]
    dad_priors = [None, None, None, None, None]

    lldda_sum_priors = [None, None, None, None, None, None, None, None, None]
    lldda_prod_priors = [None, None, None, None, None, None]

    if optimize:
#        print "optimizing prod-composite GP..."
#        lldda_prod_params,v = optimize_hyperparams(X, y, kernel="distfns_prod", start_kernel_params=lldda_prod_params, kernel_extra=[(dist_distfn, None), (azi_distfn, None), (depth_distfn, None), (ll_distfn, None)], kernel_priors = lldda_prod_priors)
 #       print "got params", lldda_prod_params , "giving ll", v

        if lld_params is not None:
            print "optimizing location-based GP..."
            lld_params,v = optimize_hyperparams(Xll, y, kernel="distfn", start_kernel_params=lld_params, kernel_extra=lon_lat_depth_distfn, kernel_priors = lld_priors)
            print "got params", lld_params, "giving ll", v

        if dad_params is not None:
            print "optimizing distance/azimuth-based GP..."
            dad_params,v = optimize_hyperparams(Xad, y, kernel="distfn", start_kernel_params=dad_params, kernel_extra=[dist_azi_depth_distfn, dist_azi_depth_distfn_deriv], kernel_priors = dad_priors)
            print "got params", dad_params , "giving ll", v

        if lldda_sum_params is not None:
            print "optimizing sum-composite GP..."
            lldda_sum_params,v = optimize_hyperparams(X, y, kernel="distfns_sum", start_kernel_params=lldda_sum_params, kernel_extra=[(dist_distfn, None), (azi_distfn, None), (depth_distfn, None), (ll_distfn, None)], kernel_priors = lldda_sum_priors)
            print "got params", lldda_sum_params , "giving ll", v

    gp_lld = None
    if lld_params is not None:
        gp_lld = GaussianProcess(Xll, y, kernel="distfn", kernel_params=lld_params, kernel_extra=lon_lat_depth_distfn)

    gp_dad = None
    if dad_params is not None:
        gp_dad = GaussianProcess(Xad, y, kernel="distfn", kernel_params=dad_params, kernel_extra=[dist_azi_depth_distfn, dist_azi_depth_distfn_deriv], ignore_pos_def_errors=False)

    gp_lldda_sum = None
    if lldda_sum_params is not None:
        gp_lldda_sum = GaussianProcess(X, y, kernel="distfns_sum", kernel_params=lldda_sum_params, kernel_extra=[(dist_distfn, None), (azi_distfn, None), (depth_distfn, None), (ll_distfn, None)], ignore_pos_def_errors=False)

#    gp_lldda_prod = GaussianProcess(X, y, kernel="distfns_prod", kernel_params=lldda_prod_params, kernel_extra=[(dist_distfn, None), (azi_distfn, None), (depth_distfn, None), (ll_distfn, None)], kernel_priors = lldda_prod_priors, ignore_pos_def_errors=False)

    regional_dist = []
    regional_y = []
    tele_dist = []
    tele_y = []
    for i in range(n):
        d = fit_data[i, FIT_DISTANCE]
        dy = target_fn(fit_data[i, :])
        if d > 2000:
            tele_dist.append(d)
            tele_y.append(dy)
        else:
            regional_dist.append(d)
            regional_y.append(dy)

    regional_dist = np.array(regional_dist)
    tele_dist = np.array(tele_dist)


    if pp is not None:
        plt.figure()
        plt.title(label + " GP_DAD + \n%s" % dad_params)
        plt.xlabel("distance (km)")
        plt.ylabel("")
        ds = np.linspace(0, 10000, 150)
        pred = np.array([ gp_dad.predict(np.array((d, 0, 0))) for d in ds])
        try:
            plt.plot(ds, pred, "k-")
            plt.plot(np.concatenate([regional_dist, tele_dist]), np.concatenate([regional_y, tele_y]), 'ro')
            pp.savefig()
        except:
            import pdb
            pdb.set_trace()


    try:
        regional_model = utils.LinearModel.LinearModel("regional", ["distance", "const"],
                                                       [regional_dist,np.ones((len(regional_dist),))],
                                                       regional_y)
    except ValueError:
        print "regional model failed", regional_dist, regional_y
        regional_model=None
    try:
        tele_model = utils.LinearModel.LinearModel("tele", ["distance", "const"],
                                                   [tele_dist,np.ones((len(tele_dist),))],
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
            t = np.linspace(0, 2000, 50)
            pred = [ regional_model[(tv,1)] for tv in t ]
            plt.plot(t, pred, "k-")
        except:
            pass
        try:

            t = np.linspace(2000, np.max(tele_dist), 50)
            pred = [ tele_model[(tv, 1)] for tv in t ]
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
        try:
            plt.figure()
            plt.title(label + " regional gaussian mean %f sigma")
            n, bins, patches = plt.hist(regional_y, normed=1)
            bincenters = 0.5*(bins[1:]+bins[:-1])
            y = mlab.normpdf( bincenters, regional_mean, np.sqrt(regional_var))
            plt.plot(bincenters, y, 'r--', linewidth=1)
            pp.savefig()
        except:
            pass
        try:
            plt.figure()
            plt.title(label + " tele gaussian")
            n, bins, patches = plt.hist(tele_y, normed=1)
            bincenters = 0.5*(bins[1:]+bins[:-1])
            y = mlab.normpdf( bincenters, tele_mean, np.sqrt(tele_var))
            plt.plot(bincenters, y, 'r--', linewidth=1)
            pp.savefig()
        except:
            pass


#    print "outputs", y
#    print "learned means ", (regional_mean, regional_var), (tele_mean, tele_var)

    m = dict()
    m["gp_lld"] = gp_lld
    m["gp_dad"] = gp_dad
    m["gp_lldda_sum"] = gp_lldda_sum
    m["gp_lldda_prod"] = gp_lldda_sum
    m["regional_linear"] = regional_model
    m["tele_linear"] = tele_model
    m["regional_gaussian"] = (regional_mean, regional_var)
    m["tele_gaussian"] = (tele_mean, tele_var)
    return m, lld_params, dad_params, lldda_sum_params, lldda_prod_params

class CodaModel:

    MODEL_TYPE_GP_LLD, MODEL_TYPE_GP_DAD, MODEL_TYPE_GP_LLDDA_SUM, MODEL_TYPE_GP_LLDDA_PROD, MODEL_TYPE_LINEAR, MODEL_TYPE_GAUSSIAN = range(6)


    target_fns = {"decay": lambda r : r[FIT_CODA_DECAY], "onset": lambda r : r[FIT_PEAK_DELAY], "amp": lambda r: r[FIT_CODA_HEIGHT] - r[FIT_MB], "amp_transfer": lambda r : r[FIT_CODA_HEIGHT] - SourceSpectrumModel().source_logamp(r[FIT_MB], int(r[FIT_PHASEID]), bandid=int(r[FIT_BANDID]))}


    def __init__(self, fit_data, band_dir, phaseids, chan, target_str="decay", ignore_evids = None, earthmodel = None, sigmodel=None, sites=None, lld_params=None, dad_params=None, lldda_sum_params = None, lldda_prod_params = None, optimize=False, debug=True):

        if target_str not in self.target_fns.keys():
            raise Exception("invalid target str %s" % target_str)
        else:
             self.target_fn = self.target_fns[target_str]

        # assume that either earthmodel and sigmodel are both given, or neither given
        if sigmodel is None:
            cursor, self.sigmodel, self.earthmodel, self.sites, dbconn = sigvisa_util.init_sigmodel()
        else:
            self.earthmodel = earthmodel
            self.sigmodel = sigmodel
            self.sites=sites

        # used for cross-validation
        if ignore_evids is not None:
            good_rows = np.array([int(fit_data[i, FIT_EVID]) not in ignore_evids for i in range(fit_data.shape[0])])
            fit_data_n = fit_data[good_rows, :]
            fit_data = fit_data_n

        self.siteid = int(fit_data[0, FIT_SITEID])
        self.slon = self.sites[self.siteid-1, 0]
        self.slat = self.sites[self.siteid-1, 1]


        pp = None
        if debug:
            outfile = os.path.join(band_dir, "model_%s_fits_%s_%s.pdf" % (target_str, ":".join([str(p) for p in phaseids]), chan))
            pp = PdfPages(outfile)
            print "saving plots to", outfile

        # learn decay rate models
        self.models, self.lld_params, self.dad_params, self.lldda_sum_params, self.lldda_prod_params = learn_models(fit_data, self.earthmodel, target_fn=self.target_fn, lld_params=lld_params, dad_params=dad_params, lldda_sum_params = lldda_sum_params, lldda_prod_params = lldda_prod_params, pp=pp, label="%s phaseids=%s chan=%s" % (target_str, phaseids, chan), optimize=optimize)
        if pp is not None:
            pp.close()

    def predict(self, ev, model_type, distance = None, azimuth=None):

        if distance is None:
            distance = utils.geog.dist_km((ev[EV_LON_COL], ev[EV_LAT_COL]), (self.slon, self.slat))
        if model_type == self.MODEL_TYPE_GP_LLD:
            return float(self.models['gp_lld'].predict((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL])))
        elif model_type == self.MODEL_TYPE_GP_DAD:
            if azimuth is None:
                azimuth = utils.geog.azimuth((self.slon, self.slat), (ev[EV_LON_COL], ev[EV_LAT_COL]))
            return float(self.models['gp_dad'].predict((distance, azimuth, ev[EV_DEPTH_COL])))
        elif model_type == self.MODEL_TYPE_GP_LLDDA_SUM:
            if azimuth is None:
                azimuth = utils.geog.azimuth((self.slon, self.slat), (ev[EV_LON_COL], ev[EV_LAT_COL]))
            return float(self.models['gp_lldda_sum'].predict((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], distance, azimuth)))
        elif model_type == self.MODEL_TYPE_GP_LLDDA_PROD:
            if azimuth is None:
                azimuth = utils.geog.azimuth((self.slon, self.slat), (ev[EV_LON_COL], ev[EV_LAT_COL]))
            return float(self.models['gp_lldda_prod'].predict((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], distance, azimuth)))
        elif model_type == self.MODEL_TYPE_LINEAR:
            if distance < 2000:
                model = self.models['regional_linear']
            else:
                model = self.models['tele_linear']
            try:
                return float(model[(distance, 1)])
            except:
                return self.predict(ev, self.MODEL_TYPE_GAUSSIAN, distance=distance, azimuth=azimuth)
        elif model_type == self.MODEL_TYPE_GAUSSIAN:
            if distance < 2000:
                (mean, var) = self.models['regional_gaussian']
            else:
                (mean, var) = self.models['tele_gaussian']
            return float(mean)

    def sample(self, ev, model_type, distance = None, azimuth=None):

        if distance is None:
            distance = utils.geog.dist_km((ev[EV_LON_COL], ev[EV_LAT_COL]), (self.slon, self.slat))

        if model_type == self.MODEL_TYPE_GP_LLD:
            return self.models['gp_lld'].sample((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL]))
        elif model_type == self.MODEL_TYPE_GP_DAD:
            if azimuth is None:
                azimuth = utils.geog.azimuth((self.slon, self.slat), (ev[EV_LON_COL], ev[EV_LAT_COL]))
            return self.models['gp_dad'].sample((distance, azimuth, ev[EV_DEPTH_COL]))
        elif model_type == self.MODEL_TYPE_GP_LLDDA_SUM:
            if azimuth is None:
                azimuth = utils.geog.azimuth((self.slon, self.slat), (ev[EV_LON_COL], ev[EV_LAT_COL]))
            return float(self.models['gp_lldda_sum'].sample((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], distance, azimuth)))
        elif model_type == self.MODEL_TYPE_GP_LLDDA_PROD:
            if azimuth is None:
                azimuth = utils.geog.azimuth((self.slon, self.slat), (ev[EV_LON_COL], ev[EV_LAT_COL]))
            return float(self.models['gp_lldda_prod'].sample((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], distance, azimuth)))
        elif model_type == self.MODEL_TYPE_LINEAR:
            raise RuntimeError("sampling not yet implemented for linear models")
            if distance < 2000:
                model = self.models['regional_linear']
            else:
                model = self.models['tele_linear']

        elif model_type == self.MODEL_TYPE_GAUSSIAN:
            if distance < 2000:
                (mean, var) = self.models['regional_gaussian']
            else:
                (mean, var) = self.models['tele_gaussian']
            return mean + np.random.randn() * np.sqrt(var)

    def log_likelihood(self, val, ev, model_type, distance = None, azimuth = None):

        ll = None

        if distance is None:
            distance = utils.geog.dist_km((ev[EV_LON_COL], ev[EV_LAT_COL]), (self.slon, self.slat))

        if model_type == self.MODEL_TYPE_GP_LLD:
            ll = self.models['gp_lld'].posterior_log_likelihood((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL]), val)
        elif model_type == self.MODEL_TYPE_GP_DAD:
            if azimuth is None:
                azimuth = utils.geog.azimuth((self.slon, self.slat), (ev[EV_LON_COL], ev[EV_LAT_COL]))
            l = self.models['gp_dad'].posterior_log_likelihood((distance, azimuth, ev[EV_DEPTH_COL]), val)
        elif model_type == self.MODEL_TYPE_GP_LLDDA_SUM:
            if azimuth is None:
                azimuth = utils.geog.azimuth((self.slon, self.slat), (ev[EV_LON_COL], ev[EV_LAT_COL]))
            ll = self.models['gp_lldda_sum'].posterior_log_likelihood((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], distance, azimuth), val)
        elif model_type == self.MODEL_TYPE_GP_LLDDA_PROD:
            if azimuth is None:
                azimuth = utils.geog.azimuth((self.slon, self.slat), (ev[EV_LON_COL], ev[EV_LAT_COL]))
            ll = self.models['gp_lldda_prod'].posterior_log_likelihood((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], distance, azimuth), val)
        elif model_type == self.MODEL_TYPE_LINEAR:
            raise RuntimeError("log likelihood not yet implemented for linear models")
            if distance < 2000:
                model = self.models['regional_linear']
            else:
                model = self.models['tele_linear']
        elif model_type == self.MODEL_TYPE_GAUSSIAN:
            if distance < 2000:
                (mean, var) = self.models['regional_gaussian']
            else:
                (mean, var) = self.models['tele_gaussian']
            ll -.5 *np.log(2*np.pi*var) - .5 *(val-mean)**2 / var

        if ll is None or np.isnan(ll):
            print "error: invalid ll", ll
            import pdb
            pdb.set_trace()

        return ll


def main():
    cm = CodaModel(sys.argv[1], ignore_evids = (5330781,))

if __name__ == "__main__":
    main()
