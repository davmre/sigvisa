import os
import numpy as np
import collections

import sigvisa.utils.geog as geog


from gpr import munge, kernels, evaluate, learn, distributions, plot
from gpr.gp import GaussianProcess

from sigvisa.models.spatial_regression.baseline_models import ParamModel
from sigvisa.source.event import Event

start_params_dad_log = {"coda_decay": [.022, .0187, 1.00, .14, .1],
                        "amp_transfer": [1.1, 3.4, 9.5, 0.1, .31],
                        "peak_offset": [2.7, 3.4, 2, .7, 0.1]
                        }
start_params_dad_cuberoot = {"coda_decay": [.022, .0187, 1.00, .14, .1],
                             "amp_transfer": [1.1, 3.4, 9.5, 0.1, .31],
                             "peak_offset": [2.7, 3.4, 2, .7, 0.1]
                             }
start_params_dad_linear = {"coda_decay": [.022, .0187, 500.00, .14, .1],
                           "amp_transfer": [1.1, 3.4, 1000.00, 0.1, .31],
                           "peak_offset": [2.7, 3.4, 500.00, .7, 0.1]
                           }
start_params_lld = {"coda_decay": [.022, .0187, 50.00],
                    "amp_transfer": [1.1, 3.4, 100.00],
                    "peak_offset": [2.7, 3.4, 50.00]
                    }
start_params = {"dad_log": start_params_dad_log,
                "dad_cuberoot": start_params_dad_cuberoot,
                "dad_linear": start_params_dad_linear,
                "lld": start_params_lld
                }


X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI = range(5)


def gp_extract_features(X, distfn):
    if distfn == "lld":
        X = X[:, [X_LON, X_LAT, X_DEPTH]]
    elif distfn.startswith("dad"):
        X = X[:, [X_DIST, X_AZI, X_DEPTH]]
    return X


def dist_azi_depth_distfn_log(dad1, dad2, params):
    azi_scale = params[0]
    depth_scale = params[1]
    dist = np.log(dad1[0] + 1) - np.log(dad2[0] + 1)
    avg_dist = (dad1[0] + dad2[0]) / 2
    azi = geog.degdiff(dad1[1], dad2[1]) * np.log(avg_dist)
    depth = np.log(dad1[2] + 1) - np.log(dad2[2] + 1)

    r = np.sqrt(dist ** 2 + (azi_scale * azi) ** 2 + (depth_scale * depth) ** 2)
    return r


def dist_azi_depth_distfn_linear(dad1, dad2, params):
    azi_scale = params[0]
    depth_scale = params[1]
    dist = dad1[0] - dad2[0]
    avg_dist = (dad1[0] + dad2[0]) / 2
    azi = geog.degdiff(dad1[1], dad2[1]) * avg_dist
    depth = dad1[2] - dad2[2]

    r = np.sqrt(dist ** 2 + (azi_scale * azi) ** 2 + (depth_scale * depth) ** 2)
    return r


def dist_azi_depth_distfn_cuberoot(dad1, dad2, params):
    azi_scale = params[0]
    depth_scale = params[1]
    dist = dad1[0] ** (1.0 / 3) - dad2[0] ** (1.0 / 3)
    avg_dist = (dad1[0] + dad2[0]) / 2
    azi = geog.degdiff(dad1[1], dad2[1]) * avg_dist ** (1.0 / 3)
    depth = dad1[2] ** (1.0 / 3) - dad2[2] ** (1.0 / 3)

    r = np.sqrt(dist ** 2 + (azi_scale * azi) ** 2 + (depth_scale * depth) ** 2)
    return r


def dist_distfn(lldda1, lldda2, params=None):
    return lldda1[3] ** (1.0 / 3) - lldda2[3] ** (1.0 / 3)


def depth_distfn(lldda1, lldda2, params=None):
    return lldda1[2] ** (1.0 / 3) - lldda2[2] ** (1.0 / 3)


def azi_distfn(lldda1, lldda2, params=None):
    avg_dist = (lldda1[2] + lldda2[2]) / 2
    return geog.degdiff(lldda1[4], lldda2[4]) * avg_dist ** (1.0 / 3)


def ll_distfn(lldda1, lldda2, params=None):
    return geog.dist_km(lldda1[0:2], lldda2[0:2])


def dist_azi_depth_distfn_deriv_cuberoot(i, dad1, dad2, params):
    azi_scale = params[0]

    depth_scale = params[1]
    dist = dad1[0] ** (1.0 / 3) - dad2[0] ** (1.0 / 3)
    avg_dist = (dad1[0] + dad2[0]) / 2
    azi = geog.degdiff(dad1[1], dad2[1]) * avg_dist ** (1.0 / 3)
    depth = dad1[2] ** (1.0 / 3) - dad2[2] ** (1.0 / 3)
    r = np.sqrt(dist ** 2 + (azi_scale * azi) ** 2 + (depth_scale * depth) ** 2)

    if i == 0:  # deriv wrt azi_scale
        deriv = azi_scale * azi ** 2 / r if r != 0 else 0
    elif i == 1:  # deriv wrt depth_scale
        deriv = depth_scale * depth ** 2 / r if r != 0 else 0
    else:
        raise Exception("unknown parameter number %d" % i)

    return deriv


def dist_azi_depth_distfn_deriv_linear(i, dad1, dad2, params):
    azi_scale = params[0]
    depth_scale = params[1]
    dist = dad1[0] - dad2[0]
    avg_dist = (dad1[0] + dad2[0]) / 2
    azi = geog.degdiff(dad1[1], dad2[1]) * avg_dist
    depth = dad1[2] - dad2[2]
    r = np.sqrt(dist ** 2 + (azi_scale * azi) ** 2 + (depth_scale * depth) ** 2)

    if i == 0:  # deriv wrt azi_scale
        deriv = azi_scale * azi ** 2 / r if r != 0 else 0
    elif i == 1:  # deriv wrt depth_scale
        deriv = depth_scale * depth ** 2 / r if r != 0 else 0
    else:
        raise Exception("unknown parameter number %d" % i)

    return deriv


def dist_azi_depth_distfn_deriv_log(i, dad1, dad2, params):
    azi_scale = params[0]
    depth_scale = params[1]
    dist = np.log(dad1[0] + 1) - np.log(dad2[0] + 1)
    avg_dist = (dad1[0] + dad2[0]) / 2
    azi = geog.degdiff(dad1[1], dad2[1]) * np.log(avg_dist + 1)
    depth = np.log(dad1[2] + 1) - np.log(dad2[2] + 1)
    r = np.sqrt(dist ** 2 + (azi_scale * azi) ** 2 + (depth_scale * depth) ** 2)

    if i == 0:  # deriv wrt azi_scale
        deriv = azi_scale * azi ** 2 / r if r != 0 else 0
    elif i == 1:  # deriv wrt depth_scale
        deriv = depth_scale * depth ** 2 / r if r != 0 else 0
    else:
        raise Exception("unknown parameter number %d" % i)

    return deriv


def lon_lat_depth_distfn(lld1, lld2, params=None):
    ll = geog.dist_km(lld1[0:2], lld2[0:2])
    depth = lld1[2] - lld2[2]
    r = np.sqrt(ll ** 2 + depth ** 2)
    return r


distfns = {
    "dad_cuberoot": [dist_azi_depth_distfn_cuberoot, dist_azi_depth_distfn_deriv_cuberoot],
"dad_linear": [dist_azi_depth_distfn_linear, dist_azi_depth_distfn_deriv_linear],
"dad_log": [dist_azi_depth_distfn_log, dist_azi_depth_distfn_deriv_log],
"lld": lon_lat_depth_distfn
}


class SpatialGP(GaussianProcess, ParamModel):

    def __init__(self, *args, **kwargs):
        try:
            self.distfn_str = kwargs.pop("distfn_str")
        except KeyError:
            self.distfn_str = "dad_log"

        try:
            ParamModel.__init__(self, sta=kwargs.pop("sta"))
        except KeyError:
            pass

        kwargs['kernel_extra'] = distfns[self.distfn_str]
        kwargs['kernel'] = "distfn"

        GaussianProcess.__init__(self, *args, **kwargs)

    def variance(self, cond):
        X1 = self.standardize_input_array(cond)

        result = GaussianProcess.variance(self, X1)
        if len(result) == 1:
            result = result[0]
        return result

    def sample(self, cond):
        X1 = self.standardize_input_array(cond)

        result = GaussianProcess.sample(self, X1)
        if len(result) == 1:
            result = result[0]
        return result

    def log_p(self, x, cond):
        X1 = self.standardize_input_array(cond)
        x = x if isinstance(x, collections.Iterable) else (x,)
        result = GaussianProcess.posterior_log_likelihood(self, X1, x)
        return result

    def predict(self, cond):
        X1 = self.standardize_input_array(cond)

        result = GaussianProcess.predict(self, X1)
        if len(result) == 1:
            result = result[0]
        return result

    def standardize_input_array(self, c):
        X1 = super(SpatialGP, self).standardize_input_array(c)
        if X1.shape[1] == 5:
            X1 = gp_extract_features(X1, self.distfn_str)
        return X1

    def save_trained_model(self, filename):
        """
        Serialize the model to a file.
        """
        kname = np.array((self.kernel_name,))
        mname = np.array((self.mean,))
        with open(filename, 'w') as f:
            np.savez(
                f, X=self.X, y=self.y, mu=np.array((self.mu,)), kernel_name=kname, kernel_params=self.kernel_params, mname=mname, alpha=self.alpha,
                Kinv=self.Kinv, L=self.L, distfn_str=self.distfn_str, ll=self.ll, base_str=super(SpatialGP, self).__repr_base_params__())

    def load_trained_model(self, filename):
        npzfile = np.load(filename)
        self.unpack_npz(npzfile)
        self.distfn_str = str(npzfile['distfn_str'])
        self.kernel_extra = distfns[self.distfn_str]
        self.kernel = kernels.setup_kernel(self.kernel_name, self.kernel_params, extra=self.kernel_extra)
        super(SpatialGP, self).__unrepr_base_params__(str(npzfile['base_str']))
        del npzfile.f
        npzfile.close()
