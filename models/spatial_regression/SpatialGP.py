import os
import time
import numpy as np
import collections
import scipy
import pyublas

from sigvisa.gpr import munge, kernels, evaluate, learn, distributions, plot
from sigvisa.gpr.gp import GaussianProcess

from sigvisa.models.spatial_regression.baseline_models import ParamModel
from sigvisa.source.event import Event

from sigvisa.utils.cover_tree import CoverTree

start_params_dad_log = {"coda_decay": [.022, .0187, 1.00, .14, .1],
                        "amp_transfer": [1.1, 3.4, 9.5, 0.1, .31],
                        "peak_offset": [2.7, 3.4, 2, .7, 0.1]
                        }

start_params_lld = {"coda_decay": [.022, .0187, 50.00, 1.0],
                    "amp_transfer": [1.1, 3.4, 100.00, 1.0],
                    "peak_offset": [2.7, 3.4, 50.00, 1.0]
                    }

start_params_composite = {"coda_decay": [.022, .01, 1.0, .01, 100.0, .01, 3.0, .01, 100.0],
                          "amp_transfer": [1.1, 3.0, 5.0, 3.0, 100.0, 3.0, 3.0, 3.0, 100.0],
                          "peak_offset": [2.7, 3.0, 5.0, 3.0, 100.0, 3.0, 3.0, 3.0, 100.0],
                          }

start_params = {"dad_log": start_params_dad_log,
                "lld": start_params_lld,
                "composite": start_params_composite
                }


X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI = range(5)

def dist_azi_depth_distfn_log(lldda1, lldda2, params):
    import sigvisa.utils.geog as geog
    import numpy as np

    azi_scale = params[0]
    depth_scale = params[1]
    dist = np.log(lldda1[3] + 1) - np.log(lldda2[3] + 1)
    avg_dist = (lldda1[3] + lldda2[3]) / 2
    azi = geog.degdiff(lldda1[4], lldda2[4]) * np.log(avg_dist)
    depth = np.log(lldda1[2] + 1) - np.log(lldda2[2] + 1)

    r = np.sqrt(dist ** 2 + (azi_scale * azi) ** 2 + (depth_scale * depth) ** 2)
    return r


def dist_azi_depth_distfn_deriv_log(i, lldda1, lldda2, params):
    import numpy as np
    import sigvisa.utils.geog as geog
    azi_scale = params[0]
    depth_scale = params[1]
    dist = np.log(lldda1[3] + 1) - np.log(lldda2[3] + 1)
    avg_dist = (lldda1[3] + lldda2[3]) / 2
    azi = geog.degdiff(lldda1[4], lldda2[4]) * np.log(avg_dist + 1)
    depth = np.log(lldda1[2] + 1) - np.log(lldda2[2] + 1)
    r = np.sqrt(dist ** 2 + (azi_scale * azi) ** 2 + (depth_scale * depth) ** 2)

    if i == 0:  # deriv wrt azi_scale
        deriv = azi_scale * azi ** 2 / r if r != 0 else 0
    elif i == 1:  # deriv wrt depth_scale
        deriv = depth_scale * depth ** 2 / r if r != 0 else 0
    else:
        raise Exception("unknown parameter number %d" % i)

    return deriv


def lon_lat_depth_distfn(lldda1, lldda2, params=None):
    import sigvisa.utils.geog as geog
    import numpy as np
    ll = geog.dist_km(tuple(lldda1[0:2]), tuple(lldda2[0:2]))
    depth = ( lldda1[2] - lldda2[2] ) * params[0]
    r = np.sqrt(ll ** 2 + depth ** 2)
    return r

def lon_lat_depth_distfn_deriv(i, lldda1, lldda2, params=None):
    import sigvisa.utils.geog as geog
    import numpy as np
    assert (i == 0)
    ll = geog.dist_km(tuple(lldda1[0:2]), tuple(lldda2[0:2]))
    depth = ( lldda1[2] - lldda2[2] ) * params[0]
    r = np.sqrt(ll ** 2 + depth ** 2)
    return ( params[0] * ( lldda1[2] - lldda2[2] )**2 ) / r if r != 0 else 0.0

def logdist_diff_distfn(lldda1, lldda2, params=None):
    import numpy as np
    dist = np.log(lldda1[3] + 1) - np.log(lldda2[3] + 1)
    return dist

def azi_diff_distfn(lldda1, lldda2, params=None):
    import sigvisa.utils.geog as geog
    import numpy as np
    azi = np.abs ( geog.degdiff(lldda1[4], lldda2[4]) )
    return azi

def logdepth_diff_distfn(lldda1, lldda2, params=None):
    import numpy as np
    depth = np.log(lldda1[2] + 1) - np.log(lldda2[2] + 1)
    return depth

def spatial_kernel_from_str(kernel_str, target=None, params=None):
    params = params if params is not None else start_params[kernel_str][target]
    priors = [None,] * len(params) # TODO: use real priors

    if kernel_str == "dad_log":
        k = kernels.setup_kernel(name='distfn',
                                 params = params,
                                 extra=[dist_azi_depth_distfn_log, dist_azi_depth_distfn_deriv_log],
                                 )
    elif kernel_str == "lld":
        noise_kernel = kernels.DiagonalKernel(params=params[0:1], priors = priors[0:1])
        local_kernel = kernels.DistFNKernel(params=params[1:4], priors=priors[1:4],
                                            distfn = lon_lat_depth_distfn, deriv=lon_lat_depth_distfn_deriv)
        k = noise_kernel + local_kernel
    elif kernel_str == "composite":
        # assume we are passed the following params/priors:
        # 0 : sigma2_n -- noise variance
        # 1 : sigma2_f_dist -- function variance wrt dist_diff
        # 2 : w_dist -- length scale for dist_diff
        # 3 : sigma2_f_azi -- function variance wrt azi_diff
        # 4 : w_azi -- length scale for azi_diff
        # 5 : sigma2_f_depth -- function variance wrt depth_diff
        # 6 : w_depth -- length scale for depth_diff
        # 7 : sigma2_f_local -- function variance wrt local_dist
        # 8 : w_local -- length scale for local_dist

        noise_kernel = kernels.DiagonalKernel(params=params[0:1], priors = priors[0:1])
        distdiff_kernel = kernels.DistFNKernel(params=params[1:3], priors=priors[1:3],
                                       distfn = logdist_diff_distfn, deriv=None)
        azidiff_kernel = kernels.DistFNKernel(params=params[3:5], priors=priors[3:5],
                                              distfn = azi_diff_distfn, deriv=None)
        depthdiff_kernel = kernels.DistFNKernel(params=params[5:7], priors=priors[5:7],
                                                distfn = logdepth_diff_distfn, deriv=None)
        local_kernel = kernels.DistFNKernel(params=params[7:10], priors=priors[7:10],
                                            distfn = lon_lat_depth_distfn, deriv=lon_lat_depth_distfn_deriv)
        k = noise_kernel + distdiff_kernel + azidiff_kernel + depthdiff_kernel + local_kernel

    return k

class SpatialGP(GaussianProcess, ParamModel):

    def __init__(self, *args, **kwargs):
        try:
            self.distfn_str = kwargs.pop("distfn_str")
        except KeyError:
            self.distfn_str = None

        try:
            ParamModel.__init__(self, sta=kwargs.pop("sta"))
        except KeyError:
            pass

        if 'fname' not in kwargs:
            kwargs['X'] = np.array(kwargs['X'], copy=False, dtype=float)

        GaussianProcess.__init__(self, *args, save_extra_info=True, **kwargs)

        if 'fname' not in kwargs:
            if self.basisfns is not None:
                r = self.y - np.dot(self.H.T, self.beta_bar)
            else:
                r = self.y
            self.alpha_r = scipy.linalg.cho_solve((self.L, True), r)
            self.build_point_tree()

    def build_point_tree(self):
        #noise_var = self.kernel.lhs.params[0]
        #signal_var = self.kernel.rhs.params[0]
        width_param = self.kernel.rhs.params[1]
        depthscale_param = self.kernel.rhs.params[2]

        ll_scale = 1.0/width_param
        d_scale = depthscale_param/width_param

        self.tree = CoverTree(self.X, "lld", np.array((ll_scale, d_scale), dtype=float))
        self.tree.set_v(0, self.alpha_r)

    def predict(self, cond, eps=1e-8):
        X1 = self.standardize_input_array(cond)

        # Kstar = self.get_query_K(X1)
        #gp_pred = self.mu + np.dot(Kstar.T, self.alpha_r)
        #gp_pred = np.array([self.mu + np.dot(k, self.alpha_r) for k in Kstar.T])

        signal_var = np.array((self.kernel.rhs.params[0],), copy=True, dtype=float)
        Kstar2 = self.tree.debug_kernel_matrix(X1, self.X, "se", signal_var, False)

        #np.savetxt('Kstar.txt', Kstar)
        #np.savetxt('Kstar2.txt', Kstar2)

        gp_pred = np.array([self.mu + self.tree.weighted_sum(0, np.reshape(x, (1,-1)), eps, "se", signal_var) for x in X1])

        if self.basisfns:
            H = np.array([[f(x) for x in X1] for f in self.basisfns], dtype=float)
            mean_pred = np.reshape(np.dot(H.T, self.beta_bar), gp_pred.shape)
            gp_pred += mean_pred

        if len(gp_pred) == 1:
            gp_pred = gp_pred[0]

        return gp_pred

    def variance(self, cond, **kwargs):
        X1 = self.standardize_input_array(cond)

        result = GaussianProcess.variance(self, X1, **kwargs)
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

    def save_trained_model(self, filename):
        """
        Serialize the model to a file.
        """
        d = self.pack_npz()
        d['alpha_r'] = self.alpha_r
        with open(filename, 'wb') as f:
            np.savez(f, base_str=super(SpatialGP, self).__repr_base_params__(), **d)

    def load_trained_model(self, filename):
        npzfile = np.load(filename)
        self.unpack_npz(npzfile)
        super(SpatialGP, self).__unrepr_base_params__(str(npzfile['base_str']))
        self.alpha_r = npzfile['alpha_r']
        del npzfile.f
        npzfile.close()
        self.n = self.X.shape[0]
        self.build_point_tree()
