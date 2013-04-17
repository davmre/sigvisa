import os
import time
import numpy as np
import collections



from sigvisa.gpr import munge, kernels, evaluate, learn, distributions, plot
from sigvisa.gpr.gp import GaussianProcess

from sigvisa.models.spatial_regression.baseline_models import ParamModel
from sigvisa.source.event import Event

start_params_dad_log = {"coda_decay": [.022, .0187, 1.00, .14, .1],
                        "amp_transfer": [1.1, 3.4, 9.5, 0.1, .31],
                        "peak_offset": [2.7, 3.4, 2, .7, 0.1]
                        }

start_params_lld = {"coda_decay": [.022, .0187, 50.00],
                    "amp_transfer": [1.1, 3.4, 100.00],
                    "peak_offset": [2.7, 3.4, 50.00]
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
    depth = lldda1[2] - lldda2[2]
    r = np.sqrt(ll ** 2 + depth ** 2)
    return r

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
        local_kernel = kernels.DistFNKernel(params=params[1:3], priors=priors[1:3],
                                    distfn = lon_lat_depth_distfn, deriv=None)
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
        local_kernel = kernels.DistFNKernel(params=params[7:9], priors=priors[7:9],
                                    distfn = lon_lat_depth_distfn, deriv=None)
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

        if self.distfn_str:
            kwargs['kernel_extra'] = distfns[self.distfn_str]

        GaussianProcess.__init__(self, *args, **kwargs)

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

    def predict(self, cond, **kwargs):
        X1 = self.standardize_input_array(cond)

        result = GaussianProcess.predict(self, X1, **kwargs)
        if len(result) == 1:
            result = result[0]
        return result

    def save_trained_model(self, filename):
        """
        Serialize the model to a file.
        """
        d = self.pack_npz()
        with open(filename, 'wb') as f:
            np.savez(f, base_str=super(SpatialGP, self).__repr_base_params__(), **d)

    def load_trained_model(self, filename):
        npzfile = np.load(filename)
        self.unpack_npz(npzfile)
        super(SpatialGP, self).__unrepr_base_params__(str(npzfile['base_str']))
        del npzfile.f
        npzfile.close()
        self.n = self.X.shape[0]
