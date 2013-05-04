import os
import time
import numpy as np
import collections
import scipy
import scipy.sparse
import pyublas
import hashlib

from sigvisa.gpr import munge, kernels, evaluate, learn, distributions, plot
from sigvisa.gpr.gp import GaussianProcess

from sigvisa.models.spatial_regression.baseline_models import ParamModel
from sigvisa.source.event import Event

from sigvisa.utils.cover_tree import VectorTree, MatrixTree

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
            if 'K' not in kwargs:
                k = kwargs['kernel']
                noise_var = k.lhs.params[0]
                width_param = k.rhs.params[1]
                depthscale_param = k.rhs.params[2]
                signal_var = np.array((k.rhs.params[0],), copy=True, dtype=float)
                ll_scale = 1.0/width_param
                d_scale = depthscale_param/width_param
                X = kwargs['X']
                vt = VectorTree(X[0:1,:], 1, "lld", np.array((ll_scale, d_scale), dtype=float))
                import time
                t0 = time.time()
                kwargs['K'] = vt.debug_kernel_matrix(pyublas.why_not(X), pyublas.why_not(X), "se", signal_var, False) + noise_var * np.eye(len(X))
                t1 = time.time()
                print "got kernel matrix in %f seconds", t1 - t0

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

        import time

        t0 = time.time()
        self.predict_tree = VectorTree(self.X, 1, "lld", np.array((ll_scale, d_scale), dtype=float))
        self.predict_tree.set_v(0, self.alpha_r)
        t1 = time.time()

        d = len(self.basisfns)
        self.cov_tree = VectorTree(self.X, d, "lld", np.array((ll_scale, d_scale), dtype=float))
        for i in range(d):
            self.cov_tree.set_v(i, self.HKinv[i, :])
        t2 = time.time()

        Kinv = np.dot(self.invL.T, self.invL)
        t3 = time.time()
        self.spKinv = scipy.sparse.csr_matrix(Kinv * (np.abs(Kinv) > 1e-20 ) )

        nzr, nzc = self.spKinv.nonzero()
        print "Kinv matrix has %d of %d nonzero elements" % (len(nzr), np.product(Kinv.shape))
        t4 = time.time()
        self.double_tree = MatrixTree(self.X, nzr, nzc, "lld", np.array((ll_scale, d_scale), dtype=float))
        t5 = time.time()
        self.double_tree.set_m(Kinv)
        t6 = time.time()

        print "predict tree time:", t1-t0
        print "cov tree time:", t2-t1
        print "matrix mult time:", t3-t2
        print "sparse matrix time:", t4-t3
        print "double tree creation time:", t5-t4
        print "double tree population time:", t6-t5

        self.Kinv = Kinv

    def get_query_K(self, X1):
        # avoid recomputing the kernel if we're evaluating at the same
        # point multiple times. This is effectively a size-1 cache.
        try:
            self.query_hsh
        except AttributeError:
            self.query_hsh = None

        hsh = hashlib.sha1(X1.view(np.uint8)).hexdigest()
        if hsh != self.query_hsh:
            signal_var = np.array((self.kernel.rhs.params[0],), copy=True, dtype=float)
            self.query_K = self.predict_tree.debug_kernel_matrix(self.X, X1, "se", signal_var, False)
            self.query_hsh = hsh

            if self.basisfns:
                H = np.array([[f(x) for x in X1] for f in self.basisfns], dtype=float)
                self.query_R = H - np.dot(self.HKinv, self.query_K)

#            print "cache fail: model %d called with " % (len(self.alpha)), X1
#        else:
#            print "cache hit!"
        return self.query_K

    def predict(self, cond, eps=1e-8):
        X1 = self.standardize_input_array(cond)

        # Kstar = self.get_query_K(X1)
        #gp_pred = self.mu + np.dot(Kstar.T, self.alpha_r)
        #gp_pred = np.array([self.mu + np.dot(k, self.alpha_r) for k in Kstar.T])

        signal_var = np.array((self.kernel.rhs.params[0],), copy=True, dtype=float)
        #Kstar2 = self.tree.debug_kernel_matrix(X1, self.X, "se", signal_var, False)

        #np.savetxt('Kstar.txt', Kstar)
        #np.savetxt('Kstar2.txt', Kstar2)

        gp_pred = np.array([self.mu + self.predict_tree.weighted_sum(0, np.reshape(x, (1,-1)), eps, "se", signal_var) for x in X1])

        if self.basisfns:
            H = np.array([[f(x) for x in X1] for f in self.basisfns], dtype=float)
            mean_pred = np.reshape(np.dot(H.T, self.beta_bar), gp_pred.shape)
            gp_pred += mean_pred

        if len(gp_pred) == 1:
            gp_pred = gp_pred[0]

        return gp_pred


    def cov_sparse(self, X1, include_obs=False, pad=1e-8):
        """
        Compute the posterior covariance matrix at a set of points given by the rows of X1.

        Default is to compute the covariance of f, the latent function values. If obs_covar
        is True, we instead compute the covariance of y, the observed values.

        By default, we add a tiny bit of padding to the diagonal to counteract any potential
        loss of positive definiteness from numerical issues. Setting pad=0 disables this.

        """

        Kstar = self.get_query_K(X1)
        tmp = self.spKinv * Kstar
        a = np.dot(Kstar.T, tmp)
        gp_cov = self.kernel(X1,X1, identical=include_obs) - a

        if self.basisfns:
            R = self.query_R
            tmp = np.dot(self.invc, R)
            mean_cov = np.dot(tmp.T, tmp)
            gp_cov += mean_cov

        gp_cov += pad * np.eye(gp_cov.shape[0])
        return gp_cov


    """
    def covariance_single_tree(self, cond, include_obs=False, parametric_only=False, pad=1e-8, eps=0):
        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]
        d = len(self.basisfns)
        n = self.n
        gp_cov = self.kernel(X1, X1, identical=include_obs)
        tmp = np.zeros((n, m))

        # compute self.invL * Kstar, where Kstar is the n x m kernel
        # matrix, where m is the number of query points (so e.g. Kstar
        # is a column vector if there's just one query point).
        import time
        t0 = time.time()
        signal_var = np.array((self.kernel.rhs.params[0],), copy=True, dtype=float)
        for i in range(n):
            for j in range(m):
                tmp[i,j] = self.cov_tree.weighted_sum(i, X1[j:j+1,:], eps, "se", signal_var)
        qf = np.dot(tmp.T, tmp)
        gp_cov -= qf
        print "single qf", qf
        t1 = time.time()

        if self.basisfns:
            H = np.array([[f(x) for x in X1] for f in self.basisfns], dtype=float)
            HKinvKstar = np.zeros((d, m))

            for i in range(d):
                for j in range(m):
                    HKinvKstar[i,j] = self.cov_tree.weighted_sum(n+i, X1[j:j+1,:], eps, "se", signal_var)
            R = H - HKinvKstar
            v = np.dot(self.invc, R)
            gp_cov += np.dot(v.T, v)
        t2 = time.time()

        self.nptime = (t1-t0)
        self.ptime = (t2-t1)

        gp_cov += pad * np.eye(m)
        return gp_cov
    """

    def covariance_double_tree(self, cond, include_obs=False, parametric_only=False, pad=1e-8, eps=1e-8):
        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]
        d = len(self.basisfns)
        n = self.n
        gp_cov = self.kernel(X1, X1, identical=include_obs)

        # compute self.invL * Kstar, where Kstar is the n x m kernel
        # matrix, where m is the number of query points (so e.g. Kstar
        # is a column vector if there's just one query point).
        import time
        t0 = time.time()
        signal_var = np.array((self.kernel.rhs.params[0],), copy=True, dtype=float)
        qf = self.double_tree.quadratic_form(X1, eps, "se", signal_var)
        gp_cov -= qf
        t1 = time.time()
        print t1-t0

        if self.basisfns:
            H = np.array([[f(x) for x in X1] for f in self.basisfns], dtype=float)
            HKinvKstar = np.zeros((d, m))

            for i in range(d):
                for j in range(m):
                    HKinvKstar[i,j] = self.cov_tree.weighted_sum(i, X1[j:j+1,:], eps, "se", signal_var)
            R = H - HKinvKstar
            v = np.dot(self.invc, R)
            gp_cov += np.dot(v.T, v)
        t2 = time.time()

        self.nptime = (t1-t0)
        self.ptime = (t2-t1)

        gp_cov += pad * np.eye(m)
        return gp_cov


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
