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
from sigvisa.gpr.util import marshal_fn, unmarshal_fn


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

X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI = range(5)


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

"""
def spatial_kernel_from_str(target=None, params=None):
    params = params if params is not None else start_params_lld[target]

    return params
"""

class SpatialGP(GaussianProcess, ParamModel):

    def init_hyperparams(self, hyperparams):
        (noise_var, signal_var, ll_scale, d_scale) = hyperparams
        self.noise_var = noise_var
        self.dfn_params = np.array((ll_scale, d_scale), dtype=np.float)
        self.wfn_params = np.array((signal_var,), copy=True, dtype=np.float)

    def build_kernel_matrix(self, X, hyperparams):
        self.init_hyperparams(hyperparams)
        vt = VectorTree(X[0:1,:], 1, "lld", self.dfn_params)
        K = vt.kernel_matrix(X, X, "se", self.wfn_params, False) + self.noise_var * np.eye(len(X), dtype=np.float64)

        K += np.eye(K.shape[0], dtype=np.float64) * 1e-8 # try to avoid losing
                                       # positive-definiteness
                                       # to numeric issues

        return K

    def invert_kernel_matrix(self, K):
        L = None
        alpha = None
        try:
            L = scipy.linalg.cholesky(K, lower=True)
            alpha = scipy.linalg.cho_solve((L, True), self.y)
            Kinv = scipy.linalg.inv(K)
        except np.linalg.linalg.LinAlgError:
            #u,v = np.linalg.eig(K)
            #print K, u
            #import pdb; pdb.set_trace()
            raise
        except ValueError:
            raise
        return alpha, L, Kinv

    def build_parametric_model(self, alpha, Kinv_sp, H, b, B):
        # notation follows section 2.7 of Rasmussen and Williams
        Binv = scipy.linalg.inv(B)
        tmp = np.dot(H, alpha) + np.dot(Binv, b)  # H * K^-1 * y + B^-1 * b

        HKinv = H * Kinv_sp
        M_inv  = Binv + np.dot(HKinv, H.T) # here M = (inv(B) +
                                           # H*K^-1*H.T)^-1 is the
                                           # posterior covariance
                                           # matrix on the params.

        c = scipy.linalg.cholesky(M_inv, lower=True) # c = sqrt(inv(B) + H*K^-1*H.T)
        beta_bar = scipy.linalg.cho_solve((c, True), tmp)
        invc = scipy.linalg.inv(c)

        return c, beta_bar, invc, HKinv

    def sparsify(self, M):
        return scipy.sparse.csr_matrix(M * (np.abs(M) > self.sparse_threshold))

    def sort_events(self, X, y):
        combined = np.hstack([X, np.reshape(y, (-1, 1))])
        combined_sorted = np.array(sorted(combined, key = lambda x: x[0]), dtype=float)
        X_sorted = np.array(combined_sorted[:, :-1], copy=True, dtype=float)
        y_sorted = combined_sorted[:, -1].flatten()
        return X_sorted, y_sorted

    def __init__(self, X=None, y=None,
                 fname=None, basisfns=None,
                 hyperparams=None,
                 param_mean=None, param_cov=None,
                 compute_ll=False,
                 compute_grad=False,
                 sparse_threshold=1e-20,
                 sta = None,
                 sort_events=True):

        try:
            ParamModel.__init__(self, sta=sta)
        except KeyError:
            pass


        if fname is not None:
            self.load_trained_model(fname)
        else:
            if sort_events:
                X, y = self.sort_events(X, y) # arrange events by
                                              # lon/lat, as a
                                              # heuristic to expose
                                              # block structure in the
                                              # kernel matrix

            self.hyperparams = np.array(hyperparams)
            self.sparse_threshold = sparse_threshold
            self.X = X
            self.n = X.shape[0]
            self.basisfns = basisfns
            mu, self.y, H = self.setup_mean("parametric", X, y)

            # train model
            #t0 = time.time()
            K = self.build_kernel_matrix(self.X, hyperparams)
            #t1 = time.time()
            self.alpha, L, Kinv = self.invert_kernel_matrix(K)
            Kinv_tri =  2 * np.tril(Kinv, k=0) - np.diag(np.diag(Kinv))
            #t2 = time.time()
            self.Kinv_sp = self.sparsify(Kinv)
            self.Kinv_sp_tri = self.sparsify(Kinv_tri)
            #t3 = time.time()
            self.c,self.beta_bar, self.invc, self.HKinv = self.build_parametric_model(self.alpha,
                                                                                      self.Kinv_sp, H,
                                                                                      b=param_mean,
                                                                                      B=param_cov)
            #t4 = time.time()


            r = self.y - np.dot(H.T, self.beta_bar)
            self.alpha_r = scipy.linalg.cho_solve((L, True), r)
            #t5 = time.time()

            self.build_point_tree(HKinv = self.HKinv, Kinv = Kinv_tri, Kinv_sp=self.Kinv_sp_tri, alpha_r = self.alpha_r)
            #t6 = time.time()

            # precompute training set log likelihood, so we don't need
            # to keep L around.
            z = np.dot(H.T, param_mean) - self.y
            B = param_cov
            if compute_ll:
                self._compute_marginal_likelihood(L=L, z=z, B=B, H=H, K=K, Kinv_sp=self.Kinv_sp_tri)
            else:
                self.ll = -np.inf
            #t7 = time.time()
            if compute_grad:
                self.ll_grad = self._log_likelihood_gradient(z=z, K=K, H=H, B=B, Kinv=Kinv)

            np.save('spatialK.npy', K)
            np.save('spatialKinv.npy', Kinv)

            #t8 = time.time()
            """
            print t1-t0
            print t2-t1
            print t3-t2
            print t4-t3
            print t5-t4
            print t6-t5
            print t7-t6
            print t8-t7
            """

    def build_point_tree(self, HKinv, Kinv, Kinv_sp, alpha_r):
        self.predict_tree = VectorTree(self.X, 1, "lld", self.dfn_params)
        self.predict_tree.set_v(0, alpha_r.astype(np.float))


        d = len(self.basisfns)
        self.cov_tree = VectorTree(self.X, d, "lld", self.dfn_params)
        HKinv = HKinv.astype(np.float)
        for i in range(d):
            self.cov_tree.set_v(i, HKinv[i, :])


        nzr, nzc = Kinv_sp.nonzero()
        self.double_tree = MatrixTree(self.X, nzr, nzc, "lld", self.dfn_params)
        kkk = np.matrix(Kinv, copy=True, dtype=np.float64)
        self.double_tree.set_m(kkk)

    def predict(self, cond, eps=1e-8):
        X1 = self.standardize_input_array(cond).astype(np.float)

        gp_pred = np.array([self.predict_tree.weighted_sum(0, np.reshape(x, (1,-1)), eps, "se", self.wfn_params) for x in X1])

        H = self.get_data_features(X1)
        mean_pred = np.reshape(np.dot(H.T, self.beta_bar), gp_pred.shape)
        gp_pred += mean_pred

        if len(gp_pred) == 1:
            gp_pred = gp_pred[0]

        return gp_pred

    def kernel(self, X1, X2, identical=False):
        K = self.predict_tree.kernel_matrix(X1, X2, "se", self.wfn_params, False)
        if identical:
            K += self.noise_var * np.eye(K.shape[0])
        return K

    def covariance(self, cond, include_obs=False, parametric_only=False, pad=1e-8):
        """
        Compute the posterior covariance matrix at a set of points given by the rows of X1.

        Default is to compute the covariance of f, the latent function values. If obs_covar
        is True, we instead compute the covariance of y, the observed values.

        By default, we add a tiny bit of padding to the diagonal to counteract any potential
        loss of positive definiteness from numerical issues. Setting pad=0 disables this.

        """
        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]

        Kstar = self.get_query_K(X1)
        if not parametric_only:
            tmp = self.Kinv_sp_tri * Kstar
            qf = np.dot(Kstar.T, tmp)
            k = self.kernel(X1,X1, identical=include_obs)
            gp_cov = k - qf
        else:
            gp_cov = np.zeros((m,m))

        R = self.query_R
        tmp = np.dot(self.invc, R)
        mean_cov = np.dot(tmp.T, tmp)
        gp_cov += mean_cov

        gp_cov += pad * np.eye(gp_cov.shape[0])
        return gp_cov

    def covariance_double_tree(self, cond, include_obs=False, parametric_only=False, pad=1e-8, eps=1e-8):
        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]
        d = len(self.basisfns)

        t0 = time.time()
        if not parametric_only:
            k = self.kernel(X1, X1, identical=include_obs)
            qf = self.double_tree.quadratic_form(X1, eps, "se", self.wfn_params)
            gp_cov = k - qf
        else:
            gp_cov = np.zeros((m,m))
        t1 = time.time()

        H = np.array([[f(x) for x in X1] for f in self.basisfns], dtype=np.float64)
        HKinvKstar = np.zeros((d, m))

        for i in range(d):
            for j in range(m):
                HKinvKstar[i,j] = self.cov_tree.weighted_sum(i, X1[j:j+1,:], eps, "se", self.wfn_params)
        R = H - HKinvKstar
        v = np.dot(self.invc, R)
        mc = np.dot(v.T, v)
        gp_cov += mc
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

    def pack_npz(self):
        d = dict()
        d['c'] = self.c
        d['beta_bar'] = self.beta_bar
        d['invc'] = self.invc
        d['HKinv'] = self.HKinv
        d['basisfns'] = np.array([marshal_fn(f) for f in self.basisfns], dtype=object)
        d['X']  = self.X,
        d['y'] =self.y,
        d['alpha'] =self.alpha,
        d['hyperparams'] = self.hyperparams
        d['Kinv_sp_tri'] =self.Kinv_sp_tri,
        #d['Kinv_sp'] =self.Kinv_sp,
        d['sparse_threshold'] =self.sparse_threshold,
        d['ll'] =self.ll,
        d['alpha_r'] = self.alpha_r

        return d

    def save_trained_model(self, filename):
        """
        Serialize the model to a file.
        """
        d = self.pack_npz()
        with open(filename, 'wb') as f:
            np.savez(f, base_str=super(SpatialGP, self).__repr_base_params__(), **d)


    def unpack_npz(self, npzfile):
        self.X = npzfile['X'][0]
        self.y = npzfile['y'][0]
        self.alpha = npzfile['alpha'][0]
        self.hyperparams = npzfile['hyperparams']
        self.init_hyperparams(self.hyperparams)
        self.Kinv_sp_tri = npzfile['Kinv_sp_tri'][0]
        #self.Kinv_sp = npzfile['Kinv_sp'][0]
        self.sparse_threshold = npzfile['sparse_threshold'][0]
        self.ll = npzfile['ll'][0]
        self.basisfns = npzfile['basisfns']
        self.basisfns = [unmarshal_fn(code) for code in self.basisfns]
        self.beta_bar = npzfile['beta_bar']
        self.c = npzfile['c']
        self.invc = npzfile['invc']
        self.HKinv = npzfile['HKinv']
        self.alpha_r = npzfile['alpha_r']

    def load_trained_model(self, filename):
        npzfile = np.load(filename)
        self.unpack_npz(npzfile)
        super(SpatialGP, self).__unrepr_base_params__(str(npzfile['base_str']))
        del npzfile.f
        npzfile.close()
        self.n = self.X.shape[0]
        self.build_point_tree(HKinv=self.HKinv, Kinv=self.Kinv_sp_tri.todense(), Kinv_sp=self.Kinv_sp_tri, alpha_r=self.alpha_r)

    def _compute_marginal_likelihood(self, L, z, B, H, K, Kinv_sp):

        # here we follow eqn 2.43 in R&W
        #
        # let z = H.T*b - y, then we want
        # .5 * z.T * (K + H.T * B * H)^-1 * z
        # minus some other stuff (dealt with below).
        # by the matrix inv lemma,
        # (K + H.T * B * H)^-1
        # = Kinv - Kinv*H.T*(Binv + H*Kinv*H.T)^-1*H*Kinv
        # = Kinv - Kinv*H.T*     invc.T * invc    *H*Kinv
        # = Kinv - (invc * HKinv)^T (invc * HKinv)
        #
        # so we have z.T * Kinv * z - z.T * (other thing) * z
        # i.e.:            term1    -     term2
        # in the notation of the code.

        tmp1 = Kinv_sp * z
        term1 = np.dot(z.T, tmp1)

        tmp2 = np.dot(self.HKinv, z)
        tmp3 = np.dot(self.invc, tmp2)
        term2 = np.dot(tmp3.T, tmp3)

        # following eqn 2.43 in R&W, we want to compute
        # log det(K + H.T * B * H). using the matrix inversion
        # lemma, we instead compute
        # log det(K) + log det(B) + log det(B^-1 + H*K^-1*H.T)

        # to compute log(det(K)), we use the trick that the
        # determinant of a symmetric pos. def. matrix is the
        # product of squares of the diagonal elements of the
        # Cholesky factor

        ld2_K = np.log(np.diag(L)).sum()
        ld2 =  np.log(np.diag(self.c)).sum() # det( B^-1 - H * K^-1 * H.T )
        ld_B = np.log(np.linalg.det(B))

        # eqn 2.43 in R&W, using the matrix inv lemma
        self.ll = -.5 * (term1 - term2 + self.n * np.log(2*np.pi) + ld_B) - ld2_K - ld2

    def _log_likelihood_gradient(self, z, K, H, B, Kinv):
        """
        Gradient of the training set log likelihood with respect to the kernel hyperparams.
        """

        nparams = 4
        grad = np.zeros((nparams,))

        #t0 = time.time()
        tmp = np.dot(self.invc, self.HKinv)
        #t1 = time.time()
        K_HBH_inv = Kinv - np.dot(tmp.T, tmp)
        #t2 = time.time()
        alpha_z = np.dot(K_HBH_inv, z)
        #t3 = time.time()

        #print "gradient: %f %f %f" % (t1-t0, t2-t1, t3-t2)

        for i in range(nparams):
            tA = time.time()
            if (i == 0):
                dKdi = np.eye(self.n)
            else:
                dKdi = self.predict_tree.kernel_deriv_wrt_i(self.X, self.X, "se", self.wfn_params, i-1)

            dlldi = .5 * np.dot(alpha_z.T, np.dot(dKdi, alpha_z))
            tB = time.time()
            # here we use the fact:
            # trace(AB) = sum_{ij} A_ij * B_ij
            dlldi -= .5 * np.sum(np.sum(K_HBH_inv.T * dKdi))

            grad[i] = dlldi
            tC = time.time()
            print "  %d: %f %f" % (i, tB-tA, tC-tB)

        return grad


def spatialgp_nll_ngrad(**kwargs):
    """
    Get both the negative log-likelihood and its gradient
    simultaneously (more efficient than doing it separately since we
    only create one new GP object, which only constructs the kernel
    matrix once, etc.).
    """

    try:
#        print "optimizing params", kernel_params
        gp = SpatialGP(compute_ll=True, compute_grad=True, **kwargs)

        nll = -1 * gp.ll
        ngrad = -1 * gp.ll_grad

    except np.linalg.linalg.LinAlgError as e:
        print "warning: lin alg error (%s) in likelihood computation, returning likelihood -inf" % str(e)
        nll = np.float("inf")
        ngrad = None
    except ValueError as e:
        print "warning: value error (%s) in likelihood computation, returning likelihood -inf" % str(e)
        nll = np.float("inf")
        ngrad = None

    return nll, ngrad
