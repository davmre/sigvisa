import os
import time
import numpy as np
import collections
import scipy
import scipy.sparse
import scikits.sparse.cholmod
import pyublas
import hashlib
import types
import marshal


from sigvisa.models.spatial_regression.baseline_models import ParamModel

from sigvisa.utils.cover_tree import VectorTree, MatrixTree


start_params_lld = {"coda_decay": [.022, .0187, 50.00, 1.0],
                    "amp_transfer": [1.1, 3.4, 100.00, 1.0],
                    "peak_offset": [2.7, 3.4, 50.00, 1.0]
                    }

start_params = {"lld": start_params_lld,
                }


X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI = range(5)


def marshal_fn(f):
    if f.func_closure is not None:
        raise ValueError("function has non-empty closure %s, cannot marshal!" % repr(f.func_closure))
    s = marshal.dumps(f.func_code)
    return s

def unmarshal_fn(dumped_code):
    f_code = marshal.loads(dumped_code)
    f = types.FunctionType(f_code, globals())
    return f

X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI = range(5)

def extract_hyperparams(kernel_str, hyperparams):
    if kernel_str=="lld_se":
        (noise_var, signal_var, ll_scale, d_scale) = hyperparams
        noise_var = noise_var
        dfn_params = np.array((ll_scale, d_scale), dtype=np.float)
        wfn_params = np.array((signal_var,), copy=True, dtype=np.float)
    elif kernel_str == "euclidean_se":
        noise_var = hyperparams[0]
        wfn_params = np.array((hyperparams[1],), copy=True, dtype=np.float)
        dfn_params = np.array((hyperparams[2:]), dtype=np.float)

    return noise_var, dfn_params, wfn_params

def prior_sample(X, hyperparams, kernel_str="lld"):
    n = X.shape[0]
    noise_var, dfn_params, wfn_params = extract_hyperparams(kernel_str, hyperparams)
    predict_tree = VectorTree(X, 1, kernel_str, dfn_params)
    spK = predict_tree.sparse_kernel(X, wfn_params) + noise_var * scipy.sparse.eye(n)
    factor = scikits.sparse.cholmod.cholesky(spK)
    L = factor.L()
    P = factor.P()
    Pinv = np.argsort(P)
    z = np.random.randn(n)
    y = (L * z)[Pinv]
    return y

class SparseGP(ParamModel):

    def build_kernel_matrix(self, X, hyperparams):
        vt = self.predict_tree
        K = vt.kernel_matrix(X, X, self.wfn_str, self.wfn_params, False) + self.noise_var * np.eye(len(X), dtype=np.float64)

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

    def get_data_features(self, X):
        H = np.array([[f(x) for x in X] for f in self.basisfns], dtype=float)
        return H

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
                 kernel_str="lld_se", hyperparams=None,
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

            self.dfn_str, self.wfn_str = kernel_str.split('_')
            self.hyperparams = np.array(hyperparams)
            self.noise_var, self.dfn_params, self.wfn_params = extract_hyperparams(kernel_str=kernel_str, hyperparams=hyperparams)
            self.sparse_threshold = sparse_threshold
            self.X = X
            self.y = y
            self.n = X.shape[0]
            self.basisfns = basisfns
            H = self.get_data_features(X)

            # train model
            print "hello"
            t0 = time.time()
            self.predict_tree = VectorTree(self.X, 1, self.dfn_str, self.dfn_params)
            t1 = time.time()
            print "built predict tree in", t1-t0
            K = self.build_kernel_matrix(self.X, hyperparams)
            t2 = time.time()
            print "got kernel matrix in", t2-t1

            self.alpha, L, Kinv = self.invert_kernel_matrix(K)
            Kinv_tri =  2 * np.tril(Kinv, k=0) - np.diag(np.diag(Kinv))
            #t2 = time.time()
            self.Kinv_sp = self.sparsify(Kinv)
            self.Kinv_sp_tri = self.sparsify(Kinv_tri)
            #t3 = time.time()

            if len(self.basisfns) > 0:
                self.c,self.beta_bar, self.invc, self.HKinv = self.build_parametric_model(self.alpha,
                                                                                          self.Kinv_sp,
                                                                                          H,
                                                                                          b=param_mean,
                                                                                          B=param_cov)
                r = self.y - np.dot(H.T, self.beta_bar)
                z = np.dot(H.T, param_mean) - self.y
                B = param_cov
            else:
                self.HKinv = None
                r = self.y
                z = self.y
                B = None

            self.alpha_r = scipy.linalg.cho_solve((L, True), r)

            self.build_point_tree(HKinv = self.HKinv, Kinv = Kinv_tri, Kinv_sp=self.Kinv_sp_tri, alpha_r = self.alpha_r)
            #t6 = time.time()

            # precompute training set log likelihood, so we don't need
            # to keep L around.
            if compute_ll:
                self._compute_marginal_likelihood(L=L, z=z, B=B, H=H, K=K, Kinv_sp=self.Kinv_sp_tri)
            else:
                self.ll = -np.inf
            #t7 = time.time()
            if compute_grad:
                self.ll_grad = self._log_likelihood_gradient(z=z, K=K, H=H, B=B, Kinv=Kinv)

            np.save('spatialK.npy', K)
            np.save('spatialKinv.npy', Kinv)

            print "trained"
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
        self.predict_tree.set_v(0, alpha_r.astype(np.float))


        d = len(self.basisfns)
        if d > 0:
            self.cov_tree = VectorTree(self.X, d, self.dfn_str, self.dfn_params)
            HKinv = HKinv.astype(np.float)
            for i in range(d):
                self.cov_tree.set_v(i, HKinv[i, :])


        nzr, nzc = Kinv_sp.nonzero()
        self.double_tree = MatrixTree(self.X, nzr, nzc, self.dfn_str, self.dfn_params)
        #kkk = np.matrix(Kinv, copy=True, dtype=np.float64)
        #self.double_tree.set_m(kkk)

    def predict(self, cond, parametric_only=False, eps=1e-8):
        X1 = self.standardize_input_array(cond).astype(np.float)

        if parametric_only:
            gp_pred = np.zeros((X1.shape[0],))
        else:
            gp_pred = np.array([self.predict_tree.weighted_sum(0, np.reshape(x, (1,-1)), eps, self.wfn_str, self.wfn_params) for x in X1])

        if len(self.basisfns) > 0:
            H = self.get_data_features(X1)
            mean_pred = np.reshape(np.dot(H.T, self.beta_bar), gp_pred.shape)
            gp_pred += mean_pred

        if len(gp_pred) == 1:
            gp_pred = gp_pred[0]

        return gp_pred

    def kernel(self, X1, X2, identical=False):
        K = self.predict_tree.kernel_matrix(X1, X2, self.wfn_str, self.wfn_params, False)
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
            qf = self.double_tree.quadratic_form(X1, eps, self.wfn_str, self.wfn_params)
            gp_cov = k - qf
        else:
            gp_cov = np.zeros((m,m))
        t1 = time.time()

        H = np.array([[f(x) for x in X1] for f in self.basisfns], dtype=np.float64)
        HKinvKstar = np.zeros((d, m))

        for i in range(d):
            for j in range(m):
                HKinvKstar[i,j] = self.cov_tree.weighted_sum(i, X1[j:j+1,:], eps, self.wfn_str, self.wfn_params)
        R = H - HKinvKstar
        v = np.dot(self.invc, R)
        mc = np.dot(v.T, v)
        gp_cov += mc
        t2 = time.time()

        self.nptime = (t1-t0)
        self.ptime = (t2-t1)

        gp_cov += pad * np.eye(m)
        return gp_cov

    def variance(self, X1, **kwargs):
        return np.diag(self.covariance(X1, **kwargs))

    def sample(self, cond, include_obs=False):
        """
        Sample from the GP posterior at a set of points given by the rows of X1.

        Default is to sample values of the latent function f. If obs=True, we instead
        sample observed values (i.e. we include observation noise)
        """

        X1 = self.standardize_input_array(cond)
        (n,d) = X1.shape
        means = np.reshape(self.predict(X1), (-1, 1))
        K = self.covariance(X1, include_obs=include_obs)
        samples = np.random.randn(n, 1)

        L = scipy.linalg.cholesky(K, lower=True)
        samples = means + np.dot(L, samples)


        if len(samples) == 1:
            samples = samples[0]

        return samples

    def param_predict(self):
        return self.beta_bar

    def param_covariance(self, chol=False):
        if chol:
            return self.invc
        else:
            return np.dot(self.invc.T, self.invc)

    def param_sample(self, n=1):
        samples = np.random.randn(len(self.beta_bar), n)
        samples = np.reshape(self.beta_bar, (1, -1)) + np.dot(self.invc.T, samples).T
        return samples


    def log_p(self, x, cond):
        """
        The log probability of the observations (X1, y) under the posterior distribution.
        """

        X1 = self.standardize_input_array(cond)
        y = x if isinstance(x, collections.Iterable) else (x,)

        y = np.array(y)
        if len(y.shape) == 0:
            n = 1
        else:
            n = len(y)

        K = self.covariance(X1)
        y = y-self.predict(X1)

        if n==1:
            var = K[0,0]
            ll1 = - .5 * ((y)**2 / var + np.log(2*np.pi*var) )

        L = scipy.linalg.cholesky(K, lower=True)
        ld2 = np.log(np.diag(L)).sum() # this computes .5 * log(det(K))
        alpha = scipy.linalg.cho_solve((L, True), y)
        ll =  -.5 * ( np.dot(y.T, alpha) + n * np.log(2*np.pi)) - ld2
        return ll




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
                dKdi = self.predict_tree.kernel_deriv_wrt_i(self.X, self.X, self.wfn_str, self.wfn_params, i-1)

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
