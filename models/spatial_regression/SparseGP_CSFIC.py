import os
import time
import numpy as np
import collections
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scikits.sparse.cholmod
import sklearn.preprocessing
import pyublas
import hashlib
import types
import marshal


from sigvisa.models.spatial_regression.baseline_models import ParamModel
from sigvisa.models.distributions import InvGamma, LogNormal
from sigvisa.utils.cover_tree import VectorTree, MatrixTree

from sigvisa.models.spatial_regression.SparseGP import *


class SparseGP_CSFIC(ParamModel):

    def build_kernel_matrix(self, X):
        K = self.kernel(X, X, identical=True)
        return K + np.eye(K.shape[0], dtype=np.float64) * 1e-8 # try to avoid losing
                                       # positive-definiteness
                                       # to numeric issues

    def sparse_build_kernel_matrix(self, X, **kwargs):
        K = self.sparse_kernel(X, identical=True)
        K = K + scipy.sparse.eye(K.shape[0], dtype=np.float64) * 1e-8 # try to avoid losing
                                       # positive-definiteness
                                       # to numeric issues
        return K.tocsc()

    def invert_kernel_matrix(self, K):
        alpha = None
        t0 = time.time()
        L = scipy.linalg.cholesky(K, lower=True) # c = sqrt(inv(B) + H*K^-1*H.T)
        factor = lambda z : scipy.linalg.cho_solve((L, True), z)
        t1 = time.time()
        self.timings['chol_factor'] = t1-t0

        alpha = factor(self.y)
        t2 = time.time()
        self.timings['solve_alpha'] = t2-t1

        Kinv = np.linalg.inv(K)
        t3 = time.time()
        self.timings['solve_Kinv'] = t3-t2

        I = np.dot(Kinv[0,:], K[:,0])
        if np.abs(I - 1) > 0.01:
            print "WARNING: poorly conditioned inverse (I=%f)" % I

        return alpha, factor, L, Kinv

    def sparse_invert_kernel_matrix(self, K):
        alpha = None
        t0 = time.time()
        factor = scikits.sparse.cholmod.cholesky(K)
        t1 = time.time()
        self.timings['chol_factor'] = t1-t0

        #unpermuted_L = factor.L()
        #P = factor.P()
        #Pinv = np.argsort(P)
        #L = unpermuted_L[Pinv,:]
        alpha = factor(self.y)
        t2 = time.time()
        self.timings['solve_alpha'] = t2-t1
        Kinv = factor(scipy.sparse.eye(K.shape[0]).tocsc())
        t3 = time.time()
        self.timings['solve_Kinv'] = t3-t2

        I = (Kinv.getrow(0) * K.getcol(0)).todense()[0,0]
        if np.abs(I - 1) > 0.01:
            print "WARNING: poorly conditioned inverse (I=%f)" % I

        return alpha, factor, factor.L(), Kinv

    def build_FITC_model(self, alpha, Kinv_sp, K_un, Kuu):
        # notation follows section 2.7 of Rasmussen and Williams
        Kuu_inv = scipy.linalg.inv(Kuu)
        tmp = np.reshape(np.asarray(np.dot(K_un, alpha)), (-1,))

        KunKinv = K_un * Kinv_sp
        M_inv  = Kuu + np.dot(KunKinv, K_un.T)  # here M = (inv(B) +
                                           # H*K^-1*H.T)^-1 is the
                                           # posterior covariance
                                           # matrix on the params.

        c = scipy.linalg.cholesky(M_inv, lower=True) # c = sqrt(inv(B) + H*K^-1*H.T)
        beta_bar = scipy.linalg.cho_solve((c, True), tmp)
        invc = scipy.linalg.inv(c)


        return c, beta_bar, invc, KunKinv

    def get_data_features(self, X):
        return self.kernel(self.Xu, X, predict_tree=self.predict_tree_fic)

    def sparsify(self, M):
        import scipy.sparse
        if scipy.sparse.issparse(M):
            M = M.copy()
            chunksize=1000000
            nchunks = len(M.data)/chunksize+1
            for i in range(nchunks):
                cond = (np.abs(M.data[i*chunksize:(i+1)*chunksize]) < self.sparse_threshold)
                M.data[i*chunksize:(i+1)*chunksize][cond] = 0
            M.eliminate_zeros()
            return M
        else:
            return scipy.sparse.csc_matrix(np.asarray(M) * (np.abs(M) > self.sparse_threshold))

    def sort_events(self, X, y):
        combined = np.hstack([X, np.reshape(y, (-1, 1))])
        combined_sorted = np.array(sorted(combined, key = lambda x: x[0]), dtype=float)
        X_sorted = np.array(combined_sorted[:, :-1], copy=True, dtype=float)
        y_sorted = combined_sorted[:, -1].flatten()
        return X_sorted, y_sorted

    def __init__(self, X=None, y=None,
                 fname=None,
                 sparse_threshold=1e-8,
                 noise_var = 1.0,
                 dfn_str = "lld",
                 dfn_params_fic = None,
                 dfn_params_cs = None,
                 wfn_str_fic = "se",
                 wfn_params_fic = None,
                 wfn_str_cs = "compact2",
                 wfn_params_cs = None,
                 Xu = None,
                 build_tree=True,
                 sort_events=True,
                 center_mean=False,
                 leaf_bin_size = 0):

        self.double_tree = None
        if fname is not None:
            self.load_trained_model(fname, build_tree=build_tree, leaf_bin_size=leaf_bin_size)
        else:
            if sort_events:
                X, y = self.sort_events(X, y) # arrange events by
                                              # lon/lat, as a
                                              # heuristic to expose
                                              # block structure in the
                                              # kernel matrix

            self.dfn_str = dfn_str
            self.wfn_str_fic, self.wfn_str_cs = wfn_str_fic, wfn_str_cs
            self.Xu = Xu
            self.noise_var, self.dfn_params_fic, self.dfn_params_cs, self.wfn_params_fic, self.wfn_params_cs = noise_var, dfn_params_fic, dfn_params_cs, wfn_params_fic, wfn_params_cs
            self.sparse_threshold = sparse_threshold

            self.timings = dict()

            if X is not None:
                self.X = np.matrix(X, dtype=float)
                self.y = np.array(y, dtype=float)
                if center_mean:
                    self.ymean = np.mean(y)
                    self.y -= self.ymean
                else:
                    self.ymean = 0.0
                self.n = X.shape[0]
            else:
                self.X = np.reshape(np.array(()), (0,0))
                self.y = np.reshape(np.array(()), (0,))
                self.n = 0
                self.ymean = 0.0
                self.K = np.reshape(np.array(()), (0,0))
                self.Kinv = np.reshape(np.array(()), (0,0))
                self.alpha_r = self.y
                self.ll = np.float('-inf')
                return

            # train model
            t0 = time.time()
            if build_tree:
                tree_X = pyublas.why_not(self.X)
            else:
                tree_X = np.array([[0.0,] * X.shape[1],], dtype=float)

            self.predict_tree_fic = VectorTree(tree_X, 1, self.dfn_str, pyublas.why_not(self.dfn_params_fic), self.wfn_str_fic, pyublas.why_not(self.wfn_params_fic))
            self.predict_tree_cs = VectorTree(self.X, 1, self.dfn_str, pyublas.why_not(self.dfn_params_cs), self.wfn_str_cs, pyublas.why_not(self.wfn_params_cs)) # it's CRUCIAL that this is self.X and not just X. otherwise it goes out of scope and causes horrific heisenbugs


            self.predict_tree = self.predict_tree_cs
            t1 = time.time()
            self.timings['build_predict_tree'] = t1-t0

            # sparse kernel matrix
            K_cs = self.sparse_kernel(X, identical=True, predict_tree=self.predict_tree_cs, max_distance=1.0)

            K_fic_uu = self.kernel(Xu, Xu, identical=False, predict_tree = self.predict_tree_fic)
            Luu  = scipy.linalg.cholesky(K_fic_uu, lower=True)
            self.Luu = Luu
            K_fic_un = self.kernel(Xu, X, identical=False, predict_tree = self.predict_tree_fic)
            self.basisfns = [None,] * Xu.shape[0]

            dc = self.covariance_diag_correction(X)
            diag_correction = scipy.sparse.dia_matrix((dc, 0), shape=K_cs.shape)

            #invuu = np.linalg.inv(K_fic_uu)
            #reinv = np.dot(K_fic_uu, invuu)
            #print "reinv error", np.max( reinv - np.diag(np.diag(reinv)) )
            #Qnn = np.dot(K_fic_un.T, np.dot(invuu, K_fic_un))
            #self.Qnn = Qnn
            #self.K_fic_uu = K_fic_uu
            #self.invuu = invuu
            #self.K_fic_un = K_fic_un

            # you'd think cholesky would be better, but in practice it seems less stable
            #chol_uu = scipy.linalg.cholesky(K_fic_uu, lower=True)
            #Qnn_sqrt = scipy.linalg.cho_solve((chol_uu, True), K_fic_un)
            #Qnn = np.dot(Qnn_sqrt.T, Qnn_sqrt)

            K_cs = K_cs + diag_correction
            self.K = K_cs
            alpha, factor, L, Kinv = self.sparse_invert_kernel_matrix(K_cs)
            self.factor = factor
            self.Kinv = self.sparsify(Kinv)
            print "Kinv is ", len(self.Kinv.nonzero()[0]) / float(self.Kinv.shape[0]**2), "full (vs diag at", 1.0/self.Kinv.shape[0], ")"

            self.c,self.beta_bar, self.invc, self.HKinv = self.build_FITC_model(alpha,
                                                                                self.Kinv,
                                                                                K_fic_un,
                                                                                K_fic_uu)
            r = self.y - np.dot(K_fic_un.T, self.beta_bar)
            z = - self.y
            #B = param_cov

            self.alpha_r = np.reshape(np.asarray(factor(r)), (-1,))

            if build_tree:
                self.build_point_tree(HKinv = self.HKinv, Kinv=self.Kinv, alpha_r = self.alpha_r, leaf_bin_size=leaf_bin_size)
            #t6 = time.time()


    def build_point_tree(self, HKinv, Kinv, alpha_r, leaf_bin_size):
        if self.n == 0: return

        fullness = len(self.Kinv.nonzero()[0]) / float(self.Kinv.shape[0]**2)
        print "Kinv is %.1f%% full (%d nonzero elements)." % (fullness * 100, len(self.Kinv.nonzero()[0]))
        if fullness > .15:
            raise Exception("not building tree, Kinv is too full!" )


        self.predict_tree.set_v(0, alpha_r.astype(np.float))

        d = len(self.basisfns)
        if d > 0:
            self.cov_tree = VectorTree(self.X, d, self.dfn_str, self.dfn_params_cs, self.wfn_str_cs, self.wfn_params_cs)
            HKinv = HKinv.astype(np.float)
            for i in range(d):
                self.cov_tree.set_v(i, HKinv[i, :])


        nzr, nzc = Kinv.nonzero()
        vals = np.reshape(np.asarray(Kinv[nzr, nzc]), (-1,))
        self.double_tree = MatrixTree(self.X, nzr, nzc, self.dfn_str, self.dfn_params_cs, self.wfn_str_cs, self.wfn_params_cs)
        self.double_tree.set_m_sparse(nzr, nzc, vals)
        print "built tree"
        if leaf_bin_size > 2:
            self.double_tree.collapse_leaf_bins(leaf_bin_size)
            print "collapsed bins"

    def predict(self, cond, parametric_only=False, eps=1e-8):
        if not self.double_tree: return self.predict_naive(cond, parametric_only)
        X1 = self.standardize_input_array(cond).astype(np.float)

        if parametric_only:
            gp_pred = np.zeros((X1.shape[0],))
        else:
            gp_pred = np.array([self.predict_tree.weighted_sum(0, np.reshape(x, (1,-1)), eps) for x in X1])

        if len(self.basisfns) > 0:
            H = self.get_data_features(X1)
            mean_pred = np.reshape(np.dot(H.T, self.beta_bar), gp_pred.shape)
            gp_pred += mean_pred

        if len(gp_pred) == 1:
            gp_pred = gp_pred[0]

        gp_pred += self.ymean
        return gp_pred

    def predict_naive(self, cond, parametric_only=False, eps=1e-8):
        X1 = self.standardize_input_array(cond).astype(np.float)

        if parametric_only:
            gp_pred = np.zeros((X1.shape[0],))
        else:
            Kstar = self.kernel(self.X, X1)
            gp_pred = np.dot(Kstar.T, self.alpha_r)

        if len(self.basisfns) > 0:
            H = self.get_data_features(X1)
            mean_pred = np.reshape(np.dot(H.T, self.beta_bar), gp_pred.shape)
            gp_pred += mean_pred

        if len(gp_pred) == 1:
            gp_pred = gp_pred[0]

        gp_pred += self.ymean
        return gp_pred


    def kernel(self, X1, X2, identical=False, predict_tree=None):
        predict_tree = self.predict_tree if predict_tree is None else predict_tree
        K = predict_tree.kernel_matrix(X1, X2, False)
        if identical:
            K += self.noise_var * np.eye(K.shape[0])
        return K

    def sparse_kernel(self, X, identical=False, predict_tree=None, max_distance=1.0):
        predict_tree = self.predict_tree if predict_tree is None else predict_tree

        if max_distance is None:
            if self.sparse_threshold ==0:
                max_distance = 1e300
            else:
                max_distance = np.sqrt(-np.log(self.sparse_threshold)) # assuming a SE kernel

        entries = predict_tree.sparse_training_kernel_matrix(X, max_distance)
        spK = scipy.sparse.coo_matrix((entries[:,2], (entries[:,1], entries[:,0])), shape=(self.n, len(X)), dtype=float)
        if identical:
            spK = spK + self.noise_var * scipy.sparse.eye(spK.shape[0])
        return spK


    def get_query_K_sparse(self, X1):
        # avoid recomputing the kernel if we're evaluating at the same
        # point multiple times. This is effectively a size-1 cache.
        try:
            self.querysp_hsh
        except AttributeError:
            self.querysp_hsh = None

        hsh = hashlib.sha1(X1.view(np.uint8)).hexdigest()
        if hsh != self.querysp_hsh:
            self.querysp_K = self.sparse_kernel(X1)
            self.querysp_hsh = hsh

            if self.basisfns:
                H = self.get_data_features(X1)
                self.querysp_R = H - np.asmatrix(self.HKinv) * self.querysp_K

#            print "cache fail: model %d called with " % (len(self.alpha)), X1
#        else:
#            print "cache hit!"
        return self.querysp_K

    def get_query_K(self, X1):
        # avoid recomputing the kernel if we're evaluating at the same
        # point multiple times. This is effectively a size-1 cache.
        try:
            self.query_hsh
        except AttributeError:
            self.query_hsh = None

        hsh = hashlib.sha1(X1.view(np.uint8)).hexdigest()
        if hsh != self.query_hsh:
            self.query_K = self.kernel(self.X, X1)
            self.query_hsh = hsh

            if self.basisfns:
                H = self.get_data_features(X1)
                self.query_R = H - np.asmatrix(self.HKinv) * self.query_K

#            print "cache fail: model %d called with " % (len(self.alpha)), X1
#        else:
#            print "cache hit!"
        return self.query_K

    def covariance_diag_correction(self, X):
        K_fic_un = self.kernel(self.Xu, X, identical=False, predict_tree = self.predict_tree_fic)
        B = scipy.linalg.solve(self.Luu, K_fic_un)
        Qvff = np.sum(B*B, axis=0)
        return self.wfn_params_fic[0] - Qvff

    def covariance_spkernel(self, cond, include_obs=False, parametric_only=False, pad=1e-8):
        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]

        Kstar = self.get_query_K_sparse(X1)

        #Kstar2 = self.get_query_K(X1)
        #if np.sum(np.abs(Kstar.todense() - Kstar2)) > .00001:
        #    import pdb; pdb.set_trace()

        if not parametric_only:
            gp_cov = self.kernel(X1,X1, identical=include_obs)
            if self.n > 0:
                tmp = self.Kinv * Kstar
                qf = (Kstar.T * tmp).todense()
                gp_cov -= qf
        else:
            gp_cov = np.zeros((m,m))

        if len(self.basisfns) > 0:
            R = self.querysp_R
            tmp = np.dot(self.invc, R)
            mean_cov = np.dot(tmp.T, tmp)
            gp_cov += mean_cov

        gp_cov += pad * np.eye(gp_cov.shape[0])
        gp_cov += self.covariance_diag_correction(X1)

        return gp_cov

    def covariance_spkernel_solve(self, cond, include_obs=False, parametric_only=False, pad=1e-8):
        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]

        Kstar = self.get_query_K_sparse(X1)
        if not parametric_only:
            gp_cov = self.kernel(X1,X1, identical=include_obs)
            if self.n > 0:

                f = self.factor(Kstar)
                qf = (Kstar.T * f).todense()

                """
                # alternate form using solve_L
                P = factor.P()
                kp = kstar[P]
                flp = factor.solve_L(kp)
                newdata = flp.data / factor.D()[flp.nonzero()[0]]
                flp2 = flp.copy()
                flp.data = newdata
                qf = (flp2.T * flp).todense()
                """

                gp_cov -= qf
        else:
            gp_cov = np.zeros((m,m))

        if len(self.basisfns) > 0:
            R = self.querysp_R
            tmp = np.dot(self.invc, R)
            mean_cov = np.dot(tmp.T, tmp)
            gp_cov += mean_cov

        gp_cov += pad * np.eye(gp_cov.shape[0])
        gp_cov += self.covariance_diag_correction(X1)

        return gp_cov

    def covariance(self, cond, include_obs=False, parametric_only=False, pad=1e-8, qf_only=False):
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
            gp_cov = self.kernel(X1,X1, identical=include_obs)
            if self.n > 0:
                tmp = self.Kinv * Kstar
                qf = np.dot(Kstar.T, tmp)
                if qf_only:
                    return qf
                gp_cov -= qf
        else:
            gp_cov = np.zeros((m,m))

        if len(self.basisfns) > 0:
            R = self.query_R
            tmp = np.dot(self.invc, R)
            mean_cov = np.dot(tmp.T, tmp)
            gp_cov += mean_cov

        gp_cov += pad * np.eye(gp_cov.shape[0])
        gp_cov += self.covariance_diag_correction(X1)

        return gp_cov

    def covariance_double_tree(self, cond, include_obs=False, parametric_only=False, pad=1e-8, eps=1e-8, eps_abs=1e-4, cutoff_rule=1, qf_only=False):
        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]
        d = len(self.basisfns)
        cutoff_rule = int(cutoff_rule)

        if not parametric_only:
            gp_cov = self.kernel(X1, X1, identical=include_obs)
            if self.n > 0:
                qf = np.zeros(gp_cov.shape)
                for i in range(m):
                    for j in range(m):
                        qf[i,j] = self.double_tree.quadratic_form(X1[i:i+1], X1[j:j+1], eps, eps_abs, cutoff_rule)
                if qf_only:
                    return qf
                gp_cov -= qf

        else:
            gp_cov = np.zeros((m,m))

        if len(self.basisfns) > 0:
            H = self.kernel(self.Xu, X1, predict_tree=self.predict_tree_fic)
            #H = np.array([[f(x) for x in X1] for f in self.basisfns], dtype=np.float64)
            HKinvKstar = np.zeros((d, m))

            for i in range(d):
                for j in range(m):
                    HKinvKstar[i,j] = self.cov_tree.weighted_sum(i, X1[j:j+1,:], eps)
            R = H - HKinvKstar
            v = np.dot(self.invc, R)
            mc = np.dot(v.T, v)
            gp_cov += mc


        gp_cov += pad * np.eye(m)
        gp_cov += self.covariance_diag_correction(X1)

        return gp_cov

    def variance(self,cond, **kwargs):
        return np.diag(self.covariance(cond, **kwargs))

    def sample(self, cond, include_obs=True):
        """
        Sample from the GP posterior at a set of points given by the rows of X1.

        Default is to sample observed values (i.e. we include observation noise). If obs=False, we instead
        sample values of the latent function f.
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

    def deriv_log_p(self, x, cond=None,lp0=None, eps=1e-4, include_obs=True, **kwargs):

        X1 = self.standardize_input_array(cond, **kwargs)
        y = x if isinstance(x, collections.Iterable) else (x,)

        y = np.array(y);
        if len(y.shape) == 0:
            n = 1
        else:
            n = len(y)

        K = self.covariance(X1, include_obs=True)
        y = y-self.predict(X1)


        L =  scipy.linalg.cholesky(K, lower=True)
        return -scipy.linalg.cho_solve((L, True), y)

        # X1: kx6 array w/ station and event locations
        # y: k-dimensional vector
        # ignore idx, cond_key, cond_idx


        # return k-dimensional vector
        # d log_p() / dy

    def log_p(self, x, cond, include_obs=True, covar="spkernel", covar_args={}, **kwargs):
        """
        The log probability of the observations (X1, y) under the posterior distribution.
        """

        X1 = self.standardize_input_array(cond, **kwargs)
        y = x if isinstance(x, collections.Iterable) else (x,)

        y = np.array(y)
        if len(y.shape) == 0:
            n = 1
        else:
            n = len(y)

        if covar=="spkernel":
            K = self.covariance_spkernel(X1, include_obs=include_obs, **covar_args)
        elif covar=="naive":
            K = self.covariance(X1, include_obs=include_obs,**covar_args)
        elif covar=="double_tree":
            K = self.covariance_double_tree(X1, include_obs=include_obs, **covar_args)
        else:
            raise Exception("unrecognized covar method %s" % covar)

        y = y-self.predict(X1)

        if n==1:
            var = K[0,0]
            ll1 = - .5 * ((y)**2 / var + np.log(2*np.pi*var) )
            return ll1

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
        d['Xu'] = self.Xu
        d['X']  = self.X,
        d['y'] =self.y,
        d['ymean'] = self.ymean,
        d['alpha_r'] =self.alpha_r,
        d['Luu'] =self.Luu,
        d['Kinv'] =self.Kinv,
        d['K'] =self.K,
        d['sparse_threshold'] =self.sparse_threshold,
        d['alpha_r'] = self.alpha_r
        d['noise_var'] = self.noise_var
        d['dfn_str'] = self.dfn_str
        d['dfn_params_cs'] = self.dfn_params_cs
        d['dfn_params_fic'] = self.dfn_params_fic
        d['wfn_str_cs'] = self.wfn_str_cs
        d['wfn_params_cs'] = self.wfn_params_cs
        d['wfn_str_fic'] = self.wfn_str_fic
        d['wfn_params_fic'] = self.wfn_params_fic

        return d

    def save_trained_model(self, filename):
        """
        Serialize the model to a file.
        """
        d = self.pack_npz()
        with open(filename, 'wb') as f:
            if 'site_lon' in self.__dict__:
                base_str = super(SparseGP, self).__repr_base_params__()
            else:
                base_str = ""
            np.savez(f, base_str=base_str, **d)


    def unpack_npz(self, npzfile):
        self.X = npzfile['X'][0]
        self.y = npzfile['y'][0]
        if 'ymean' in npzfile:
            self.ymean = npzfile['ymean'][0]
        else:
            self.ymean = 0.0
        self.dfn_str  = npzfile['dfn_str'].item()
        self.wfn_str_cs  = npzfile['wfn_str_cs'].item()
        self.wfn_str_fic  = npzfile['wfn_str_fic'].item()
        self.noise_var = npzfile['noise_var'].item()
        self.dfn_params_cs = npzfile['dfn_params_cs']
        self.dfn_params_fic = npzfile['dfn_params_fic']
        self.wfn_params_cs = npzfile['wfn_params_cs']
        self.wfn_params_fic = npzfile['wfn_params_fic']

        self.Kinv = npzfile['Kinv'][0]
        self.K = npzfile['K'][0]
        self.sparse_threshold = npzfile['sparse_threshold'][0]
        self.Xu = npzfile['Xu']
        self.beta_bar = npzfile['beta_bar']
        self.c = npzfile['c']
        self.invc = npzfile['invc']
        self.HKinv = npzfile['HKinv']
        self.alpha_r = npzfile['alpha_r']
        self.Luu = npzfile['Luu'][0]

    def load_trained_model(self, filename, build_tree=True, cache_dense=False, leaf_bin_size=0):
        npzfile = np.load(filename)
        self.unpack_npz(npzfile)
        if len(str(npzfile['base_str'])) > 0:
            super(SparseGP, self).__unrepr_base_params__(str(npzfile['base_str']))
        del npzfile.f
        npzfile.close()
        self.n = self.X.shape[0]

        self.basisfns = [None,] * self.Xu.shape[0]
        self.dfn_params = self.dfn_params_cs

        self.predict_tree_cs = VectorTree(self.X, 1, self.dfn_str, self.dfn_params_cs, self.wfn_str_cs, self.wfn_params_cs)
        self.predict_tree_fic = VectorTree(self.X, 1, self.dfn_str, self.dfn_params_fic, self.wfn_str_fic, self.wfn_params_fic)
        self.predict_tree = self.predict_tree_cs

        if build_tree:
            self.factor = scikits.sparse.cholmod.cholesky(self.K)
            self.build_point_tree(HKinv = self.HKinv, Kinv=self.Kinv, alpha_r = self.alpha_r, leaf_bin_size=leaf_bin_size)
        if cache_dense and self.n > 0:
            self.Kinv_dense = self.Kinv.todense()


    def log_likelihood(self):
        return self.ll
