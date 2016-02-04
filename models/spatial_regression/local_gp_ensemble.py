import os
import time
import numpy as np
import collections
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scikits.sparse.cholmod
import sklearn.preprocessing
import hashlib
import types
import marshal
import pyublas

from sigvisa.treegp.gp import GP, GPCov
from sigvisa.treegp.cover_tree import VectorTree
from sigvisa.models.spatial_regression.baseline_models import ParamModel

class LocalGPEnsemble(ParamModel):

    def _build_local_gps(self):
        cluster_distances = self.cluster_tree.kernel_matrix(pyublas.why_not(self.X), 
                                                            pyublas.why_not(self.cluster_centers), True)
        closest_cluster = np.argmin(cluster_distances, axis=1)

        local_gps = []
        for i, ctr in enumerate(self.cluster_centers):
            cov = self.cluster_covs[i]
            noise_var = self.cluster_noise_vars[i]

            cluster_idxs = (closest_cluster == i)
            cluster_X = self.X[cluster_idxs]
            cluster_y = self.y[cluster_idxs]
            cluster_yvars = self.yvars[cluster_idxs]

            lgp = GP(X=cluster_X, y=cluster_y, y_obs_variances=cluster_yvars,
                     cov_main=cov, noise_var=noise_var,
                     compute_ll = (self.basis is None),
                     compute_grad = True)
            local_gps.append(lgp)

        return local_gps
        
    def __init__(self, X, y,  
                 cluster_centers,
                 cluster_covs,
                 cluster_noise_vars,
                 yvars = None,
                 basis=None,
                 extract_dim = None,
                 param_mean=None,
                 param_cov=None,
                 featurizer_recovery=None,
                 **kwargs):

        ParamModel.__init__(self, **kwargs)

        self.X = X
        self.y = y
        if yvars is None:
            yvars = np.zeros(y.shape)
        self.yvars = yvars

        self.cluster_centers = cluster_centers
        self.cluster_covs = cluster_covs
        self.cluster_noise_vars = cluster_noise_vars

        cluster_metric = GPCov(wfn_str="se", dfn_str="lld", dfn_params=(1.0, 1.0), wfn_params=(1.0,))
        self.cluster_tree = VectorTree(cluster_centers, 1, *cluster_metric.tree_params())


        self.basis = basis
        self.extract_dim = extract_dim
        self.param_mean = param_mean
        self.param_cov = param_cov
        self.featurizer_recovery = featurizer_recovery

        # setup parametric features if needed
        H = None
        self.featurizer = None
        self.featurizer_recovery = None
        if featurizer_recovery is None:
            if basis is not None:
                H, self.featurizer, self.featurizer_recovery = featurizer_from_string(X, basis, extract_dim=extract_dim, transpose=True)
        else:
            self.featurizer, self.featurizer_recovery = recover_featurizer(basis, featurizer_recovery, transpose=True)
            H = self.featurizer(X)


        self.local_gps = self._build_local_gps()
        self.Kinv = scipy.sparse.block_diag([gp.Kinv for gp in self.local_gps])
        self.L = scipy.sparse.block_diag([gp.L for gp in self.local_gps])
        self.alpha = np.concatenate([gp.alpha_r.flatten() for gp in self.local_gps])
        self.local_cumpts = np.cumsum([lgp.n for lgp in self.local_gps])

        def build_low_rank_model(self, alpha, Kinv_sp, H, b, Binv):
            """
            let n be the training size; we'll use an additional rank-m approximation.
            the notation here follows section 2.7 in Rasmussen & Williams. For
            simplicity, K refers to the observation covariance matrix rather than the
            underlying function covariance (i.e. it might really be K+noise_var*I, or
            K+diag(y_obs_variances), etc.)

            takes:
             alpha: n x 1, equal to K^-1 y
             Kinv_sp: n x n sparse matrix, equal to K^-1
             H: n x m features of training data (this is Qfu for FIC)
             b: m x 1 prior mean on feature weights (this is 0 for FIC)
             B: m x m prior covariance on feature weights (this is Quu for FIC)

            returns:
             invc = inv(chol(M)), where M = (B^-1 + H K^-1 H^T)^-1 is the
                    posterior covariance matrix on feature weights
             beta_bar = M (HK^-1y + B^-1 b) gives the weights for the correction
                        of the low-rank component to the mean prediction
             HKinv = HK^-1 comes up in the marginal likelihood computation, so we
                     go ahead and remember the value we compute now.
            """

            # tmp = H * K^-1 * y + B^-1 * b
            tmp = np.reshape(np.asarray(np.dot(H, alpha)), (-1,))
            tmp += np.dot(Binv, b)

            HKinv = H * Kinv_sp
            M_inv  = Binv + np.dot(HKinv, H.T)
            c = scipy.linalg.cholesky(M_inv, lower=True)
            beta_bar = scipy.linalg.cho_solve((c, True), tmp)
            invc = scipy.linalg.inv(c)
            return c, invc, beta_bar, HKinv

        if self.param_mean is None:
            self.n_features = 0
        else:
            self.n_features = len(self.param_mean)
            b = self.param_mean
            Binv = np.linalg.inv(self.param_cov)
            self.c, self.invc,self.beta_bar, self.HKinv = build_low_rank_model(self.alpha,
                                                                               self.Kinv,
                                                                               H, b, Binv)
            self.z = np.dot(H.T, b) - self.y
        
        
    def _x_to_cluster(self, X1):
        dists = self.cluster_tree.kernel_matrix(X1, self.cluster_centers, True)
        return np.argmin(dists, axis=1)


    def get_data_features(self, X):
        # compute the full set of features for a matrix X of test points
        features = np.zeros((self.n_features, X.shape[0]))

        i = 0
        if self.featurizer is not None:
            F = self.featurizer(X)
            i = F.shape[0]
            features[:i,:] = F

        return features

    def predict(self, cond):

        X1 = self.standardize_input_array(cond).astype(np.float)
        cluster_idx = self._x_to_cluster(X1)
        lgp = self.local_gps[cluster_idx]
        gp_pred = lgp.predict(X1)

        #if self.n_features > 0:
        #    H = self.get_data_features(X1)
        #    mean_pred = np.reshape(np.dot(H.T, self.beta_bar), gp_pred.shape)
        #    gp_pred += mean_pred

        if self.n_features > 0:
            query_K = lgp.get_query_K(X1, no_R=True)
            H = self.get_data_features(X1)

            k = self.local_cumpts[cluster_idx]
            local_HKinv = np.matrix(self.HKinv[:, k:k+lgp.n])
            R = H - local_HKinv * query_K

            gp_pred += np.dot(R.T, self.beta_bar)

        return gp_pred

    def variance(self, cond, **kwargs):

        X1 = self.standardize_input_array(cond).astype(np.float)
        cluster_idx = self._x_to_cluster(X1)
        lgp = self.local_gps[cluster_idx]

        gp_variance = lgp.variance(X1, **kwargs)
        

        if self.n_features > 0:
            query_K = lgp.get_query_K(X1, no_R=True)
            H = self.get_data_features(X1)

            k = self.local_cumpts[cluster_idx]
            local_HKinv = np.matrix(self.HKinv[:, k:k+lgp.n])
            R = H - local_HKinv * query_K

            tmp = np.dot(self.invc, R)
            mean_cov = np.dot(tmp.T, tmp)
            gp_variance += mean_cov

        return gp_variance

    def _compute_marginal_likelihood(self):

        if self.n_features == 0:
            return np.sum([gp.log_likelihood() for gp in self.local_gps])

        Kinv = self.Kinv
        z = self.z

        tmp1 = Kinv * z
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

        ld2_K = np.log(ldiag).sum()
        ld2 =  np.log(np.diag(self.c)).sum() # det( B^-1 - H * K^-1 * H.T )
        ld_B = -np.log(np.linalg.det(Binv))

        # eqn 2.43 in R&W, using the matrix inv lemma
        self.ll = -.5 * (term1 - term2 + self.n * np.log(2*np.pi) + ld_B) - ld2_K - ld2

    def _log_likelihood_gradient(self):

        nparams = np.sum([len(c.flatten())+1 for c in self.cluster_covs])
        grad = np.zeros((nparams,))

        if self.n_features > 0:
            tmp = np.dot(self.invc, self.HKinv)
            K_HBH_inv = self.Kinv - np.dot(tmp.T, tmp)
            alpha = np.matrix(np.reshape(np.dot(K_HBH_inv, self.z), (-1, 1)))
            M = np.matrix(K_HBH_inv)
        else:
            M = self.Kinv.todense()
            alpha = self.alpha
        
        npts = 0
        nparams = 0
        for k, lgp in enumerate(self.local_gps):
            lgp.distance_cache_XX = lgp.predict_tree.kernel_matrix(lgp.X, lgp.X, True)
            n_main_params = len(lgp.cov_main.flatten())

            local_alpha = alpha[npts:npts+lgp.n]
            local_M = M[npts:npts+lgp.n][:, npts:npts+lgp.n]
            npts += lgp.n

            for i in range(n_main_params+1):
                dKdi = lgp.get_dKdi_dense(i, n_main_params, 0)
                dlldi = .5 * np.dot(local_alpha.T, np.dot(dKdi, local_alpha))

                # here we use the fact:
                # trace(AB) = sum_{ij} A_ij * B_ij
                dlldi -= .5 * np.sum(np.sum(np.multiply(local_M.T, dKdi)))

                grad[nparams ] = dlldi
                nparams += 1

        return grad

    def get_flat_params(self):
        params = []
        for lgp in self.local_gps:
            params += list(lgp.cov_main.flatten())
            params.append(lgp.noise_var)
        return params

    
    def unflatten_to_covs(self, params):
        covs = []
        noise_vars = []

        n_params = 0
        for lgp in self.local_gps:
            new_cov = lgp.cov_main.copy()
            
            n_wfn = len(new_cov.wfn_params)
            new_cov.wfn_params = np.array(params[n_params:n_params+n_wfn])
            n_params += n_wfn

            n_dfn = len(new_cov.dfn_params)
            new_cov.dfn_params = np.array(params[n_params:n_params+n_dfn])
            n_params += n_dfn
            
            nv = params[n_params]
            n_params += 1

            covs.append(new_cov)
            noise_vars.append(nv)

        return covs, noise_vars
        
