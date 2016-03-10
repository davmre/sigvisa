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
import cPickle as pickle

from sigvisa.treegp.gp import GP, GPCov
from sigvisa.treegp.features import featurizer_from_string, recover_featurizer
from sigvisa.treegp.cover_tree import VectorTree
from sigvisa.models.spatial_regression.baseline_models import ParamModel

class LocalGPEnsemble(ParamModel):

    def _build_local_gps(self, X, y, yvars, **kwargs):
        cluster_distances = self.cluster_tree.kernel_matrix(pyublas.why_not(X), 
                                                            pyublas.why_not(self.cluster_centers), True)
        closest_cluster = np.argmin(cluster_distances, axis=1)

        local_gps = []
        sorted_X = []
        sorted_y = []
        sorted_yvars = []
        for i, ctr in enumerate(self.cluster_centers):
            cov = self.cluster_covs[i]
            noise_var = self.cluster_noise_vars[i]

            cluster_idxs = (closest_cluster == i)
            cluster_X = X[cluster_idxs]
            cluster_y = y[cluster_idxs]
            cluster_yvars = yvars[cluster_idxs]

            if len(cluster_y) == 0:
                cluster_X = np.zeros((1, 5,))
                cluster_y = np.zeros((1,))
                cluster_yvars = np.ones((1,)) * 1e20

            sorted_X.append(cluster_X)
            sorted_y.append(cluster_y)
            sorted_yvars.append(cluster_yvars)

            lgp = GP(X=cluster_X, y=cluster_y, y_obs_variances=cluster_yvars,
                     cov_main=cov, noise_var=noise_var, 
                     sort_events=False,
                     sparse_invert=False,
                     **kwargs)
            local_gps.append(lgp)

        sorted_X = np.vstack(sorted_X)
        sorted_y = np.concatenate(sorted_y)
        sorted_yvars = np.concatenate(sorted_yvars)
            
        return local_gps, sorted_X, sorted_y, sorted_yvars
        
    def __init__(self, X, y,  
                 cluster_centers,
                 cluster_covs,
                 cluster_noise_vars,
                 yvars = None,
                 basis=None,
                 extract_dim = None,
                 prior_mean=None,
                 prior_cov=None,
                 featurizer_recovery=None,
                 **kwargs):

        ParamModel.__init__(self, **kwargs)

        self.cluster_centers = cluster_centers
        self.cluster_covs = cluster_covs
        self.cluster_noise_vars = cluster_noise_vars
        self.cluster_metric = GPCov(wfn_str="se", dfn_str="lld", dfn_params=(1.0, 1.0), wfn_params=(1.0,))
        self.cluster_tree = VectorTree(cluster_centers, 1, *self.cluster_metric.tree_params())

        if yvars is None:
            yvars = np.zeros(y.shape)
        self.local_gps, self.X, self.y, self.yvars = self._build_local_gps(X, y, yvars,
                                                                           compute_ll=(basis is None))
        self.n = len(self.y)

        self.basis = basis
        self.extract_dim = extract_dim
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.featurizer_recovery = featurizer_recovery

        # setup parametric features if needed
        H = None
        self.featurizer = None
        self.featurizer_recovery = None
        if featurizer_recovery is None:
            if basis is not None:
                H, self.featurizer, self.featurizer_recovery = featurizer_from_string(self.X, basis, extract_dim=extract_dim, transpose=True)
        else:
            self.featurizer, self.featurizer_recovery = recover_featurizer(basis, featurizer_recovery, transpose=True)
            H = self.featurizer(self.X)


        self.Kinv = scipy.sparse.block_diag([gp.Kinv for gp in self.local_gps])
        self.L = scipy.sparse.block_diag([gp.L for gp in self.local_gps])
        self.alpha = np.concatenate([gp.alpha_r.flatten() for gp in self.local_gps])
        self.local_cumpts = np.cumsum([lgp.n for lgp in self.local_gps])


        def build_low_rank_model(alpha, Kinv_sp, H, b, Binv):
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

        if self.basis is None:
            self.n_features = 0
        else:
            self.n_features = len(self.prior_mean)
            b = self.prior_mean
            Binv = np.linalg.inv(self.prior_cov)
            self.c, self.invc,self.beta_bar, self.HKinv = build_low_rank_model(self.alpha,
                                                                               self.Kinv,
                                                                               H, b, Binv)
            self.z = np.dot(H.T, b) - self.y
        
        
    def _x_to_cluster(self, X1):
        dists = self.cluster_tree.kernel_matrix(X1, self.cluster_centers, True)
        return np.argmin(dists, axis=1)

    def param_mean(self):
        try:
            return self.beta_bar
        except:
            return np.zeros((0,))

    def param_covariance(self, chol=False):
        if chol:
            return self.invc
        else:
            return np.dot(self.invc.T, self.invc)

    def get_data_features(self, X):
        # compute the full set of features for a matrix X of test points
        features = np.zeros((self.n_features, X.shape[0]))

        i = 0
        if self.featurizer is not None:
            F = self.featurizer(X)
            i = F.shape[0]
            features[:i,:] = F

        return features

    def sample(self, cond, include_obs=True, **kwargs):

        mean = self.predict(cond)
        variance = self.variance(cond, include_obs=include_obs)

        return np.random.randn() * np.sqrt(variance) + mean

    def log_p(self, x, cond, include_obs=True, **kwargs):
        y = float(x)

        mean = float(self.predict(cond))
        variance = float(self.variance(cond, include_obs=include_obs))

        return - .5 * ((y-mean)**2 / variance + np.log(2*np.pi*variance) )
        

    def predict(self, cond, **kwargs):

        # TODO: cache features and R between predict and variance calls..

        X1 = self.standardize_input_array(cond).astype(np.float)
        cluster_idx = self._x_to_cluster(X1)
        lgp = self.local_gps[cluster_idx]
        gp_pred = float(lgp.predict(X1))

        if self.n_features > 0:
            query_K = lgp.get_query_K(X1, no_R=True)
            H = self.get_data_features(X1)

            k = self.local_cumpts[cluster_idx-1] if cluster_idx > 0 else 0
            local_HKinv = np.matrix(self.HKinv[:, k:k+lgp.n])
            R = H - local_HKinv * query_K

            gp_pred += float(np.dot(R.T, self.beta_bar))

        return gp_pred

    def variance(self, cond, **kwargs):

        X1 = self.standardize_input_array(cond).astype(np.float)
        assert(X1.shape[0] == 1)

        cluster_idx = self._x_to_cluster(X1)
        lgp = self.local_gps[cluster_idx]

        gp_variance = float(lgp.variance(X1, **kwargs))
        

        if self.n_features > 0:
            query_K = lgp.get_query_K(X1, no_R=True)
            H = self.get_data_features(X1)

            k = self.local_cumpts[cluster_idx-1] if cluster_idx > 0 else 0
            local_HKinv = np.matrix(self.HKinv[:, k:k+lgp.n])
            R = H - local_HKinv * query_K

            tmp = np.dot(self.invc, R)
            mean_cov = np.dot(tmp.T, tmp)
            gp_variance += float(mean_cov)

        return gp_variance

    def log_likelihood(self):

        if self.n_features == 0:
            ll = np.sum([gp.log_likelihood() for gp in self.local_gps])
            return ll


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

        ldiag = self.L.diagonal()
        ld2_K = np.log(ldiag).sum()
        ld2 =  np.log(np.diag(self.c)).sum() # det( B^-1 - H * K^-1 * H.T )

        ld_B = np.linalg.slogdet(self.prior_cov)[1]

        # eqn 2.43 in R&W, using the matrix inv lemma
        ll = -.5 * (term1 - term2 + self.n * np.log(2*np.pi) + ld_B) - ld2_K - ld2

        return ll

    def log_likelihood_gradient(self):

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
            params.append(lgp.noise_var)
            params += list(lgp.cov_main.flatten())
        return params

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["cluster_tree"]

        try:
            del d["featurizer"]
        except:
            pass

        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.cluster_tree = VectorTree(self.cluster_centers, 1, *self.cluster_metric.tree_params())
        if self.basis is not None:
            self.featurizer, self.featurizer_recovery = recover_featurizer(self.basis, self.featurizer_recovery, transpose=True)
        else:
            self.featurizer = None
            self.featurizer_recovery = None

    def save_trained_model(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self, f)



def optimize_localgp_hyperparams(noise_prior=None,
                                 cov_main=None, 
                                 cluster_centers=None,
                                 y_list = None,
                                 yvars_list = None,
                                 force_unit_var=False,
                                 **kwargs):

    n_clusters = len(cluster_centers)

    n_wfn = len(cov_main.wfn_params) 
    n_dfn = len(cov_main.dfn_params) 
    nparams = 1 + n_wfn + n_dfn 
    nparams *= n_clusters

    if y_list is None:
        y_list = [kwargs["y"],]
        del kwargs["y"]

    if yvars_list is None:
        if "yvars" in kwargs:
            yvars_list = [kwargs["yvars"]]
            del kwargs["yvars"]
        else:
            yvars_list = [None,] * len(y_list)

    def expand_reduced_params(rparams):
        # given a set of params that includes only the signal/noise
        # ratio, expand to the full parameterization assuming unit
        # total variance.
        
        # standard param order:
        # noise var, signal var, lscale horiz, lscale depth

        params = []
        for i in range(0, len(rparams), 3):
            # ratio = nv/sv = nv / (1-nv)
            ratio10 = rparams[i]
            ratio = ratio10 / 10.0

            nv = ratio / (1.+ratio)
            if nv == 1.0:
                nv = 1.-1e-10
            elif nv == 0.0:
                nv = 1e-10

            sv = 1.0-nv

            lscale_horiz = rparams[i+1]
            lscale_depth = rparams[i+2]
            
            params.append(nv)
            params.append(sv)
            params.append(lscale_horiz)
            params.append(lscale_depth)


        return np.array(params)

    def reduce_params(params):
        rparams = []
        for i in range(0, len(params), 4):
            # ratio = nv/sv = nv / (1-nv)
            nv = params[i]
            sv = params[i+1]
            ratio = nv/sv
            ratio10 = ratio * 10

            lscale_horiz = params[i+2]
            lscale_depth = params[i+3]
            
            rparams.append(ratio10)
            rparams.append(lscale_horiz)
            rparams.append(lscale_depth)
        
        return np.array(rparams)

    def grad_reduced_params(gfull, params):
        rgrad = []
        
        for i in range(0, len(gfull), 4):
            d_nv = gfull[i]
            d_sv = gfull[i+1]
            d_lhoriz = gfull[i+2]
            d_ldepth = gfull[i+3]

            nv = params[i]
            sv = params[i+1]
            ratio = nv/sv

            # dll_dratio = dll_dnv dnv_dratio + dll_dsv dsv_dratio            
            d_ratio = d_nv * 1./(ratio+1.)**2 + d_sv * -1. / (ratio+1.)**2

            d_ratio10 = d_ratio / 10.0

            rgrad.append(d_ratio10)
            rgrad.append(d_lhoriz)
            rgrad.append(d_ldepth)

        return np.array(rgrad)

            
    def covs_from_vector(params):
        covs = []
        noise_vars = []

        k = 0
        for c in cluster_centers:
            new_cov = cov_main.copy()

            nv = params[k]
            k += 1
            
            new_cov.wfn_params = np.array(params[k:k+n_wfn])
            k += n_wfn

            new_cov.dfn_params = np.array(params[k:k+n_dfn])
            k += n_dfn
            
            covs.append(new_cov)
            noise_vars.append(nv)

        return covs, noise_vars

    def nllgrad(v):

        if not np.all(np.isfinite(v)):
            return np.float('inf'), np.zeros(v.shape)

        try:
            expv = np.exp(v)
            if force_unit_var:
                expv = expand_reduced_params(expv)

            cluster_covs, cluster_noise_vars = covs_from_vector(expv)

            grad_expv = np.zeros(expv.shape)

            ll = 0.0
            for i, (y, yvars) in enumerate(zip(y_list, yvars_list)):

                lgps = LocalGPEnsemble(cluster_centers=cluster_centers, 
                                       cluster_covs=cluster_covs,
                                       cluster_noise_vars=cluster_noise_vars, 
                                       y=y, yvars=yvars, **kwargs)

                param_ll = lgps.log_likelihood()
                ll += param_ll
                grad_expv += lgps.log_likelihood_gradient()
                del lgps


            prior_grad = []
            priorll = 0.0
            for i, cc in enumerate(cluster_centers):
                priorll += noise_prior.log_p(cluster_noise_vars[i])
                priorll += cluster_covs[i].prior_logp()
                
                prior_grad.append(noise_prior.deriv_log_p(cluster_noise_vars[i]))
                prior_grad += list(cluster_covs[i].prior_grad())
            
            prior_grad = np.array(prior_grad)
            grad_expv += prior_grad
            ll += priorll


            if force_unit_var:
                grad_expv = grad_reduced_params(grad_expv, expv)

            grad_v = grad_expv * np.exp(v)

            
            #print "expv", expv, "ll", ll
        
            if np.isinf(ll):
                import pdb; pdb.set_trace()

            if np.isinf(np.sum(grad_v)):
                import pdb; pdb.set_trace()

            if np.isnan(grad_v).any():
                print "warning: nans in gradient", grad_v
                grad_v[np.isnan(grad_v)] = 0.0
                ll = min(ll, -1e100)



        except FloatingPointError as e:
            print "warning: floating point error (%s) in likelihood computation, returning likelihood -inf" % str(e)
            ll = np.float("-inf")
            grad_v = np.zeros((len(v),))
        except np.linalg.linalg.LinAlgError as e:
            print "warning: lin alg error (%s) in likelihood computation, returning likelihood -inf" % str(e)
            ll = np.float("-inf")
            grad_v = np.zeros((len(v),))
        except scikits.sparse.cholmod.CholmodError as e:
            print "warning: cholmod error (%s) in likelihood computation, returning likelihood -inf" % str(e)
            ll = np.float("-inf")
            grad_v = np.zeros((len(v),))
        #except ValueError as e:
        #    print "warning: value error (%s) in likelihood computation, returning likelihood -inf" % str(e)
        #    ll = np.float("-inf")
        #    grad = np.zeros((len(v),))
        #print "hyperparams", v, "ll", ll, 'grad', grad


        return -1 * ll, (-1 * grad_v  if grad_v is not None else None)

    def build_gp(v, **kwargs2):
        expv = np.exp(v)
        if force_unit_var:
            expv = expand_reduced_params(expv)

        cluster_covs, cluster_noise_vars = covs_from_vector(expv)
        kw = dict(kwargs.items() + kwargs2.items())

        gps = []

        for (y, yvars) in zip(y_list, yvars_list):
            gp = LocalGPEnsemble(cluster_centers=cluster_centers,
                                 cluster_noise_vars=cluster_noise_vars, 
                                 cluster_covs=cluster_covs, 
                                 y=y, yvars=yvars, **kw)
            gps.append(gp)

        if len(gps) == 1:
            return gp
        else:
            return gps

    noise_var_default = noise_prior.predict()
    if force_unit_var:
        x0 = np.concatenate([[0.4, 0.6,] + list(cov_main.flatten())[1:] for i in range(n_clusters)])
        x0 = reduce_params(x0)
    else:
        x0 = np.concatenate([[noise_var_default,] + list(cov_main.flatten()) for i in range(n_clusters)])


    x0 = np.log(x0)


    

    return nllgrad, x0, build_gp, covs_from_vector


