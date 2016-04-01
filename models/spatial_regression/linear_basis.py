import numpy as np
import scipy.stats

from sigvisa import Sigvisa
from sigvisa.source.event import Event
from sigvisa.models import Distribution
import sigvisa.utils.geog as geog
import collections
import hashlib
import marshal
import types

from sigvisa.models.spatial_regression.baseline_models import ParamModel
from sigvisa.treegp.features import featurizer_from_string, recover_featurizer

class LinearBasisModel(ParamModel):

    def __init__(self, fname=None, sta=None, X=None, y=None, yvars=None, basis="poly1", param_mean=None, param_cov=None, param_var=1.0, noise_std=1, featurizer_recovery=None, compute_ll=False, H=None, extract_dim=0):
        super(LinearBasisModel, self).__init__(sta=sta)

        if fname is not None:
            self.load_trained_model(fname)
            return

        X = self.standardize_input_array(X)
        n = X.shape[0]

        if featurizer_recovery is None:
           H, self.featurizer, self.featurizer_recovery = featurizer_from_string(X, basis, extract_dim=extract_dim)
        else:
           self.featurizer, self.featurizer_recovery = recover_featurizer(basis, featurizer_recovery)
           H = self.featurizer(X)

        self.basis = basis

        d = H.shape[1]
        b = np.zeros((d,)) if param_mean is None else param_mean
        B = np.eye(d) * param_var if param_cov is None else param_cov

        self.noise_var  = float(noise_std**2)
        noise_diag = np.ones((n,)) * self.noise_var
        if yvars is not None:
            noise_diag += yvars
        self.noise_prec_diag = 1.0/noise_diag

        B_chol = scipy.linalg.cholesky(B, lower=True)
        B_chol_inv = scipy.linalg.inv(B_chol)
        prior_precision = np.dot(B_chol_inv.T, B_chol_inv)
        precision = np.dot( self.noise_prec_diag*H.T, H) + prior_precision
        precision_chol = scipy.linalg.cholesky(precision, lower=True)
        precision_chol_inv = scipy.linalg.inv(precision_chol)
        covar = np.dot(precision_chol_inv.T, precision_chol_inv)
        self.mean = np.dot(covar, np.dot(prior_precision, b) + np.dot(H.T, y * self.noise_prec_diag))
        self.sqrt_covar=precision_chol_inv
        self.ll=None
        if compute_ll:
            # marginal_covar = self.noise_var * np.eye(n) + np.dot(H, np.dot(B, H.T))
            # using the matrix inversion lemma:
            # (sigma2*I + H*B*H.T)^(-1) = 1/sigma2 * I - 1/sigma2**2 * H(Binv + 1/sigma2 H.T*H)^(-1)H.T

            N_diag = np.diag(self.noise_prec_diag)

            empirical_covar = np.dot(self.noise_prec_diag * H.T, H)
            inner = prior_precision + empirical_covar
            inner_chol = scipy.linalg.cholesky(inner, lower=True)
            inner_chol_inv = scipy.linalg.inv(inner_chol)
            inner_inv = np.dot(inner_chol_inv.T, inner_chol_inv)

            # outer = np.dot(N_diag, np.dot(H, np.dot(inner_inv, np.dot(H.T, N_diag))))
            outer_tmp = np.dot(H, np.dot(inner_inv, np.dot(H.T, N_diag)))
            outer = (self.noise_prec_diag*outer_tmp.T).T
            marginal_precision = N_diag - outer

            # also using the matrix determinant lemma:
            # |sigma2*I + H*B*H.T| = |Binv + 1/sigma2 * H.T * H| * |B| * |sigma2 * I|
            inner_logdet = 2*np.log(np.diag(inner_chol)).sum()
            B_logdet = 2*np.log(np.diag(B_chol)).sum()
            s2I_logdet = np.sum(-np.log(self.noise_prec_diag))
            marginal_covar_logdet = inner_logdet + B_logdet + s2I_logdet

            r = y - np.dot(H, b)
            tmp = np.dot(marginal_precision, r)
            data_term = np.dot(r.T, tmp)

            self.ll =  -.5 * (data_term + n * np.log(2*np.pi) + marginal_covar_logdet)
            self.ll_deriv = noise_std * (np.dot(tmp.T, tmp) - np.trace(marginal_precision))



    def save_trained_model(self, fname):
        with open(fname, 'wb') as f:
            base_str = super(LinearBasisModel, self).__repr_base_params__()
            np.savez(f,
                     sqrt_covar=self.sqrt_covar,
                     mean=self.mean,
                     noise_var=self.noise_var,
                     ll=self.ll,
                     base_str = base_str,
                     desc = self.basis,
                     **(self.featurizer_recovery)
            )

    def load_trained_model(self, fname):
        npzfile = np.load(fname)
        self.sqrt_covar = npzfile['sqrt_covar']
        self.mean = npzfile['mean']
        self.noise_var = npzfile['noise_var']
        self.ll = npzfile['ll']
        self.basis = str(npzfile['desc'])
        self.featurizer, self.featurizer_recovery = recover_featurizer(self.basis, npzfile)
        super(LinearBasisModel, self).__unrepr_base_params__(str(npzfile['base_str']))
        del npzfile.f
        npzfile.close()

    def predict(self, cond, features=None, **kwargs):

        if features is None:
            X1 = self.standardize_input_array(cond, **kwargs)
            X2 = self.featurizer(X1)
        else:
            X2 = features

        return np.dot(X2, self.mean)

    def log_likelihood(self):
        return self.ll

    def covariance(self, cond, return_sqrt=False, include_obs=True, features=None, **kwargs):

        if features is None:
            X1 = self.standardize_input_array(cond, **kwargs)
            X2 = self.featurizer(X1)
        else:
            X2 = features

        tmp = np.dot(self.sqrt_covar, X2.T)

        if return_sqrt and not include_obs:
            return tmp.T
        else:
            covar = np.dot(tmp.T, tmp) + (self.noise_var if include_obs else 1e-8) * np.eye(X2.shape[0])
            if return_sqrt:
                return scipy.linalg.cholesky(covar, lower=True)
            else:
                return covar

    def variance(self, cond, include_obs=True):
        return np.diag(self.covariance(cond=cond, include_obs=include_obs))

    def log_p(self, x, cond, **kwargs):
        cond = self.standardize_input_array(cond, **kwargs)
        X2 = self.featurizer(cond)

        mean = self.predict(cond, features=X2, **kwargs)
        n = len(mean)
        covar = self.covariance(cond, features=X2, include_obs=True)

        r = x - mean

        if n==1:
            var = covar[0,0]
            ll = - .5 * ((r[0])**2 / var + np.log(2*np.pi*var) )
            return ll

        chol = scipy.linalg.cholesky(covar, lower=True)
        ld2 = np.log(np.diag(chol)).sum()
        csi = scipy.linalg.inv(chol)
        tmp = np.dot(csi, r)
        d = np.dot(tmp.T, tmp)
        ll =  -.5 * ( d + n * np.log(2*np.pi)) - ld2



        return ll

    def sample(self, cond, include_obs=True):
        mean = self.predict(cond)
        n = len(mean)
        covar_sqrt = self.covariance(cond, return_sqrt=True, include_obs=include_obs)

        samples = np.random.randn(covar_sqrt.shape[1], 1)
        samples = mean + np.dot(covar_sqrt, samples).flatten()
        return samples

    def param_mean(self):
        return self.mean

    def param_covariance(self, chol=False):
        if chol:
            return self.sqrt_covar
        else:
            return np.dot(self.sqrt_covar.T, self.sqrt_covar)

    def __str__(self):
        return "LinearBasis(param_mean=%s, noise_std=%.2f)" % (self.mean, np.sqrt(self.noise_var))
