import numpy as np
import scipy.stats

from sigvisa.models.statespace import StateSpaceModel

class CompactSupportSSM(StateSpaceModel):

    def __init__(self, bases, coef_prior_means=None, coef_prior_vars=None):
        self.bases=bases
        self.n_basis, self.n_steps = bases.shape

        def active_at(k, bases):
            return np.arange(bases.shape[0])[np.abs(bases[:, k]) > 0]

        self.active_bases = [active_at(k, bases) for k in range(self.n_steps)]

        self.max_dimension = np.max([len(b) for b in self.active_bases])

        # precompute the state-space indices of each basis vector at each timestep
        def basis_activation_index(i, k):
            try:
                return list(active_at(k, bases)).index(i)
            except ValueError as e:
                return -1
        self.active_indices = np.array([[basis_activation_index(i, k) for k in range(self.n_steps)] for i in range(self.n_basis)], dtype=np.int32)

        if coef_prior_means is None:
            self.coef_means = np.zeros((self.n_basis,))
        else:
            assert(len(coef_prior_means)==self.n_basis)
            self.coef_means = coef_prior_means
        if coef_prior_vars is None:
            self.coef_vars = np.ones((self.n_basis,))
        else:
            assert(len(coef_prior_vars)==self.n_basis)
            self.coef_vars = coef_prior_vars

    def apply_transition_matrix(self, x, k, x_new):
        # compute F_k*x for the given x, where
        # F_k is the transition matrix from time
        # k-1 to k. The result is stored in
        # the provided vector x_new. If x_new
        # has extra dimensions, they are not
        # touched (so may still contain garbage).

        assert(k>0)
        active = self.active_bases[k]

        for i, idx in enumerate(self.active_bases[k]):
            prev_idx = self.active_indices[idx, k-1]
            x_new[i] = 0 if prev_idx < 0 else x[prev_idx]


    def transition_bias(self, k, x):
        """
        in general this will be the prior means of the new coefs.
        for now we assume these are all zero.
        """
        for i, idx in enumerate(self.active_bases[k]):
            prev_idx = self.active_indices[idx, k-1]
            if prev_idx < 0:
                assert(x[i])==0
                x[i] += self.coef_means[idx]

    def transition_noise_diag(self, k, noise):
        noise[:] = 0
        for i, idx in enumerate(self.active_bases[k]):
            prev_idx = self.active_indices[idx, k-1]
            noise[i] = self.coef_vars[idx] if prev_idx < 0 else 0.0

    def apply_observation_matrix(self, x, k, result=None):
        active = self.active_bases[k]
        obs = self.bases[active, k]

        if len(x.shape)==1:
            # matrix-vector multiplication, produces a scalar in this case since the observation "matrix" is really a row vector
            return np.dot(obs, x[:len(obs)])
        else:
            assert(len(x.shape)==2)
            # matrix-matrix multiplication, produces a (1 x x.shape[1])
            r = np.dot(obs, x[:len(obs), :], result[:x.shape[1]])


    def observation_noise(self, k):
        return 0.01

    def prior_mean(self):
        p = np.zeros((self.max_dimension,))
        for j, idx in enumerate(self.active_bases[0]):
            p[j] = self.coef_means[idx]
        return p

    def prior_vars(self):
        p = np.zeros((self.max_dimension,))
        for j, idx in enumerate(self.active_bases[0]):
            p[j] = self.coef_vars[idx]
        return p
