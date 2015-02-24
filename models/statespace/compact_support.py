import numpy as np
import scipy.stats

from sigvisa.models.statespace import StateSpaceModel


class CompactSupportSSM(StateSpaceModel):

    def __init__(self, basis, coef_prior_means=None, coef_prior_vars=None, bias=0.0, obs_noise=0.01):
        self.basis=basis
        self.bias=bias
        self.obs_noise = obs_noise
        self.n_basis, self.n_steps = basis.shape

        def active_at(k, basis):
            return np.arange(basis.shape[0])[np.abs(basis[:, k]) > 0]

        self.active_basis = [active_at(k, basis) for k in range(self.n_steps)]

        self.max_dimension = np.max([len(b) for b in self.active_basis])

        # precompute the state-space indices of each basis vector at each timestep
        def basis_activation_index(i, k):
            try:
                return list(active_at(k, basis)).index(i) + 1
            except ValueError as e:
                return 0
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
        active = self.active_basis[k]

        for i, idx in enumerate(active):
            prev_idx = self.active_indices[idx, k-1]
            x_new[i] = 0 if prev_idx < 1 else x[prev_idx-1]
            #print "transition k %d i %d idx %d prev_idx %d x[prev_idx] %.4f" % (k, i, idx, prev_idx, x_new[i])
        return len(active)

    def transition_matrix_debug(self, k):
        x = np.zeros((self.max_dimension,))
        F = np.zeros((self.max_dimension,self.max_dimension))
        for i in range(self.max_dimension):
            x[:] = 0
            x[i] = 1.0
            self.apply_transition_matrix(x, k, F[:,i])
        return F

    def transition_bias(self, k, x):
        """
        in general this will be the prior means of the new coefs.
        for now we assume these are all zero.
        """
        for i, idx in enumerate(self.active_basis[k]):
            prev_idx = self.active_indices[idx, k-1]
            if prev_idx < 1:
                assert(x[i])==0
                x[i] += self.coef_means[idx]

    def transition_noise_diag(self, k, noise):
        noise[:] = 0
        for i, idx in enumerate(self.active_basis[k]):
            prev_idx = self.active_indices[idx, k-1]
            noise[i] = self.coef_vars[idx] if prev_idx < 1 else 0.0

    def apply_observation_matrix(self, x, k, result=None):
        active = self.active_basis[k]
        obs = self.basis[active, k]

        if len(x.shape)==1:
            # matrix-vector multiplication, produces a scalar in this case since the observation "matrix" is really a row vector
            return np.dot(obs, x[:len(obs)])
        else:
            assert(len(x.shape)==2)
            # matrix-matrix multiplication, produces a (1 x x.shape[1])
            r = np.dot(obs, x[:len(obs), :], result[:x.shape[1]])

    def obs_vector_debug(self, k):
        active = self.active_basis[k]
        obs = self.basis[active, k]
        H = np.zeros((self.max_dimension,))
        H[:len(obs)] = obs
        return H

    def observation_bias(self, k):
        return self.bias

    def observation_noise(self, k):
        return self.obs_noise

    def stationary(self, k):
        return False

    def prior_mean(self):
        p = np.zeros((self.max_dimension,))
        for j, idx in enumerate(self.active_basis[0]):
            p[j] = self.coef_means[idx]
        return p

    def prior_vars(self):
        p = np.zeros((self.max_dimension,))
        for j, idx in enumerate(self.active_basis[0]):
            p[j] = self.coef_vars[idx]
        return p

    def extract_coefs(self, x, P, k, coef_means, coef_vars):
        # given a state estimate at some time k, extract marginals for
        # whatever coefficients we can reasonably do so for at the
        # current time. Earlier estimates will always be overwritten
        # by later ones.
        for i, basis_idx in enumerate(self.active_basis[k]):
            state_idx = self.active_indices[basis_idx, k]
            coef_means[basis_idx] = x[state_idx-1]
            coef_vars[basis_idx] = P[state_idx-1,state_idx-1]

    def filtered_coef_marginals(self, z):
        coef_means = np.empty((self.n_basis,))
        coef_vars = np.empty((self.n_basis,))


        for k, (x, U, d) in enumerate(self.filtered_states(z)):
            P = np.dot(d*U, U.T)
            self.extract_coefs(x, P, k, coef_means, coef_vars)
            if k % 100 == 0:
                print "filtering step", k

        return coef_means, coef_vars

class ImplicitCompactSupportSSM(CompactSupportSSM):

    def __init__(self, start_idxs, end_idxs, identities, basis_prototypes, coef_prior_means=None, coef_prior_vars=None, bias=0.0, obs_noise=0.01):

        self.basis_prototypes=basis_prototypes
        self.start_idxs = np.asarray(start_idxs, dtype=int)
        self.end_idxs = np.asarray(end_idxs, dtype=int)
        self.identities = np.asarray(identities, dtype=int) # for each basis, an integer index into the array of basis prototypes

        self.bias=bias
        self.obs_noise = obs_noise

        self.n_basis = len(start_idxs)
        self.n_steps = np.max(end_idxs)


        # we have a giant list of start and end times
        # how about
        self.active_basis = [[] for i in range(self.n_steps)]

        rows = []
        cols = []
        entries = []
        for (i, st, et) in zip(range(self.n_basis), self.start_idxs, self.end_idxs):
            if i % 1000 == 0:
                print "preprocessed up to basis", i
            for j in range(max(st, 0), min(et, self.n_steps)):
                self.active_basis[j].append(i)

                rows.append(i)
                cols.append(j)
                entries.append(len(self.active_basis[j]))

        entries = np.asarray(entries, dtype=int)
        self.active_indices = scipy.sparse.coo_matrix((entries, (rows, cols)), shape=(self.n_basis, self.n_steps))
        self.active_indices = self.active_indices.todok()

        #def active_at(k):
        #    return np.arange(self.n_basis)[(self.start_idxs <= k) * (self.end_idxs > k)]

        #self.active_basis = [active_at(k) for k in range(self.n_steps)]

        self.max_dimension = np.max([len(b) for b in self.active_basis])

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

    def apply_observation_matrix(self, x, k, result=None):
        active = self.active_basis[k]


        #print "time", k, "active", active
        #for basis_idx in active:
        #    print " basis", basis_idx, "identity", self.identities[basis_idx], "st", self.start_idxs[basis_idx], "adjusted", k-self.start_idxs[basis_idx],
        #    ident = self.identities[basis_idx]
        #    pt = self.basis_prototypes[ident]
        #    print " pt len", len(pt),
        #    print " et", self.end_idxs[basis_idx],
        #    print "obs", pt[k-self.start_idxs[basis_idx]]

        obs = [self.basis_prototypes[self.identities[basis_idx]][k-self.start_idxs[basis_idx]] for basis_idx in active]

        if len(x.shape)==1:
            # matrix-vector multiplication, produces a scalar in this case since the observation "matrix" is really a row vector
            return np.dot(obs, x[:len(obs)])
        else:
            assert(len(x.shape)==2)
            # matrix-matrix multiplication, produces a (1 x x.shape[1])
            r = np.dot(obs, x[:len(obs), :], result[:x.shape[1]])
