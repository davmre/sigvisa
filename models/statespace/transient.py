import numpy as np
import scipy.stats
import copy

from sigvisa.models.statespace import StateSpaceModel


class TransientCombinedSSM(StateSpaceModel):
    """
    State space model consisting of a bunch of submodels that come and go, each model
    being active for a given interval of time. The observation is taken to be
    a (scalar) linear combination of the outputs of whatever submodels are currently
    active.
    """

    def __init__(self, components, obs_noise=0.0):
        """
        components: list of tuples (ssm, start_idx, npts, scale), where
          ssm: StateSpaceModel object
          start_idx: the first timestep at which this ssm should be active.
          npts: number of timesteps for which this ssm should be active.
          scale: either None, or an array of length >npts containing a scale factor for each timestep.
        """

        # save basic info
        self.n_ssms = len(components)
        self.obs_noise = obs_noise
        self.ssms = []
        self.ssm_starts = []
        self.ssm_ends = []
        self.scales = []
        for (ssm, start_idx, npts, scale) in components:
            if scale is not None:
                assert(len(scale) >= npts)
            self.ssms.append(ssm)
            self.ssm_starts.append(start_idx)
            self.ssm_ends.append(start_idx+npts)
            self.scales.append(scale)

        # compute a list of changepoints, with the set of ssms active at each point
        self.changepoints = []
        self.active_sets = []
        starts = [(st, i, True) for (i, st) in enumerate(self.ssm_starts)]
        ends = [(et, i, False) for (i, et) in enumerate(self.ssm_ends)]
        events = sorted(starts+ends)
        active_set = [ ]
        t_prev = 0
        for (t, i_ssm, start) in events:
            if t != t_prev:
                self.changepoints.append(t_prev)
                self.active_sets.append(copy.copy(active_set))
                t_prev = t
            if start:
                active_set.append(i_ssm)
            else:
                active_set.remove(i_ssm)
        self.changepoints.append(t)
        self.active_sets.append(active_set)
        self.changepoints = np.array(self.changepoints, dtype=int)

        # compute dimension of each active set to find the max dimension
        self.max_dimension = np.max([int(np.sum([self.ssms[i].max_dimension for i in s])) for s in self.active_sets])

        # set up caches and tmp arrays
        self.active_ssm_cache1_k = -1
        self.active_ssm_cache1_v = []
        self.active_ssm_cache2_k = -1
        self.active_ssm_cache2_v = []

        self.ssm_tmp = np.empty((self.n_ssms,), dtype=int)

        # storage for arbitrary temp arrays. Keys are array sizes (so
        # you can just get a tmp array of the required size).
        self.tmp_arrays = dict()



    def active_ssms(self, k):
        """Return the (integer) indices of the ssms active at the given
        timestep.  It caches the previous two calls, so assuming the
        filter is running in sequence, timesteps k and k-1 should
        almost always be in cache.
        When the cache misses, we perform a binary search on the list
        of changepoints, so this is approximately O(log n) in the total
        number of ssms. It's probably possible to do better than this...
        """

        if k == self.active_ssm_cache1_k:
            return self.active_ssm_cache1_v
        elif k == self.active_ssm_cache2_k:
            return self.active_ssm_cache2_v

        i = np.searchsorted(self.changepoints, k, side="right")-1
        active_ssms = self.active_sets[i]

        self.active_ssm_cache2_k = self.active_ssm_cache1_k
        self.active_ssm_cache2_v = self.active_ssm_cache1_v
        self.active_ssm_cache1_k = k
        self.active_ssm_cache1_v = active_ssms
        return active_ssms

    def apply_transition_matrix(self, x, k, x_new):

        # first, loop over ssms active at the *previous*
        # timestep in order to cache the location of each
        # ssm in the previous state space.
        j = 0
        old_ssm_indices = self.active_ssms(k-1)
        for i_ssm in old_ssm_indices:
            ssm = self.ssms[i_ssm]
            state_size = ssm.max_dimension
            self.ssm_tmp[i_ssm] = j
            j += state_size

        # now apply the transition to the current time
        i = 0
        ssm_indices = self.active_ssms(k)
        for i_ssm in ssm_indices:
            ssm = self.ssms[i_ssm]
            state_size = ssm.max_dimension

            if self.ssm_starts[i_ssm] == k:
                # new ssms just get filled in as zero
                # (prior means will be added by the
                #  transition_bias operator)
                x_new[i:i+state_size] = 0.0
            else:
                # this ssm is persisting from the
                # previous timestep, so just run the
                # transition
                j = self.ssm_tmp[i_ssm]
                ssm.apply_transition_matrix(x[j:j+state_size], k-self.ssm_starts[i_ssm], x_new[i:i+state_size])
            i += state_size
        return i

    def transition_bias(self, k, x):
        i = 0
        ssm_indices = self.active_ssms(k)
        for j in ssm_indices:
            ssm = self.ssms[j]
            state_size = ssm.max_dimension
            if self.ssm_starts[j] == k:
                x[i:i+state_size] += ssm.prior_mean()
            else:
                ssm.transition_bias(k-self.ssm_starts[j], x[i:i+state_size])
            i += state_size

    def transition_noise_diag(self, k, noise):
        i = 0
        ssm_indices = self.active_ssms(k)
        for j in ssm_indices:
            ssm = self.ssms[j]
            state_size = ssm.max_dimension
            if self.ssm_starts[j] == k:
                noise[i:i+state_size] = ssm.prior_vars()
            else:
                ssm.transition_noise_diag(k-self.ssm_starts[j], noise[i:i+state_size])
            i += state_size

    def apply_observation_matrix(self, x, k, result=None):
        """
        We define the observation for the combined SSM as a (weighted) sum of the
        observations from the currently active components.
        """

        i = 0

        # vector case
        ssm_indices = self.active_ssms(k)
        if len(x.shape)==1:
            r = 0
            for j in ssm_indices:
                ssm, scale = self.ssms[j], self.scales[j]
                state_size = ssm.max_dimension
                ri = ssm.apply_observation_matrix(x[i:i+state_size], k-self.ssm_starts[j])
                if scale is not None:
                    r += ri * scale[k-self.ssm_starts[j]]
                else:
                    r += ri
                i += state_size
            return r

        # matrix case
        else:
            assert(len(x.shape)==2)

            try:
                rr = self.tmp_arrays[len(result)]
            except KeyError:
                rr = np.empty((len(result),))
                self.tmp_arrays[len(result)] = rr

            result[:] = 0
            for j in ssm_indices:
                ssm, scale = self.ssms[j], self.scales[j]
                state_size = ssm.max_dimension
                ssm.apply_observation_matrix(x[i:i+state_size,:], k-self.ssm_starts[j], rr)
                if scale is not None:
                    rr *= scale[k-self.ssm_starts[j]]
                result += rr
                i += state_size

    def obs_vector_debug(self, k):
        H = np.zeros((self.max_dimension,))
        ssm_indices = self.active_ssms(k)
        i = 0
        for j in ssm_indices:
            v = self.ssms[j].obs_vector_debug(k-self.ssm_starts[j])
            if self.scales[j] is not None:
                v *= self.scales[j][k-self.ssm_starts[j]]
            H[i:i+len(v)] = v
            i += len(v)
        return H

    def observation_bias(self, k):
        bias = 0.0
        ssm_indices = self.active_ssms(k)
        for j in ssm_indices:
            kk = k-self.ssm_starts[j]
            b = self.ssms[j].observation_bias(kk)
            if self.scales[j] is not None:
                b *= self.scales[j][kk]
            bias += b
        return bias

    def observation_noise(self, k):
        return self.obs_noise

    def stationary(self, k):
        """
        The combined model is stationary as long as *all* active
        models are stationary, the set of active models hasn't changed,
        and there are no scaling factors active (since scaling factors
        are nonstationary in general).
        """
        s1 = self.active_ssms(k)

        if k > 0:
            s2 = self.active_ssms(k-1)
            if s2 != s1:
                return False
        for j in s1:
            if self.scales[j] is not None:
                return False
            if not self.ssms[j].stationary(k-self.ssm_starts[j]):
                return False
        return True

    def component_means(self, z):
        """
        Given an observed signal, decompose it into component signals
        corresponding to the state space models. The return value
        "means" is a list, indexed in the same way as the component
        ssms, in which each entry is an array, of length equal to the
        activation time for that component, containing the mean
        observations from the (filtered) mean states of that
        component.
        """

        # pre-allocate space by initializing each component to its prior mean.
        # This is only really relevant if a component starts before time 0, in
        # which case those unobserved timesteps will never be updated and so
        # will remain at the prior mean.
        means = []
        for i in range(self.n_ssms):
            means.append(self.ssms[i].mean_obs(self.ssm_ends[i]-self.ssm_starts[i]))

        # run the Kalman filter, and compute the observation generated by each
        # ssm at each timestep.
        for k, (x, U, d) in enumerate(self.filtered_states(z)):
            ssm_indices = self.active_ssms(k)
            i=0
            for j in ssm_indices:
                ssm = self.ssms[j]
                state_size = ssm.max_dimension
                ix = k-self.ssm_starts[j]
                means[j][ix] = ssm.apply_observation_matrix(x[i:i+state_size], ix)
                means[j][ix] += ssm.observation_bias(ix)
                i += state_size
        return means

    def prior_mean(self):
        """
        The prior mean of the combined model is just the concatenation
        of prior mean for all submodels active at step 0. In the
        special case that a model's start time is negative, we
        propagate its prior mean through the transition model for
        the appropriate number of steps to get the (exact) prior mean at stepx 0.
        """

        priors = []
        for i in self.active_ssms(0):
            ssm = self.ssms[i]
            prior = ssm.prior_mean()
            if self.ssm_starts[i] < 0:
                p2 = prior.copy()
                for k in range(-self.ssm_starts[i]):
                    state_size = ssm.apply_transition_matrix(p2, k+1, prior)
                    ssm.transition_bias(k, prior)
                    p2 = prior
            priors.append(prior)
        return np.concatenate(priors)

    def prior_vars(self):
        """
        The prior variance of the combined model is just the concatenation
        of prior variances for all submodels active at step 0. In the
        special case that a model's start time is negative, we
        propagate its prior variance through the transition model for
        the appropriate number of steps to get an (approximate)
        diagonal variance at step 0.
        """
        priors = []
        for i in self.active_ssms(0):
            ssm = self.ssms[i]
            prior = ssm.prior_vars()

            if self.ssm_starts[i] < 0:
                P = np.diag(prior)
                P2 = P.copy()
                for k in range(-self.ssm_starts[i]):
                    ssm.transition_covariance(P2, k+1, P)
                    ssm.transition_noise_diag(k+1, prior)
                    np.fill_diagonal(P, np.diag(P) + prior)
                    P2 = P

                # since the interface only supports independent
                # priors, return a diagonal approximation of the true
                # prior
                prior = np.diag(P)
            priors.append(prior)
        return np.concatenate(priors)

    def component_state_indices(self, k, component_idx):
        ssms = self.active_ssms(k)
        i = 0
        for ssm in ssms:
            next_i = i + self.ssms[ssm].max_dimension
            if ssm == component_idx:
                return i, next_i
            i = next_i
        raise ValueError("component %d is not active at timestep %d" % (component_idx, k))

    def filtered_cssm_coef_marginals(self, z, component_idx):
        # return the marginal means and variances on the basis
        # coefficients for a CompactSupportSSM component.

        ssm = self.ssms[component_idx]
        start = self.ssm_starts[component_idx]
        end = self.ssm_ends[component_idx]
        coef_means = np.empty((ssm.n_basis,))
        coef_vars = np.empty((ssm.n_basis,))

        for k, (x, U, d) in enumerate(self.filtered_states(z)):
            if k < start: continue
            if k >= end: break

            i1, i2 = self.component_state_indices(k, component_idx)
            P = np.dot(d*U, U.T)
            ssm.extract_coefs(x[i1:i2], P[i1:i2,i1:i2], k-start, coef_means, coef_vars)

        return coef_means, coef_vars

    def all_filtered_cssm_coef_marginals(self, z):

        cssms = []
        marginals = dict()
        for i, ssm in enumerate(self.ssms):
            if "extract_coefs" in dir(ssm):
                cssms.append(i)
                # initialize marginals to the prior
                marginals[i] = (ssm.coef_means.copy(), ssm.coef_vars.copy() )

        for k, (x, U, d) in enumerate(self.filtered_states(z)):
            for i in cssms:
                ssm = self.ssms[i]
                start = self.ssm_starts[i]
                end = self.ssm_ends[i]

                if k < start or k >= end: continue

                i1, i2 = self.component_state_indices(k, i)
                P = np.dot(d*U, U.T)
                ssm.extract_coefs(x[i1:i2], P[i1:i2,i1:i2], k-start, marginals[i][0], marginals[i][1])

        return marginals
