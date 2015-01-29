import numpy as np
import scipy.stats

from sigvisa.models.statespace import StateSpaceModel

class CombinedSSM(StateSpaceModel):

    def __init__(self, ssms, scales):
        # ssms: list of n_ssms StateSpaceModel objects
        # scales: list of functions giving the scale for each ssm at time k.

        assert (len(ssms) == len(scales))
        self.n_ssms = len(ssms)
        self.ssms = ssms
        self.scales=scales

        # all ssms must have a fixed state size
        self.state_sizes = np.array([ssm.max_dimension for ssm in ssms])
        self.max_dimension = np.sum(self.state_sizes)

        # indexed by size of the array
        self.tmp_arrays = dict()

    def apply_transition_matrix(self, x, k, x_new):
        i = 0
        for state_size, ssm in zip(self.state_sizes, self.ssms):
            ssm.apply_transition_matrix(x[i:i+state_size], k, x_new[i:i+state_size])
            i += state_size

    def transition_bias(self, k, x):
        i = 0
        for state_size, ssm in zip(self.state_sizes, self.ssms):
            ssm.transition_bias(k, x[i:i+state_size])
            i += state_size

    def transition_noise_diag(self, k, noise):
        i = 0
        for state_size, ssm in zip(self.state_sizes, self.ssms):
            ssm.transition_noise_diag(k, noise[i:i+state_size])
            i += state_size

    def apply_observation_matrix(self, x, k, result=None):
        i = 0

        if len(x.shape)==1:
            r = 0
            for state_size, ssm, scale in zip(self.state_sizes, self.ssms, self.scales):
                ri = ssm.apply_observation_matrix(x[i:i+state_size], k)
                r += ri * scale(k)
                i += state_size
            return r
        else:
            assert(len(x.shape)==2)

            try:
                rr = self.tmp_arrays[len(result)]
            except KeyError:
                rr = np.empty((len(result),))
                self.tmp_arrays[len(result)] = rr

            result[:] = 0
            for state_size, ssm, scale in zip(self.state_sizes, self.ssms, self.scales):
                ssm.apply_observation_matrix(x[i:i+state_size], k, rr)
                rr *= scale(k)
                result += rr
                i += state_size

    def observation_noise(self, k):
        return 0.01

    def prior_mean(self):
        return np.concatenate([ssm.prior_mean() for ssm in self.ssms])

    def prior_vars(self):
        return np.concatenate([ssm.prior_vars() for ssm in self.ssms])
