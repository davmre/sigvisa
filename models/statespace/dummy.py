import numpy as np
import scipy.stats

from sigvisa.models.statespace import StateSpaceModel

class DummySSM(StateSpaceModel):

    def __init__(self, bias=0.0):
        self.bias = bias
        self.max_dimension = 0

    def apply_transition_matrix(self, x, k, x_new):
        return self.max_dimension

    def transition_matrix_debug(self, k):
        return np.eye(0)

    def transition_bias(self, k, x):
        pass

    def transition_noise_diag(self, k, noise):
        pass

    def apply_observation_matrix(self, x, k, result=None):
        if len(x.shape)==1:
            return 0.0
        else:
            assert(len(x.shape)==2)
            result[:] = 0.0

    def obs_vector_debug(self, k):
        H = np.zeros((self.max_dimension,))
        return H

    def observation_bias(self, k):
        return self.bias

    def observation_noise(self, k):
        return 0.0

    def prior_mean(self):
        return np.zeros((self.max_dimension,))

    def prior_vars(self):
        p = np.zeros((self.max_dimension,))
        return p
