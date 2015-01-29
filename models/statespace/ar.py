import numpy as np
import scipy.stats

from sigvisa.models.statespace import StateSpaceModel

class ARSSM(StateSpaceModel):

    def __init__(self, params, error_var):
        self.params = np.asarray(params) # [a1, a2, ..., an] where a1 is the first-order param
        self.error_var = error_var
        self.max_dimension = len(params)

    def apply_transition_matrix(self, x, k, x_new):
        x_new[1:] = x[:-1]
        x_new[0] = np.dot(self.params, x)

    def transition_bias(self, k, x):
        pass

    def transition_noise_diag(self, k, noise):
        noise[:] = 0
        noise[0] = self.error_var

    def apply_observation_matrix(self, x, k, result=None):
        if len(x.shape)==1:
            return x[0]
        else:
            assert(len(x.shape)==2)
            result[:] = x[0,:]

    def observation_noise(self, k):
        return 0.0

    def prior_mean(self):
        return np.zeros((self.max_dimension,))

    def prior_vars(self):
        p = np.zeros((self.max_dimension,))
        p[0] = self.error_var
        return p
