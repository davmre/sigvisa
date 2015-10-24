import numpy as np
import scipy.stats

from sigvisa.models.statespace import StateSpaceModel

class ARSSM(StateSpaceModel):

    def __init__(self, params, error_var, mean=0.0):
        self.params = np.asarray(params) # [a1, a2, ..., an] where a1 is the first-order param

        assert(error_var > 0)
        self.error_var = error_var

        self.max_dimension = len(params)
        self.mean = mean

    def apply_transition_matrix(self, x, k, x_new):
        x_new[1:] = x[:-1]
        x_new[0] = np.dot(self.params, x)
        return self.max_dimension

    def transition_matrix_debug(self, k):
        F = np.zeros((self.max_dimension,self.max_dimension))
        F[0,:] = self.params
        for i in range(1, self.max_dimension):
            F[i, i-1] = 1
        return F

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

    def obs_model_changed(self, k1, k2):
        return False

    def obs_vector_debug(self, k):
        H = np.zeros((self.max_dimension,))
        H[0] = 1
        return H

    def observation_bias(self, k):
        return self.mean

    def observation_noise(self, k):
        return 0.0

    def stationary(self, k):
        return True

    def prior_mean(self):
        return np.zeros((self.max_dimension,))

    def prior_vars(self):
        p = np.ones((self.max_dimension,))*self.error_var
        return p
