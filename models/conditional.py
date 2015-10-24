import numpy as np
from sigvisa.models import Distribution

class ConditionalGaussian(Distribution):

    def __init__(self, parent_key, std):
        self.parent_key = parent_key
        self.std = std
        self.var = std**2

    def log_p(self, x, cond,  **kwargs):
        mu = cond[self.parent_key]
        sigma2 = self.var
        lp = -.5 * np.log(2*np.pi*sigma2) - .5 * (x - mu)**2 / sigma2
        if sigma2==0:
            lp = np.float("-inf")
        return lp

    def deriv_log_p(self, x, cond, **kwargs):
        mu = cond[self.parent_key]
        return -(x - self.mean)/self.var

    def predict(self, cond, **kwargs):
        return cond[self.parent_key]

    def variance(self, **kwargs):
        return self.var

    def sample(self, cond, **kwargs):
        return cond[self.parent_key] + np.random.randn() * self.std

    def __str__(self):
        return "Gaussian(key=%s, std=%f)" % (self.parent_key, self.std)
