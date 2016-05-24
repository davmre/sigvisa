import numpy as np
from sigvisa.models import Distribution
from scipy.special import erf

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

class ConditionalTruncatedGaussian(Distribution):

    def __init__(self, parent_key, std, a, b, eps=1e-2):
        self.parent_key = parent_key
        self.std = std
        self.var = std**2
        self.a=a
        self.b=b
        self.eps=eps

        self.Z = .5 * (erf((self.b)/(std*np.sqrt(2))) -  erf((self.a)/(std*np.sqrt(2))))
        self.logZ = np.log(self.Z)


    def log_p(self, x, cond,  **kwargs):
        mu = cond[self.parent_key]
        sigma2 = self.var

        r = x-mu

        if sigma2==0:
            lp = np.float("-inf")

        eps = self.eps
        if r <= self.a-eps or r >= self.b +eps:
            return -np.inf

        lp = -.5 * np.log(2*np.pi*sigma2) - .5 * r**2 / sigma2 - self.logZ

        if np.isnan(lp):
            lp = np.float("-inf")

        if r < self.a:
            d = ( (self.a-r)/eps ) ** (1/1024.0)
            penalty =  d / (1 - d )
            return lp - penalty
        elif r > self.b:
            # at x=b, we have 0
            # at x=b+eps
            d = ( (r-self.b)/eps ) ** (1/1024.0)
            penalty =  d / (1 - d )
            return lp - penalty

        return lp

    def deriv_log_p(self, x, cond, **kwargs):
        mu = cond[self.parent_key]
        r = x-mu

        eps=self.eps
        if r <= self.a-eps or r >= self.b +eps:
            return 0.0

        d_lp = -r/(self.std**2)

        if r < self.a:
            d = ( (self.a-r)/eps ) ** (1/1024.0)
            dd_dx = (1/1024.0) * ( (self.a-r)/eps )**(1/1024.0 - 1.0) * -1.0/eps
            penalty =  d / (1 - d )
            dpenalty_dd = 1.0/(1-d)**2
            dpenalty_dx = dpenalty_dd * dd_dx
            return d_lp - dpenalty_dx
        elif r > self.b:
            # at x=b, we have 0
            # at x=b+eps
            d = ( (r-self.b)/eps ) ** (1/1024.0)
            dd_dx = (1/1024.0) * ( (r-self.b)/eps )**(1/1024.0 - 1.0) * 1.0/eps
            penalty =  d / (1 - d )
            dpenalty_dd = 1.0/(1-d)**2
            dpenalty_dx = dpenalty_dd * dd_dx
            return d_lp - dpenalty_dx

        return d_lp

    def predict(self, cond, **kwargs):
        return cond[self.parent_key]

    def variance(self, **kwargs):
        return self.var

    def sample(self, cond, **kwargs):
        return cond[self.parent_key] + np.random.randn() * self.std

    def __str__(self):
        return "CondTruncatedGaussian(key=%s, std=%f)" % (self.parent_key, self.std)
