import numpy as np
import scipy
import scipy.stats as stats
from scipy.misc import factorial
from scipy.special import erf
from sigvisa.models import Distribution

class Gamma(Distribution):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def log_p(self, x, **kwargs):
        alpha = self.alpha
        beta = self.beta

        if x < 0.0: return np.log(1e-300)
        # the special case of an exponential distribution is defined even when x==0
        if alpha == 1: return np.log(beta) - beta*x
        if x == 0.0: return np.log(1e-300)
        lp = alpha*np.log(beta) - scipy.special.gammaln(alpha) + (alpha-1)*np.log(x) - beta*x
        if np.isnan(lp):
            lp = np.float("-inf")
        return lp

    def deriv_log_p(self, x):
        alpha = self.alpha
        beta = self.beta
        if x == 0.0: return 1000.0
        return (alpha-1)/x - beta

    def predict(self, **kwargs):
        return self.alpha/self.beta

    def sample(self, **kwargs):
        return stats.gamma.rvs(self.alpha, scale=1.0/self.beta, loc=0., n=1)

class InvGamma(Distribution):
    # mean: beta/(alpha-1)
    # mode: beta/(alpha+1)
    # variance: beta^2 / ( (alpha-1)^2 (alpha-2) )

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def predict(self):
        return self.beta / (self.alpha+1) # return the mode, since the mean isn't always defined

    def log_p(self, x):
        alpha = self.alpha
        beta = self.beta
        if alpha <= 0 or beta <= 0:
            return np.float("-inf")
        if x == 0.0: return np.log(1e-300)
        lp = alpha*np.log(beta) - scipy.special.gammaln(alpha) - (alpha+1)*np.log(x) - beta/x
        if np.isnan(lp):
            lp = np.float("-inf")

        return lp

    def deriv_log_p(self, x):
        alpha = self.alpha
        beta = self.beta
        if x == 0.0: return 1000.0
        return beta/(x**2) - (alpha+1)/x

class LogNormal(Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def log_p(self, x):
        mu = self.mu
        sigma = self.sigma
        if x == 0.0: return np.log(1e-300)
        lp = -1 * np.log(x) - .5 * np.log(2*np.pi) - np.log(sigma) - .5 * (np.log(x) - mu)**2 / sigma**2
        if np.isnan(lp):
            lp = np.float("-inf")
        return lp

    def deriv_log_p(self, x):
        mu = self.mu
        sigma = self.sigma
        if x == 0.0: return 1000.0
        return (-1 -(np.log(x) - mu)/(sigma**2)) / x

    def predict(self):
        return np.exp(self.mu)

class Uniform(Distribution):
    def __init__(self, lbound, rbound):
        assert ( lbound < rbound)
        self.lbound = lbound
        self.rbound = rbound

    def log_p(self, x,  **kwargs):
        if self.lbound <= x <= self.rbound:
            return -np.log(self.rbound - self.lbound)
        else:
            return np.float("-inf")

    def predict(self, **kwargs):
        return self.lbound + (self.rbound - self.lbound) / 2

    def sample(self, **kwargs):
        return self.lbound + np.random.rand() * (self.rbound - self.lbound)

class Gaussian(Distribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def log_p(self, x,  **kwargs):
        mu = self.mean
        sigma = self.std
        lp = -.5 * np.log(2*np.pi*sigma*sigma) - .5 * (x - mu)**2 / sigma**2
        if np.isnan(lp):
            lp = np.float("-inf")
        return lp

    def predict(self, **kwargs):
        return self.mean

    def sample(self, **kwargs):
        return self.mean + np.random.randn() * self.std

    def __str__(self):
        return "Gaussian(mean=%f, std=%f)" % (self.mean, self.std)

class TruncatedGaussian(Distribution):
    def __init__(self, mean, std, a=-np.inf, b=np.inf):
        self.mean = mean
        self.std = std
        self.a=a
        self.b=b

        self.Z = .5 * (erf((self.b-mean)/(std*np.sqrt(2))) -  erf((self.a-mean)/(std*np.sqrt(2))))
        self.logZ = np.log(self.Z)

    def log_p(self, x,  **kwargs):
        if x < self.a or x > self.b:
            return -np.inf

        mu = self.mean
        sigma = self.std
        lp = -.5 * np.log(2*np.pi*sigma*sigma) - .5 * (x - mu)**2 / sigma**2 - self.logZ
        if np.isnan(lp):
            lp = np.float("-inf")
        return lp

    def predict(self, **kwargs):
        return self.mean

    def sample(self, **kwargs):

        # just do rejection sampling since in practice I only plan to
        # cut off very unlikely values.
        sample = self.a
        while sample <= self.a or sample >= self.b:
            sample = self.mean + np.random.randn() * self.std
        return sample


class Laplacian(Distribution):
    def __init__(self, center, scale):
        self.center = center
        self.scale = scale

    def log_p(self, x,  **kwargs):
        center = self.center
        scale = self.scale
        lp = -np.log(2*scale) - np.abs(x-center)/scale
        if np.isnan(lp):
            lp = np.float("-inf")
        return lp

    def predict(self, **kwargs):
        return self.center

    def sample(self, **kwargs):
        u = np.random.rand() - .5
        return self.center - self.scale * np.sign(u) * np.log(1-2*np.abs(u))



class Exponential(Distribution):
    def __init__(self, rate, min_value=0.0):
        self.rate = float(rate)
        self.min_value = min_value

    def log_p(self, x,  **kwargs):
        rate = self.rate
        x = x - self.min_value
        if x < 0:
            lp = np.float("-inf")

        lp = np.log(rate) - rate * x

        return lp

    def predict(self, **kwargs):
        return 1.0/self.rate + self.min_value

    def sample(self, **kwargs):
        u = np.random.rand()
        return -np.log(u) / self.rate + self.min_value



class Poisson(Distribution):
    def __init__(self, mu):
        self.mu = mu

    def log_p(self, x, **kwargs):
        return -self.mu  + x * np.log(self.mu) - scipy.special.gammaln(x+1)

    def predict(self, **kwargs):
        return self.mu

    def sample(self, **kwargs):
        return stats.poisson.rvs(self.mu)

class Bernoulli(Distribution):

    def __init__(self, p):
        self.p = p

    def log_p(self, x, **kwargs):
        if x:
            return np.log(self.p)
        else:
            return np.log(1-self.p)

    def predict(self, **kwargs):
        return self.p >= .5

    def sample(self, **kwargs):
        u = np.random.rand()
        return u < self.p

class Negate(Distribution):
    def __init__(self, dist):
        self.dist = dist

    def dim(self, *args, **kwargs):
        return self.dist.dim( *args, **kwargs)

    def predict(self, *args, **kwargs):
        return -1 * self.dist.predict(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return -1 * self.dist.sample(*args, **kwargs)

    def log_p(self, x, *args, **kwargs):
        return self.dist.log_p(-x, *args, **kwargs)

    def deriv_log_p(self, x, *args, **kwargs):
        return self.dist.deriv_log_p(-x, *args, **kwargs)
