import numpy as np
import scipy
import scipy.stats as stats
import scipy.linalg
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

    def log_p(self, x, **kwargs):
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

    def sample(self):
        s = scipy.stats.invgamma.rvs(a=self.alpha, size=1)
        return s * self.beta

    def variance(self):
        assert (self.alpha > 2)
        return self.beta**2 / ((self.alpha-1)**2 * (self.alpha - 2))

class LogNormal(Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def log_p(self, x, **kwargs):
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

    def sample(self):
        return np.exp(np.random.randn() * self.sigma + self.mu)

    def variance(self):
        return (np.exp(self.sigma**2) - 1 ) * ( np.exp(2*self.mu + self.sigma**2))

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
        self.var = std**2

    def log_p(self, x,  **kwargs):
        mu = self.mean
        sigma2 = self.var
        lp = -.5 * np.log(2*np.pi*sigma2) - .5 * (x - mu)**2 / sigma2
        if sigma2==0:
            lp = np.float("-inf")
        return lp

    def deriv_log_p(self, x, **kwargs):
        return -(x - self.mean)/self.var

    def entropy(self):
        return .5 * np.log(2*np.pi*np.e*self.var)

    def predict(self, *args, **kwargs):
        return self.mean

    def variance(self, *args, **kwargs):
        return self.var

    def sample(self, *args, **kwargs):
        return self.mean + np.random.randn() * self.std

    def product(self, other):
        # given two Gaussian distributions, return
        # the Gaussian resulting from normalizing the product
        # N(x; u1, v1) N(x; u2, v2).
        # This corresponds to computing a Bayesian posterior
        # given a Gaussian prior and Gaussian likelihood.

        prec1 = 1.0/self.var
        prec2 = 1.0/other.var
        prec = prec1+prec2

        mean = (prec1*self.mean + prec2*other.mean)/prec
        return Gaussian(mean, 1.0/np.sqrt(prec))

    def __str__(self):
        return "Gaussian(mean=%f, std=%f)" % (self.mean, self.std)

class TruncatedGaussian(Distribution):
    def __init__(self, mean, std, a=-np.inf, b=np.inf, eps=1e-2):
        self.mean = mean
        self.std = std
        self.a=a
        self.b=b
        self.eps = eps

        self.Z = .5 * (erf((self.b-mean)/(std*np.sqrt(2))) -  erf((self.a-mean)/(std*np.sqrt(2))))
        self.logZ = np.log(self.Z)

    def log_p(self, x,  **kwargs):
        eps = self.eps
        if x <= self.a-eps or x >= self.b +eps:
            return -np.inf

        mu = self.mean
        sigma = self.std
        lp = -.5 * np.log(2*np.pi*sigma*sigma) - .5 * (x - mu)**2 / sigma**2 - self.logZ
        if np.isnan(lp):
            lp = np.float("-inf")

        if x < self.a:

            d = ( (self.a-x)/eps ) ** (1/1024.0)
            penalty =  d / (1 - d )
            return lp - penalty
        elif x > self.b:
            # at x=b, we have 0
            # at x=b+eps
            d = ( (x-self.b)/eps ) ** (1/1024.0)
            penalty =  d / (1 - d )
            return lp - penalty


        return lp

    def deriv_log_p(self, x, **kwargs):
        eps=self.eps
        if x <= self.a-eps or x >= self.b +eps:
            return 0.0

        d_lp = -(x - self.mean)/(self.std**2)

        if x < self.a:
            d = ( (self.a-x)/eps ) ** (1/1024.0)
            dd_dx = (1/1024.0) * ( (self.a-x)/eps )**(1/1024.0 - 1.0) * -1.0/eps
            penalty =  d / (1 - d )
            dpenalty_dd = 1.0/(1-d)**2
            dpenalty_dx = dpenalty_dd * dd_dx
            return d_lp - dpenalty_dx
        elif x > self.b:
            # at x=b, we have 0
            # at x=b+eps
            d = ( (x-self.b)/eps ) ** (1/1024.0)
            dd_dx = (1/1024.0) * ( (x-self.b)/eps )**(1/1024.0 - 1.0) * 1.0/eps
            penalty =  d / (1 - d )
            dpenalty_dd = 1.0/(1-d)**2
            dpenalty_dx = dpenalty_dd * dd_dx
            return d_lp - dpenalty_dx


        return d_lp

    def predict(self, **kwargs):
        return self.mean

    def sample(self, **kwargs):

        # just do rejection sampling since in practice I only plan to
        # cut off very unlikely values.
        sample = self.a
        while sample <= self.a or sample >= self.b:
            sample = self.mean + np.random.randn() * self.std
        return sample

    def variance(self, **kwargs):
        return self.std**2

class MultiGaussian(Distribution):
    def __init__(self, mean, cov, pre_inv=False):
        self.mean = mean
        self.n_d = len(mean)
        self.cov = cov
        self.L = scipy.linalg.cholesky(cov, True)

        if pre_inv:
            self.invL = np.linalg.inv(self.L)
        else:
            self.invL = None

    def predict(self, **kwargs):
        return self.mean.copy()

    def log_p(self, x, **kwargs):
        assert(len(x) == self.n_d)
        r = x-self.mean

        if self.invL is not None:
            l = np.dot(self.invL, r)
            rr = np.dot(l.T, l)
        else:
            tmp = scipy.linalg.cho_solve((self.L, True), r)
            rr = np.dot(r.T, tmp)

        logdet2 = np.log(np.diag(self.L)).sum()
        ll =  -.5 * (rr + len(self.mean) * np.log(2*np.pi)) - logdet2
        return ll

    def sample(self):
        n = len(self.mean)
        z = np.random.randn(n)
        return np.dot(self.L, z) + self.mean

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

    def variance(self, **kwargs):
        return 2*self.scale**2


class Exponential(Distribution):
    def __init__(self, rate, min_value=0.0):
        self.rate = float(rate)
        self.min_value = min_value

    def log_p(self, x,  **kwargs):
        rate = self.rate
        x = x - self.min_value
        if x < 0:
            lp = np.float("-inf")
        else:
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

class Beta(Distribution):
    def __init__(self, alpha, beta):
        assert(alpha >= 0 and beta >= 0)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def log_p(self, x, **kwargs):
        if x <= 0 or x >= 1:
            return -np.inf
        alpha , beta = self.alpha, self.beta

        lp = (alpha-1)*np.log(x) + (beta-1)*np.log(1-x) - scipy.special.betaln(alpha, beta)
        return lp

    def deriv_log_p(self, x, *args, **kwargs):
        return (self.alpha-1.) / x + (self.beta-1.) / (x-1.)

    def entropy(self):
        alpha, beta = self.alpha, self.beta

        B = scipy.special.betaln(alpha, beta)
        dga = scipy.special.digamma(alpha)
        dgb = scipy.special.digamma(beta)
        dgab = scipy.special.digamma(alpha+beta)

        entropy = B - (alpha-1)*dga - (beta-1)*dgb - (alpha+beta-2)*dgab
        return entropy

    def predict(self, *args, **kwargs):
        return self.alpha/(self.alpha+self.beta)

    def sample(self, *args, **kwargs):
        return scipy.stats.beta(self.alpha, self.beta, loc=0, scale=1).rvs(1)[0]

    def variance(self, **kwargs):
        alpha, beta = self.alpha, self.beta
        return (alpha*beta)/((alpha+beta)**2 * (alpha+beta+1))

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

class PiecewiseLinear(Distribution):
    def __init__(self, xs, lps, mix_dist=None, mix_weight=0.0):
        self.mix_dist = mix_dist
        self.mix_weight = mix_weight
        self.ps = np.exp(lps - np.max(lps))
        self.xs = xs
        self._normalize()

    def _normalize(self):
        pts = len(self.xs)
        areas = np.empty((pts-1,))
        for i in range(pts-1):
            gap = self.xs[i+1]-self.xs[i]
            areas[i] = gap * (self.ps[i+1]+self.ps[i])/2.0
        Z = np.sum(areas)

        if Z == 0 or not np.isfinite(Z):
            import pdb; pdb.set_trace()

        self.areas = areas / Z
        self.ps /= Z

    def log_p(self, x, *args, **kwargs):
        idx = np.searchsorted(self.xs, x)
        if idx == 0:
            p = 0
        elif idx==len(self.xs):
            p = 0
        else:
            p1 = self.ps[idx-1]
            p2 = self.ps[idx]
            z = x-self.xs[idx-1]
            gap = self.xs[idx]-self.xs[idx-1]

            p = (1.0 - z/gap) * p1 + (z/gap)*p2

        if self.mix_weight > 0:
            mix_p = np.exp(self.mix_dist.log_p(x))
            p = p*(1.0-self.mix_weight) + self.mix_weight * mix_p

        lp = np.log(p)
        #assert(np.isfinite(lp))
        return lp

    def sample(self, *args, **kwargs):
        if self.mix_weight > 0:
            u = np.random.rand()
            if u < self.mix_weight:
                return self.mix_dist.sample(*args, **kwargs)

        pts = len(self.xs)
        segment = np.random.choice(np.arange((pts-1),), p=self.areas)
        area = self.areas[segment]
        gap = self.xs[segment+1] - self.xs[segment]
        p1 = self.ps[segment]
        p2 = self.ps[segment+1]

        u = np.random.rand() * area
        if np.abs(p1-p2) < 1e-10:
            z = (u/area) * gap
        else:
            z = (np.sqrt(p1**2 * gap**2 + 2*(p2-p1)*u*gap) - p1*gap) / (p2-p1)

        if np.isnan(z):
            import pdb; pdb.set_trace()
        
        if z > gap:
            import pdb; pdb.set_trace()

        x =  self.xs[segment] + z
        return x

