import autograd.numpy as np
#import numpy as np
from autograd import grad

import scipy.stats
import scipy.optimize

import pdb, traceback, sys

def gaussian_kl(m1, C1, m2, C2):
    # http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence

    # TODO: should cache inv_C2 and logdetC2 since the prior doesn't change

    K = len(m1)

    r = m2-m1

    inv_C2 = np.linalg.inv(C2)
    tr_prod = np.sum(inv_C2*C1)

    quadratic = np.dot(r, np.dot(inv_C2, r))

    _, logdet1 = np.linalg.slogdet(C1)
    _, logdet2 = np.linalg.slogdet(C2)

    return .5*(tr_prod + quadratic - K + logdet2 - logdet1)

def gaussian_kl_factored(q_xmeans, q_xcovs, p_xmeans, p_xcovs):
    n = len(q_xmeans)
    kl = 0
    for i in range(n):
        kl += gaussian_kl(q_xmeans[i], q_xcovs[i], p_xmeans[i], p_xcovs[i])
    return kl

def flatten_cov(C):
    n = C.shape[0]
    flat = np.empty(((n*(n+1))/2))

    idx = 0
    for i in range(n):
        row = C[i, i:]
        flat[idx:idx+len(row)] = row
        idx += len(row)
    return flat

def unflatten_cov(flat):
    n = int((np.sqrt(1+8*flat.shape[0]) - 1)/2)
    C = np.empty((n,n))

    rows = []
    idx = 0
    for i in range(n):
        row = np.concatenate((np.zeros((i,)), flat[idx:idx+(n-i)]))
        rows.append(row)
        idx += n-i
    C = np.array(rows)

    C = C + C.T - np.diag(np.diag(C))
    return C

def flatten_cov_chol(C):
    n = C.shape[0]
    flat = np.empty(((n*(n+1))/2))
    L = np.linalg.cholesky(C)

    idx = 0
    for i in range(n):
        col = L[i:, i]
        flat[idx:idx+len(col)] = col
        idx += len(col)
    return flat

def unflatten_cov_chol(flat):
    n = int((np.sqrt(1+8*flat.shape[0]) - 1)/2)

    idx = 0
    rows = []
    for i in range(n):
        row = np.concatenate((np.zeros((i,)), flat[idx:idx+(n-i)]))
        rows.append(row)
        idx += n-i
    L = np.array(rows)
    C = np.dot(L, L.T)
    return C


class UncertainInputGP(object):
    # n: number of training points
    # m: number of inducing inputs
    # q: dimension of input space
    # p: dimension of output space

    def __init__(self, prior_xmeans, prior_xcovs, Y):
        # xmeans: list of n vectors, each of length q
        # xvars: list of cov matrices of size q * q
        # prior_xmeans: list of mean vectors
        # Y: matrix of size n x p, each row the output observed at a training point

        self.prior_xmeans = prior_xmeans
        self.prior_xcovs = prior_xcovs
        self.Y = Y

        # precompute quantities used in
        self.prior_cov_cache = []
        for C in self.prior_xcovs:
            self.prior_cov_cache.append((np.linalg.inv(C), np.linalg.slogdet(C)[1]))


    def kl_vs_prior(self, means, covs):

        n = len(means)
        kl = 0
        for i in range(n):
            m1, C1 = means[i], covs[i]
            m2 = self.prior_xmeans[i]
            inv_C2, logdet2 = self.prior_cov_cache[i]
            K = m1.shape[0]
            r = m2-m1
            tr_prod = np.sum(inv_C2*C1)
            quadratic = np.dot(r, np.dot(inv_C2, r))

            #
            #_, logdet1 = np.linalg.slogdet(C1)
            logdet1 = np.log(np.linalg.det(C1))

            kl += .5*(tr_prod + quadratic - K + logdet2 - logdet1)
        return kl


    def likelihood(self, X, theta, sigma2_n):
        n, p = self.Y.shape

        ll = 0
        for i, x in enumerate(X):
            ll += scipy.stats.multivariate_normal(mean=self.prior_xmeans[i], cov=self.prior_xcovs[i]).logpdf(x)

        K = cov_matrix(X, X, theta) + np.eye(n)*sigma2_n
        for y in self.Y.T:
            ll += scipy.stats.multivariate_normal(mean=np.zeros((n,)), cov=K).logpdf(y)
        return ll

    def mh_step(self, ll_old, X, theta, sigma2_n, std=0.4):
        r = np.random.randn(*X.shape) * std
        X_new = X+r
        ll_new = self.likelihood(X_new, theta, sigma2_n)
        u = np.random.rand()
        if ll_new-ll_old > np.log(u):
            return X_new, ll_new, True
        else:
            return X, ll_old, False

    def mh(self, X, theta, sigma2_n, steps=2000, std=0.4):
        lls = []
        Xs = []
        accepts = []
        cll = self.likelihood(X, theta, sigma2_n)
        for step in range(steps):
            X, cll, accept = self.mh_step(cll, X, theta, sigma2_n, std=std)
            Xs.append(X)
            lls.append(cll)
            accepts.append(accept)
            if step%50==0:
                print "step %d, accept rate %f, lp %f" % (step, np.mean(accepts), cll)
        return np.array(Xs), np.array(lls)

    def varobj_ll(self, Xu, q_xmeans, q_xcovs, theta, sigma2_n):
        # Xu: matrix of size m x q, each row the location of an inducing point

        # theta: set of hyperparams passed to the cov_matrix functions
        # sigma: observation noise variance

        sigma2_f = theta[0]
        n, p = self.Y.shape

        # NEED DERIVS wrt Xu and to theta, for all of these
        Kuu = cov_matrix(Xu, Xu, theta)
        psi0 = n*sigma2_f # expected_kernel_trace(q_xmeans, q_xcovs, theta)

        # also need derivs wrt the qs
        Psi1 = compute_Psi1(q_xmeans, q_xcovs, Xu, theta)
        Psi2 = compute_Psi2(q_xmeans, q_xcovs, Xu, theta)


        #Luu = np.linalg.cholesky(Kuu)
        #logdet_Kuu = 2*np.sum(np.log(np.diag(Luu)))
        # TODO: find a better differentiable logdet
        logdet_Kuu = np.log(np.linalg.det(Kuu))

        KuuInv = np.linalg.inv(Kuu)
        tr_KuuInv_Psi2 = np.sum(KuuInv * Psi2)

        Psi2_Kuu = 1./sigma2_n * Psi2 + Kuu


        #L1 = np.linalg.cholesky(Psi2_Kuu)
        Psi2_Kuu_inv = np.linalg.inv(Psi2_Kuu)
        #logdet_Psi2_Kuu = 2*np.sum(np.log(np.diag(L1)))
        logdet_Psi2_Kuu = np.log(np.linalg.det(Psi2_Kuu))

        F = -n/2.0 * np.log(sigma2_n) + .5*logdet_Kuu # numerator of eqn 24
        F -= -n/2.0 * np.log(2*np.pi) - .5*logdet_Psi2_Kuu # denominator of eqn 24
        F += 1.0/(2*sigma2_n) * (tr_KuuInv_Psi2 - psi0)
        F *= p # F currently contains all the non-j-dependent terms of eqn 24,
               # so we multiply by p to simulate the sum over j from eqn 29.

        for y in self.Y.T: # iterate over output dimensions (cols of Y)
            # L1 is the factorization of Psi2_Kuu
            v1 = np.dot(Psi1.T, y)
            v2 = np.dot(Psi2_Kuu_inv, v1)
            v3 = 1./sigma2_n**2 * np.dot(v1.T, v2)
            v = np.dot(y.T, y)/sigma2_n - v3

            F -= .5 * v

        # need deriv
        kl = self.kl_vs_prior(q_xmeans, q_xcovs)

        return F - kl

    def locate_inputs(self, Xu, theta, sigma2_n, x_init=None, init_std=None, **kwargs):
        # x_init: list of location vectors to initialize the X beliefs at
        #         (default is to use prior means)
        #         (variances are also assumed to be initialized at the prior)

        x_init = x_init if x_init is not None else self.prior_xmeans
        n = len(x_init)
        p = len(x_init[0])

        nu = Xu.shape[0]

        means = np.concatenate(x_init)

        init_covs = [np.eye(p)*init_std**2 for i in range(n)] \
                    if init_std is not None else self.prior_xcovs
        covs = np.concatenate([flatten_cov_chol(c) for c in init_covs])

        x0 = np.concatenate(( Xu.flatten(), means, covs))
        s = (p*(p+1))/2 # size of a flattened covariance matrix

        def unflatten(xparams):
            Xu = xparams[:nu*p]
            xus = [Xu[p*k:p*(k+1)] for k in range(nu)]
            Xu = np.array(xus).reshape((nu, p))

            means = xparams[nu*p:(n+nu)*p]
            xmeans = [means[p*k:p*(k+1)] for k in range(n)]

            covs = xparams[(n+nu)*p:]
            xcovs = [unflatten_cov_chol(covs[s*k:s*(k+1)]) for k in range(n)]
            return xmeans, xcovs, Xu

        def xnll(xparams):
            xmeans, xcovs, Xu = unflatten(xparams)
            return -self.varobj_ll(Xu, xmeans, xcovs, theta, sigma2_n)

        gll = grad(xnll)
        f = lambda xp: (xnll(xp), gll(xp))

        g1 = gll(x0)

        # should be [ -6.11907326  10.8705521  -14.56096521  10.27840101   6.22187982    4.31318128]
        g2 = scipy.optimize.approx_fprime(x0, xnll, 1e-6)

        print g2
        print g1
        #print g2-g1
        #print np.max(np.abs(g2-g1))


        r = scipy.optimize.minimize(f, x0, jac=True, **kwargs)
        xmeans, xcovs, Xu = unflatten(r.x)

        #xmeans, xcovs = self.prior_xmeans, self.prior_xcovs

        return xmeans, xcovs, Xu


def cov_matrix(X1, X2, theta):
    # SE cov with ARD, theta specifies variance and lengthscales (diagonals of W matrix)
    n1, d = X1.shape
    n2, d = X2.shape
    assert(len(theta)==d+1)
    s2 = theta[0]
    lscales = theta[1:]

    K = np.zeros((n1, n2))
    rows = []
    for i in range(n1):
        row = []
        for j in range(n2):
            diff = X1[i,:] - X2[j,:]
            sdiff = np.sum(diff**2 / lscales)
            row.append(np.exp(-.5*sdiff))
        rows.append(row)
    K = np.array(rows)
    return s2*K


def compute_Psi1(q_xmeans, q_xcovs, Xu, theta):
    sigma2_f = theta[0]
    lscales = theta[1:]

    n = len(q_xmeans)
    m, q = Xu.shape

    #Psi1 = np.zeros((n, m))

    # uses eqn (7) from Dallaire et al.,
    # "Learning Gaussian Process Models from Uncertain Data"
    # or equiv, "computing psi1" from my note
    rows = []
    for i in range(n):
        mu = q_xmeans[i]
        S = q_xcovs[i]
        Winv_S = (1.0/lscales*S.T).T
        normalizer = sigma2_f / np.sqrt(np.linalg.det(np.eye(q) + Winv_S))
        inv_WS = np.linalg.inv(np.diag(lscales) + S)
        row = []
        for k in range(m):
            xu = Xu[k,:]
            r = xu-mu
            qq = -.5*np.dot(r, np.dot(inv_WS, r))
            v = np.exp(qq) * normalizer
            row.append(v)
        rows.append(row)
    Psi1 = np.array(rows)
    return Psi1

def compute_Psi2(q_xmeans, q_xcovs, Xu, theta):
    sigma2_f = theta[0]
    lscales = theta[1:]

    n = len(q_xmeans)
    m, q = Xu.shape

    Psi2 = np.zeros((m,m))
    for i in range(n):
        mu = q_xmeans[i]
        S = q_xcovs[i]
        Winv_S = (1.0/lscales*2*S.T).T
        normalizer = sigma2_f**2 / np.sqrt(np.linalg.det(np.eye(q) + Winv_S))
        inv_WS = np.linalg.inv(np.diag(lscales/2) + S)

        rows = []
        for k in range(m):
            row = []
            for kp in range(m):
                xu1 = Xu[k,:]
                xu2 = Xu[kp,:]
                xbar = (xu1+xu2)/2.0

                qq1 = -.25*np.sum((xu1-xu2)**2/lscales)

                r = mu-xbar
                qq2 = -.5*np.dot(r, np.dot(inv_WS, r))
                row.append(np.exp(qq1+qq2) * normalizer)
                #Psi2[k,kp] +=
            rows.append(row)
        P2 = np.array(rows)
        Psi2 = Psi2 + P2

    return Psi2



def test():
    p_means = [np.array((-1.0,)), np.array((0.0,)), np.array((1.0,))]
    p_covs = [np.array(((1.0,),)), np.array(((1.0,),)), np.array(((1.0,),))]
    theta = np.array([1.0, 1.0])

    Y = np.array(((4,), (4,), (0,)))

    Xu = np.array((-1, 0, 1)).reshape((-1, 1))

    sigma2_n=0.1

    uigp = UncertainInputGP(p_means, p_covs, Y)
    xm, xv = uigp.locate_inputs(Xu, theta, sigma2_n)


if __name__ == "__main__":
    try:
        test()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
