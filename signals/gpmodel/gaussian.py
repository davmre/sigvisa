import numpy as np
import matplotlib.pyplot as plt

"""
domain: lists of (multi-dimensional) input points
params: len_scale, var_signal, var_noise

self.domain: 2D array where points are indexed by row
self.X: 2D array where points are index by col (used for matrix calc)
self.l: length scale
self.vs: variance of signal
self.vn: variance of noise
"""
class GPModel:
    def __init__(self, domain, params=None):
        domain = np.array(domain)
        dim = len(domain.shape)
        if dim == 0:
            self.X = np.array([[domain]])
        elif dim == 1:
            self.X = np.array([domain])
        elif dim == 2:
            self.X = np.transpose(domain)
        else:
            raise Exception("Inputs need to be 2D or less.")
        self.domain = np.transpose(self.X)
        self.n, self.xd = self.domain.shape # n=number of points, xd=in dim
        if params == None:
            self.l = 0.8
            self.vs = 1
            self.vn = 0.01
        else:
            self.l, self.vs, self.vn = params
        self.K = self.covmat(self.X, self.X, same=True)

    def covvec(self, x1, x2, same=False):
        if x1.shape != x2.shape:
            raise Exception("Vectors need to have the same dimension.")

        n = np.size(x1,0)
        M = 1/np.square(self.l)*np.eye(n)
        t1 = -0.5*np.dot(np.transpose(x1-x2), np.dot(M, x1-x2))
        t2 = 0
        if same:
            t2 = self.vn
        return self.vs*np.exp(t1)+t2

    def covmat(self, X1, X2, same=False):
        if np.size(X1,0,) != np.size(X2,0):
            raise Exception("Matrices need to have the same input dimension.")

        n1 = np.size(X1, 1)
        n2 = np.size(X2, 1)
        K = np.zeros([n1, n2])
        for i in range(n1):
            for j in range(n2):
                K[i][j] = self.covvec(X1[:,i:i+1], X2[:,j:j+1], same and i==j)
        return K

    def cov(self, x):
        return self.covmat(self.X, x)

    def sample(self, means=None):
        d = np.size(self.X,1)
        if means == None:
            means = np.zeros(d)
        return np.random.multivariate_normal(means, self.K)

"""
gpm: a GPModel instance to be learned
range: array of 1D output points

y: column vector of output points for calculation.
"""
class GPLearner:
    def __init__(self, gpm, range):
        self.gpm = gpm
        self.X = gpm.X
        range = np.array(range)
        dim = len(range.shape)
        if dim == 0:
            self.y = np.array([[range]])
        elif dim == 1:
            self.y = np.transpose(np.array([range]))

        if gpm.n != np.size(self.y, 0):
            print gpm.domain
            print np.size(self.y,0)
            raise Exception("Number of training points mismatch.")

        self.K = gpm.K
        self.l = gpm.l
        self.vs = gpm.vs
        self.vn = gpm.vn
        self.n = gpm.n

    """
    calculate predicted output means and variances for each test input points

    tp: test points indexed by row
    """
    def predict(self, tp):
        tp = np.array(tp)
        dim = len(tp.shape)
        xs = None
        if dim == 0:
            xs = np.array([[tp]])
        elif dim == 1:
            xs = np.array([tp])
        elif dim == 2:
            xs = np.transpose(tp)
        else:
            raise Exception("test points need to be 2D at most.")
        xd, m = xs.shape

        if xd != self.gpm.xd:
            raise Exception("input dimension mismatch.")

        # misc. values
        L = np.linalg.cholesky(self.K+self.vn*np.eye(self.n))
        L_inv = np.linalg.inv(L)
        L_t_inv = np.linalg.inv(np.transpose(L))
        a = np.dot(L_t_inv,np.dot(L_inv,self.y))

        # log(p(y|X))
        logp1 = -0.5*np.dot(np.transpose(self.y),a)-(self.n/2)*np.log(2*np.pi)
        logp2 = np.sum([L[i][i] for i in range(self.n)])
        logp = logp1-logp2

        # means and variances of xs
        f_bars = np.zeros(m)
        vars = np.zeros(m)
        for i in range(m):
            x = xs[:,i:i+1]
            k = self.gpm.cov(x)
            f_bar = np.dot(np.transpose(k), a)
            v = np.dot(L_inv, k)
            var = self.gpm.covvec(x,x, same=True)-np.dot(np.transpose(v),v)
            f_bars[i] = f_bar
            vars[i] = var

        return (f_bars, vars, logp)

    def lklhood(self, tp, target):
        n = len(tp)
        pd = np.zeros(n)



        def single_lklhood(mean, var, x):
            c = 1/(np.sqrt(2*np.pi*var))
            e = -0.5*np.square(x-mean)/var
            return c*np.exp(e)

        f_bars, vars, logp = self.predict(tp)


        for i in range(n):
            pd[i] = single_lklhood(f_bars[i], vars[i], target)

        return pd
