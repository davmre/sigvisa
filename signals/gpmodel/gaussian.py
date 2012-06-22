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
    def __init__(self, domain, params):
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
    

"""
def cov(x1, x2, params, same):
    len_scale, var_signal, var_noise = params
    n = np.size(x1,0)
    M = 1/np.square(len_scale)*np.eye(n)
    t1 = -0.5*np.dot(np.transpose(x1-x2), np.dot(M, x1-x2))
    t2 = 0
    if same:
        t2 = var_noise
    return var_signal*np.exp(t1)+t2

def covmat(X1, X2, params, same):
    n1 = np.size(X1, 1)
    n2 = np.size(X2, 1)
    K = np.zeros([n1, n2])
    for i in range(n1):
        for j in range(n2):
            K[i][j] = cov(X1[:,i:i+1], X2[:,j:j+1], params, same and i==j)
    return K
"""
"""
Gaussian process of d-dimensional input space, n training points, then:
X is d x n (input space x number of training points)
y is n x 1 (outputs for n training points)
var_noise is the variance of the noise
x is d x 1 test point
"""

"""
def gp(X, y, params, xs):
    assert np.size(X,0) == np.size(xs, 0)  
    dim = np.size(X,0)
    
    # misc. values
    len_scale, var_signal, var_noise = params
    n = np.size(X,1) # number of training points
    K = covmat(X,X, params, True)
    L = np.linalg.cholesky(K+var_noise*np.eye(n))
    L_inv = np.linalg.inv(L)
    L_t_inv = np.linalg.inv(np.transpose(L))
    a = np.dot(L_t_inv,np.dot(L_inv,y))

    # log(p(y|X)) 
    logp1 = -0.5*np.dot(np.transpose(y),a)-(n/2)*np.log(2*np.pi)
    logp2 = np.sum([L[i][i] for i in range(n)])
    logp = logp1-logp2

    # means and variances of xs
    m = np.size(xs, 1) # number of test points
    f_bars = np.zeros(m)
    vars = np.zeros(m)
    for i in range(m):
        x = xs[:,i:i+1]
        k = covmat(X,x, params, False)
        f_bar = np.dot(np.transpose(k), a)
        v = np.dot(L_inv, k)
        var = cov(x,x, params, True)-np.dot(np.transpose(v),v)
        f_bars[i] = f_bar
        vars[i] = var
    
    return (f_bars, vars, logp)
"""

"""
initially intended for multi-dimensional f_bars, but apparently
f_bars is always 1D... Still works though.
"""
"""
def lklhood(f_bars, vars, target):
    dim = f_bars.shape
    out = np.zeros(dim)
    
    # precondition: var > 0
    def single_lklhood(mean, var, x):
        c = 1/(np.sqrt(2*np.pi*var))
        e = -0.5*np.square(x-mean)/var
        return c*np.exp(e)

    def iterate(dim_i, addr):
        if dim_i == len(dim):
            mean = f_bars[tuple(addr)]
            var = vars[tuple(addr)]
            if var == 0:
                out[tuple(addr)] = 0
            else:
                out[tuple(addr)] = single_lklhood(mean, var, target)
        else:
            for i in range(dim[dim_i]):
                iterate(dim_i+1, np.append(addr, i))
    iterate(0,[])
    return out
"""
"""
X = d x n matrix then
sample a point from n-dimensional Gaussian distribution
"""
"""
def sample(X, params, mean=0):
    d = np.size(X,1)
    C = covmat(X, X, params, True)
    means = mean*np.ones(d)
    return np.random.multivariate_normal(means, C)

def heatmap2(X, y, res=100):
    d,N = np.shape(X)
    X_t = np.transpose(X)
    x1s = np.zeros(N)
    x2s = np.zeros(N)
    for i in range(N):
        x1s[i], x2s[i] = X_t[i]
    table, x1edges, x2edges = np.histogram2d(x1s, x2s, bins=res)
    extent = [x1edges[0], x1edges[-1], x2edges[0], x2edges[-1]]
    plt.clf()
    print table
    plt.imshow(table, extent=extent, interpolation='none')

def getgrid(x1range, x2range, res1=100, res2=100):
    x1grid = np.linspace(x1range[0],x1range[1],res1)
    x2grid = np.linspace(x2range[0],x2range[1],res2)
    return np.transpose([[x1,x2] for x1 in x1grid for x2 in x2grid])

def heatmap(X, y, params, x1range, x2range, res1=100, res2=100, target=None):
    x1grid = np.linspace(x1range[0],x1range[1],res1)
    x2grid = np.linspace(x2range[0],x2range[1],res2)
    list = [(x1,x2) for x1 in x1grid for x2 in x2grid]
    x1ind = range(res1)
    x2ind = range(res2)
    indices = [(x1,x2) for x1 in x1ind for x2 in x2ind]
    xs = np.transpose(list)
    f_bars, vars, logp = gp(X, y, params, xs)
    
    result = np.zeros([res1,res2])
    for i in range(len(indices)):
        x1,x2 = indices[i]
        result[x1][x2] = f_bars[i]
    plt.imshow(result, interpolation='none')
    plt.show()
    if target != None:
        p = lklhood(f_bars,vars, target)
        ptable = np.zeros([res1,res2])
        for i in range(len(indices)):
            x1,x2 = indices[i]
            ptable[x1][x2] = p[i]
        plt.imshow(ptable, interpolation='none')
        plt.show()

len_scale = 0.5
var_signal = 0.8
var_noise = 0.01
params = len_scale, var_signal, var_noise
"""
"""
X = np.array([[0,0,1,1],[0,1,0,1]])
y = np.transpose(np.array([[0,10,3,2]]))
"""

"""
N = 101
grid = np.linspace(0,1,N)
X_t = [[x1] for x1 in grid]
X = np.transpose(X_t)
y = sample(X, params)
test_i = int(N*(np.random.rand()))
testy = np.transpose([y[test_i]])
testx = np.array([X_t[test_i]])
trainy = np.transpose([np.append(y[:test_i], y[test_i+1:])])
trainx = np.array([np.append(X_t[:test_i], X_t[test_i+1:])])
f_bar, var, logp = gp(trainx, trainy, params, testx)

plt.plot(X[0], y)
plt.plot(testx, f_bar, 'o')
plt.show()

N2 = 1001
grid2 = np.linspace(0,1,N2)
xs_t = [[x1] for x1 in grid2]
xs = np.transpose(xs_t)
f_bars, vars, logp = gp(trainx, trainy, params, xs)
ps = lklhood(f_bars, vars, testy)
plt.plot(xs[0], ps)
plt.plot(testx, ps[test_i*(N2-1)/(N-1)], 'o')
plt.show()
"""

"""
N = 11
grid = np.linspace(0,1,N)
X_t = [[x1,x2] for x1 in grid for x2 in grid]
X = np.transpose(X_t)
y = sample(X, params)
test_i = int(N*N*(np.random.rand()))
testy = np.transpose([y[test_i]])
testx = np.array([X_t[test_i]])
trainy = np.transpose([np.append(y[:test_i], y[test_i+1:])])
trainx = np.array([np.append(X_t[:test_i], X_t[test_i+1:])])
f_bar, var, logp = gp(trainx, trainy, params, testx)

plt.plot(X[0], y)
plt.plot(testx, f_bar, 'o')
plt.show()
"""

#heatmap(X,y,params,[-1,2],[-1,2],target=1)