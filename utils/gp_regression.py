from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pickle

import numpy as np
import scipy.linalg
import hashlib

from scipy.spatial import KDTree

# cache weight vector for kernels on the same dataset


class GaussianProcess:

    def __init__(self, kernel, sigma2):
        self.kernel = kernel
        self.sigma2 = sigma2

    def train(self, X, y):
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.Ks = self.sigma2 * np.eye(self.n,self.n)
        for i in range(self.n):
            for j in range(self.n):
                self.Ks[i,j] += self.kernel(X[i, :], X[j, :])

        L = scipy.linalg.cholesky(self.Ks, lower=True)
        self.alpha = scipy.linalg.cho_solve((L, True), y)
        invL = scipy.linalg.inv(L)
        self.invKs = np.dot(invL.T, invL)

    def __kernel_vector(self, x):
        k = np.zeros((self.n,))
        for i in range(self.n):
            k[i] = self.kernel(x, self.X[i, :])
        return k

    def predict(self, x):
        return np.dot(self.alpha, self.__kernel_vector(x))

    def variance(self, x):
        k = self.__kernel_vector(x)
        return self.kernel(x,x) - np.dot(k.T, np.dot(self.invKs, k))

    def log_likelihood(self, x, y):
        k = self.__kernel_vector(x)
        mu = np.dot(self.alpha, k)
        var = self.kernel(x,x) - np.dot(k.T, np.dot(self.invKs, k))
        return -.5 * np.log(2*np.pi * var) - .5 * ((y - mu)**2 )/var



def plot_interpolated_surface(gp, X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xmin = np.min(X, 0)
    xmax = np.max(X, 0)

    u = np.linspace(xmin[0], xmax[0], 20)
    v = np.linspace(xmin[1], xmax[1], 20)

    xc = np.outer(np.ones((20,)), u)
    yc = np.outer(v, np.ones((20,)))

    k = np.zeros(xc.shape)
    for i in range(xc.shape[0]):
        for j in range(xc.shape[1]):
            k[i,j] = gp.predict((xc[i,j], yc[i,j]))

    #print xmin, xmax
    #print u, v
    #print x, y, k
    
    ax.plot_surface(xc, yc, k,  color='b')

    plt.show()

def main():
    X = np.array(((0,0), (0,1), (1,0), (1,1)))
    y = np.array((1, 50, 50, 15))


    kernel = lambda x, y : np.exp(-.5 * np.linalg.norm(x-y)**2)
    gp = GaussianProcess(kernel, 0.01)
    gp.train(X, y)
    print gp.predict((0,0))

    print pickle.dumps(gp)

#    plot_interpolated_surface(gp, X, y)


if __name__ == "__main__":
    main()
