from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import hashlib

from scipy.spatial import KDTree

# cache weight vector for kernels on the same dataset

cache = dict()

def knn_predict(x, X, y, k):
    (n, d) = X.shape

    tree = KDTree(X)
    closest_d, closest_i = tree.query(x, k=k)

    pred = np.mean(y[closest_i])
    return pred

# Computes Watson-Nadaraya kernel smoothing estimate
# x: d-dimensional vector representing a data point
# X: n x d matrix with rows giving the n training inputs
# Y: n-dimensional vector with rows giving the n training outputs
# kernel: a function taking a tuple of two d-dimensional vectors. Default is a Gaussian kernel with width chosen by Silverman's rule (avg distance of 3 nearest neighbors).
def kernel_predict(x, X, y, kernel = None):
    (n, d) = X.shape


    if kernel is not None:
        k = np.array([kernel((X[i, :], x)) for i in range(n)])
    else:
        tree = KDTree(X)

        # compute (or load from cache) kernel width at each point
        Xhash = hashlib.sha1(X.view(np.uint8)) 
        if Xhash in cache:
            w = cache[Xhash]

        else:
            w = np.zeros((n, 1))
            for i in range(n):
                closest_d, closest_i = tree.query(x, k=3)
                w[i] = np.mean(closest_d)
            cache[Xhash] = w

        # evaluate the kernel
        k = np.zeros((n, 1))
        for i in range(n):
            k[i] = 1/np.sqrt(2*np.pi*w[i]) * np.exp(-.5 * np.linalg.norm(x - X[i,:])**2 / w[i]**2)

    pred = np.dot(k.T, y) / np.sum(k)
    return pred

def plot_interpolated_surface(X, y):
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
            k[i,j] = knn_predict((xc[i,j], yc[i,j]), X, y, 3)

    #print xmin, xmax
    #print u, v
    #print x, y, k
    

    ax.plot_surface(xc, yc, k,  color='b')

    plt.show()

def main():
    X = np.array(((0,0), (0,1), (1,0), (1,1)))
    y = np.array((1, 50, 50, 15))
    
    print kernel_predict((0,0), X, y)

    plot_interpolated_surface(X, y)


if __name__ == "__main__":
    main()
