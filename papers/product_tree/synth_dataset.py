import os, errno
import sys
import numpy as np
import scipy.linalg
import matplotlib
import itertools
import time
from scikits.sparse.cholmod import cholesky as sp_cholesky
import scipy.sparse

from sigvisa.models.spatial_regression.SparseGP import SparseGP, prior_sample

basedir = os.path.join(os.getenv("SIGVISA_HOME"), "papers", "product_tree", "benchmarks")
#lscales = (0.00005, 0.0005, 0.005, 0.01, 0.02)
lscales = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5, 10)
#cluster_ns = (50, 100, 200, 400, 1000, 1500, 2000, 2500, 3000, 4000, 8000)
#cluster_ns = (50, 100, 200, 400)
#cluster_ns = (8000,)

#Ns = (1000, 2000, 4000, 8000, 20000, 40000, 80000, 160000) #,50000, 60000, 80000, 160000)#
Ns = (1000, 2000, 4000, 8000,16000, 24000, 32000, 48000, 64000,) #,50000, 60000, 80000, 160000)
#Ns = (20000, 40000, 60000, 80000)
#Ns = (4000,)

wfn_str = "se"

def genX_clusters(dim, n_clusters, cluster_pts, cluster_width):
    centers = [ np.array([np.random.rand() for d in range(dim)]) for i in range(n_clusters)] # generate n_clusters points within the unit cube of dimension dim
    pts = [  c +  np.array([np.random.randn() * cluster_width for d in range(dim)])  for (c,i) in itertools.product(centers, range(cluster_pts))]
    return np.array(pts)

def sort_events(X):
    X_sorted = np.array(sorted(X, key = lambda x: x[0]), dtype=float)
    return X_sorted

def genX(dim, npts):
    pts = np.array([[np.random.rand() for d in range(dim)] for i in range(npts)]) # generate n_clusters points within the unit cube of dimension dim
    return pts


def eval_gp(bdir=None, gp=None, testX=None, resultfile=None, errorfile=None, test_n=None):
    if bdir is not None:
        gp = SparseGP(fname=os.path.join(bdir, 'trained.gp'))
        testX = np.load(os.path.join(bdir, "testX.npy"))
        resultfile = os.path.join(bdir, 'results.txt')
        errorfile = os.path.join(bdir, "error.npz")


    if test_n is None:
        test_n = len(testX)

    naive_predict = np.zeros(test_n)

    tree_predict = np.zeros(test_n)
    tree_predict_terms = np.zeros(test_n, dtype=int)

    sparse_covar = np.zeros(test_n)
    sparse_covar_spkernel = np.zeros(test_n)

    tree_covar8 = np.zeros(test_n)
    tree_covar8_terms = np.zeros(test_n, dtype=int)
    tree_covar8_distevals = np.zeros(test_n, dtype=int)

    t0 = time.time()
    for i in range(test_n):
        naive_predict[i] = gp.predict_naive(testX[i:i+1,:])
    t1 = time.time()
    naive_predict_time = t1-t0

    t2 = time.time()
    for i in range(test_n):
        tree_predict[i] = gp.predict(testX[i:i+1,:], eps=1e-4)
        tree_predict_terms[i] = gp.predict_tree.fcalls
    t3 = time.time()
    tree_predict_time = t3-t2

    t4 = time.time()
    for i in range(test_n):
        sparse_covar[i] = gp.covariance_dense(testX[i:i+1,:])
    t5 = time.time()
    sparse_covar_time = t5-t4
    sparse_covar_spkernel = sparse_covar

    t41 = time.time()
    """
    for i in range(test_n):
        sparse_covar_spkernel[i] = gp.covariance_spkernel(testX[i:i+1,:])
    """
    t51 = time.time()
    sparse_covar_spkernel_time = t51-t41

    def stats(v):
        return (scipy.stats.scoreatpercentile(v, 10), np.mean(v), np.median(v) , scipy.stats.scoreatpercentile(v, 90) )

    f = open(resultfile, 'w')
    f.write("naive predict time: %f\n" % (naive_predict_time/test_n))

    f.write("tree predict time: %f\n" % (tree_predict_time/test_n))
    f.write("tree predict terms:  10th %.2f mean %.2f median %.2f 90th %.2f \n" % stats(tree_predict_terms))
    f.write("tree predict errors: 10th %.9f mean %.9f median %.9f 90th %.9f \n" % stats(np.abs(tree_predict - naive_predict) ))

    f.write("sparse covar time: %f\n" % (sparse_covar_time/test_n))
    f.write("sparse covar spkernel time: %f\n" % (sparse_covar_spkernel_time/test_n))

    for epsm in (2,4,8,16):
        for eps_abs in (4,8):
            t6 = time.time()
            for i in range(test_n):
                tree_covar8[i] = gp.covariance_double_tree(testX[i:i+1,:], eps=10**(-epsm), eps_abs=10**(-eps_abs))
                tree_covar8_terms[i] = gp.double_tree.fcalls
                tree_covar8_distevals[i] = gp.double_tree.dfn_evals
            t7 = time.time()
            tree_covar8_time = t7-t6
            f.write("tree covar%d_%d time: %f\n" % (epsm, eps_abs, tree_covar8_time/test_n))
            f.write("tree covar%d_%d terms: 10th %.2f mean %.2f median %.2f 90th %.2f \n" % ((epsm, eps_abs) + stats(tree_covar8_terms)))
            f.write("tree covar%d_%d dfnevals: 10th %.2f mean %.2f median %.2f 90th %.2f \n" %  ((epsm, eps_abs) + stats(tree_covar8_distevals)))
            f.write("tree covar%d_%d rel errors: 10th %.9f mean %.9f median %.9f 90th %.9f \n" %  ((epsm, eps_abs) + stats(np.abs(tree_covar8 - sparse_covar_spkernel)/(1-sparse_covar_spkernel))))
            f.write("tree covar%d_%d var-rel errors: 10th %.9f mean %.9f median %.9f 90th %.9f \n" %  ((epsm, eps_abs) + stats(np.abs(tree_covar8 - sparse_covar_spkernel)/sparse_covar_spkernel)))
            f.write("tree covar%d_%d abs errors: 10th %.9f mean %.9f median %.9f 90th %.9f \n" %  ((epsm, eps_abs) + stats(np.abs(tree_covar8 - sparse_covar_spkernel))))

    np.savez(errorfile, tree_covar8=tree_covar8, sparse_covar=sparse_covar, tree_predict=tree_predict, naive_predict=naive_predict, sparse_covar_spkernel=sparse_covar_spkernel)


"""
def geny(X, cluster_width, sigma2_n):
    # sample from a GP prior with SE kernel

    n = X.shape[0]
    print "starting y"
    K, spK = kernel(X, cluster_width, sigma2_n)
    print "got kernel"
    z = np.random.randn(n)

    t0  = time.time()
    factor = sp_cholesky(spK)
    t1 = time.time()
    L = factor.L()
    P = factor.P()
    Pinv = np.argsort(P)

    t2 = time.time()
    y = (L * z)[Pinv]
    t3 = time.time()
    print "chol on %d elements took time %.2f" % (n, t1-t0)
    print "mult on %d elements took time %.2f" % (n, t3-t2)
    return y
"""

def scatter2d(X, y, fname):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    f = Figure()
    ax = f.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=2, edgecolors='none')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig(fname)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def build_lscale_benchmark():
    sigma2_n = 0.000001
    sigma2_f = 1.0
    npts=1000

    for lengthscale in lscales:
        bdir = os.path.join(basedir, "%s_%f_%d" % (wfn_str,lengthscale,npts))
        create_bench(bdir=bdir, dim=2, test_n=1000, npts=npts, lengthscale=lengthscale, sigma2_n=sigma2_n, sigma2_f=sigma2_f)

def run_lscale_benchmark():
    npts=1000

    for lengthscale in lscales:
        bdir = os.path.join(basedir, "%s_%f_%d" % (wfn_str,lengthscale,npts))
        eval_gp(bdir)


def create_bench(bdir, dim, test_n, npts, lengthscale, sigma2_n, sigma2_f):
    mkdir_p(bdir)
    sparse_threshold=1e-8
    hyperparams = [sigma2_n, sigma2_f, lengthscale, lengthscale]
    X = genX(dim, npts + test_n)
    test_p = np.random.permutation(len(X))
    test_X = X[test_p[:test_n], :]
    X = np.array(X[test_p[test_n:], :])

    X = sort_events(X)

    y, spK = prior_sample(X, hyperparams=hyperparams, dfn_str="euclidean", wfn_str=wfn_str, sparse_threshold=sparse_threshold, return_kernel=True)

    np.save(os.path.join(bdir, 'testX.npy'), test_X)
    np.save(os.path.join(bdir, 'hyperparams.npy'), hyperparams)
    np.save(os.path.join(bdir, 'K.npy'), spK)
    np.save(os.path.join(bdir, 'X.npy'), X)
    np.save(os.path.join(bdir, 'y.npy'), y)

    gp = SparseGP(X=X, y=y, basisfns = [],
                  dfn_str="euclidean", wfn_str=wfn_str,
                  hyperparams = hyperparams,
                  K = spK, sparse_threshold=sparse_threshold,
                  sort_events=False, build_tree=False)


    gp.save_trained_model(os.path.join(bdir, 'trained.gp'))

    with open(os.path.join(bdir, 'stats.txt'), 'w') as f:
        f.write("hyperparams: %s\n" % hyperparams)
        nzr, nzc = spK.nonzero()
        f.write("K nonzero: %d\n" % len(nzr))

        nzr, nzc = gp.Kinv.nonzero()
        f.write("Kinv nonzero: %d\n" % len(nzr))
        for k in gp.timings:
            f.write("timing %s: %.3f\n" % (k, gp.timings[k]))


def build_size_benchmark():
    sigma2_n = 1.0
    sigma2_f = 1.0
    #extra_nc = test_n / cluster_size

    for points_within_lscale in (0.25, 5.0, ):
        for npts in Ns:
            lengthscale = np.sqrt(points_within_lscale/npts) / np.sqrt(np.pi)
            bdir = os.path.join(basedir, "%s_%.2fpts_%d" % (wfn_str,points_within_lscale,npts))
            if not os.path.exists(os.path.join(bdir, "trained.gp")):
                print "building", npts, points_within_lscale, "at", lengthscale
                print bdir
                create_bench(bdir=bdir, dim=2, test_n=1000, npts=npts, lengthscale=lengthscale, sigma2_n=sigma2_n, sigma2_f=sigma2_f)

def run_size_benchmark():
    for npts in Ns:
        for points_within_lscale in (1.0,5.0):
            bdir = os.path.join(basedir, "%s_%.2fpts_%d" % (wfn_str,points_within_lscale,npts))
            print bdir
            eval_gp(bdir)

def build_highd_benchmark():
    sigma2_n = 1.0
    sigma2_f = 1.0
    npts = 10000
    points_within_lscale=1.0
    import scipy.misc
    for dim in (2, 6, 12, 20):
        k = dim/2
        lengthscale = (scipy.misc.factorial(k) * points_within_lscale/npts) ** (.5/k) / np.sqrt(np.pi)
        bdir = os.path.join(basedir, "highd_%d_%s_base%f_%d" % (dim, wfn_str,points_within_lscale,npts))
        if not os.path.exists(bdir):
            create_bench(bdir=bdir, dim=dim, test_n=1000, npts=npts, lengthscale=lengthscale, sigma2_n=sigma2_n, sigma2_f=sigma2_f)


def main():

    sigma2_n = 0.000001
    lengthscale = 0.000005
    sigma2_f = 1.0
    hyperparams = [sigma2_n, sigma2_f, lengthscale, lengthscale]

    X = genX(20, 2000, 20, lengthscale)
    print "got X"
    y, spK = prior_sample(X, hyperparams=hyperparams, dfn_str="euclidean", wfn_str=wfn_str, return_kernel=True)
    print "got y"

    gp = SparseGP(X=X, y=y, basisfns = [],
                  kernel_str="euclidean_se",
                  hyperparams = hyperparams,
                  K = spK,
                  sort_events=False)

    ymax = np.max(y)
    ymin = np.min(y)
    normy = (y - ymin) / (ymax-ymin)

    scatter2d(X, normy, 'plot.png')
    print "scattered"

    recovered = gp.predict(X)
    normr = (recovered - ymin) / (ymax - ymin)
    np.savetxt('recovered.txt', recovered)
    scatter2d(X, normr, 'recovered.png')

    print "covar", gp.covariance(X[0:1, :])

if __name__ == "__main__":
    #build_lscale_benchmark()
    #run_lscale_benchmark()
    build_size_benchmark()
    #run_size_benchmark()
    #build_highd_benchmark()
    #main()
