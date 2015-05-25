import numpy as np
import scipy.stats

from collections import defaultdict
from multiprocessing import Pool

from sigvisa.treegp.gp import GP, GPCov, mcov, dgaussian, dgaussian_rank1

from sigvisa.treegp.cover_tree import VectorTree
import pyublas


class Blocker(object):

    def __init__(self, block_centers):
        self.block_centers = block_centers
        self.n_blocks = len(block_centers)

    def get_block(self, X_new):
        dists = [np.linalg.norm(X_new - center) for center in self.block_centers]
        return np.argmin(dists)

    def sort_by_block(self, X, Y=None, YY=None):
        if Y is not None:
            assert(Y.shape[0] == X.shape[0])
        else:
            assert(YY.shape[0] == X.shape[0])
            assert(YY.shape[1] == X.shape[0])

        n = X.shape[0]
        blocks = np.array([self.get_block(xp) for xp in X])
        idxs = np.arange(n)
        perm = np.asarray(sorted(idxs, key = lambda i : blocks[i]))
        sorted_X = X[perm]
        sorted_blocks = blocks[perm]
        block_idxs = [idxs[sorted_blocks==block] for block in np.arange(self.n_blocks)]
        block_boundaries = [ (np.min(bi), np.max(bi)+1)  for bi in block_idxs  ]

        if Y is not None:
            sorted_Y = Y[perm]
            return sorted_X, sorted_Y, perm, block_boundaries
        else:
            tmp = YY[perm]
            sorted_YY = tmp[:, perm]
            return sorted_X, sorted_YY, perm, block_boundaries


def sample_synthetic(seed=1, n=400, xd=2, yd=10, lscale=0.1, noise_var=0.01):
    # sample data from the prior
    np.random.seed(seed)
    X = np.random.rand(n, xd)

    cov = GPCov(wfn_params=[1.0], dfn_params=[lscale, lscale], dfn_str="euclidean", wfn_str="se")
    KK = mcov(X, cov, noise_var)

    y = scipy.stats.multivariate_normal(mean=np.zeros((X.shape[0],)), cov=KK).rvs(yd).T.reshape((-1, yd))

    return X, y, cov

def vis(X, y):
    for i in range(y.shape[0]):
        figure()
        scatter(X[:, 0], X[:, 1], c=y[i,:], cmap="jet")


class MultiSharedBCM(object):

    def __init__(self, X, Y, block_boundaries, cov, noise_var, kernelized=False, dy=None, neighbor_threshold=1e-5):
        self.X = X

        if kernelized:
            self.kernelized = True
            self.YY = Y
            assert(dy is not None)
            self.dy = dy
        else:
            self.kernelized = False
            self.Y = Y
        self.block_boundaries = block_boundaries
        self.n_blocks = len(block_boundaries)

        self.cov = cov
        self.noise_var = noise_var
        dummy_X = np.array([[0.0,] * self.X.shape[1],], dtype=float)
        self.predict_tree = VectorTree(dummy_X, 1, cov.dfn_str, cov.dfn_params, cov.wfn_str, cov.wfn_params)

        self.compute_neighbors(threshold=neighbor_threshold)

    def compute_neighbors(self, threshold=1e-5):
        neighbor_count = defaultdict(int)
        neighbors = []
        for i in range(self.n_blocks):
            i_start, i_end = self.block_boundaries[i]
            Xi = self.X[i_start:i_end]
            ni = Xi.shape[0]
            for j in range(i):
                j_start, j_end = self.block_boundaries[j]
                Xj = self.X[j_start:j_end]
                Kij = self.kernel(Xi, X2=Xj)
                maxk = np.max(np.abs(Kij))
                if maxk > threshold:
                    neighbors.append((i,j))
                    neighbor_count[i] += 1
                    neighbor_count[j] += 1
        self.neighbor_count = neighbor_count
        self.neighbors = neighbors

    def llgrad_local(self, **kwargs):
        unaries = [self.llgrad_unary(i, **kwargs) for i in range(self.n_blocks)]
        pairs = [self.llgrad_joint(i, j, **kwargs) for (i, j) in self.neighbors]
        unary_lls, unary_grads = zip(*unaries)
        pair_lls, pair_grads = zip(*pairs)

        ll = np.sum(pair_lls)
        ll -= np.sum([(self.neighbor_count[i]-1)*ull for (i, ull) in enumerate(unary_lls) ])

        grads = np.zeros(self.X.shape)
        pair_idx = 0
        for i in range(self.n_blocks):
            i_start, i_end = self.block_boundaries[i]
            grads[i_start:i_end, :] -= (self.neighbor_count[i]-1)*unary_grads[i]

        for pair_idx, (i,j) in enumerate(self.neighbors):
            i_start, i_end = self.block_boundaries[i]
            j_start, j_end = self.block_boundaries[j]
            ni = i_end-i_start
            grads[i_start:i_end] += pair_grads[pair_idx][:ni]
            grads[j_start:j_end] += pair_grads[pair_idx][ni:]
        return ll, grads

    def llgrad(self, **kwargs):
        # overall likelihood is the pairwise potentials for all (unordered) pairs,
        # where each block is involved in (n-1) pairs. So we subtract each unary potential n-1 times.
        # Then finally each unary potential gets added in once.

        unaries = [self.llgrad_unary(i, **kwargs) for i in range(self.n_blocks)]
        pairs = [self.llgrad_joint(i, j, **kwargs) for i in range(self.n_blocks) for j in range(i)]

        unary_lls, unary_grads = zip(*unaries)
        pair_lls, pair_grads = zip(*pairs)

        ll = np.sum(pair_lls) - (self.n_blocks - 2)*np.sum(unary_lls)

        grads = -(self.n_blocks-2) * np.vstack(unary_grads)
        pair_idx = 0
        for i in range(self.n_blocks):
            i_start, i_end = self.block_boundaries[i]
            ni = i_end-i_start
            for j in range(i):
                j_start, j_end = self.block_boundaries[j]
                grads[i_start:i_end, :] += pair_grads[pair_idx][:ni, :]
                grads[j_start:j_end, :] += pair_grads[pair_idx][ni:, :]
                pair_idx += 1
        return ll, grads

    def llgrad_unary(self, i, **kwargs):
        i_start, i_end = self.block_boundaries[i]
        X = self.X[i_start:i_end]

        if self.kernelized:
            YY = self.YY[i_start:i_end, i_start:i_end]
            return self.gaussian_llgrad_kernel(X, YY, dy=self.dy, **kwargs)
        else:
            Y = self.Y[i_start:i_end]
            return self.gaussian_llgrad(X, Y, **kwargs)

    def llgrad_joint(self, i, j, **kwargs):
        i_start, i_end = self.block_boundaries[i]
        j_start, j_end = self.block_boundaries[j]
        Xi = self.X[i_start:i_end]
        Xj = self.X[j_start:j_end]

        ni = Xi.shape[0]
        nj = Xj.shape[0]
        X = np.vstack([Xi, Xj])

        if self.kernelized:
            YY = np.empty((ni+nj, ni+nj))
            YY[:ni, :ni] = self.YY[i_start:i_end, i_start:i_end]
            YY[ni:, ni:] = self.YY[j_start:j_end, j_start:j_end]
            YY[:ni, ni:] = self.YY[i_start:i_end, j_start:j_end]
            YY[ni:, :ni]  = YY[:ni, ni:].T
            return self.gaussian_llgrad_kernel(X, YY, dy=self.dy, **kwargs)
        else:
            Yi = self.Y[i_start:i_end]
            Yj = self.Y[j_start:j_end]
            Y = np.vstack([Yi, Yj])
            return self.gaussian_llgrad(X, Y, **kwargs)


    def kernel(self, X, X2=None):
        if X2 is None:
            n = X.shape[0]
            K = self.predict_tree.kernel_matrix(X, X, False)
            K += np.eye(n) * self.noise_var
        else:
            K = self.predict_tree.kernel_matrix(X, X2, False)
        return K

    def dKdx(self, X, p, i, return_vec=False):
        # derivative of kernel(X1, X2) wrt i'th coordinate of p'th point in X1.
        if return_vec:
            dKv = self.predict_tree.kernel_deriv_wrt_xi_row(X, p, i)
            dKv[p] = 0
            return dKv
        else:
            dK = self.predict_tree.kernel_deriv_wrt_xi(X, X, p, i)
            dK[p,p] = 0
            dK = dK + dK.T
            return dK

    def gaussian_llgrad(self, X, Y, grad_X = False):
        n, dx = X.shape
        dy = Y.shape[1]

        K = mcov(X, self.cov, self.noise_var)
        prec = np.linalg.inv(K)
        Alpha = np.dot(prec, Y)

        ll = -.5 * np.sum(Y*Alpha)
        ll += -.5 * dy * np.linalg.slogdet(K)[1]
        ll += -.5 * dy * n * np.log(2*np.pi)

        if not grad_X:
            return ll, np.array(())

        llgrad = np.zeros((n, dx))
        for p in range(n):
            for i in range(dx):
                dll = 0
                dcv = self.dKdx(X, p, i, return_vec=True)
                #t1 = -np.outer(prec[p,:], dcv)
                #t1[:, p] = -np.dot(prec, dcv)
                #dll_dcov = .5*ny*np.trace(t1)

                dll = -dy * np.dot(prec[p,:], dcv)

                for j in range(dy):
                    alpha = Alpha[:,j]
                    dK_alpha = dcv * alpha[p]
                    dK_alpha[p] = np.dot(dcv, alpha)
                    dll_dcov = .5*np.dot(alpha, dK_alpha)
                    dll += dll_dcov

                llgrad[p,i] = dll

        return ll, llgrad

    def gaussian_llgrad_kernel(self, X, YY, dy=None, grad_X=False):
        n, dx = X.shape
        if dy is None:
            dy = self.dy

        K = mcov(X, self.cov, self.noise_var)
        Kinv = np.linalg.inv(K)
        prec = Kinv

        KYYK = np.dot(np.dot(Kinv, YY), Kinv)

        ll =  -.5 * np.sum(Kinv * YY)
        ll += -.5 * dy * np.linalg.slogdet(K)[1]
        ll += -.5 * dy * n * np.log(2*np.pi)
        if not grad_X:
            return ll, np.array(())

        llgrad = np.zeros((n, dx))
        for p in range(n):
            for i in range(dx):
                #dcv_full = self.dKdx(X, p, i)
                #dll = -.5*np.sum(KYYK * dcv_full)
                dcv = self.dKdx(X, p, i, return_vec=True)
                dll = np.dot(KYYK[p,:], dcv)

                dll += -dy * np.dot(prec[p,:], dcv)

                llgrad[p,i] = dll

                #t1 = -np.outer(prec[p,:], dcov_v)
                #t1[:, p] = -np.dot(prec, dcov_v)

        return ll, llgrad
