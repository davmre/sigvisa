import numpy as np
import scipy.stats
from sigvisa.treegp.gp import GP, GPCov, mcov
from collections import defaultdict

class LocalGPs(object):

    def __init__(self, block_centers, cov_block_params, X=None, y=None, X_blocks=None, y_blocks=None):
        self.n_blocks = len(block_centers)
        self.block_centers = block_centers

        if X is not None:
            X_blocks, y_blocks = self.blocks_from_centers(X, y)
        self.X_blocks = X_blocks
        self.y_blocks = y_blocks
        self.cov_block_params = cov_block_params

        cov_blocks = []
        for cbp in cov_block_params:
            noise_var = cbp[0]
            cov = GPCov(wfn_params=cbp[1:2], dfn_params=cbp[2:], dfn_str="euclidean", wfn_str="se")
            cov_blocks.append((noise_var, cov))
        if len(cov_blocks) == 1:
            cov_blocks = cov_blocks * self.n_blocks
            self.tied_params = True
        else:
            self.tied_params = False

        self.cov_blocks = cov_blocks

        self.gps = []
        for (X, y, nc) in zip(X_blocks, y_blocks, cov_blocks):
            noise_var, cov = nc
            gp = GP(X, y, cov_main=cov,
                    noise_var=noise_var,
                    compute_ll=True,
                    compute_grad=True,
                    sort_events=False,
                    sparse_invert=False)
            self.gps.append(gp)

    def blocks_from_centers(self, X, y):
        X_blocks = [[] for i in range(self.n_blocks)]
        y_blocks = [[] for i in range(self.n_blocks)]
        for xp, yp in zip(X, y):
            block = self.get_block(xp)
            X_blocks[block].append(xp)
            y_blocks[block].append(yp)
        X_blocks = [np.array(xb) for xb in X_blocks]
        y_blocks = [np.array(yb) for yb in y_blocks]

        return X_blocks, y_blocks

    def get_block(self, X_new):
        dists = [np.linalg.norm(X_new - center) for center in self.block_centers]
        return np.argmin(dists)

    def likelihood(self):
        return np.sum([gp.log_likelihood() for gp in self.gps])

    def likelihood_grad(self):
        grads = [gp.ll_grad for gp in self.gps]
        if self.tied_params:
            return np.sum(grads, axis=0)
        else:
            return np.concatenate(grads)

    def predict(self, X_new):
        p = []
        for xn in X_new:
            block = self.get_block(xn)
            p.append(self.gps[block].predict(xn.reshape(1, -1)))
        return np.array(p)

    def covariance(self, X_new):

        Xs, perm, b = self.sort_by_block(X_new)

        nx = len(perm)
        cov = np.zeros((nx, nx))

        i_old = 0
        for block, i in enumerate(b):
            Xs_block = Xs[i_old:i,:]
            cov[i_old:i, i_old:i] = self.gps[block].covariance(Xs_block)
            i_old = i

        rperm = np.argsort(perm)
        tmp = cov[:, rperm]
        rcov = tmp[rperm, :]

        return rcov

    def predict_dist(self, X_new):
        return self.predict(X_new), self.covariance(X_new)

    def sort_by_block(self, X):
        # given: set of input locations X
        # returns: tuple (Xs, p, b)
        # Xs: X locations sorted by block, i.e. first all points in block 0,
        #     then block 1, etc.
        # p: permutation such that Xs = X[p, :]
        # b: endpoints of each block in Xs. For example, if we have three
        #    points in block 0, none in block 1, and one in block 2, we'd
        #    return b = [3, 3, 4]
        blocks = np.array([self.get_block(xn) for xn in X])
        idx = np.arange(len(blocks))
        perm = np.array(sorted(idx, key = lambda i : blocks[i]))

        sorted_blocks = blocks[perm]
        b = []
        i = 0
        for block in range(self.n_blocks):
            i += np.sum(sorted_blocks==block)
            b.append(i)

        return X[perm,:], perm, b

class BCM(LocalGPs):

    def __init__(self, test_cov, **kwargs):
        super(BCM, self).__init__(**kwargs)
        self.test_cov = test_cov

    def predict_dist(self, X_new, diag=False, source_gps=None, predict_cov=None, noise_var=0.0):

        if diag:
            means = []
            vs = []
            for x in X_new:
                m, v = self.predict_dist(x.reshape((1, -1)), diag=False, source_gps=source_gps, noise_var=noise_var)
                means.append(m)
                vs.append(v)
            return np.array(means), np.array(vs)

        if source_gps is None:
            source_gps = self.gps

        if predict_cov is None:
            predict_cov = self.test_cov


        means = [lgp.predict(X_new) for lgp in source_gps]
        covs = [lgp.covariance(X_new, include_obs=True) for lgp in source_gps]
        precs = [np.linalg.inv(cov) for cov in covs]
        combined_prec = np.sum(precs, axis=0)

        target_prior_cov = mcov(X_new, predict_cov, noise_var)
        target_prior_prec = np.linalg.inv(target_prior_cov)
        prior_covs = [mcov(X_new, lgp.cov_main, lgp.noise_var) for lgp in source_gps]
        prior_precs = [np.linalg.inv(cov) for cov in prior_covs]
        combined_prec += target_prior_prec
        combined_prec -= np.sum(prior_precs, axis=0)
        cov = np.linalg.inv(combined_prec)

        assert( (np.diag(cov) > 0).all() )

        weighted_means = [np.dot(prec, mean) for (mean, prec) in zip(means, precs)]
        mean = np.dot(cov, np.sum(weighted_means, axis=0))
        return mean, cov

    def _local_pred_gradient(self, target_X, target_y, source_gp, subtract_prior=True):
        def dgaussian(r, prec, dcov, dmean=None):
            r = r.reshape((-1, 1))

            dprec = -np.dot(prec, np.dot(dcov, prec))
            dll_dcov = -.5*np.dot(r.T, np.dot(dprec, r))
            dll_dcov -= .5*np.trace(np.dot(prec, dcov))
            dll = dll_dcov

            if dmean is not None:
                dmean = dmean.reshape((-1, 1))
                dll_dmean = np.dot(r.T, np.dot(prec, dmean))
                dll += dll_dmean

            return dll

        mean = source_gp.predict(target_X)
        cov = source_gp.covariance(target_X, include_obs=True)

        nparams = len(source_gp.ll_grad)
        llgrad = np.zeros((nparams,))

        r = target_y - mean
        prec = np.linalg.inv(cov)
        ll = -.5 * np.dot(r.T, np.dot(prec, r))
        ll -= .5*np.log(np.linalg.det(cov))
        ll -= .5*len(r)*np.log(2*np.pi)

        if subtract_prior:
            prior_cov = source_gp.kernel(target_X, target_X, identical=True)
            prior_prec = np.linalg.inv(prior_cov)
            ll += .5 * np.dot(target_y.T, np.dot(prior_prec, target_y))
            ll += .5*np.log(np.linalg.det(prior_cov))
            ll += .5*len(target_y)*np.log(2*np.pi)

        for i_hparam in range(nparams):
            dmean, dcov = source_gp.grad_prediction(target_X, i_hparam)

            llgrad[i_hparam] = dgaussian(r, prec, dcov, dmean)

            if subtract_prior:
                dcov = source_gp.dK(target_X, target_X, i_hparam, identical=True)
                llgrad[i_hparam] -= dgaussian(target_y, prior_prec, dcov)

        return ll, llgrad

    def llgrad(self, block_perm = None, pseudo=False):
        pseudo_ll = False
        if block_perm is None:
            X_blocks = self.X_blocks
            y_blocks = self.y_blocks
            gps = self.gps
        else:
            X_blocks = []
            y_blocks = []
            gps = []

            for i, b in enumerate(block_perm):
                X_blocks.append(self.X_blocks[b])
                y_blocks.append(self.y_blocks[b])
                gps.append(self.gps[b])

        # overall likelihood is sum of predictive likelihoods, *minus* prior likelihoods, plus
        ll = 0
        if self.tied_params:
            nparams = len(self.cov_block_params[0])
            llgrad = np.zeros((nparams,))
        else:
            llgrad = [np.zeros((len(gps[i].ll_grad),)) for i in range(self.n_blocks)]

        for source_i in range(self.n_blocks):

            min_target = source_i+1
            include_prior = True
            if pseudo:
                min_target = max(min_target, self.n_blocks-1)
                include_prior = (source_i == self.n_blocks-1)

            if include_prior:
                gp = gps[source_i]
                ll += gp.log_likelihood()
                pllgrad_prior = gp.ll_grad
                #print "including prior at target", source_i, "ll", gp.log_likelihood()
                if self.tied_params:
                    llgrad += pllgrad_prior
                else:
                    llgrad[source_i] += pllgrad_prior

            for target_i in range(min_target, self.n_blocks):
                target_X = X_blocks[target_i]
                target_y = y_blocks[target_i]
                source_gp = gps[source_i]

                pll, pllgrad = self._local_pred_gradient(target_X, target_y, source_gp, subtract_prior=True)
                ll += pll
                #print "predicting %d from %d: ll %.2f grad %s" % (target_i, source_i, pll, pllgrad)
                if self.tied_params:
                    llgrad += pllgrad
                else:
                    llgrad[source_i] += pllgrad

        if not self.tied_params:
            if block_perm is not None:
                llgrad_orig = []
                rperm = np.argsort(block_perm)
                for b in rperm:
                    llgrad_orig.append(llgrad[b])
                llgrad = llgrad_orig
            llgrad = np.concatenate(llgrad)

        return ll, llgrad

    def stochastic_llgrad(self, samples=1, pseudo=True):
        ll = 0
        llgrad = None
        for i in range(samples):
            p = np.random.permutation(self.n_blocks)
            sll, sllgrad = self.llgrad(pseudo=pseudo, block_perm=p)
            ll += sll
            llgrad = sllgrad if llgrad is None else llgrad+sllgrad
        return ll/samples, llgrad/samples

    def cycle_pseudoll(self):
        ll = 0
        llgrad = None
        n = self.n_blocks
        p = np.arange(n)
        for i in range(n):
            p[-1], p[i] = p[i], p[-1]
            sll, sllgrad = self.llgrad(pseudo=True, block_perm=p)
            ll += sll
            llgrad = sllgrad if llgrad is None else llgrad+sllgrad
            p[-1], p[i] = p[i], p[-1]
        return ll/n, llgrad/n

    def likelihood(self):
        ll = self.gps[0].log_likelihood()
        for i in range(1, self.n_blocks):
            nv, pc = self.cov_blocks[i]
            m, c = self.predict_dist(self.X_blocks[i], source_gps = self.gps[:i], noise_var=nv, predict_cov=pc)
            llb = scipy.stats.multivariate_normal(mean=m, cov=c).logpdf(self.y_blocks[i])
            print "block %d conditional ll %.2f vs prior %.2f" % (i, llb, self.gps[i].log_likelihood())
            ll += llb
        return ll
