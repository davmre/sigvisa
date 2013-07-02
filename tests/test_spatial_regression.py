import numpy as np
import unittest

from sigvisa.learn.train_param_common import learn_model, load_model
from sigvisa.infer.optimize.optim_utils import construct_optim_params

from sigvisa.models.spatial_regression.SparseGP import SparseGP, start_params
from sigvisa.models.spatial_regression.baseline_models import LinearBasisModel, poly_basisfns

from sigvisa.utils.cover_tree import VectorTree
import pyublas

class TestModels(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [120, 30, 0, 1000, 32],
            [118, 31, 0, 1050, 32],
            [120, 29, 40, 1000, 34],
            [110, 30, 20, 3000, 34],
        ], dtype=float)
        self.y = np.array([
            -0.02,
            -0.01,
            -0.015,
            -0.005,
        ])
        self.evids = np.array([
            1,
            2,
            3,
            4,
        ])
        self.testX1 = np.array([[120, 30, 0, 1025, 32], ], dtype=float)
        self.testX2 = np.array([[119, 31, 0, 1000, 33], ], dtype=float)

    def test_constant(self):
        model = learn_model(self.X, self.y, model_type="constant_gaussian", sta="AAK")
        pred1 = model.predict(self.testX1)
        pred2 = model.predict(self.testX2)
        self.assertAlmostEqual(pred1, np.mean(self.y))
        self.assertAlmostEqual(pred1, pred2)

        fname = "test_constant_model"
        model.save_trained_model(fname)
        nmodel = load_model(fname, "constant_gaussian")
        pred3 = nmodel.predict(self.testX1)
        self.assertAlmostEqual(pred1, pred3)

        ll = nmodel.log_likelihood()
        ll1 = nmodel.log_p(cond=self.X, x=self.y)
        self.assertAlmostEqual(ll, ll1)

        s = nmodel.sample(self.X)

    def test_linear_distance(self):
        model = learn_model(self.X, self.y, model_type="linear_distance", sta="AAK")
        pred1 = model.predict(self.testX1)
        pred2 = model.predict(self.testX2)
        self.assertAlmostEqual(pred1, -0.01375)
        self.assertAlmostEqual(pred2, -0.0175)

        fname = "test_linear_model"
        model.save_trained_model(fname)
        nmodel = load_model(fname, "linear_distance")
        pred3 = nmodel.predict(self.testX1)
        self.assertAlmostEqual(pred1, pred3)

        ll = nmodel.log_likelihood()
        ll1 = nmodel.log_p(cond=self.X, x=self.y)
        self.assertAlmostEqual(ll, ll1)

        s = nmodel.sample(self.X)


    def test_poly_regression(self):
        N = 10
        bfn = poly_basisfns(3)
        X = np.reshape(np.linspace(-5, 5, N), (-1, 1))
        H = np.array([[f(x) for f in bfn] for x in X])
        coeffs = np.array([50, 1, 3, -3])
        sigma_n = 1
        np.random.seed(0)
        y = np.dot(H, coeffs) + np.random.randn(N) * sigma_n

        b = np.zeros((4,))
        B = np.eye(4) * 100000
        B[0,0] = (1000000)**2

        model = LinearBasisModel(X=X, y=y, basisfns=bfn, param_mean=b, param_covar=B, noise_std=sigma_n, H=H, compute_ll=True, sta="AAK")

        tol = np.array((1, .4, .05, .05))
        self.assertTrue( (np.abs(model.mean - coeffs) < tol ).all())

        # test that the variances are reasonable: we should be most
        # certain about the high-order terms and least certain about
        # the low-order terms.
        v = np.diag(np.dot(model.sqrt_covar.T, model.sqrt_covar))
        self.assertGreater(v[0], v[1])
        self.assertGreater(v[1], v[2])
        self.assertGreater(v[2], v[3])

        # test that loading and saving work
        p = model.predict(X)
        c = model.covariance(X, include_obs=True)

        model.save_trained_model('model.npz')
        m = LinearBasisModel(fname='model.npz')
        p1 = m.predict(X)
        c1 = m.covariance(X, include_obs=True)

        self.assertTrue( (np.abs(p-p1) < .001).all() )
        self.assertTrue( (np.abs(c-c1) < .001).flatten().all() )


class TestCTree(unittest.TestCase):

    def test_cover_tree_multiplication(self):

        # generate random data points
        np.random.seed(6)
        cluster = np.random.normal(size=(10, 2)) * 10
        cluster_locations = np.random.normal(size=(100, 2)) * 1000
        X = np.zeros((1000, 2))
        for (i,cl) in enumerate(cluster_locations):
            X[10*i:10*(i+1),:] = cluster + cl

        # build a cover tree
        dfn_param = np.array((1.0, 1.0), dtype=float)
        weight_param = np.array((1.0,), dtype=float)
        tree = VectorTree(X, 1, "euclidean", dfn_param, 'se', weight_param)

        # assign an arbitrary value to each data point
        v = np.array([1,2,3,4,5,6,7,8,9,10] * 100, dtype=float)
        tree.set_v(0, v)

        query_pt = np.matrix(cluster_locations[29,:], dtype=float, copy=True)

        #w = lambda x1, x2 : np.exp(-1 * np.linalg.norm(x1-x2, 2)**2 )
        #k = [w(query_pt, x) for x in X]
        #kv = np.dot(k, v)

        kv_tree = tree.weighted_sum(0, query_pt, 1e-4)
        self.assertAlmostEqual(0.893282181527, kv_tree, places=4)
        self.assertEqual(tree.fcalls, 54)


class TestGP(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [120, 30, 0, 1000, 32],
            [118, 31, 0, 1050, 32],
            [120, 29, 40, 1000, 34],
            [110, 30, 20, 3000, 34],
        ], dtype=float)
        self.y = np.array([
            -0.02,
            -0.01,
            -0.015,
            -0.005,
        ])
        self.evids = np.array([
            1,
            2,
            3,
            4,
        ])
        self.testX1 = np.array([[120, 30, 0, 1025, 32], ], dtype=float)
        self.testX2 = np.array([[119, 31, 0, 1000, 33], ], dtype=float)


        self.basisfns = ()
        self.b = None #np.array(())
        self.B = None #np.array(((),))
        self.hparams = start_params["lld"]["coda_decay"]
        self.gp = SparseGP(X=self.X, y=self.y, basisfns=self.basisfns, param_mean=self.b, param_cov=self.B, hyperparams=self.hparams, compute_ll=True, compute_grad=True)

    def test_sparse_gradient(self):
        g_sparse = self.gp._log_likelihood_gradient(None, None, None, self.gp.Kinv)
        g_dense = self.gp._log_likelihood_gradient(None, None, None, self.gp.Kinv.todense())
        self.assertTrue( (np.abs(g_sparse - g_dense) < 0.0001 ).all() )

    def test_SE_gradient(self):
        grad = self.gp.ll_grad
        n = len(self.hparams)
        kp  = self.hparams
        eps = 1e-4
        empirical_grad = np.zeros(n)
        for i in range(n):
            kp[i] -= eps
            gp1 = SparseGP(X=self.X, y=self.y, hyperparams=kp, basisfns=self.basisfns, param_mean=self.b, param_cov=self.B, compute_ll=True)
            l1 = gp1.log_likelihood()
            kp[i] += 2*eps
            gp2 = SparseGP(X=self.X, y=self.y, hyperparams=kp, basisfns=self.basisfns, param_mean=self.b, param_cov=self.B, compute_ll=True)
            l2 = gp2.log_likelihood()
            kp[i] -= eps
            empirical_grad[i] = (l2 - l1)/ (2*eps)

        self.assertTrue( (np.abs(grad - empirical_grad) < 0.01 ).all() )

    """
    def test_prior_gradient(self):
        kernel = "se"
        pgrad = k.param_prior_grad()

        n = len(self.start_params)
        kp  = self.start_params
        eps = 1e-4
        empirical_pgrad = np.zeros(n)
        for i in range(n):
            kp[i] -= eps
            k = kernels.setup_kernel(name=kernel, params=kp, extra=None, priors = [InvGamma(1.0, 1.0), InvGamma(1.0, 1.0), LogNormal(3.0, 2.0)])
            l1 = k.param_prior_ll()
            kp[i] += 2*eps
            k = kernels.setup_kernel(name=kernel, params=kp, extra=None, priors = [InvGamma(1.0, 1.0), InvGamma(1.0, 1.0), LogNormal(3.0, 2.0)])
            l2 = k.param_prior_ll()
            kp[i] -= eps
            empirical_pgrad[i] = (l2 - l1)/ (2*eps)

        self.assertTrue( (np.abs(pgrad - empirical_pgrad) < 0.01 ).all() )
    """

    def test_GPsave(self):
        model = learn_model(self.X, self.y, model_type="gp_lld", target="coda_decay", sta='AAK', optim_params=construct_optim_params("'method': 'none', 'normalize': False"))
        pred1 = model.predict(self.testX1)
        pred2 = model.predict(self.testX2)

        fname = "test_gp_model"
        model.save_trained_model(fname)
        nmodel = load_model(fname, "gp_lld")
        pred3 = nmodel.predict(self.testX1)
        self.assertAlmostEqual(pred1, pred3)

        ll = nmodel.log_likelihood()
        ll1 = nmodel.log_p(cond=self.X, x=self.y)
        # unlike with the other models, these likelihoods are not
        # necessarily equal in the GP case

        s = nmodel.sample(self.X)


class TestSemiParametric(unittest.TestCase):


    def setUp(self):
        N = 10
        x = np.linspace(-5,5,N)
        self.X = np.reshape(x, (-1, 1))

        H = np.vstack([x**3, x**2, x, np.ones(N)]) # basis functions for a cubic regression
        self.beta = [9.91898792e-01,  -1.62113090e+00,   3.15437605e+00,   1.25732838e+00]
        self.B = np.eye(4) * 9
        self.b = np.zeros((4,))

        p = np.dot(H.T, self.beta)

        def covar_matrix(x, k):
            n = len(x)
            K = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    K[i,j] = k(x[i], x[j])
            return K
        k1 = lambda x1, x2 : .001*np.exp( - ((x1-x2)/1.5)**2 ) + (.001 if x1==x2 else 0)
        K1 = covar_matrix(x, k1)

        np.random.seed(0)
        f1 = np.random.multivariate_normal(mean=np.zeros((len(x),)), cov=K1)
        self.y1 = f1 + p

        self.basisfns = [lambda x : x**3, lambda x : x**2, lambda x : x, lambda x : 1]

        self.hparams = np.array((.001, 0.001, 1.5), dtype=float)

        self.gp = SparseGP(X=self.X,
                           y=self.y1,
                           hyperparams=self.hparams,
                           basisfns=self.basisfns,
                           param_mean=self.b,
                           param_cov=self.B,
                           dfn_str="euclidean",
                           compute_ll=True,
                           compute_grad=True,
                           sparse_threshold=0,)


    def test_param_recovery(self):
        gp = self.gp
        inferred_beta = gp.param_predict()
        self.assertTrue( ( np.abs(inferred_beta - self.beta) < .1 ).all() )

        # make sure the posterior covariance matrix is reasonable
        posterior_covar = gp.param_covariance()
        self.assertTrue( np.max(posterior_covar.flatten()) < 1e-2 )

        # we should have the most uncertainty about the low-order
        # params (e.g. the constant term), and the least uncertainty
        # about the high-order terms, since a small change in a high-
        # order term has a larger effect on the resulting function.
        posterior_var = np.diag(posterior_covar)
        self.assertTrue( posterior_var[3] > posterior_var[2] )
        self.assertTrue( posterior_var[2] > posterior_var[1] )
        self.assertTrue( posterior_var[1] > posterior_var[0] )


    def test_likelihood(self):

        # in the limit of a prior forcing the parameters to be zero,
        # the semiparametric likelihood should match that of a
        # standard GP.
        gp_smallparam = SparseGP(X=self.X, y=self.y1, hyperparams=self.hparams, dfn_str="euclidean", basisfns=self.basisfns, param_mean=self.b, param_cov=np.eye(len(self.b)) * 0.000000000000001, compute_ll=True, sparse_threshold=0)
        gp_noparam = SparseGP(X=self.X, y=self.y1, hyperparams=self.hparams, dfn_str="euclidean", basisfns=(), compute_ll=True, sparse_threshold=0)

        self.assertGreater(self.gp.ll, gp_smallparam.ll)
        self.assertAlmostEqual(gp_smallparam.ll, gp_noparam.ll, places=-1)

    def test_gradient(self):
        grad = self.gp.ll_grad
        n = len(self.hparams)
        kp  = self.hparams
        eps = 1e-6
        empirical_grad = np.zeros(n)
        for i in range(n):
            kp[i] -= eps
            gp1 = SparseGP(X=self.X, y=self.y1, hyperparams=kp, basisfns=self.basisfns, param_mean=self.b, param_cov=self.B, compute_ll=True, dfn_str="euclidean", sparse_threshold=0)
            l1 = gp1.log_likelihood()
            kp[i] += 2*eps
            gp2 = SparseGP(X=self.X, y=self.y1, hyperparams=kp, basisfns=self.basisfns, param_mean=self.b, param_cov=self.B, compute_ll=True, dfn_str="euclidean", sparse_threshold=0)
            l2 = gp2.log_likelihood()
            kp[i] -= eps
            empirical_grad[i] = (l2 - l1)/ (2*eps)

        self.assertTrue( (np.abs(grad - empirical_grad) < 0.01 ).all() )


    def test_load_save(self):
        gp1 = self.gp
        gp1.save_trained_model("test_semi.npz")
        gp2 = SparseGP(fname="test_semi.npz")

        pts = np.reshape(np.linspace(-5, 5, 20), (-1, 1))
        p1 = gp1.predict(pts)
        v1 = gp1.variance(pts)
        p2 = gp2.predict(pts)
        v2 = gp2.variance(pts)
        self.assertTrue((p1 == p2).all())
        self.assertTrue((v1 == v2).all())



if __name__ == '__main__':
    unittest.main()
