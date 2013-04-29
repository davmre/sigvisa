import numpy as np
import unittest

from sigvisa.learn.train_param_common import learn_model, load_model
from sigvisa.infer.optimize.optim_utils import construct_optim_params

from sigvisa.models.spatial_regression.SpatialGP import SpatialGP, start_params
from sigvisa.models.spatial_regression.baseline_models import LinearBasisModel, poly_basisfns
from sigvisa.gpr.gp import GaussianProcess

from sigvisa.utils.cover_tree import CoverTree
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

    def test_GPsave(self):
        model = learn_model(self.X, self.y, model_type="gp_lld", target="coda_decay", sta='AAK', optim_params=construct_optim_params("'method': 'none', 'normalize': False"))
        pred1 = model.predict(self.testX1)
        pred2 = model.predict(self.testX2)

        fname = "test_gp_model"
        model.save_trained_model(fname)
        nmodel = load_model(fname, "gp_dad_log")
        pred3 = nmodel.predict(self.testX1)
        self.assertAlmostEqual(pred1, pred3)

        ll = nmodel.log_likelihood()
        ll1 = nmodel.log_p(cond=self.X, x=self.y)
        # unlike with the other models, these likelihoods are not
        # necessarily equal in the GP case

        s = nmodel.sample(self.X)


    def test_GPtree(self):
        model = learn_model(self.X, self.y, model_type="gp_lld", target="coda_decay", sta='AAK', optim_params=construct_optim_params("'method': 'none', 'normalize': False"))
        pred1 = model.predict(self.testX1)
        pred2 = model.predict(self.testX2)

        K1 = model.kernel(self.testX1, self.X)
        K2 = model.tree.debug_kernel_matrix(self.testX1,
                                            self.X,
                                            "se",
                                            np.array((model.kernel.rhs.params[0],), dtype=float),
                                            False)
        rel_err = (np.abs(K1 - K2)/K1).flatten()
        rel_err[K2.flatten() < 1e-100] = 0
        print rel_err
        self.assertTrue( ( rel_err < .01 ).all() )


        pred3 = GaussianProcess.predict(model, self.testX1)[0]
        pred4 = GaussianProcess.predict(model, self.testX2)[0]
        self.assertAlmostEqual(pred1, pred3)
        self.assertAlmostEqual(pred2, pred4)




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
        dummy_param = np.array((), dtype=float)
        tree = CoverTree(X, "pair", dummy_param)

        # assign an arbitrary value to each data point
        v = np.array([1,2,3,4,5,6,7,8,9,10] * 100, dtype=float)
        tree.set_v(0, v)

        query_pt = np.matrix(cluster_locations[29,:], dtype=float, copy=True)

        #w = lambda x1, x2 : np.exp(-1 * np.linalg.norm(x1-x2, 2)**2 )
        #k = [w(query_pt, x) for x in X]
        #kv = np.dot(k, v)

        weight_param = np.array((1.0,), dtype=float)
        kv_tree = tree.weighted_sum(0, query_pt, 1e-4, 'se', weight_param)
        self.assertAlmostEqual(0.893282181527, kv_tree, places=4)
        self.assertEqual(tree.fcalls, 54)


if __name__ == '__main__':
    unittest.main()
