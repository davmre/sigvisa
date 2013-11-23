import numpy as np
import unittest

from sigvisa.learn.train_param_common import learn_model, load_model
from sigvisa.infer.optimize.optim_utils import construct_optim_params

from sigvisa.models.spatial_regression.SparseGP import SparseGP, start_params
from sigvisa.models.spatial_regression.linear_basis import LinearBasisModel

class TestModels(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [120, 30, 0, 1000, 32],
            [118, 31, 0, 1050, 32],
            [120, 29, 40, 1000, 34],
            [110, 30, 20, 3000, 34],
        ], dtype=float)
        self.y = np.array([
            1.0,
            -2.0,
            5.0,
            20.0,
            -3.0,
        ])
        self.evids = np.array([
            1,
            2,
            3,
            4,
        ])
        self.testX1 = np.array([[120, 30, 0, 1025, 32], ], dtype=float)
        self.testX2 = np.array([[119, 31, 0, 1000, 33], ], dtype=float)

    def test_gaussian(self):
        model = learn_model(self.X, self.y, model_type="constant_gaussian", sta="AAK")
        pred1 = model.predict(self.testX1)
        pred2 = model.predict(self.testX2)
        self.assertAlmostEqual(pred1, np.mean(self.y), 3)
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

        eps = 1e-6
        llA = nmodel.log_p(x=-.01-eps)
        llB = nmodel.log_p(x=-.01+eps)
        d_hat = (llB-llA)/(2*eps)
        d = nmodel.deriv_log_p(-.01)
        self.assertAlmostEqual(d_hat, d)

    def test_laplacian(self):
        model = learn_model(self.X, self.y, model_type="constant_laplacian", sta="AAK")
        pred1 = model.predict(self.testX1)
        pred2 = model.predict(self.testX2)
        self.assertAlmostEqual(pred1, np.median(self.y))
        self.assertAlmostEqual(pred1, pred2)

        fname = "test_constant_model"
        model.save_trained_model(fname)
        nmodel = load_model(fname, "constant_laplacian")
        pred3 = nmodel.predict(self.testX1)
        self.assertAlmostEqual(pred1, pred3)

        ll = nmodel.log_likelihood()
        ll1 = nmodel.log_p(cond=self.X, x=self.y)
        self.assertAlmostEqual(ll, ll1)

        eps = 1e-4
        llA = nmodel.log_p(x=.1)
        llB = nmodel.log_p(x=.1+eps)
        d_hat = (llB-llA)/eps
        d = nmodel.deriv_log_p(.1)
        self.assertAlmostEqual(d_hat, d)

        s = nmodel.sample(self.X)

if __name__ == '__main__':
    unittest.main()
