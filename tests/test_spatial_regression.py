import numpy as np
import unittest

from sigvisa.learn.train_param_common import learn_model, load_model, gp_extract_features


class TestModels(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [120, 30, 0, 1000, 32],
            [118, 31, 0, 1050, 32],
            [120, 29, 40, 1000, 34],
            [110, 30, 20, 3000, 34],
        ])
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
        self.testX1 = np.array([[120, 30, 0, 1025, 32], ])
        self.testX2 = np.array([[119, 31, 0, 1000, 33], ])

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

    def test_GP(self):
        model = learn_model(self.X, self.y, model_type="gp_dad_log", target="coda_decay", sta='AAK')
        pred1 = model.predict(self.testX1)
        pred2 = model.predict(self.testX2)

        fname = "test_gp_model"
        model.save_trained_model(fname)
        nmodel = load_model(fname, "gp_dad_log")
        pred3 = nmodel.predict(self.testX1)
        self.assertAlmostEqual(pred1, pred3)

        ll = nmodel.log_likelihood()
        ll1 = nmodel.log_p(cond=gp_extract_features(self.X, "dad_log"), x=self.y)
        # unlike with the other models, these likelihoods are not
        # necessarily equal in the GP case

        s = nmodel.sample(self.X)


if __name__ == '__main__':
    unittest.main()
