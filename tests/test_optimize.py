import numpy as np
import unittest

import sigvisa.infer.optimize.optim_utils as optimize

class TestOptimize(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_minimize(self):
        f = lambda x : (x-17)**2 + 4
        x0 = np.array([13])
        for method in ["bfgscoord", "bfgs", "tnc", "simplex"]:
            optim_params = optimize.construct_optim_params("'normalize': False, 'method': '%s'" % method)
            x_min, f_min = optimize.minimize(f, x0, optim_params=optim_params)
            self.assertAlmostEqual(x_min, 17, places=1)
            self.assertAlmostEqual(f_min, 4, places=1)

    def test_minimize_matrix(self):

        f = lambda X : np.dot(X.flatten(), X.flatten())
        x0 = np.array([[1, 2], [3, 4]])
        x_opt = np.array([[0, 0], [0, 0]])
        x_fixedcol_opt = np.array([[1, 0], [3, 0]])
        for method in ["bfgscoord", "bfgs", "tnc"]:
            optim_params = optimize.construct_optim_params("'normalize': False, 'method': '%s', 'fix_first_cols': 0" % method)
            x_min, f_min = optimize.minimize_matrix(f, start=x0, optim_params=optim_params)
            self.assertAlmostEqual(np.sum(np.abs((x_min - x_opt)).flatten()), 0, places=1)
            self.assertAlmostEqual(f_min, 0, places=1)

            optim_params = optimize.construct_optim_params("'normalize': False, 'method': '%s', 'fix_first_cols': 1" % method)
            x_min, f_min = optimize.minimize_matrix(f, start=x0, optim_params=optim_params)
            self.assertAlmostEqual(np.sum(np.abs((x_min - x_fixedcol_opt)).flatten()), 0, places=1)
            self.assertAlmostEqual(f_min, 10, places=1)

if __name__ == '__main__':
    unittest.main()
