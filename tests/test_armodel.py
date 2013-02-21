import numpy as np
import numpy.ma as ma

from sigvisa.models.noise.armodel.model import ARModel, ErrorModel
from sigvisa.models.noise.armodel.learner import ARLearner

import unittest


class TestAutoregressiveModels(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_fastAR(self):
        true_params = np.array([.8, .1])
        true_std = .11
        errormodel = ErrorModel(mean=0, std=true_std)
        true_model = ARModel(true_params, errormodel)
        sampled_data = ma.masked_array(true_model.sample(1000), mask=[False, ] * 1000)

        origll = true_model.fastAR_missingData(sampled_data, 0, true_std)
        orig_slowll = true_model.slow_AR(sampled_data, 0)
        orig_fastll = true_model.fast_AR(sampled_data, 0, true_std)
        self.assertAlmostEqual(origll, orig_slowll)
        print origll, orig_slowll, orig_fastll

        sampled_data[100:200] = ma.masked
        sampled_data[400:600] = ma.masked
        fastll = true_model.fastAR_missingData(sampled_data, 0, true_std)
        slowll = true_model.slow_AR(sampled_data, 0)
        print fastll, slowll
        self.assertAlmostEqual(fastll, slowll, places=6)

    def test_AR_learning(self):
        true_params = np.array([.8, .1])
        true_std = .11
        errormodel = ErrorModel(mean=0, std=true_std)
        true_model = ARModel(true_params, errormodel)
        sampled_data = true_model.sample(1000)

        learner = ARLearner(sampled_data)
        learned_params, learned_std = learner.yulewalker(2)

        self.assertAlmostEqual(np.sum(true_params - learned_params), 0, places=1)
        self.assertAlmostEqual(true_std, learned_std, places=1)

    def test_pickle(self):
        true_params = np.array([.8, .1])
        true_std = .11
        errormodel = ErrorModel(mean=0, std=true_std)
        true_model = ARModel(true_params, errormodel)
        true_model.dump_to_file("test.armodel")

        loaded_model = ARModel.load_from_file("test.armodel")
        self.assertAlmostEqual(np.sum(true_model.params - loaded_model.params), 0)
        self.assertAlmostEqual(true_model.em.std, loaded_model.em.std)

if __name__ == '__main__':
    unittest.main()
