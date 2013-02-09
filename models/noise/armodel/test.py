import numpy as np
from models.noise.armodel.model import ARModel, ErrorModel
from models.noise.armodel.learne import ARLearner

class TestAutoregressiveModels(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

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

        loaded_model = load_armodel_from_file("test.armodel")
        self.assertAlmostEqual(np.sum(true_model.params - loaded_model.params), 0)
        self.assertAlmostEqual(true_model.em.std, loaded_model.em.std)
