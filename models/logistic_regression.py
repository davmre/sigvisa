import numpy as np

class LogisticRegressionModel(object):
    def __init__(self, weights, b, x_mean=None, x_scale=None, sta=None, phase=None):

        self.weights = weights
        self.b = b

        if x_mean is None:
            x_mean = np.zeros(weights.shape)
        self.x_mean = x_mean

        if x_scale is None:
            x_scale = np.ones(weights.shape)
        self.x_scale = x_scale

        self.sta = sta
        self.phase = phase

    def predict_prob(self, x):
        centered = (x-self.x_mean) / self.x_scale
        log_odds = np.dot(centered, self.weights) + self.b
        return 1.0 / (1.0 + np.exp(-log_odds))
