import numpy as np
from scipy.stats import laplace

from sigvisa.models.noise.noise_model import NoiseModel


def train_l1_model(x):
    median = np.median(x)
    b = np.mean(np.abs(x - median))

    return L1IIDModel(median=median, b=b)

class L1IIDModel(NoiseModel):

    def __init__(self, median, b):
        self.median = median
        self.b = b

        self.normalizer = np.log(.5 * self.b)

    def predict(self, n):
        return np.ones((n,)) * self.median

    def sample(self, n):
        return laplace.rvs(size=n, loc=self.median, scale=self.b)

    def log_p(self, x, zero_mean=False):
        n = len(x)
        c = 0 if zero_mean else self.median
        normed = (x - c) / self.b
        normedp = np.sum(np.abs( normed  ) )
        n1 = self.normalizer * n
        return n1 - normedp

    def dump_to_file(self, fname):
        with open(fname, 'w') as f:
            f.write("median %f\n" % self.median)
            f.write("b %f\n" % self.b)

    @staticmethod
    def load_from_file(fname):
        with open(fname, 'r') as f:
            median = float(f.readline().split(" ")[1])
            b = float(f.readline().split(" ")[1])
        return L1IIDModel(median=median, b=b)

    def location(self):
        return self.median

    def scale(self):
        return self.b

    def noise_model_type(self):
        return "l1"

    def order(self):
        return 0
