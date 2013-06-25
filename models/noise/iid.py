import numpy as np
import numpy.ma as ma
from scipy.stats import laplace

from sigvisa.models.noise.noise_model import NoiseModel
import scipy.weave as weave
from scipy.weave import converters


def train_l1_model(x):
    median = np.median(x)
    b = np.mean(np.abs(x - median))

    return L1IIDModel(median=median, b=b)

class L1IIDModel(NoiseModel):

    def __init__(self, median, b):
        self.median = median
        self.b = b
        self.normalizer = np.log(.5 * self.b)
        self.nomask = np.array([False,] * 5000, dtype=bool)

    def predict(self, n):
        return np.ones((n,)) * self.median

    def sample(self, n):
        return laplace.rvs(size=n, loc=self.median, scale=self.b)

    def log_p(self, x, zero_mean=False):
        n = len(x)
        c = 0 if zero_mean else self.median

        if not isinstance(x, ma.masked_array):
            x = ma.masked_array(x, mask=False)


        try:
            x.mask[0]
            m = x.mask
        except (TypeError,IndexError):
            if len(x) > len(self.nomask):
                self.nomask = np.array([False,] * (len(self.nomask)*2), dtype=bool)
            m = self.nomask

        d = x.data - c

        b = self.b

        code = """
double ll = 0;
for (int t=0; t < n; ++t) {
  if (m(t)) {
    continue;
  }
  ll = ll - fabs(d(t) / b);
}

return_val = ll;
"""
        ll = weave.inline(code,
                          ['n', 'b', 'd', 'm',],
                          type_converters=converters.blitz,
                          compiler='gcc')
        ll += self.normalizer * n
        return ll

    def slow_log_p(self, x, zero_mean=False):
        n = len(x)
        c = 0 if zero_mean else self.median
        normed = (x - c) / self.b
        normedp = np.sum(np.abs( normed  ) )
        n1 = self.normalizer * n
        ll = n1 - normedp
        return ll

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
