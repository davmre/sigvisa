import numpy as np
import cPickle



class Distribution(object):

    def dim(self):
        raise NotImplementedError('abstract base class')

    def predict(self, cond=None):
        raise NotImplementedError('abstract base class')

    def sample(self, cond=None, key_prefix=""):
        raise NotImplementedError('abstract base class')

    def log_p(self, x, cond=None, key_prefix=""):
        raise NotImplementedError('abstract base class')

    def deriv_log_p(self, x, idx=None, cond=None, cond_key=None, cond_idx=None, lp0=None, eps=1e-4, **kwargs):
        """

        Derivative of log P(X = x | cond = cond) with
        respect to x_idx (if idx is not None) or with
        respect to cond[cond_key]_{cond_idx} (if those
        quantities are not None).

        The default implementation computes a numerical
        approximation to the derivative:
        df/dx ~= f(x + eps)

        """

        lp0 = lp0 if lp0 else self.log_p(x=x, cond=cond, **kwargs)
        if cond_key is None:
            # we're computing df/dx

            if idx is None:
                # assume x is scalar
                deriv = ( self.log_p(x = x + eps, cond=cond, **kwargs) - lp0 ) / eps
            else:
                x[idx] += eps
                deriv = ( self.log_p(x = x, cond=cond, **kwargs) - lp0 ) / eps
                x[idx] -= eps

        else:
            # we're computing df/dcond[cond_key]

            if cond_idx is None:
                cond[cond_key] += eps
                deriv = ( self.log_p(x = x, cond=cond, **kwargs) - lp0 ) / eps
                cond[cond_key] -= eps
            else:
                cond[cond_key][cond_idx] += eps
                deriv = ( self.log_p(x = x, cond=cond, **kwargs) - lp0 ) / eps
                cond[cond_key][cond_idx] -= eps

        return deriv

    def dump_to_file(self, fname):
        with open(fname, 'wb') as f:
            cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(fname):
        raise NotImplementedError('abstract base class')

    def save_to_db(self, dbconn):
        raise NotImplementedError('abstract base class')

    @staticmethod
    def load_from_db(dbconn, return_extra=False):
        raise NotImplementedError('abstract base class')



class TimeSeriesDist(Distribution):
    def predict(self, n):
        raise NotImplementedError('abstract base class')

    def sample(self, n):
        raise NotImplementedError('abstract base class')



class DummyModel(Distribution):

    def __init__(self, default_value = 0.0, **kwargs):
        super(DummyModel, self).__init__(**kwargs)
        self.default_value = default_value

    def log_p(self, x, **kwargs):
        return 0.0

    def sample(self, **kwargs):
        return self.default_value

    def predict(self, **kwargs):
        return self.default_value
