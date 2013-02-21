import numpy as np
import cPickle


class Distribution(object):

    def dim(self):
        raise NotImplementedError('abstract base class')

    def mean(self, cond=None):
        raise NotImplementedError('abstract base class')

    def sample(self):
        raise NotImplementedError('abstract base class')

    def log_p(self, x):
        raise NotImplementedError('abstract base class')

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


    def unflatten(self, x):
        return x

    def flatten(self, x):
        return x

class TimeSeriesDist(Distribution):
    def mean(self, n):
        raise NotImplementedError('abstract base class')

    def sample(self, n):
        raise NotImplementedError('abstract base class')



class ConditionalDist(Distribution):
    def mean(self, cond=None):
        raise NotImplementedError('abstract base class')

    def sample(self, cond=None):
        raise NotImplementedError('abstract base class')

    def log_p(self, cond=None):
        raise NotImplementedError('abstract base class')

