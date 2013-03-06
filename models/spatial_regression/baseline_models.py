import numpy as np
import scipy.stats

from sigvisa import Sigvisa
from sigvisa.source.event import Event
from sigvisa.models import ConditionalDist
import sigvisa.utils.geog as geog
import collections

X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI = range(5)


class ParamModel(ConditionalDist):

    def __init__(self, sta=None, **kwargs):
        self.ev_cache = dict()

        if sta is not None:
            (self.site_lon, self.site_lat, self.site_elev, siteid) = Sigvisa().stations[sta]

    def __repr_base_params__(self):
        return repr({'site_lon': self.site_lon, 'site_lat': self.site_lat, 'site_elev': self.site_elev})

    def __unrepr_base_params__(self, d_str):
        d = eval(d_str)
        self.site_lon = d['site_lon']
        self.site_lat = d['site_lat']
        self.site_elev = d['site_elev']

    def save_trained_model(self, fname):
        raise Exception("not implemented")

    def load_trained_model(self, fname):
        raise Exception("not implemented")

    def predict(self, cond):
        raise Exception("not implemented")

    def sample(self, cond):
        raise Exception("not implemented")

    def log_likelihood(self):
        raise Exception("not implemented")

    def log_p(self, x, cond):
        raise Exception("not implemented")

    def event_to_array(self, event):
        if event in self.ev_cache:
            a = self.ev_cache[event]
        else:
            distance = geog.dist_km((event.lon, event.lat), (self.site_lon, self.site_lat))
            azimuth = geog.azimuth((self.site_lon, self.site_lat), (event.lon, event.lat))
            a = np.array(((event.lon, event.lat, event.depth, distance, azimuth),))
            self.ev_cache[event] = a
        return a

    def standardize_input_array(self, c):
        if isinstance(c, np.ndarray):
            X1 = c
        elif isinstance(c, Event):
            X1 = self.event_to_array(c)
        elif isinstance(c, dict):
            assert(len(c) == 1)
            event = c.values()[0]
            X1 = self.event_to_array(event)
        else:
            raise ValueError("unknown event object type %s input to spatial regression model!" % type(c))
        assert(len(X1.shape) == 2)
        return X1





class ConstGaussianModel(ParamModel):

    def __init__(self, X=None, y=None, sta=None, fname=None):
        super(ConstGaussianModel, self).__init__(sta=sta)

        if fname is not None:
            self.load_trained_model(fname)
            return

        self.mean = np.mean(y)
        self.std = np.std(y)

        self.ll = np.sum([scipy.stats.norm.logpdf((z - self.mean) / self.std) for z in y])

    def save_trained_model(self, fname):
        with open(fname, 'w') as f:
            f.write(repr({'mean': self.mean, 'std': self.std, 'll': self.ll}) + "\n")
            f.write(super(ConstGaussianModel, self).__repr_base_params__())

    def load_trained_model(self, fname):
        with open(fname, 'r') as f:
            l = f.readlines()
            p_dict = eval(l[0])
            self.mean = p_dict['mean']
            self.std = p_dict['std']
            self.ll = p_dict['ll']
            super(ConstGaussianModel, self).__unrepr_base_params__(l[1])

    def predict(self, cond):
        X1 = self.standardize_input_array(cond)
        if len(X1.shape) == 1 or X1.shape[0] == 1:
            return self.mean
        n = X1.shape[1]
        return self.mean * np.ones((n, 1))

    def sample(self, cond):
        X1 = self.standardize_input_array(cond)
        return scipy.stats.norm.rvs(size=X1.shape[0], loc=self.mean, scale=self.std)

    def log_likelihood(self):
        return self.ll

    def log_p(self, x, cond):
        X1 = self.standardize_input_array(cond)
        x = x if isinstance(x, collections.Iterable) else (x,)

        return np.sum([scipy.stats.norm.logpdf((z - self.mean) / self.std) for z in x])


class LinearModel(ParamModel):

    def __init__(self, sta=None, X=None, y=None, fname=None):
        super(LinearModel, self).__init__(sta=sta)

        if fname is not None:
            self.load_trained_model(fname)
            return

        n = len(y)

        self.tele_cutoff = 2000

        mean = np.mean(y)
        std = np.std(y)

        regional_dist = []
        regional_y = []
        tele_dist = []
        tele_y = []
        for i in range(n):
            d = X[i, X_DIST]
            dy = y[i]
            if d > self.tele_cutoff:
                tele_dist.append(d)
                tele_y.append(dy)
            else:
                regional_dist.append(d)
                regional_y.append(dy)

        regional_dist = np.array(regional_dist)
        tele_dist = np.array(tele_dist)

        try:
            regional_data = np.vstack([regional_dist, np.ones((len(regional_dist),))]).T
            self.regional_coeffs, residues, rank, sing = np.linalg.lstsq(regional_data, regional_y, 1e-6)

            regional_residuals = regional_y - np.array([self.predict_dist(d) for d in regional_dist])
            self.regional_std = np.std(regional_residuals) if len(regional_residuals) > 1 else std

            regional_items = [scipy.stats.norm.logpdf(
                (y1 - self.predict_dist(z)) / self.regional_std) for (z, y1) in zip(regional_dist, regional_y)]
            regional_ll = np.sum(regional_items)

        except ValueError:
            self.regional_coeffs = np.array([0, mean])
            self.regional_std = std
            regional_ll = 0

        try:
            tele_data = np.vstack([tele_dist, np.ones((len(tele_dist),))]).T
            self.tele_coeffs, residues, rank, sing = np.linalg.lstsq(tele_data, tele_y, 1e-6)

            tele_residuals = tele_y - np.array([self.predict_dist(d) for d in tele_dist])
            self.tele_std = np.std(tele_residuals) if len(tele_residuals) > 1 else std

            tele_items = [scipy.stats.norm.logpdf((y1 - self.predict_dist(z)) / self.tele_std) for (z, y1) in zip(tele_dist, tele_y)]
            tele_ll = np.sum(tele_items)

        except ValueError:
            self.tele_coeffs = np.array([0, mean])
            self.tele_std = std
            tele_ll = 0

        self.ll = regional_ll + tele_ll

    def save_trained_model(self, fname):
        f = open(fname, 'w')
        f.write("regional %.16f %.16f %.16f\n" % (tuple(self.regional_coeffs) + (self.regional_std,)))
        f.write("tele %.16f %.16f %.16f\n" % (tuple(self.tele_coeffs) + (self.tele_std,)))
        f.write("cutoff_km %.16f\n" % self.tele_cutoff)
        f.write("ll %.16f\n" % self.ll)
        f.write(super(LinearModel, self).__repr_base_params__() + '\n')
        f.close()

    def load_trained_model(self, fname):
        f = open(fname, 'r')
        regional_line = f.readline()
        tele_line = f.readline()
        cutoff_line = f.readline()
        ll_line = f.readline()
        base_param_line = f.readline()
        f.close()

        self.regional_coeffs = [float(x) for x in regional_line.split()[1:-1]]
        self.regional_std = float(regional_line.split()[-1])
        self.tele_coeffs = [float(x) for x in tele_line.split()[1:]]
        self.tele_std = float(tele_line.split()[-1])
        self.tele_cutoff = float(cutoff_line.split()[1])
        self.ll = float(ll_line.split()[1])
        super(LinearModel, self).__unrepr_base_params__(base_param_line)

    def predict_dist(self, d):
        if d > self.tele_cutoff:
            v = self.tele_coeffs[0] * d + self.tele_coeffs[1]
        else:
            v = self.regional_coeffs[0] * d + self.regional_coeffs[1]
        return v

    def predict_item(self, x):
#        import pdb; pdb.set_trace()
        d = x[X_DIST]
        return self.predict_dist(d)

    def predict(self, cond):
        X1 = self.standardize_input_array(cond)
        results = np.array([self.predict_item(x) for x in X1])
        return results

    def log_likelihood(self):
        return self.ll

    def ll_item(self, x1, y1):

        d = x1[X_DIST]
        if d > self.tele_cutoff:
            ll = scipy.stats.norm.logpdf((y1 - self.predict_item(x1)) / self.tele_std)
        else:
            ll = scipy.stats.norm.logpdf((y1 - self.predict_item(x1)) / self.regional_std)
        return ll

    def log_p(self, x, cond):
        X1 = self.standardize_input_array(cond)
        x = x if isinstance(x, collections.Iterable) else (x,)
        items = [self.ll_item(x1, y1) for (x1, y1) in zip(X1, x)]
        return np.sum(items)

    def std(self, x):
        std = self.tele_std if x[X_DIST] > self.tele_cutoff else self.regional_std
        return std

    def sample_item(self, x):
        mean = self.predict_item(x)
        s = mean + scipy.stats.norm.rvs(scale=self.std(x))
        return s

    def sample(self, cond):
        X1 = self.standardize_input_array(cond)
        return np.array([self.sample_item(x1) for x1 in X1])

