import numpy as np
import scipy.stats

from sigvisa import Sigvisa
from sigvisa.source.event import Event
from sigvisa.models import Distribution
import sigvisa.utils.geog as geog
import collections
import hashlib
import marshal
import types

X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI = range(5)

def marshal_fn(f):
    if f.func_closure is not None:
        raise ValueError("function has non-empty closure %s, cannot marshal!" % f.func_closure)
    s = marshal.dumps(f.func_code)
    return s

def unmarshal_fn(dumped_code):
    try:
        f_code = marshal.loads(dumped_code)
    except:
        import pdb; pdb.set_trace()
    f = types.FunctionType(f_code, globals())
    return f


class ParamModel(Distribution):

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

    def event_dict_to_array(self, ev_dict):
#        if ev_dict in self.ev_cache:
#            a = self.ev_cache[ev_dict]
#        else:
        distance = geog.dist_km((ev_dict['lon'], ev_dict['lat']), (self.site_lon, self.site_lat))
        azimuth = geog.azimuth((self.site_lon, self.site_lat), (ev_dict['lon'], ev_dict['lat']))
        a = np.array(((ev_dict['lon'], ev_dict['lat'], ev_dict['depth'], distance, azimuth),))
        return a

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
            if len(c) == 1:
                X1 = self.standardize_input_array(c=c.values()[0])
            else:
                X1 = self.event_dict_to_array(ev_dict=c)
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

        self.ll = np.sum([scipy.stats.norm.logpdf(z,  loc= self.mean, scale= self.std) for z in y])

        self.l1 = -.5 * np.log( 2 * np.pi * self.std * self.std )

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
            self.l1 = -.5 * np.log( 2 * np.pi * self.std * self.std )
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


        # HERE FOR BACKWARDS COMPATIBILITY WITH OLD PICKLED OBJECTS: REMOVE EVENTUALLY
        self.l1 = -.5 * np.log( 2 * np.pi * self.std * self.std )


        r1 = np.sum([self.l1 -.5 * ( (z - self.mean) / self.std )**2 for z in x])
        # r2 = np.sum([scipy.stats.norm.logpdf(z, loc=self.mean, scale=self.std) for z in x])
        # assert( np.abs(r1 - r2) < 0.0001)

        return r1

    def deriv_log_p(self, x, idx=None, cond=None, cond_key=None, cond_idx=None, lp0=None, eps=1e-4):
        assert(idx == None)
        X1 = self.standardize_input_array(cond)
        x = x if isinstance(x, collections.Iterable) else (x,)

        if cond_key is not None:
            deriv = 0
        else:
            deriv = np.sum( [ (self.mean - z) / (self.std ** 2)  for z in x ] )

        return deriv

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
                y1, loc= self.predict_dist(z), scale= self.regional_std) for (z, y1) in zip(regional_dist, regional_y)]
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

            tele_items = [scipy.stats.norm.logpdf(y1, loc= self.predict_dist(z), scale= self.tele_std) for (z, y1) in zip(tele_dist, tele_y)]
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
            ll = scipy.stats.norm.logpdf(y1, loc= self.predict_item(x1), scale=self.tele_std)
        else:
            ll = scipy.stats.norm.logpdf(y1, loc= self.predict_item(x1), scale=self.regional_std)
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



def poly_basisfns(order):
    basisfn_strs = ["lambda x : " + ("1" if d==0 else "x**%d" % d)   for d in range(order+1)]
    return [eval(s) for s in basisfn_strs]

class LinearBasisModel(ParamModel):

    def __init__(self, fname=None, sta=None, X=None, y=None, basisfns=None, param_mean=None, param_covar=None, noise_std=1, compute_ll=False, H=None):
        super(LinearBasisModel, self).__init__(sta=sta)

        if fname is not None:
            self.load_trained_model(fname)
            return

        X = self.standardize_input_array(X)
        n = X.shape[0]
        self.basisfns = basisfns
        H = H if H is not None else np.array([[f(x) for f in basisfns] for x in X], dtype=float)
        b = param_mean
        B = param_covar

        self.noise_var = float(noise_std**2)

        B_chol = scipy.linalg.cholesky(B, lower=True)
        B_chol_inv = scipy.linalg.inv(B_chol)
        prior_precision = np.dot(B_chol_inv.T, B_chol_inv)
        precision = 1.0/self.noise_var * np.dot(H.T, H) + prior_precision
        precision_chol = scipy.linalg.cholesky(precision, lower=True)
        precision_chol_inv = scipy.linalg.inv(precision_chol)
        covar = np.dot(precision_chol_inv.T, precision_chol_inv)
        self.mean = np.dot(covar, np.dot(prior_precision, b) + 1.0/self.noise_var * np.dot(H.T, y))
        self.sqrt_covar=precision_chol_inv
        self.ll=None
        if compute_ll:
            # marginal_covar = self.noise_var * np.eye(n) + np.dot(H, np.dot(B, H.T))
            # using the matrix inversion lemma:
            # (sigma2*I + H*B*H.T)^(-1) = 1/sigma2 * I - 1/sigma2**2 * H(Binv + 1/sigma2 H.T*H)^(-1)H.T

            empirical_covar = np.dot(H.T, H)
            inner = prior_precision + 1.0/self.noise_var * empirical_covar
            inner_chol = scipy.linalg.cholesky(inner)
            inner_chol_inv = scipy.linalg.inv(inner_chol)
            inner_inv = np.dot(inner_chol_inv.T, inner_chol_inv)
            outer = 1.0/self.noise_var**2 * np.dot(H, np.dot(inner_inv, H.T))
            marginal_precision = np.eye(outer.shape[0]) * 1/self.noise_var - outer

            # also using the matrix determinant lemma:
            # |sigma2*I + H*B*H.T| = |Binv + 1/sigma2 * H.T * H| * |B| * |sigma2 * I|
            inner_logdet = 2*np.log(np.diag(inner_chol)).sum()
            B_logdet = 2*np.log(np.diag(B_chol)).sum()
            s2I_logdet = n * np.log(self.noise_var)
            marginal_covar_logdet = inner_logdet + B_logdet + s2I_logdet

            r = y - np.dot(H, b)
            tmp = np.dot(marginal_precision, r)
            data_term = np.dot(r.T, tmp)

            self.ll =  -.5 * (data_term + n * np.log(2*np.pi) + marginal_covar_logdet)
            self.ll_deriv = noise_std * (np.dot(tmp.T, tmp) - np.trace(marginal_precision))

    def save_trained_model(self, fname):
        with open(fname, 'wb') as f:
            np.savez(f,
                     sqrt_covar=self.sqrt_covar,
                     mean=self.mean,
                     noise_var=self.noise_var,
                     ll=self.ll,
                     basisfns = np.array([marshal_fn(f) for f in self.basisfns], dtype=object))

    def load_trained_model(self, fname):
        npzfile = np.load(fname)
        self.sqrt_covar = npzfile['sqrt_covar']
        self.mean = npzfile['mean']
        self.noise_var = npzfile['noise_var']
        self.ll = npzfile['ll']
        self.basisfns = [unmarshal_fn(code) for code in npzfile['basisfns']]
        del npzfile.f
        npzfile.close()

    def predict(self, cond):
        X1 = self.standardize_input_array(cond)
        X2 = np.array([[f(x) for f in self.basisfns] for x in X1])
        return np.dot(X2, self.mean)

    def log_likelihood(self):
        return self.ll

    def covariance(self, cond, return_sqrt=False, include_obs=False):
        X1 = self.standardize_input_array(cond)
        X2 = np.array([[f(x) for f in self.basisfns] for x in X1], dtype=float)
        tmp = np.dot(self.sqrt_covar, X2.T)

        if return_sqrt and not include_obs:
            return tmp.T
        else:
            covar = np.dot(tmp.T, tmp) + (self.noise_var if include_obs else 1e-8) * np.eye(X1.shape[0])
            if return_sqrt:
                return scipy.linalg.cholesky(covar, lower=True)
            else:
                return covar

    def variance(self, cond, include_obs=False):
        return np.diag(self.covariance(cond=cond, include_obs=include_obs))

    def log_p(self, x, cond):
        mean = self.predict(cond)
        n = len(mean)
        covar = self.covariance(cond, include_obs=True)

        r = x - mean

        if n==1:
            var = covar[0,0]
            ll1 = - .5 * ((r)**2 / var + np.log(2*np.pi*var) )

        chol = scipy.linalg.cholesky(covar, lower=True)
        ld2 = np.log(np.diag(chol)).sum()
        csi = scipy.linalg.inv(chol)
        tmp = np.dot(csi, r)
        d = np.dot(tmp.T, tmp)
        ll =  -.5 * ( d + n * np.log(2*np.pi)) - ld2

        return ll

    def sample(self, cond, include_obs=False):
        mean = self.predict(cond)
        n = len(mean)
        covar_sqrt = self.covariance(cond, return_sqrt=True, include_obs=include_obs)

        samples = np.random.randn(covar_sqrt.shape[1], 1)
        samples = mean + np.dot(covar_sqrt, samples).flatten()
        return samples
