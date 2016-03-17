import numpy as np
import scipy.stats

from sigvisa import Sigvisa
from sigvisa.source.event import Event
from sigvisa.models import Distribution
from sigvisa.models.distributions import Beta
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
            self.site_lon, self.site_lat, self.site_elev, _, _, _, _ = Sigvisa().earthmodel.site_info(sta, 0)


    def __repr_base_params__(self):
        return repr({'site_lon': self.site_lon, 'site_lat': self.site_lat, 'site_elev': self.site_elev})

    def __unrepr_base_params__(self, d_str):
        d = eval(d_str)
        self.site_lon = d['site_lon']
        self.site_lat = d['site_lat']
        self.site_elev = d['site_elev']

    """
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
    """

    def event_dict_to_array(self, ev_dict):
        #dictkey = frozenset(ev_dict.items() +[self.site_lon,self.site_lat])
        #if dictkey in self.s.global_dict_cache:
        #    return self.s.global_dict_cache[dictkey]


        # this method is designed to accept parent_values dicts from a Sigvisa graph, which may
        # have keys of the form "eid;lon" where eid is an unknwon integer.
        for (k,v) in ev_dict.items():
            if 'lon' in k:
                lon = v
            elif 'lat' in k:
                lat = v
            elif 'depth' in k:
                depth = v
            elif 'mb' in k:
                mb = v

        distance = geog.dist_km((lon, lat), (self.site_lon, self.site_lat))
        a = np.array(((lon, lat, depth, distance, mb),), dtype=float)
        #self.s.global_dict_cache[dictkey] = a
        return a

    def event_to_array(self, event):
        #if event in self.ev_cache:
        #    a = self.ev_cache[event]
        #else:
        distance = geog.dist_km((event.lon, event.lat), (self.site_lon, self.site_lat))
        a = np.array(((event.lon, event.lat, event.depth, distance, event.mb),), dtype=float)
        #self.ev_cache[event] = a
        return a

    def standardize_input_array(self, c, **kwargs):
        if isinstance(c, np.ndarray):
                X1 = np.array(c, dtype=float, copy=False)
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

    def __init__(self, X=None, y=None, yvars=None, sta=None, fname=None, mean=None, std=None):
        super(ConstGaussianModel, self).__init__(sta=sta)

        if fname is not None:
            self.load_trained_model(fname)
            return

        # assume our points are samples from a single Gaussian with some mean and var
        # we observe each point with addition noise variance yvar.

        # I'm too lazy to work out a proper Bayesian analysis, so instead we'll do a hack where
        # we just assume var=1
        if mean is not None:
            self.mean = mean
            self.std = std
        else:
            weights = 1.0/(yvars+1.0)
            self.mean = np.average(y, weights=weights)
            variance = np.average((y-self.mean)**2, weights=weights)
            self.std = np.sqrt(variance)

        self.l1 = -.5 * np.log( 2 * np.pi * self.std * self.std )
        if y is not None:
            self.ll = np.sum(self.l1 - .5 * ((y - self.mean)/self.std)**2)
        else:
            self.ll=0

        self.s = None

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

    def log_p(self, x, cond=None, **kwargs):
        #X1 = self.standardize_input_array(cond, **kwargs)
        x = x if isinstance(x, collections.Iterable) else (x,)



        r1 = np.sum([self.l1 -.5 * ( (z - self.mean) / self.std )**2 for z in x])
        # r2 = np.sum([scipy.stats.norm.logpdf(z, loc=self.mean, scale=self.std) for z in x])
        # assert( np.abs(r1 - r2) < 0.0001)

        return r1

    def deriv_log_p(self, x, idx=None, cond=None, cond_key=None, cond_idx=None, lp0=None, eps=1e-4, **kwargs):
        assert(idx == None)
        #X1 = self.standardize_input_array(cond, **kwargs)
        x = x if isinstance(x, collections.Iterable) else (x,)

        if cond_key is not None:
            deriv = 0
        else:
            deriv = np.sum( [ (self.mean - z) / (self.std ** 2)  for z in x ] )

        return deriv

    def variance(self, cond, **kwargs):
        return self.std**2

    def __str__(self):
        return "ConstGaussian(mean=%.2f, std=%.2f)" % (self.mean, self.std)


class ConstLaplacianModel(ParamModel):

    def __init__(self, X=None, y=None, yvars=None, sta=None, fname=None, center=None, scale=None):
        super(ConstLaplacianModel, self).__init__(sta=sta)

        if fname is not None:
            self.load_trained_model(fname)
            return

        if yvars is not None:
            print "warning: ignoring message variances in estimating Laplacian"

        self.center = np.median(y) if center is None else center
        self.scale = np.mean(np.abs(y-self.center)) if scale is None else scale

        self.ll = np.sum(-np.log(2*self.scale) - np.abs(y-self.center)/self.scale)


    def save_trained_model(self, fname):
        with open(fname, 'w') as f:
            f.write(repr({'center': self.center, 'scale': self.scale, 'll': self.ll}) + "\n")
            f.write(super(ConstLaplacianModel, self).__repr_base_params__())

    def load_trained_model(self, fname):
        with open(fname, 'r') as f:
            l = f.readlines()
            p_dict = eval(l[0])
            self.center = p_dict['center']
            self.scale = p_dict['scale']
            self.ll = p_dict['ll']
            super(ConstLaplacianModel, self).__unrepr_base_params__(l[1])

    def predict(self, cond):
        X1 = self.standardize_input_array(cond)
        if len(X1.shape) == 1 or X1.shape[0] == 1:
            return self.center
        n = X1.shape[1]
        return self.center * np.ones((n, 1))

    def sample(self, cond):
        X1 = self.standardize_input_array(cond)
        return scipy.stats.laplace.rvs(size=X1.shape[0], loc=self.center, scale=self.scale)

    def log_likelihood(self):
        return self.ll

    def log_p(self, x, cond=None, **kwargs):
        #X1 = self.standardize_input_array(cond, **kwargs)
        x = x if isinstance(x, collections.Iterable) else np.array((x,))

        scale = self.scale
        center = self.center

        r1 = np.sum(-np.log(2*scale) - np.abs(x-center)/scale)
        #r2 = np.sum([scipy.stats.laplace.logpdf(z,  loc=self.center, scale= self.scale) for z in x])

        return r1

    def deriv_log_p(self, x, idx=None, cond=None, cond_key=None, cond_idx=None, lp0=None, eps=1e-4, **kwargs):
        assert(idx == None)
        #X1 = self.standardize_input_array(cond, **kwargs)
        x = x if isinstance(x, collections.Iterable) else (x,)

        if cond_key is not None:
            deriv = 0
        else:
            deriv = np.sum( [ 0.0 if z ==self.center else float(np.sign(self.center - z))/ self.scale for z in x ] )
        return deriv

    def variance(self, cond, **kwargs):
        return 2 * self.scale**2

    def __str__(self):
        return "ConstLaplacian(center=%.3f, scale=%.3f)" % (self.center, self.scale)

class ConstBetaModel(ParamModel):

    def __init__(self, X=None, y=None, yvars=None, sta=None, fname=None, alpha=None, beta=None):
        super(ConstBetaModel, self).__init__(sta=sta)

        if fname is not None:
            self.load_trained_model(fname)
            return

        if yvars is not None:
            print "warning: ignoring message variances in estimating Laplacian"


        def lp((a, b)):
            d = Beta(a, b)
            return -np.sum(d.log_p(yy) for yy in y.flatten())
        if alpha is None:
            r = scipy.optimize.minimize(lp, np.array((2.0, 3.0)), bounds=((0, None), (0, None)))
            alpha, beta = r.x
            ll = -r.fun
        else:
            ll = -lp((alpha, beta))
        self.alpha, self.beta = alpha, beta
        self.ll = ll
        self.model = Beta(alpha, beta)

    def save_trained_model(self, fname):
        with open(fname, 'w') as f:
            f.write(repr({'alpha': self.alpha, 'beta': self.beta, 'll': self.ll}) + "\n")
            f.write(super(ConstBetaModel, self).__repr_base_params__())

    def load_trained_model(self, fname):
        with open(fname, 'r') as f:
            l = f.readlines()
            p_dict = eval(l[0])
            self.alpha = p_dict['alpha']
            self.beta = p_dict['beta']
            self.ll = p_dict['ll']
            super(ConstBetaModel, self).__unrepr_base_params__(l[1])
        self.model = Beta(self.alpha, self.beta)

    def predict(self, cond):
        X1 = self.standardize_input_array(cond)
        if len(X1.shape) == 1 or X1.shape[0] == 1:
            return self.model.predict()
        n = X1.shape[1]
        return self.model.predict() * np.ones((n, 1))

    def sample(self, cond):
        X1 = self.standardize_input_array(cond)
        return scipy.stats.beta.rvs(self.alpha, self.beta, size=X1.shape[0])

    def log_likelihood(self):
        return self.ll

    def log_p(self, x, cond=None, **kwargs):
        #X1 = self.standardize_input_array(cond, **kwargs)
        x = x if isinstance(x, collections.Iterable) else np.array((x,))
        lps = self.model.log_p(x)
        if not isinstance(lps, float) and len(lps) ==1:
            return lps[0]
        else:
            return lps 

    def deriv_log_p(self, x, idx=None, cond=None, cond_key=None, cond_idx=None, lp0=None, eps=1e-4, **kwargs):
        assert(idx == None)
        #X1 = self.standardize_input_array(cond, **kwargs)
        x = x if isinstance(x, collections.Iterable) else (x,)

        if cond_key is not None:
            deriv = 0
        else:
            raise Exception("not implemented")
        return deriv

    def variance(self, cond, **kwargs):
        return self.model.variance()

    def __str__(self):
        return "ConstBeta(alpha=%.3f, beta=%.3f)" % (self.alpha, self.beta)
