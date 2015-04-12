import numpy as np
from sigvisa.treegp.gp import GP
import scipy.stats

"""
Model a specific parameter at a specific station, jointly over many events.
(it'd also maybe be nice to allow jointness over stations?)

Should be able to
a) compute prior marginal for any given event
b) collect messages upwards from events
c) compute posterior marginal for any given event given other messages

This means we have to know the set of events we're modeling?
When we create a joint model, we create it for a station. So actually every wave node of that station will have the *same* joint model; there's nothing to project.
The issue is when we do a CSSM for a particular event.

I'm gonna need to special-case this in the signal model code already. So let's not project at all, just call things directly there.

Where do messages come from? Another way is to say, how do we know the set of events that the joint GP model is *currently* defined over?

Each wave_node will be affiliated with a list (indexed by param) of JGPs covering that station.
Each wave_node also gets updated whenever its set of arrivals changes. So we ought to be able to remove messages from events as they disappear from the wave_node.
When a new arrival appears, we'll default to creating a CSSM for it. When we do that, we need means and variances for the CSSM. So we need to ask the JGP for those.

When and how does a message get passed up?
Obvious times are:
   - when we build the CSSM, and would need to communicate about a posterior anyway
   - when we actually call log_p
It doesn't seem like there's any reason to do this during log_p.
"""

def multiply_scalar_gaussian(m1, v1, m2, v2):
    """
    Unnormalized product of two Gaussian densities: precisions add, means are averaged weighted by precision.
    """

    assert(v1 >= 0 or v2 >= 0)
    # without loss of generality, ensure v1 is nonnegative
    if v1 < 0 and v2 > 0:
        v2, v1 = v1, v2
        m2, m1 = m1, m2

    if v1 == 0:
        v1 = 1e-10
    if v2 == 0:
        v2 = 1e-10

    prec = (1.0/v1 + 1.0/v2)
    v = 1.0/prec if ( np.abs(prec) > 1e-10) else 1e10
    m = v * (m1/v1 + m2/v2)

    if v2 > 0:
        normalizer_var = v1 + v2
        normalizer = -.5*np.log(2*np.pi*normalizer_var) - .5*(m2-m1)**2/normalizer_var
    else:
        normalizer_var = -(v1+v2)
        if normalizer_var < 1e-20:
            # if v1 and v2 are essentially the same, we're passing a uniform message.
            # So we need to just cancel out the (very negative) logp of the message distribution evaluated at its mean,
            # so that the final logp at the mean is 0.
            normalizer = .5*np.log(2*np.pi*v)
        else:
            # http://davmre.github.io/statistics/2015/03/27/gaussian_quotient/
            normalizer = .5*np.log(2*np.pi*normalizer_var) + .5*(m2-m1)**2/normalizer_var
            normalizer += np.log( -v2/normalizer_var  )

    return m, v, normalizer

class JointGP(object):

    def __init__(self, param, sta, ymean, noise_var, cov):
        self.param = param
        self.sta = sta
        self.noise_var = noise_var
        self.ymean = ymean
        self.cov = cov

        self.messages = dict()
        self.evs = dict()
        self._input_cache=False
        self._cached_gp = dict()
        self.Z = 0

    def generic_upwards_message(self, v, cond):
        eid, evdict = self._ev_from_pv(cond)
        self.evs[eid] = evdict
        self.messages[eid] = v, 0.0, 0.0
        self._clear_cache()

    def message_from_arrival(self, eid, evdict, prior_mean, prior_var, posterior_mean, posterior_var):
        self.evs[eid] = evdict
        m, v, Z = multiply_scalar_gaussian(posterior_mean, posterior_var, prior_mean, -prior_var)
        self.messages[eid] = m,v, Z
        self._clear_cache()
        return m,v

    def get_message(self, eid):
        return self.messages[eid]

    def remove_arrival(self, eid):
        del self.evs[eid]
        del self.messages[eid]
        del self._cached_gp[eid]
        self._clear_cache()

    def prior(self):
        return self.ymean, self.cov.wfn_params[0] + self.noise_var

    def posterior(self, holdout_eid, evdict=None):
        """
        posterior on value for event eid, given messages from all other events
        """
        gp = self.train_gp(holdout_eid=holdout_eid)
        if gp is None:
            return self.prior()
        evdict = self.evs[holdout_eid] if evdict is None else evdict
        x = self._ev_features(evdict).reshape(1, -1)
        posterior = gp.predict(x), gp.variance(x, include_obs=True, pad=0)
        #print holdout_eid, "posterior", posterior
        return posterior

    def log_likelihood(self):
        gp = self.train_gp()
        if gp is None:
            return 0.0
        return gp.log_likelihood() + self.Z

    def train_gp(self, holdout_eid=None):
        if holdout_eid not in self._cached_gp:
            eids, X, y, yvar = self._gp_inputs()
            if len(eids) == 0:
                return None

            compute_ll = True
            if holdout_eid is not None:
                mask = eids != holdout_eid
                X = X[mask,:]
                y = y[mask]
                yvar = yvar[mask]
                compute_ll = False
            else:
                self.Z = np.sum([self.messages[eid][2] for eid in eids])
            gp = GP(X=X, y=y, y_obs_variances=yvar, cov_main=self.cov, ymean=self.ymean, compute_ll=compute_ll, sparse_invert=False, noise_var=self.noise_var, sort_events=False)

            self._cached_gp[holdout_eid] = gp
        return self._cached_gp[holdout_eid]


    def _clear_cache(self):
        self._input_cache=False
        self._cached_gp = dict()
        self.Z = 0

    def _gp_inputs(self):
        if not self._input_cache:
            eids = np.array(sorted(self.evs.keys()))
            self._eids = eids
            self._X = np.array([self._ev_features(self.evs[eid]) for eid in eids])
            self._y = np.array([self.messages[eid][0] for eid in eids])
            self._y_obs_variances = np.array([self.messages[eid][1] for eid in eids])
            self._input_cache=True

        return self._eids, self._X, self._y, self._y_obs_variances

    def _ev_features(self, evdict):
        return np.array((evdict['lon'], evdict['lat'], evdict['depth'], 0.0, evdict['mb']))

    def _ev_from_pv(self, parent_values):
        eid = int(parent_values.keys()[0].split(';')[0])
        evdict = {'lon': parent_values['%d;lon' % eid],
                  'lat': parent_values['%d;lat' % eid],
                  'depth': parent_values['%d;depth' % eid],
                  'mb': parent_values['%d;mb' % eid]}

        return eid, evdict

    def __getstate__(self):
        self._clear_cache()
        d = self.__dict__.copy()
        return d
