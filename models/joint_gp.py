import numpy as np
from sigvisa.treegp.gp import GP, GPCov
from sigvisa.models.spatial_regression.baseline_models import ConstGaussianModel, ConstLaplacianModel
from sigvisa.models.spatial_regression.linear_basis import LinearBasisModel
import scipy.stats
from collections import defaultdict

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
    if prec > 1e-10:
        v = 1.0/prec
        m = v * (m1/v1 + m2/v2)
        if v2 > 0:
            normalizer_var = v1 + v2
            normalizer = -.5*np.log(2*np.pi*normalizer_var) - .5*(m2-m1)**2/normalizer_var
        else:
            normalizer_var = -(v1+v2)
            # http://davmre.github.io/statistics/2015/03/27/gaussian_quotient/
            normalizer = .5*np.log(2*np.pi*normalizer_var) + .5*(m2-m1)**2/normalizer_var
            normalizer += np.log( -v2/normalizer_var  )
    else:
        v = 1e10
        m = 0.0
        normalizer = .5*np.log(2*np.pi*v)

    assert (v >= 0)
    assert(np.isfinite(normalizer))

    return m, v, normalizer



class JointGP(object):

    def __init__(self, param, sta, ymean, hparam_nodes, param_model=None):
        self.param = param
        self.sta = sta
        self.ymean = ymean

        self.hparam_nodes = hparam_nodes
        self._gpinit_params = gpinit_from_model(param_model)

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

    def message_from_arrival(self, eid, evdict, prior_mean, prior_var, posterior_mean, posterior_var, coef=None):
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
        noise_var, cov = self._get_cov()
        return self.ymean, cov.wfn_params[0] + noise_var

    def posterior(self, holdout_eid, x=None, evdict=None):
        """
        posterior on value for event eid, given messages from all other events
        """
        gp = self.train_gp(holdout_eid=holdout_eid)
        if gp is None:
            return self.prior()
        if x is None:
            evdict = self.evs[holdout_eid] if evdict is None else evdict
            x = self._ev_features(evdict).reshape(1, -1)
        posterior = gp.predict(x), gp.variance(x, include_obs=True, pad=0)
        #print holdout_eid, "posterior", posterior
        return posterior

    def log_likelihood(self):
        """Explanation of likelihood calculations:

        Suppose we have two signals s1, s2, each described by a
        corresponding param c1, c2 with a joint Gaussian prior p(c) =
        p(c1, c2). We also have the ability to compute p_T(si) = \int
        p(si|ci)p_T(ci) dci for any Gaussian distribution T on the
        coef ci, along with the coef posterior p_T(ci|si) (in my case
        this is the Kalman filter that runs inside the signal model).

        We want to compute the likelihood p(s) = p(s1, s2) which
        integrates over c1, c2 with respect to their priors. As stated
        above it's easy to do this for any single signal; the hard
        part is tracking the dependence between the two coefs. A
        simple solution would be to apply the chain rule, p(s1,s2) =
        p(s1)p(s2|s1) where we can use the Kalman filter to compute
        p(s1) (letting T be the true prior), and then use the
        posterior on c1 generated from s1 to get a GP posterior on c2,
        which we can now set as the Kalman filter prior T to evaluate
        p(s2|s1).

        TODO: why was this bad?

        Instead, we think of a message-passing algorithm, in which we
        eliminate the signal nodes and pass messages representing
        their observations upwards to the GP. The log-likelihood of an
        entire model is the product of elimination messages converging
        at any node -- in our case we use the GP on c as the root
        node.

        Each signal sends a message to this node, which is just the
        likelihood p(si|ci) considered as a function of ci. Since the
        signal model is Gaussian, this likelihood will have Gaussian
        form, but I don't have any code to efficiently compute it
        directly. Instead we can compute the posterior p(ci|si), and
        recover the likelihood:

        p(si|ci) = Z * p(ci|si)/p(ci)

        (this is just Bayes' rule rearranged, where Z is the local
        model evidence int_ci p(si|ci p(ci) dci).

        To get an (unnormalized) Gaussian form for the likelihood, we
        divide the posterior by the prior -- this will give a mean,
        variance and a normalizing constant (from
        multiply_scalar_gaussian above).  This normalizing constant is
        then *multiplied* by Z (or added in logspace) to get the full
        normalizing constant for the Gaussian message.  (here Z is the
        upwards_message_normalizer computed by Kalman filtering under
        the prior distribution -- the new normalizing constant is the
        "correction" from having run the filter under the actual prior
        to having run under a flat prior, which is what I originally
        wanted to do but runs into numerical issues).

        Finally we train a GP using the means and variances of the
        messages, and gp.log_likelihood() computes the product of all
        those messages integrated over c, which would be the model log-likelihood
        if we ignored normalizing constants. But we do actually have to
        include normalizing constants, which means that we add in the
        "correction" normalizers here as well as adding in (elsewhere in the code)
        the Z's computed as upwards_message_normalizers.

        """

        try:
            gp = self.train_gp()
            if gp is None:
                return 0.0
            return gp.log_likelihood() + self.Z
        except np.linalg.linalg.LinAlgError:
            return -np.inf

    def train_gp(self, holdout_eid=None):

        noise_var, cov = self._get_cov()

        k = (holdout_eid, noise_var, cov)

        if k not in self._cached_gp:
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
            gp = GP(X=X, y=y, y_obs_variances=yvar, cov_main=cov, ymean=self.ymean, compute_ll=compute_ll, sparse_invert=False, noise_var=noise_var, sort_events=False, **self._gpinit_params)

            self._cached_gp[k] = gp
        return self._cached_gp[k]

    def holdout_evidence(self, eid):
        """
        NOTE: not used in inference, just post-hoc analysis.

        We want to compute the model evidence int_coef p(signal | coef) p(coef),
        using the unconditional prior on coefs, and the conditional prior in which we regress from
        all other events.

        We have available the message passed upwards from the signal, which is a normalized version of
        p(signal|coef). (and we don't care about the normalization constant since the thing we'll


        be changing is the prior, p(coef), so however we normalize the message will just cancel out).
        So this is as simple as just multiplying the Gaussian message by the two Gaussian priors,
        and comparing the resulting normalization constants.
        """

        m1, v1 = self.prior()
        m2, v2 = self.posterior(eid)

        message_mean, message_var, _ = self.messages[eid]

        _, _, Z1 = multiply_scalar_gaussian(message_mean, message_var, m1, v1)
        _, _, Z2 = multiply_scalar_gaussian(message_mean, message_var, m2, v2)

        return Z2-Z1


    def pairwise_evidence(self, eid1, eid2):
        """
        NOTE: not used in inference, just post-hoc analysis.

        Compute the evidence ratio for the arrival of eid2, given eid1, vs eid2 given the prior.
        Also do the reverse: eid1|eid2 vs unconditional.
        """

        m1, v1 = self.prior()

        noise_var, cov = self._get_cov()

        # train a GP only on a single eid
        def single_input_gp(eeid):
            X = np.array([self._ev_features(self.evs[eid]) for eid in [eeid,]])
            y = np.array([self.messages[eid][0] for eid in [eeid,]])
            y_obs_variances = np.array([self.messages[eid][1] for eid in [eeid,]])
            return GP(X=X, y=y, y_obs_variances=y_obs_variances, cov_main=cov, ymean=self.ymean, compute_ll=False, sparse_invert=False, noise_var=noise_var, sort_events=False, **self._gpinit_params)

        gp1 = single_input_gp(eid1)
        gp2 = single_input_gp(eid2)
        evdict1 = self.evs[eid1]
        evdict2 = self.evs[eid2]
        x1 = self._ev_features(evdict1).reshape(1, -1)
        x2 = self._ev_features(evdict2).reshape(1, -1)

        posterior12 = gp1.predict(x2), gp1.variance(x2, include_obs=True, pad=0)
        posterior21 = gp2.predict(x1), gp2.variance(x1, include_obs=True, pad=0)


        # ratio for eid1
        message_mean, message_var, _ = self.messages[eid1]
        _, _, Z1 = multiply_scalar_gaussian(message_mean, message_var, m1, v1)
        _, _, Z2 = multiply_scalar_gaussian(message_mean, message_var, posterior21[0], posterior21[1])
        ratio1 = Z2-Z1

        # ratio for eid2
        message_mean, message_var, _ = self.messages[eid2]
        _, _, Z1 = multiply_scalar_gaussian(message_mean, message_var, m1, v1)
        _, _, Z2 = multiply_scalar_gaussian(message_mean, message_var, posterior12[0], posterior12[1])
        ratio2 = Z2-Z1

        return ratio1, ratio2

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
        for k in parent_values.keys():
            try:
                eid = int(k.split(';')[0])
            except ValueError:
                continue

        evdict = {'lon': parent_values['%d;lon' % eid],
                  'lat': parent_values['%d;lat' % eid],
                  'depth': parent_values['%d;depth' % eid],
                  'mb': parent_values['%d;mb' % eid]}

        return eid, evdict

    def _get_cov(self):

        noise_var = self.hparam_nodes["noise_var"].get_value()

        # the marginal variance of wavelet params on raw signals is 
        # redundant with coda height and thus not identifiable. 
        try:
            signal_var = self.hparam_nodes["signal_var"].get_value()
        except:
            assert(0 <= noise_var <= 1)
            signal_var = 1 - noise_var

        depth_lscale = self.hparam_nodes["depth_lscale"].get_value()
        horiz_lscale = self.hparam_nodes["horiz_lscale"].get_value()
        #wfn_str = parent_values[param_prefix + "wfn_str"]
        wfn_str = "se"

        cov = GPCov(wfn_str=wfn_str, wfn_params=np.array((signal_var,)),
                    dfn_str="lld", dfn_params=np.array((horiz_lscale, depth_lscale)))
        return noise_var, cov




    def __getstate__(self):
        self._clear_cache()
        d = self.__dict__.copy()

        # avoid problems with pickling recursive structures.
        # this pointer is reset upon unpickling
        # by the setstate() method of SigvisaGraph
        del d['hparam_nodes']

        return d

class JointIndepGaussian(JointGP):
    def __init__(self, param, sta, ymean, hparam_nodes, param_model=None):
        self.param = param
        self.sta = sta
        self.ymean = ymean

        self.hparam_nodes = hparam_nodes
        #self._gpinit_params = gpinit_from_model(param_model)

        self.messages = defaultdict(dict)


    def generic_upwards_message(self, v, cond):
        self.messages[eid] = 0.0
        self._clear_cache()

    def message_from_arrival(self, eid, evdict, prior_mean, prior_var, posterior_mean, posterior_var, coef):
        m, v, Z = multiply_scalar_gaussian(posterior_mean, posterior_var, prior_mean, -prior_var)
        self.messages[eid] = Z

    def get_message(self, eid):
        return self.messages[eid]

    def remove_arrival(self, eid):
        del self.messages[eid]
        self._clear_cache()

    def prior(self):
        noise_var = self.hparam_nodes["level_var"].get_value()
        return self.ymean, noise_var

    def posterior(self, holdout_eid, x=None, evdict=None):
        return self.prior()

    def log_likelihood(self):
        Z = np.sum(self.messages.values())
        return Z

    def _clear_cache(self):
        self._input_cache=False

    def holdout_evidence(self, *args, **kwargs):
        return 0.0


def gpinit_from_model(model):
    gpinit_dict = dict()
    if isinstance(model, ConstGaussianModel):
        gpinit_dict['basis']="mlinear"
        gpinit_dict['extract_dim']=np.array((), dtype=int)
        gpinit_dict['param_mean']=model.mean
        gpinit_dict['param_cov']=np.asarray(model.variance(None)).reshape((1,1))
    elif isinstance(model, ConstLaplacianModel):
        gpinit_dict['basis']="mlinear"
        gpinit_dict['extract_dim']=np.array((), dtype=int)
        gpinit_dict['param_mean']=model.center
        gpinit_dict['param_cov']=np.asarray(model.variance(None)).reshape((1,1))
    elif isinstance(model, LinearBasisModel):
        gpinit_dict["basis"] = model.basis
        gpinit_dict["featurizer_recovery"] = model.featurizer_recovery
        gpinit_dict["param_mean"] = model.param_mean()
        gpinit_dict["param_cov"] = model.param_covariance()
    elif model is None:
        pass
    else:
        raise Exception("unrecognized model type %s" % str(type(model)))

    return gpinit_dict
