import numpy as np
import obspy.signal.filter

from sigvisa.graph.nodes import Node
from sigvisa import Sigvisa
from sigvisa.models.distributions import TruncatedGaussian, Gaussian, MultiGaussian, InvGamma
from sigvisa.models.noise.armodel.model import ARModel, ErrorModel
from sigvisa.models.noise.armodel.learner import ARLearner
from sigvisa.models.noise.noise_util import get_noise_model
from sigvisa.models.noise.noise_model import NoiseModel


class NoiseModelNode(Node):
    # a codaHeightNode is the descendent of an amp_transfer node.
    # it adds in the event source amplitude, deterministically

    def __init__(self, waveform, nmid=None, is_env=False, **kwargs):
        self.is_env = is_env
        if nmid is not None:
            self.prior_nmid = nmid
            self.prior_nm = NoiseModel.load_by_nmid(Sigvisa().dbconn, nmid)            
        else:
            self.prior_nm, self.prior_nmid, _ = get_noise_model(waveform=waveform, model_type="ar", return_details=True)

        if is_env:
            self.prior_mean_dist = TruncatedGaussian(self.prior_nm.c, std=0.1, a=0.0)
        else:
            self.prior_mean_dist = Gaussian(self.prior_nm.c, std=0.1)

        prior_alpha = 100
        self.prior_var_dist = InvGamma(prior_alpha, (self.prior_nm.em.std**2) * (prior_alpha - 1))

        n_p = len(self.prior_nm.params)
        self.prior_param_dist = MultiGaussian(self.prior_nm.params, np.eye(n_p) * 0.01, pre_inv=True)
        nm = self.prior_nm.copy()

        super(NoiseModelNode, self).__init__(initial_value=nm, **kwargs)
        self.set_value(nm)


    def log_p(self, parent_values=None, *kwargs):
        nm = self.get_value()
        lp = self.prior_mean_dist.log_p(nm.c)
        lp += self.prior_var_dist.log_p(nm.em.std**2)
        lp += self.prior_param_dist.log_p(nm.params)
        return lp

    def parent_predict(self, set_new_value=True):
        nm = self.prior_nm.copy()
        if set_new_value:
            self.set_value(nm)
        else:
            return nm

    def parent_sample(self, set_new_value=True):
        nm = self.prior_nm.copy()
        nm.c = self.prior_mean_dist.sample()
        nm.emd.std = np.sqrt(self.prior_var_dist.sample())
        nm.params = self.prior_param_dist.sample()
        if set_new_value:
            self.set_value(nm)
        else:
            return nm

    def set_value(self, nm, **kwargs):
        super(NoiseModelNode, self).set_value(value=nm, **kwargs)
        if self.is_env:
            self.nm_env = nm
        else:
            self._gen_nm_env(nm)

    def _gen_nm_env(self, nm, seed=0, len_s=120):
        # given a non-env noise model, generate (deterministically) a
        # corresponding noise model for envelopes. this is not part of
        # the probability model, but is used as part of various
        # proposal distributions.

        rng_state = np.random.get_state()
        np.random.seed(seed)
        
        srate = nm.sf
        n = int(len_s * srate)
        sampled_noise = nm.sample(n)
        env = obspy.signal.filter.envelope(sampled_noise)
        L = ARLearner(env, sf=srate)
        params, std = L.yulewalker(nm.p)
        em = ErrorModel(0, std)
        nm_env = ARModel(params, em, c=L.c, sf=srate)
        np.random.set_state(rng_state)


        self.nm_env = nm_env

