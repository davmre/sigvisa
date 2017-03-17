import numpy as np
import obspy.signal.filter
import logging

from sigvisa.graph.nodes import Node
from sigvisa import Sigvisa
from sigvisa.models.distributions import TruncatedGaussian, Gaussian, MultiGaussian, InvGamma
from sigvisa.models.noise.armodel.model import ARModel, ErrorModel
from sigvisa.models.noise.armodel.learner import ARLearner
from sigvisa.models.noise.noise_util import get_noise_model
from sigvisa.models.noise.noise_model import NoiseModel

from sigvisa.learn.train_param_common import load_model

from sigvisa.database.dataset import sql_multi_str

def load_noise_model_prior(sta, chan=None, band=None, hz=None, runids=None, env=False):
    # currently no way to specify n_p or env status except implicitly through runid

    s = Sigvisa()
    
    if chan is not None:
        chan_list = s.equivalent_channels(chan)
        chan_cond = "and " + sql_multi_str("chan", chan_list)
    else:
        chan_cond = ""

    band_cond = ("and band='%s'" % band) if band is not None else ""    
    runid_cond = ("and (%s)" % " or ".join(["fitting_runid=%d" % runid for runid in runids])) if runids is not None else ""
    #hz_cond = ("and hz=%.2f" % hz) if hz is not None else ""
    phase_name = "noise_%s" % ("env" if env else "raw")
    conds = "site='%s' and phase='%s' %s %s %s" % (sta, phase_name, chan_cond, band_cond, runid_cond)
    
    mean_query = "select model_fname, model_type from sigvisa_param_model where param='armean' and %s" % conds
    var_query = "select model_fname, model_type from sigvisa_param_model where param='arvar' and %s" % conds
    params_query = "select model_fname, model_type from sigvisa_param_model where param='arparams' and %s" % conds

    fmean, tmean = s.sql(mean_query)[0]
    fvar, tvar = s.sql(var_query)[0]
    fparams, tparams = s.sql(params_query)[0]

    mean_model = load_model(fmean, tmean)
    var_model = load_model(fvar, tvar)
    params_model = load_model(fparams, tparams)
    
    return mean_model, var_model, params_model

def waveform_dummy_prior(waveform, is_env=True, n_p=None):
    srate = waveform["srate"]
    n_p = int(np.ceil(srate/3.0)) if n_p is None else n_p

    prior_params = np.zeros((n_p,))
    prior_params[0] = 0.3

    wave_mean = np.mean(waveform.data)
    wave_std = np.std(waveform.data) + 1e-6

    if is_env:
        if waveform["sta"] =="PD31":
            wave_mean = 60
        prior_mean_dist = TruncatedGaussian(wave_mean, std=wave_mean/2.0, a=0.0)
    else:
        prior_mean_dist = Gaussian(0.0, std=wave_std/5.0)

    prior_alpha = 100.0
    prior_var_dist = InvGamma(prior_alpha, (wave_std/4.0)**2 * (prior_alpha - 1))
    prior_param_dist = MultiGaussian(prior_params, np.eye(n_p), pre_inv=True)


    return prior_mean_dist, prior_var_dist, prior_param_dist

class NoiseModelNode(Node):
    # a codaHeightNode is the descendent of an amp_transfer node.
    # it adds in the event source amplitude, deterministically

    def __init__(self, waveform, force_dummy=False, is_env=False, 
                 runids=None, nmid=None, init_extra_noise=True, 
                 dummy_fallback=False, **kwargs):

        self.is_env = is_env
        
        nm = None
        if nmid is not None:
            nm = NoiseModel.load_by_nmid(Sigvisa().dbconn, nmid)

        if runids is None or len(runids)==0:
            force_dummy = True

        if force_dummy:
            n_params = None if nm is None else len(nm.params)
            self.prior_mean_dist, self.prior_var_dist, self.prior_param_dist = waveform_dummy_prior(waveform, is_env, n_p = n_params)
        else:
            try:
                self.prior_mean_dist, self.prior_var_dist, self.prior_param_dist = \
                 load_noise_model_prior(sta=waveform["sta"], chan=waveform["chan"],
                                        band=waveform["band"], hz=waveform["srate"],
                                      runids=runids,
                                      env = is_env)
            except Exception as e:
                if dummy_fallback:
                    n_params = None if nm is None else len(nm.params)
                    self.prior_mean_dist, self.prior_var_dist, self.prior_param_dist = waveform_dummy_prior(waveform, is_env, n_p = n_params)
                    print e
                    logging.warning("falling back to dummy noise prior for %s" % str(waveform))
                else:
                    raise e



        v_pred, v_std = self.prior_var_dist.predict(), np.sqrt(self.prior_var_dist.variance())
        init_std = np.sqrt(v_pred + 2*v_std) if init_extra_noise else np.sqrt(v_pred)
        em = ErrorModel(mean=0.0, std=init_std)

        m_pred, m_std = self.prior_mean_dist.predict(), np.sqrt(self.prior_mean_dist.variance())
        init_mean = m_pred + (2*m_std if is_env else 0.0) if init_extra_noise else m_pred
        self.prior_nm = ARModel(params=self.prior_param_dist.predict(), em=em, 
                                c=init_mean, sf=waveform["srate"])

        if nmid is None:
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

    def parent_sample(self, set_new_value=True, max_tries=20):
        nm = self.prior_nm.copy()
        nm.c = float(self.prior_mean_dist.sample())
        nm.em.std = float(np.sqrt(self.prior_var_dist.sample()))

        stationary = False
        tries = 0
        while not stationary and tries < max_tries:
            nm.params = np.asarray(self.prior_param_dist.sample(), dtype=np.float)
            stationary = nm.stationary()
            tries += 1
        if not stationary:
            raise Exception("Gibbs sampling new AR coefficients failed")

        if set_new_value:
            self.set_value(nm)
        else:
            return nm

    def set_value(self, nm, **kwargs):
        assert(len(nm.params) == len(self.prior_nm.params))

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

