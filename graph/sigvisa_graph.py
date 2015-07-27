import time
import numpy as np
import os
import sys
import shutil
import errno
import re

from collections import defaultdict
from functools32 import lru_cache
from sigvisa import Sigvisa

from sigvisa.database.signal_data import get_fitting_runid, ensure_dir_exists, RunNotFoundException

from sigvisa.source.event import get_event
from sigvisa.learn.train_param_common import load_modelid as tpc_load_modelid
import sigvisa.utils.geog as geog
from sigvisa.models import DummyModel
from sigvisa.models.distributions import Uniform, Poisson, Gaussian, Exponential, TruncatedGaussian, LogNormal, InvGamma, Beta
from sigvisa.models.conditional import ConditionalGaussian
from sigvisa.models.ev_prior import setup_event, event_from_evnodes
from sigvisa.models.ttime import tt_predict, tt_log_p, ArrivalTimeNode
from sigvisa.models.joint_gp import JointGP, JointIndepGaussian
from sigvisa.graph.nodes import Node
from sigvisa.graph.dag import DirectedGraphModel, ParentConditionalNotDefined
from sigvisa.graph.graph_utils import extract_sta_node, predict_phases_sta, create_key, get_parent_value, parse_key
from sigvisa.models.signal_model import ObservedSignalNode, update_arrivals
from sigvisa.graph.array_node import ArrayNode
from sigvisa.models.templates.load_by_name import load_template_generator
from sigvisa.database.signal_data import execute_and_return_id
from sigvisa.models.wiggles.wavelets import construct_full_basis_implicit
from sigvisa.plotting.plot import plot_with_fit
from sigvisa.signals.common import Waveform

from sigvisa.utils.fileutils import clear_directory, mkdir_p

class ModelNotFoundError(Exception):
    pass

import cPickle as pickle
#import pickle, logging, traceback
#class SpyingPickler(pickle.Pickler, object):
#    def save(self, obj):
#        print "saving stuff"
#        logging.critical("depth: %d, obj_type: %s, obj: %s",
#                         len(traceback.extract_stack()),
#                         type(obj), repr(obj))
#        super(SpyingPickler, self).save(obj)

MAX_TRAVEL_TIME = 2000.0

@lru_cache(maxsize=1024)
def get_param_model_id(runids, sta, phase, model_type, param,
                       template_shape, chan=None, band=None):

    s = Sigvisa()
    cursor = s.dbconn.cursor()



    if len(runids) is None:
        raise ModelNotFoundError("no runid specified, so not loading parameter model.")

    chan_cond = "and chan='%s'" % chan if chan else ""
    band_cond = "and band='%s'" % band if band else ""
    for runid in runids:
        # get a DB modelid for a previously-trained parameter model
        sql_query = "select modelid, shrinkage_iter from sigvisa_param_model where model_type = '%s' and site='%s' %s %s and phase='%s' and fitting_runid=%d and template_shape='%s' and param='%s'" % (model_type, sta, chan_cond, band_cond, phase, runid, template_shape, param)
        try:
            cursor.execute(sql_query)
            results = cursor.fetchall()
            modelid = sorted(results, key = lambda x : -x[1])[0][0] # use the model with the most shrinkage iterations
        except:
            continue

        cursor.close()
        return modelid

    cursor.close()
    raise ModelNotFoundError("no model found matching model_type = '%s' and site='%s' %s %s and phase='%s' and fitting_runid in %s and template_shape='%s' and param='%s'" % (model_type, sta, chan_cond, band_cond, phase, runids, template_shape, param))



dummyPriorModel = {
"tt_residual": TruncatedGaussian(mean=0.0, std=1.0, a=-25.0, b=25.0),
"amp_transfer": Gaussian(mean=0.0, std=2.0),
"peak_offset": TruncatedGaussian(mean=-0.5, std=1.0, b=4.0),
"mult_wiggle_std": Beta(4.0, 1.0),
"coda_decay": Gaussian(mean=0.0, std=1.0),
"peak_decay": Gaussian(mean=0.0, std=1.0)
}

class SigvisaGraph(DirectedGraphModel):


    """
    Construct the Sigvisa graphical model.

    """

    def _tm_type(self, param, site=None):

        try:
            tmtype = self.template_model_type[param]
        except TypeError:
            tmtype = self.template_model_type

        if site is None: return tmtype

        s = Sigvisa()
        if s.is_array_station(site) and self.arrays_joint:
            return tmtype.replace('lld', 'lldlld')
        else:
            return tmtype


    def __init__(self, template_model_type="dummy", template_shape="paired_exp",
                 wiggle_model_type="dummy", wiggle_family="dummy", skip_levels=1,
                 dummy_fallback=False,
                 nm_type="ar",
                 run_name=None, iteration=None, runids = None,
                 phases="auto", base_srate=40.0,
                 assume_envelopes=True, smoothing=None,
                 arrays_joint=False, gpmodel_build_trees=False,
                 absorb_n_phases=False, hack_param_constraint=True,
                 uatemplate_rate=1e-3,
                 fixed_arrival_npts=None,
                 dummy_prior=None,
                 jointgp_hparam_prior=None,
                 jointgp_param_run_init=None,
                 force_event_wn_matching=False):
        """

        phases: controls which phases are modeled for each event/sta pair
                "auto": model all phases for which we have travel-time predictions from an event to station.
                "leb": model all phases whose arrival is recorded in the LEB. (works only on LEB training data)
                [list of phase names]: model a fixed set of phases
        """

        super(SigvisaGraph, self).__init__()

        self.gpmodel_build_trees = gpmodel_build_trees
        self.absorb_n_phases = absorb_n_phases
        self.hack_param_constraint = hack_param_constraint


        if template_model_type=="param":
            # sensible defaults
            self.template_model_type = {'tt_residual': 'constant_laplacian',
                                        'peak_offset': 'param_linear_mb',
                                        'coda_decay': 'param_linear_distmb',
                                        'peak_decay': 'param_linear_distmb',
                                        'mult_wiggle_std': 'constant_beta',
                                        'amp_transfer': 'param_sin1'}
        else:
            self.template_model_type = template_model_type
        self.template_shape = template_shape
        self.tg = dict()
        if type(self.template_shape) == dict:
            for (phase, ts) in self.template_shape.items():
                self.tg[phase] = load_template_generator(ts)


        self.wiggle_model_type = wiggle_model_type
        if self.wiggle_model_type=="gp_joint":
            self.jointgp = True
        else:
            self.jointgp=False
        self.wiggle_family = wiggle_family
        self.wavelet_basis_cache = dict()
        self.skip_levels = skip_levels

        self.base_srate = base_srate
        self.assume_envelopes = assume_envelopes
        self.smoothing = smoothing



        self.dummy_fallback = dummy_fallback

        self.dummy_prior = dummy_prior if dummy_prior is not None else dummyPriorModel

        self.nm_type = nm_type
        self.phases = phases

        self.phases_used = set()

        self.runids = runids if runids is not None else ()
        if run_name is not None and iteration is not None:
            cursor = Sigvisa().dbconn.cursor()
            try:
                runid = get_fitting_runid(cursor, run_name, iteration, create_if_new = False)
                self.runids = (runid,)
            except RunNotFoundException:
                self.runids = ()
            cursor.close()
        assert(isinstance(self.runids, tuple))

        self.template_nodes = []
        self.wiggle_nodes = []


        self.station_waves = defaultdict(list) # (sta) -> list of ObservedSignalNodes
        self.site_elements = dict() # site (str) -> set of elements (strs)
        self.site_bands = dict()
        self.site_chans = dict()
        self.arrays_joint = arrays_joint
        self.start_time = np.float('inf')
        self.event_start_time = np.float('inf')
        self.end_time = np.float('-inf')

        self.optim_log = ""

        self.event_rate = 0.00126599049
        self.next_eid = 1

        self.next_uatemplateid = 1
        self.uatemplate_rate = uatemplate_rate
        self.uatemplate_ids = defaultdict(set) # keys are (sta,chan,band) tuples, vals are sets of ids
        self.uatemplates = dict() # keys are ids, vals are param:node dicts.

        self.evnodes = dict() # keys are eids, vals are attribute:node dicts
        self.extended_evnodes = defaultdict(list) # keys are eids, vals are list of all nodes for an event, including templates.

        self.fixed_arrival_npts = fixed_arrival_npts

        self._joint_gpmodels = defaultdict(dict)

        self.jointgp_hparam_prior=jointgp_hparam_prior
        self.jointgp_param_run_init=jointgp_param_run_init

        if self.jointgp_hparam_prior is None:
            # todo: different priors for different params
            self.jointgp_hparam_prior = {'horiz_lscale': LogNormal(mu=3.0, sigma=3.0),
                                         'depth_lscale': LogNormal(mu=3.0, sigma=3.0),
                                         'signal_var': InvGamma(beta=3.0, alpha=4.0),
                                         'noise_var': InvGamma(beta=1.0, alpha=3.0),
                                         'level_var': InvGamma(beta=3.0, alpha=4.0),}

        self.force_event_wn_matching = force_event_wn_matching

    def joint_gpmodel(self, sta, phase, band, chan, param):
        if (param, band, chan, phase) not in self._joint_gpmodels[sta]:
            #if param.startswith("db"):
            #    noise_var, gpcov = self.jointgp_prior['wiggle']
            #else:
            #    noise_var, gpcov = self.jointgp_prior[param]

            nodes = {}
            for hparam in ("noise_var", "signal_var", "horiz_lscale", "depth_lscale"):
                prior = self.jointgp_hparam_prior[hparam]
                n = Node(label="gp;%s;%s;%s" % (sta, param, hparam), model=prior, initial_value=prior.predict())
                nodes[hparam]=n
                self.add_node(n)

            model = None
            if self.jointgp_param_run_init is not None:
                runid, tmtypes = self.jointgp_param_run_init
                if param in tmtypes:
                    modelid = self.get_param_model_id(runids=(runid,),
                                                      sta=sta,
                                                      phase=phase,
                                                      model_type=tmtypes[param],
                                                      param=param,
                                                      template_shape=self.template_shape,
                                                      chan=chan, band=band)
                    model = self.load_modelid(modelid, gpmodel_build_trees=self.gpmodel_build_trees)

            jgp = JointGP(param, sta, 0.0, hparam_nodes=nodes, param_model=model)

            self._joint_gpmodels[sta][(param, band, chan, phase)] = jgp, nodes
        return self._joint_gpmodels[sta][(param, band, chan, phase)]

    def level_coef_model(self, sta, phase, band, chan, param):
        if (param, band, chan, phase) not in self._joint_gpmodels[sta]:
            #if param.startswith("db"):
            #    noise_var, gpcov = self.jointgp_prior['wiggle']
            #else:
            #    noise_var, gpcov = self.jointgp_prior[param]

            nodes = {}
            for hparam in ("level_var",):
                prior = self.jointgp_hparam_prior[hparam]
                n = Node(label="gp;%s;%s;%s" % (sta, param, hparam), model=prior, initial_value=prior.predict())
                nodes[hparam]=n
                self.add_node(n)

            model = None
            jgp = JointIndepGaussian(param, sta, 0.0, hparam_nodes=nodes, param_model=model)

            self._joint_gpmodels[sta][(param, band, chan, phase)] = jgp, nodes
        return self._joint_gpmodels[sta][(param, band, chan, phase)]

    def ev_arriving_phases(self, eid, sta=None, site=None):
        if sta is None and site is not None:
            sta = next(iter(self.site_elements[site]))
        sta_keys = [n.label for n in self.extended_evnodes[eid] if sta in n.label]
        phases = set([parse_key(k)[1] for k in sta_keys])
        return list(phases)

    def template_generator(self, phase):
        if phase not in self.tg and type(self.template_shape) == str:
            self.tg[phase] = load_template_generator(self.template_shape)
        return self.tg[phase]

    def wavelet_basis(self, srate):
        if self.wiggle_family is None or self.wiggle_family=="dummy" or self.wiggle_family=="iid":
            return None

        if srate not in self.wavelet_basis_cache:
            self.wavelet_basis_cache[srate] = \
                construct_full_basis_implicit(srate=srate,
                                              wavelet_str=self.wiggle_family,
                                              c_format=True)
            # TODO: make sure this basis has marginal variance 1.0
        return self.wavelet_basis_cache[srate]

    def get_template_nodes(self, eid, sta, phase, band, chan):
        tg = self.template_generator(phase)
        nodes = dict()
        for param in tg.params() + ('arrival_time',):
            try:
                k, node = get_parent_value(eid=eid, sta=sta, phase=phase, param_name=param, chan=chan, band=band, parent_values=self.nodes_by_key, return_key=True)
                nodes[param]=(k, node)
            except KeyError:
                if param =="mult_wiggle_std":
                    pass
                else:
                    raise

        # if this template is a real event, also load the latent event variables
        for param in ('amp_transfer', 'tt_residual'):
            try:
                k, node = get_parent_value(eid=eid, sta=sta, phase=phase, param_name=param, chan=chan, band=band, parent_values=self.nodes_by_key, return_key=True)
                nodes[param]=(k, node)
            except KeyError:
                continue

        return nodes

    def get_template_vals(self, eid, sta, phase, band, chan):
        nodes = self.get_template_nodes(eid, sta, phase, band, chan)
        vals = dict([(p, n.get_value(k)) for (p,(k, n)) in nodes.iteritems()])
        return vals

    def get_template_nodes_byphase(self, eid, sta, band, chan):
        allnodes = dict()
        for phase in self.phases_used:
            try:
                tmnodes = self.get_template_nodes(eid, sta, phase, band, chan)
                allnodes[phase] = tmnodes
            except KeyError:
                continue
        return allnodes

    def set_template(self, eid, sta, phase, band, chan, values):
        for (param, value) in values.items():
            if param in ("arrival_time", 'amp_transfer', 'tt_residual'):
                b = None
                c = None
            else:
                b = band
                c = chan
            self.set_value(key=create_key(param=param, eid=eid,
                                          sta=sta, phase=phase,
                                          band=b, chan=c),
                           value=value)

    def get_arrival_wn(self, sta, eid, phase, band, chan, revert_to_atime=False):
        # this method and the next (_by_atime) are super naive, we
        # should track this info and just look it up instead of
        # searching every time.
        wns = self.station_waves[sta]

        matching_wns = []
        for wn in wns:
            if (eid, phase) in wn.arrivals():
                if band is None and chan is None:
                    matching_wns.append(wn)
                elif band in wn.label and chan in wn.label:
                    return wn
        if len(matching_wns) > 0:
            return matching_wns

        if revert_to_atime:
            ev = self.get_event(eid)
            atime = ev.time + tt_predict(ev, sta, phase=phase)
            return self.get_wave_node_by_atime(sta, band, chan, atime, allow_out_of_bounds=True)

        raise KeyError("couldn't find wave node for eid %d phase %s at station %s chan %s band %s" % (eid, phase, sta, chan, band))


    def get_wave_node_by_atime(self, sta, band, chan, atime, allow_out_of_bounds=False):
        wns = self.station_waves[sta]
        first_wn = None
        last_wn = None
        for wn in wns:
            if band in wn.label and chan in wn.label:
                if first_wn is None or wn.st < first_wn.st:
                    first_wn = wn
                if last_wn is None or wn.et > last_wn.et:
                    last_wn = wn

                if wn.st <= atime and wn.et >= atime:
                    return wn

        if allow_out_of_bounds:
            if first_wn is not None and atime < first_wn.st:
                return first_wn
            if last_wn is not None and atime > last_wn.et:
                return last_wn

        raise KeyError("no wave node exists for station %s chan %s band %s at time %.1f" % (sta, chan, band, atime))


    def nevents_log_p(self, n=None):
        if n is None:
            n = len(self.evnodes)

        # Poisson cancellation here works similarly to in the
        # uatemplate case (described below in ntemplates_sta_log_p)
        lp = -(self.event_rate * (self.end_time - self.event_start_time))  + n * np.log(self.event_rate)
        return lp

    def ntemplates_log_p(self):
        lp = 0
        for (site, elements) in self.site_elements.items():
            for sta in elements:
                lp += self.ntemplates_sta_log_p(sta)
        return lp

    def ntemplates_sta_log_p(self, sta, n=None):
        """

        Return the log probability of having n unassociated templates
        at station sta. If n is unspecified, use the current
        value. The probability incorporates the Poisson rate as well
        as the arrival time probabilities. (arrival times in a Poisson
        process are uniformly distributed but exchangeable, so rather
        than just having a uniform distribution for each template, we
        have to multiply in a factor of n!/T^n to account for the
        exchangeability)

        """

        if n is None:

            s = Sigvisa()
            site = s.get_array_site(sta)
            assert (len(list(self.site_bands[site])) == 1)
            band = list(self.site_bands[site])[0]
            assert (len(list(self.site_chans[site])) == 1)
            chan = list(self.site_chans[site])[0]

            n = len(self.uatemplate_ids[(sta, chan, band)])


        valid_len = 0
        for wn in self.station_waves[sta]:
            valid_len += wn.valid_len

        #n_template_dist = Poisson(self.uatemplate_rate * wn.valid_len)
        #poisson_lp = n_template_dist.log_p(n)
        #atimes_lp = scipy.special.gammaln(n+1) -n * np.log(wn.valid_len)
        #lp = poisson_lp + atimes_lp


        # we can cancel out a lot of terms from Poisson(n) * n!/T^n
        # let Y = R*T = self.uatemplate_rate * wn.valid_len
        # Poisson(n) = Y^n/n! * e^-Y
        # overall: Y^n/n! * e^-Y * n!/T^n
        # log: n log Y - Y - n log T
        #      = n log RT - RT - n log T
        #      = n log R + (n log T - n log T) - RT
        #      = n log R - RT
        lp = n * np.log(self.uatemplate_rate) - (self.uatemplate_rate * valid_len)

        return lp


    def current_log_p_breakdown(self):
        nt_lp = self.ntemplates_log_p()
        ne_lp = self.nevents_log_p()

        ua_peak_offset_lp = 0.0
        ua_coda_height_lp = 0.0
        ua_coda_decay_lp = 0.0
        ua_peak_decay_lp = 0.0
        ua_mult_wiggle_std_lp = 0.0
        for ((sta, band, chan), tmid_set) in self.uatemplate_ids.items():
            for tmid in tmid_set:
                uanodes = self.uatemplates[tmid]
                ua_peak_offset_lp += uanodes['peak_offset'].log_p()
                ua_coda_height_lp += uanodes['coda_height'].log_p()
                ua_peak_decay_lp += uanodes['peak_decay'].log_p()
                ua_coda_decay_lp += uanodes['coda_decay'].log_p()
                ua_mult_wiggle_std_lp += uanodes['mult_wiggle_std'].log_p()


        ev_prior_lp = 0.0
        ev_obs_lp = 0.0
        ev_tt_lp = 0.0
        ev_amp_transfer_lp = 0.0
        ev_peak_offset_lp = 0.0
        ev_coda_decay_lp = 0.0
        ev_peak_decay_lp = 0.0
        ev_mult_wiggle_std_lp = 0.0
        for (eid, evdict) in self.evnodes.items():
            evnode_set = set(evdict.values())
            for node in evnode_set:
                ev_prior_lp += node.log_p()

            for node in self.extended_evnodes[eid]:
                if node in evnode_set:
                    continue
                if node.deterministic():
                    continue
                if "tt_residual" in node.label:
                    try:
                        ev_tt_lp += node.log_p()
                    except ParentConditionalNotDefined:
                        pass
                elif  "amp_transfer" in node.label:
                    ev_amp_transfer_lp += node.log_p()
                elif  "coda_decay" in node.label:
                    ev_coda_decay_lp += node.log_p()
                elif  "peak_decay" in node.label:
                    ev_peak_decay_lp += node.log_p()
                elif  "peak_offset" in node.label:
                    ev_peak_offset_lp += node.log_p()
                elif  "mult_wiggle_std" in node.label:
                    ev_mult_wiggle_std_lp += node.log_p()
                elif "obs" in node.label:
                    ev_obs_lp += node.log_p()
                else:
                    raise Exception('unexpected node %s' % node.label)

        signal_lp = 0.0
        for (sta_, wave_list) in self.station_waves.items():
            for wn in wave_list:
                try:
                    signal_lp += wn.log_p()
                except ParentConditionalNotDefined:
                    signal_lp += wn.upwards_message_normalizer()

        jointgp_lp = 0.0
        for sta in self._joint_gpmodels.keys():
            for k in self._joint_gpmodels[sta].keys():
                jgp, _ = self._joint_gpmodels[sta][k]
                jointgp_lp  += jgp.log_likelihood()

        print "n_uatemplate: %.1f" % nt_lp
        print "n_event: %.1f" % ne_lp
        print "ev priors: ev %.1f" % (ev_prior_lp)
        print "ev observations: ev %.1f" % (ev_obs_lp)
        print "tt_residual: ev %.1f" % (ev_tt_lp)
        print "ev global cost (n + priors + tt): %.1f" % (ev_prior_lp + ev_tt_lp + ne_lp,)
        print "coda_decay: ev %.1f ua %.1f total %.1f" % (ev_coda_decay_lp, ua_coda_decay_lp, ev_coda_decay_lp+ua_coda_decay_lp)
        print "peak_decay: ev %.1f ua %.1f total %.1f" % (ev_peak_decay_lp, ua_peak_decay_lp, ev_peak_decay_lp+ua_peak_decay_lp)
        print "peak_offset: ev %.1f ua %.1f total %.1f" % (ev_peak_offset_lp, ua_peak_offset_lp, ev_peak_offset_lp+ua_peak_offset_lp)
        print "coda_height: ev %.1f ua %.1f total %.1f" % (ev_amp_transfer_lp, ua_coda_height_lp, ev_amp_transfer_lp+ua_coda_height_lp)
        print "mult_std_wiggle: ev %.1f ua %.1f total %.1f" % (ev_mult_wiggle_std_lp, ua_mult_wiggle_std_lp, ev_mult_wiggle_std_lp+ua_mult_wiggle_std_lp)
        print "coef jointgp: %.1f" % jointgp_lp
        ev_total = ev_coda_decay_lp + ev_peak_decay_lp + ev_peak_offset_lp + ev_amp_transfer_lp +ev_mult_wiggle_std_lp + jointgp_lp
        ua_total = ua_coda_decay_lp + ua_peak_decay_lp + ua_peak_offset_lp + ua_mult_wiggle_std_lp + ua_coda_height_lp
        print "total param: ev %.1f ua %.1f total %.1f" % (ev_total, ua_total, ev_total+ua_total)
        ev_total += ev_prior_lp + ev_obs_lp + ev_tt_lp + ne_lp
        ua_total += nt_lp
        print "priors+params: ev %.1f ua %.1f total %.1f" % (ev_total, ua_total, ev_total + ua_total)
        print "station noise (observed signals): %.1f" % (signal_lp)
        print "overall: %.1f" % (ev_total + ua_total + signal_lp)
        print "official: %.1f" % self.current_log_p()

    def joint_gp_ll(self, verbose=False):
        lp = 0
        for sta in self._joint_gpmodels.keys():
            for k in self._joint_gpmodels[sta].keys():
                jgp, _ = self._joint_gpmodels[sta][k]
                jgpll = jgp.log_likelihood()
                lp += jgpll
                if verbose:
                    print "jgp %s %s: %.3f" % (sta, k, jgpll)
        return lp

    def current_log_p(self, **kwargs):
        lp = super(SigvisaGraph, self).current_log_p(**kwargs)
        lp += self.ntemplates_log_p()
        lp += self.nevents_log_p()
        lp += self.joint_gp_ll()

        #if lp < -1000000:
        #import pdb; pdb.set_trace()

        if np.isnan(lp):
            raise Exception('current_log_p is nan')
        return lp

    def add_node(self, node, template=False):
        if template:
            self.template_nodes.append(node)
        super(SigvisaGraph, self).add_node(node)

    def remove_node(self, node):
        if node in self.template_nodes:
            self.template_nodes.remove(node)
        super(SigvisaGraph, self).remove_node(node)

    def get_event(self, eid):
        return event_from_evnodes(self.evnodes[eid])

    def set_event(self, eid, ev, params_changed=None, preserve_templates=True, illegal_phase_action="raise", node_lps=None):

        # Updates the given eid to have the parameters specified by the event object ev.

        # If preserve_templates is True, then the tt_residual and
        # amp_transfer nodes for each phase arrival are updated so
        # that arrival_time and coda_height remain invariant, i.e. so
        # that the actual predicted signal does not change. Otherwise,
        # arrival_time and coda_height will automatically update to
        # reflect the tt_residual and amp_transfer for the new event
        # location/mb. If preserve_templates is a list of node labels
        # (e.g., arrival_time nodes for a subset of phase arrivals),
        # then only the listed nodes will have their values preserved.

        # if some of the phases currently generated by the given event
        # are illegal at the new location, the travel-time
        # model will raise an error. Choices for "illegal_phase_action":
        # - "raise": raise the error
        # - "delete": forcibly remove the illegal phases
        # - "ignore": silently ignore the error, creating an inconsistent state(!)

        # params_changed can be None (in which case we assume all
        # event parameters need to be updated), or an expicit list of
        # parameters, e.g. ["lon", "lat"]. In the latter case, we only
        # update the given parameters.

        # node_lps is an optional UpdatedNodeLPs object tracking changes to the state, used for
        # efficiently computing MH proposal ratios

        def get_preserved_nodes(preserve_templates, params_changed):
            # return the set of nodes whose values we want to fix

            # determine if we're selectively preserving values at a list
            # of template nodes, or just boolean all-or-nothing.
            try:
                'a' in preserve_templates
                preserve_set = preserve_templates
                preserve_templates=True
            except TypeError:
                if preserve_templates:
                    preserve_set = set([n.label for n in self.extended_evnodes[eid]])
                else:
                    preserve_set = None

            # the only nodes we might need to preserve values for are arrival_time and coda_height.
            # we only need coda_height if we updated the mb.
            # and we only need arrival_time if we updated lon/lat/depth/time.
            # so here we prune preserve_set to reflect this

            atimes_updated = params_changed is None or "lon" in params_changed or "lat" in params_changed or "depth" in params_changed or "time" in params_changed
            coda_heights_updated = params_changed is None or "mb" in params_changed or "natural_source" in params_changed
            if preserve_set is not None:
                if  coda_heights_updated:
                    preserve_set_mb = set((nl for nl in preserve_set if "coda_height" in nl))
                else:
                    preserve_set_mb = set()
                if atimes_updated:
                    preserve_set_loc = set((nl for nl in preserve_set if "arrival_time" in nl))
                else:
                    preserve_set_loc = set()
                preserve_set = preserve_set_mb | preserve_set_loc
            return preserve_set, atimes_updated, coda_heights_updated

        evdict = ev.to_dict()
        evnodes = self.evnodes[eid]

        preserved_nodes, atimes_updated, coda_heights_updated = get_preserved_nodes(preserve_templates, params_changed)

        perturbed_wns = set()
        if node_lps is not None:
            for sw in self.station_waves.values():
                for wn in sw:
                    phases = [phase for (arr_eid, phase) in wn.arrivals() if arr_eid==eid]
                    for phase in phases:
                        # if we're updating a parameter that changes
                        # the atime or coda_height, and we didn't fix
                        # the atime/coda_height for this arrival, then
                        # the predicted signal (thus wn.log_p()) will
                        # change.
                        if atimes_updated:
                            k, v = get_parent_value(eid, phase, wn.sta, "arrival_time", self.all_nodes, wn.chan, wn.band, return_key=True)
                            if k not in preserved_nodes:
                                perturbed_wns.add(wn)
                                break
                        if coda_heights_updated:
                            k, v = get_parent_value(eid, phase, wn.sta, "coda_height", self.all_nodes, wn.chan, wn.band, return_key=True)
                            if k not in preserved_nodes:
                                perturbed_wns.add(wn)
                                break
        for wn in perturbed_wns:
            if wn not in node_lps.nodes_changed_old:
                node_lps.nodes_changed_old[wn] = wn.log_p()

        # record the values of the nodes we're going to preserve, so
        # we can re-set them later.
        fixed_vals = dict()
        if preserved_nodes is not None:
            for n in self.extended_evnodes[eid]:
                if n.label in preserved_nodes:
                    fixed_vals[n.label] = n.get_value()

        # set the new event values
        for k in (params_changed if params_changed is not None else evdict.keys()):
            evnodes[k].set_local_value(evdict[k], key=k, force_deterministic_consistency=False)

        # now re-set the values we promised to fix
        impossible_phases = []
        if preserved_nodes is None:
            for n in self.extended_evnodes[eid]:
                if n.deterministic():
                    try:
                        n.parent_predict()
                    except ValueError as e: # requesting travel time for impossible phase
                        if illegal_phase_action == "raise":
                            raise e
                        _,nphase,nsta,nchan,nband,nparam = parse_key(n.label)
                        impossible_phases.append((nsta, nphase))
        else:
            for n in self.extended_evnodes[eid]:
                try:
                    if n.label not in preserved_nodes:
                        if n.deterministic():
                            n.parent_predict()
                    else:
                        n.set_value(fixed_vals[n.label])
                except ValueError as e: # requesting travel time for impossible phase
                    if illegal_phase_action == "raise":
                        raise e
                    _,nphase,nsta,nchan,nband,nparam = parse_key(n.label)
                    impossible_phases.append((nsta, nphase))

        if illegal_phase_action=="delete":
            print "deleting impossible phases", impossible_phases
            for (sta, phase) in impossible_phases:
                print "removing impossible %s at %s" % (sta, phase)
                self.delete_event_phase(eid, sta, phase)

        for wn in perturbed_wns:
            node_lps.nodes_changed_new[wn] = wn.log_p()

    def delete_event_phase(self, eid, sta, phase):
        extended_evnodes = self.extended_evnodes[eid][:]
        for node in extended_evnodes:
            if sta not in node.label: continue
            neid,nphase,nsta,nchan,nband,nparam = parse_key(node.label)
            if nsta==sta and nphase==phase:
                self.remove_node(node)
                self.extended_evnodes[eid].remove(node)
        self._topo_sort()

    def predict_phases_site(self, ev, site):
        phase_set = None
        for sta in self.site_elements[site]:
            phases = set(predict_phases_sta(ev, sta, self.phases))
            phase_set = phases if phase_set is None else phase_set.intersection(phases)
        return phase_set

    def remove_event(self, eid):

        del self.evnodes[eid]
        for node in self.extended_evnodes[eid]:
            self.remove_node(node)
        del self.extended_evnodes[eid]

        self._topo_sort()

    def event_is_fixed(self, eid):
        return np.prod([n._fixed for n in self.evnodes[eid].values()])

    def fix_event(self, eid):
        for n in self.evnodes[eid].values():
            n.fix_value()


    def add_event(self, ev, tmshapes=None, sample_templates=False, fixed=False, eid=None,
                  observed=False, stddevs=None):
        """

        Add an event node to the graph and connect it to all waves
        during which its signals might arrive.

        tmshapes: optional dictionary of template shapes (strings), keyed on phase name.

        """
        if eid is None:
            eid = self.next_eid
            self.next_eid += 1
        ev.eid = eid
        evnodes = setup_event(ev, fixed=fixed)
        self.evnodes[eid] = evnodes

        # use a set here to ensure we don't add the 'loc' node
        # multiple times, since it has multiple keys
        for n in set(evnodes.itervalues()):
            self.extended_evnodes[eid].append(n)
            self.add_node(n)

        for (site, element_list) in self.site_elements.iteritems():
            for phase in self.predict_phases_site(ev=ev, site=site):
                #print "adding phase", phase, "at site", site
                self.phases_used.add(phase)
                if self.absorb_n_phases:
                    if phase == "Pn":
                        phase = "P"
                    elif phase == "Sn":
                        phase = "S"
                tg = self.template_generator(phase)
                try:
                    tt = tt_predict(ev, site, phase=phase)
                except Exception as e:
                    print e
                    continue

                self.add_event_site_phase(tg, site, phase, evnodes, sample_templates=sample_templates)

        if observed != False:
            if observed == True:
                observed_ev = ev
            else:
                observed_ev = observed
            self.observe_event(eid, observed_ev, stddevs=stddevs)

        self._topo_sort()
        return evnodes

    def observe_event(self, eid, ev, stddevs=None):
        if stddevs is None:
            stddevs = {"lon": 0.2, "lat": 0.2, "depth": 20.0, "time": 3.0, "mb": 0.3}

        evnodes = self.evnodes[eid]
        obs_dict = ev.to_dict()
        for n in set(evnodes.itervalues()):
            for k in n.keys():
                k_local = k.split(";")[1]
                if k_local not in stddevs: continue

                n_obs = Node(label=k+"_obs", model=ConditionalGaussian(k, stddevs[k_local]), parents=(n,), fixed=True, initial_value=obs_dict[k_local])
                self.extended_evnodes[eid].append(n_obs)
                self.add_node(n_obs)


    def destroy_unassociated_template(self, nodes=None, tmid=None, nosort=False):
        if tmid is not None:
            nodes = self.uatemplates[tmid]
        eid, phase, sta, chan, band, param = parse_key(nodes.values()[0].label)
        if tmid is None:
            tmid = -eid
        del self.uatemplates[tmid]
        self.uatemplate_ids[(sta,chan,band)].remove(tmid)

        for (param, node) in nodes.items():
            self.remove_node(node)
        if not nosort:
            self._topo_sort()

    def create_unassociated_template(self, wave_node, atime, nosort=False, tmid=None, initial_vals=None):

        """

        Add a new unassociated template to a particular wave node
        (TODO: generalize this to multiple wave nodes, across
        bands/chans).

        Optional: nosort, tmid, and initial_vals are used by MCMC
        moves that need to temporarily remove/reconstruct a template.

        """

        if tmid is None:
            tmid = self.next_uatemplateid
            self.next_uatemplateid += 1

        phase="UA"
        eid=-tmid
        tg = self.template_generator(phase=phase)

        tnodes = dict()
        wnodes = dict()
        at_label = create_key(param="arrival_time", sta=wave_node.sta,
                           phase=phase, eid=eid,
                           chan=wave_node.chan, band=wave_node.band)

        tnodes['arrival_time'] = Node(label=at_label, model=DummyModel(atime),
                                      initial_value=atime, children=(wave_node,),
                                      low_bound=wave_node.st, high_bound=wave_node.et)
        self.add_node(tnodes['arrival_time'], template=True)
        for param in tg.params():
            label = create_key(param=param, sta=wave_node.sta,
                               phase=phase, eid=eid,
                               chan=wave_node.chan, band=wave_node.band)
            model = tg.unassociated_model(param, nm=wave_node.nm)
            lb = tg.low_bounds()[param]
            hb = tg.high_bounds()[param]

            tnodes[param] = Node(label=label, model=model, children=(wave_node,), low_bound=lb, high_bound=hb)
            self.add_node(tnodes[param], template=True)


        for (param, node) in tnodes.items():
            node.tmid = tmid
            if initial_vals is None:
                node.parent_sample()
            else:
                node.set_value(initial_vals[param])

        nodes = tnodes

        self.uatemplates[tmid] = nodes
        self.uatemplate_ids[(wave_node.sta,wave_node.chan,wave_node.band)].add(tmid)

        if not nosort:
            self._topo_sorted_list = nodes.values() + self._topo_sorted_list
            self._gc_topo_sorted_nodes()

        return nodes

    """
    The following definitions are just here to allow the model-loading machinery to be
    monkeypatched if we need to (e.g. when using synthetic data).
    """
    def load_modelid(self, modelid, **kwargs):
        return tpc_load_modelid(modelid, **kwargs)

    def get_param_model_id(self, *args, **kwargs):
        return get_param_model_id(*args, **kwargs)

    def load_node_from_modelid(self, modelid, label, **kwargs):
        model = self.load_modelid(modelid, gpmodel_build_trees=self.gpmodel_build_trees)
        node = Node(model=model, label=label, **kwargs)
        node.modelid = modelid
        return node

    def load_array_node_from_modelid(self, modelid, label, **kwargs):
        model = self.load_modelid(modelid, gpmodel_build_trees=self.gpmodel_build_trees)
        node = ArrayNode(model=model, label=label, st=self.start_time, **kwargs)
        node.modelid = modelid
        return node

    def setup_site_param_node(self, **kwargs):
        if self.arrays_joint:
            return self.setup_site_param_node_joint(**kwargs)
        else:
            return self.setup_site_param_node_indep(**kwargs)

    def setup_site_param_node_joint(self, param, site, phase, parents, model_type,
                              chan=None, band=None,
                              modelid=None,
                              children=(), low_bound=None,
                              high_bound=None, initial_value=None, **kwargs):

        if not model_type.startswith("dummy") and modelid is None:
            try:
                modelid = self.get_param_model_id(runids=self.runids, sta=site,
                                             phase=phase, model_type=model_type,
                                             param=param, template_shape=self.template_shape,
                                             chan=chan, band=band)
            except ModelNotFoundError:
                if self.dummy_fallback:
                    print "warning: falling back to dummy model for %s, %s, %s phase %s param %s" % (site, chan, band, phase, param)
                    model_type = "dummyPrior"
                else:
                    raise
        label = create_key(param=param, sta="%s_arr" % site,
                           phase=phase, eid=parents[0].eid,
                           chan=chan, band=band)
        if model_type.startswith("dummy"):
            return self.setup_site_param_indep(param=param, site=site, phase=phase, parents=parents, chan=chan, band=band, model_type=model_type, children=children, low_bound=low_bound, high_bound=high_bound, initial_value=initial_value, **kwargs)
        else:
            sorted_elements = sorted(self.site_elements[site])
            sk = [create_key(param=param, eid=parents[0].eid, sta=sta, phase=phase, chan=chan, band=band) for sta in sorted_elements]
            if initial_value is None:
                initial_value = 0.0
            if type(initial_value) != dict:
                initial_value = dict([(k, initial_value) for k in sk])
            node = self.load_array_node_from_modelid(modelid=modelid, parents=parents, children=children, initial_value=initial_value, low_bound=low_bound, high_bound=high_bound, sorted_keys=sk, label=label)
            self.add_node(node, **kwargs)
            return node



    def setup_site_param_node_indep(self, param, site, phase, parents, model_type,
                              chan=None, band=None,
                              modelid=None,
                              children=(), low_bound=None,
                              high_bound=None, initial_value=None, **kwargs):


        # for each station at this site, create a node with the
        # appropriate parameter model.
        nodes = dict()
        for sta in self.site_elements[site]:
            if not model_type.startswith("dummy") and modelid is None and model_type != "gp_joint":
                try:
                    modelid = self.get_param_model_id(runids=self.runids, sta=sta,
                                                      phase=phase, model_type=model_type,
                                                      param=param, template_shape=self.template_shape,
                                                      chan=chan, band=band)
                except ModelNotFoundError:
                    if self.dummy_fallback:
                        print "warning: falling back to dummy model for %s, %s, %s phase %s param %s" % (site, chan, band, phase, param)
                        model_type = "dummyPrior"
                    else:
                        raise
            label = create_key(param=param, sta=sta,
                               phase=phase, eid=parents[0].eid,
                               chan=chan, band=band)
            my_children = [wn for wn in children if wn.sta==sta]
            if model_type.startswith("dummy") or model_type=="gp_joint":
                if model_type=="dummyPrior":
                    model = self.dummy_prior[param]
                elif model_type == "gp_joint":
                    model = None
                else:
                    if "tt_residual" in label:
                        model = Gaussian(mean=0.0, std=10.0)
                    elif "amp" in label:
                        model = Gaussian(mean=0.0, std=0.25)
                    else:
                        model = DummyModel(default_value=initial_value)

                node = Node(label=label, model=model, parents=parents, children=my_children, initial_value=initial_value, low_bound=low_bound, high_bound=high_bound, hack_param_constraint=self.hack_param_constraint)
                if model_type=="gp_joint":
                    jgp, hparam_nodes = self.joint_gpmodel(sta=sta, param=param, chan=chan, band=band, phase=phase)
                    node.params_modeled_jointly.add(jgp)
                    for n in hparam_nodes.values():
                        node.addParent(n)
            else:
                node = self.load_node_from_modelid(modelid, label, parents=parents, children=my_children, initial_value=initial_value, low_bound=low_bound, high_bound=high_bound, hack_param_constraint=self.hack_param_constraint)

            nodes[sta] = node
            self.add_node(node, **kwargs)
        return nodes

    """

    okay, so to set up an arrival time node, I need to load the GP residual model. and that's great.
    but I also need to, somehow, add in the IASPEI predictions, and the event time. That's what the TravelTimeModel node already does.

    for the coda_height node, I'll need to do something similar. it depends on the amp_transfer node -- which is itself just a GP model -- but has to add in the event source stuff, and maybe projections onto the three components.

    also of course the wiggle node is a special case at the moment.

    any new template shape model / templategenerator would need to keep a notion of arrival time. it might add some other params, and it might define multiple height params (say a coda height and a peak height). I think maybe amp_transfer should be something more fundamental (though obviously the values we learn for it will be mediated by the way in which it gets used), and then the actual height params used by the shape model can be something else.
    so my generic code should set up an arrivalTime node and an amp_transfer node.
    then it loops through the params of the templateGenerator.
    for each one, it calls a templateGenerator.create_node() function.
    in the generic case, the templateGenerator just calls right back to the SG node creator.
    but for stuff like a coda_height node, it can do something custom.

    how do we know which params should be shared across channels/bands/arrays?
    some params ("coda_decay") should be joint across arrays, and exist above chan/band
    some params ("coda_height") should be specific to a specific sta/chan/band

    """

    def setup_tt(self, site, phase, evnodes, tt_residual_node, children):
        nodes = dict()
        eid = evnodes['mb'].eid
        for sta in self.site_elements[site]:
            ttrn = extract_sta_node(tt_residual_node, sta)
            label = create_key(param="arrival_time", sta=sta, phase=phase, eid=eid)

            my_children = [wn for wn in children if wn.sta==sta]
            arrtimenode = ArrivalTimeNode(eid=eid, sta=sta,
                                          phase=phase, parents=[evnodes['loc'], evnodes['time'], ttrn],
                                          label=label, children=my_children)
            self.add_node(arrtimenode, template=True)
            nodes[sta] = arrtimenode
        return nodes

    def add_event_site_phase(self, tg, site, phase, evnodes, sample_templates=False):
        # the "nodes" we create here can either be
        # actual nodes (if we are modeling these quantities
        # jointly across an array) or sta:node dictionaries (if we
        # are modeling them independently).
        def extract_sta_node_list(n):
            try:
                return n.values()
            except AttributeError:
                return [n,]

        eid = evnodes['mb'].eid

        child_wave_nodes = set()

        ev_time = evnodes['time'].get_value()
        for sta in self.site_elements[site]:
            for wave_node in self.station_waves[sta]:
                if wave_node.st > ev_time + MAX_TRAVEL_TIME: continue
                if wave_node.et < ev_time: continue

                if self.force_event_wn_matching:
                    ev = self.get_event(eid)
                    pred_time = ev.time + tt_predict(ev, sta, "P")
                    if pred_time < wave_node.st or pred_time > wave_node.et: continue


                child_wave_nodes.add(wave_node)

                # wave nodes depend directly on event location
                # since they need to know what GP prior
                # to use on wiggles.
                evnodes["loc"].addChild(wave_node)
                evnodes["mb"].addChild(wave_node)

                if self.force_event_wn_matching:
                    break

        #if self.force_event_wn_matching:
        #    assert(len(child_wave_nodes)==1)

        # create nodes common to all bands and channels: travel
        # time/arrival time, and amp_transfer.
        tt_model_type = self._tm_type(param="tt_residual", site=site)
        tt_residual_node = tg.create_param_node(self, site, phase,
                                                band=None, chan=None, param="tt_residual",
                                                model_type=tt_model_type,
                                                evnodes=evnodes,
                                                low_bound = -15,
                                                high_bound = 15)
        arrival_time_node = self.setup_tt(site, phase, evnodes=evnodes,
                                          tt_residual_node=tt_residual_node,
                                          children= child_wave_nodes)
        ampt_model_type = self._tm_type(param="amp_transfer", site=site)
        amp_transfer_node = tg.create_param_node(self, site, phase,
                                                 band=None, chan=None, param="amp_transfer",
                                                 model_type=ampt_model_type,
                                                 evnodes=evnodes,
                                                 low_bound=-4.0, high_bound=10.0)

        nodes = dict()
        nodes["arrival_time"] = arrival_time_node

        # create all other shape param nodes, specific to each band and channel
        for band in self.site_bands[site]:
            for chan in self.site_chans[site]:
                for param in tg.params():

                    if param == "coda_height":
                        model_type = None
                    else:
                        model_type = self._tm_type(param, site)
                    # here the "create param node" creates, potentially, a single node or a dict of nodes
                    nodes[(band, chan, param)] = tg.create_param_node(self, site, phase, band,
                                                                      chan, model_type=model_type, param=param,
                                                                      evnodes=evnodes,
                                                                      atime_node=arrival_time_node,
                                                                      amp_transfer_node=amp_transfer_node,
                                                                      children=child_wave_nodes,
                                                                      low_bound = tg.low_bounds()[param],
                                                                      high_bound = tg.high_bounds()[param],
                                                                      initial_value = tg.default_param_vals()[param])

        fullnodes = []
        for ni in [tt_residual_node, amp_transfer_node] + nodes.values():
            for n in extract_sta_node_list(ni):
                if sample_templates:
                    n.parent_sample()
                    # hacks to deal with Gaussians occasionally being negative
                    if "peak_offset" in n.label:
                        v = n.get_value()
                        if isinstance(v, float):
                            v = 0.5 if v <= 0 else v
                        else:
                            invalid_offsets = (v <= 0)
                            v[invalid_offsets] = 0.5
                    if "mult_wiggle_std" in n.label:
                        v = n.get_value()
                        if isinstance(v, float):
                            v = 0.5 if v <= 0 else v
                        else:
                            invalid_offsets = (v <= 0)
                            v[invalid_offsets] = 0.5
                    if "coda_decay" in n.label:
                        v = n.get_value()
                        if isinstance(v, float):
                            v = 0.01 if v >= 0 else v
                        else:
                            invalid_offsets = (v >= 0)
                            v[invalid_offsets] = -0.01
                else:
                    n.parent_predict()

                try:
                    n.upwards_message_normalizer()
                except:
                    pass

                fullnodes.append(n)
        self.extended_evnodes[eid].extend(fullnodes)
        return fullnodes


    def add_wave(self, wave, fixed=True, disable_conflict_checking=False, **kwargs):
        """
        Add a wave node to the graph. Assume that all waves are added before all events.
        """

        basis = self.wavelet_basis(wave['srate'])

        """
        To ensure no confusion, don't allow wave nodes at the same station
        within an hour of each other.
        (in the current model, the only legitimate time to allow multiple wns from
         the same station is when doing event relocation via waveform matching,
         in which case the wns will potentially be many years apart).
        """
        for wn in self.station_waves[wave['sta']]:
            if not disable_conflict_checking and \
               wave['chan']==wn.chan and wave['band']==wn.band and \
               wave['stime'] < wn.et + MAX_TRAVEL_TIME and \
               wave['etime'] > wn.st - MAX_TRAVEL_TIME:
                raise Exception("adding new wave at %s,%s,%s from time %.1f-%.1f potentially conflicts with existing wave from %.1f-%.1f" % (wn.sta, wn.chan, wn.band, wave['stime'], wave['etime'], wn.st, wn.et) )

        param_models = {}
        hparam_nodes = set()
        has_jointgp = False

        if basis is not None:
            levels = basis[-1]
            n_params = len(basis[0])
            n_joint_params = np.sum(levels[:len(levels) - self.skip_levels])
            joint_params = [self.wiggle_family + "_%d" % i for i in range(n_joint_params)]
        if self.jointgp:
            for phase in self.phases:
                param_models[phase] = []
                for param in joint_params:
                    jgp, nodes = self.joint_gpmodel(sta=wave['sta'], param=param, chan=wave['chan'], band=wave['band'], phase=phase)
                    param_models[phase].append(jgp)
                    hparam_nodes = hparam_nodes | set(nodes.values())

                # if we're skipping two levels, start with level 2 since its params are listed first
                for level in range(self.skip_levels, 0, -1):
                    level_size = levels[-(level)]
                    jgp, nodes = self.level_coef_model(sta=wave['sta'], param=self.wiggle_family + "_level%d" % level, chan=wave['chan'], band=wave['band'], phase=phase)
                    hparam_nodes = hparam_nodes | set(nodes.values())
                    for i in range(level_size):
                        param_models[phase].append(jgp)

            has_jointgp = True
        elif self.wiggle_model_type != "dummy":
            for phase in self.phases:
                param_models[phase] = []
                for param in joint_params:
                    modelid = self.get_param_model_id(runids=self.runids, sta=wave['sta'],
                                                      phase=phase, model_type=self.wiggle_model_type,
                                                      param=param, template_shape=self.template_shape,
                                                      chan=wave['chan'], band=wave['band'])
                    #except ModelNotFoundError as e:
                    #    print e
                    #    continue
                    model = self.load_modelid(modelid, gpmodel_build_trees=self.gpmodel_build_trees)
                    param_models[phase].append(model)
                for level in range(self.skip_levels, 0, -1):
                    level_size = levels[-(level)]
                    param = self.wiggle_family + "_level%d" % level
                    modelid = self.get_param_model_id(runids=self.runids, sta=wave['sta'],
                                                      phase=phase, model_type="constant_gaussian",
                                                      param=param, template_shape=self.template_shape,
                                                      chan=wave['chan'], band=wave['band'])
                    model = self.load_modelid(modelid, gpmodel_build_trees=self.gpmodel_build_trees)
                    for i in range(level_size):
                        param_models[phase].append(model)
        else:
            try:
                n_params = len(basis[0])
                for phase in self.phases:
                    param_models[phase] = [Gaussian(0.0, 1.0),]*n_params
            except TypeError:
                pass


        wave_node = ObservedSignalNode(model_waveform=wave, graph=self, nm_type=self.nm_type, observed=fixed, label=self._get_wave_label(wave=wave), wavelet_basis=basis, wavelet_param_models=param_models, has_jointgp = has_jointgp, **kwargs)

        for n in hparam_nodes:
            wave_node.addParent(n)

        s = Sigvisa()
        sta = wave['sta']
        _, _, _, isarr, _, _, ref_site_id = s.earthmodel.site_info(sta, wave['stime'])
        ref_site_name = s.siteid_minus1_to_name[ref_site_id-1]
        if ref_site_name not in self.site_elements:
            self.site_elements[ref_site_name] = set()
            self.site_bands[ref_site_name] = set()
            self.site_chans[ref_site_name] = set()
        self.site_elements[ref_site_name].add(sta)
        self.site_bands[ref_site_name].add(wave['band'])
        self.site_chans[ref_site_name].add(wave['chan'])

        if sta not in self.station_waves:
            self.station_waves[sta] = []
        self.station_waves[sta].append(wave_node)

        self.start_time = min(self.start_time, wave_node.st)
        self.event_start_time = self.start_time - MAX_TRAVEL_TIME
        self.end_time = max(self.end_time, wave_node.et)

        self.add_node(wave_node)
        self._topo_sort()
        return wave_node

    def _get_wave_label(self, wave):
        return 'wave_%s_%s_%s_%.1f' % (wave['sta'], wave['chan'], wave['band'], wave['stime'])

    """
    def get_template_node(self, **kwargs):
        lbl = self._get_interior_node_label(**kwargs)
        return self.all_nodes["template_%s" % lbl]

    def get_wiggle_node(self, **kwargs):
        lbl = self._get_interior_node_label(**kwargs)
        return self.all_nodes["wiggle_%s" % lbl]
   """

    def get_wave_node(self, wave):
        return self.all_nodes[self._get_wave_label(wave=wave)]


    def get_wave_node_log_p(self, wave_node):
        log_p = 0
        parents = wave_node.parents.values()
        for p in parents:
            if p.deterministic():
                for pp in p.parents.values():
                    log_p = pp.log_p()
            else:
                log_p += p.log_p()
        log_p += wave_node.log_p()
        return log_p


    """
    def get_partner_node(self, n):
        if n.label.startswith("template_"):
            lbl = n.label[9:]
            return self.all_nodes["wiggle_%s" % lbl]
        elif n.label.startswith("wiggle_"):
            lbl = n.label[7:]
            return self.all_nodes["template_%s" % lbl]
        else:
            raise ValueError("node %s has no partner!" % n.label)


    def fix_arrival_times(self, fixed=True):
        for tm_node in self.template_nodes:
            if fixed:
                tm_node.fix_value(key='arrival_time')
            else:
                tm_node.unfix_value(key='arrival_time')
    """


    def optimize_templates(self, optim_params):
        st = time.time()
        self.joint_optimize_nodes(node_list=self.template_nodes, optim_params=optim_params)
        et = time.time()
        ll = self.current_log_p()
        self.optim_log += ("optimize_templates: t=%.1fs ll=%.1f\n" % (et-st, ll))
        return ll

    def optimize_with_seed_time_and_depth(lon, lat, t, depth):
        assert(len(self.toplevel_nodes) == 1)
        ev_node = list(self.toplevel_nodes)[0]
        ev_node.set_index(key="lon", value=lon)
        ev_node.set_index(key="lat", value=lat)
        ev_node.set_index(key="time", value=t)
        ev_node.set_index(key="depth", value=depth)
        ev_node.fix_value(key="lon")
        ev_node.fix_value(key="lat")
        self.fix_arrival_times(fixed=False)

        # initialize
        for n in self.template_nodes:
            n.parent_predict()

        # TODO: optimize

    def prior_sample_event(self, min_mb=3.5, stime=None, etime=None):
        s = Sigvisa()

        stime = self.event_start_time if stime is None else stime
        etime = self.end_time if etime is None else etime

        event_time_dist = Uniform(stime, etime)
        event_mag_dist = Exponential(rate=np.log(10.0), min_value=min_mb)

        origin_time = event_time_dist.sample()

        s.sigmodel.srand(np.random.randint(sys.maxint))
        lon, lat, depth = s.sigmodel.event_location_prior_sample()
        mb = event_mag_dist.sample()
        natural_source = True # TODO : sample from source prior

        ev = get_event(lon=lon, lat=lat, depth=depth, time=origin_time, mb=mb, natural_source=natural_source)

        return ev

    def prior_sample_events(self, min_mb=3.5, force_mb=None, stime=None, etime=None, n_events=None):
        # assume a fresh graph, i.e. no events already exist

        if n_events is None:
            n_event_dist = Poisson(self.event_rate * (etime - stime))
            n_events = n_event_dist.sample()

        evs = []

        for i in range(n_events):
            ev = self.prior_sample_event(min_mb, stime, etime)
            if force_mb is not None and force_mb != False:
                ev.mb = force_mb
            self.add_event(ev, sample_templates=True)
            evs.append(ev)
        return evs

    def prior_sample_uatemplates(self, wn, n_templates=None, **kwargs):

        if not n_templates:
            n_template_dist = Poisson(self.uatemplate_rate * (wn.et-wn.st))
            n_templates = n_template_dist.sample()

        template_time_dist = Uniform(wn.st, wn.et)
        templates = []
        for i in range(n_templates):
            atime = template_time_dist.sample()
            tnodes = self.create_unassociated_template(wave_node=wn, atime=atime, nosort=True, **kwargs)
            for node in tnodes.values():
                node.parent_sample()
            templates.append(tnodes)
        self._topo_sort()
        wn.unfix_value()
        wn.parent_sample()
        wn.fix_value()
        return templates

    def dump_event_signals(self, eid, dump_path):
        mkdir_p(dump_path)

        for (sta, waves) in self.station_waves.items():
            for wn in waves:
                for (aeid, phase) in wn.arrivals():
                    if eid != aeid: continue
                    params, tg = wn.get_template_params_for_arrival(eid, phase)
                    atime = params['arrival_time']
                    stime = atime-10.0
                    etime = atime + 100.0
                    plot_with_fit(os.path.join(dump_path, "%s_%d_%s.png" % (sta, eid, phase)), wn,
                                  highlight_eid=eid, stime=stime, etime=etime)


    def debug_dump(self, dump_dirname=None, dump_path=None, pickle_graph=True, pickle_only=False):


        #sys.setrecursionlimit(5000)

        if dump_path is None:
            assert(dump_dirname is not None)
            dump_path = os.path.join('logs', 'dumps', dump_dirname)

        clear_directory(dump_path)
        print "saving debug dump to %s..." % dump_path


        if pickle_graph:
            with open(os.path.join(dump_path, 'pickle.sg'), 'wb') as f:
                #spypickle = SpyingPickler(f, 2)
                #spypickle.dump(self)
                #import pickle as ppickle
                pickle.dump(self, f, 2)
            print "saved pickled graph"
        if pickle_only:
            return

        for (sta, waves) in self.station_waves.items():
            for wn in waves:
                plot_with_fit(os.path.join(dump_path, "%s.png" % (wn.label)), wn)

                with open(os.path.join(dump_path, "%s_arrivals.txt" % (wn.label)), 'w') as f:
                    for (eid, phase) in sorted(wn.arrivals()):
                        v, tg = wn.get_template_params_for_arrival(eid=eid, phase=phase)
                        f.write("eid %d, phase %s:\n" % (eid, phase))
                        for (key, val) in v.items():
                            f.write(" %s: %s\n" % (key, val))
                        f.write("\n")
                print "saved plot and arrival info for %s" % (wn.label)

        with open(os.path.join(dump_path, "nodes.txt"), 'w') as f:
            for (k, n) in sorted(self.all_nodes.items()):
                if n.deterministic():
                    f.write("%s: deterministic\n" % k)
                else:
                    f.write("%s: lp %.1f\n" % (k, n.log_p()))
                for key in sorted(n.keys()):
                    f.write(" %s: %s\n" % (key, n.get_value(key)))
                f.write("\n")
        print "saved node values and probabilities"

        os.system("tar cfz %s.tgz %s/*" % (dump_path, dump_path))
        print "generated tarball"

    def __setstate__(self, d):
        self.__dict__ = d
        for sta in self.station_waves.keys():

            gpmodels = self._joint_gpmodels[sta]
            ks = gpmodels.keys()

            try:
                for wn in self.station_waves[sta]:
                    for k in ks:
                        jgp, hparam_nodes = gpmodels[k]

                        # fill in pointers discarded by
                        # getstate() of JointGP
                        jgp.hparam_nodes = hparam_nodes

                        # fill in the parent pointers discarded
                        # during pickling by the getstate() method
                        # of ObservedSignalNode

                        #for n in hparam_nodes.values():
                        #    wn.parents[n.single_key] = n
                self.recover_parents_from_children()
            except TypeError:
                # backwards compatibility if we're loading sggraphs with no hparams
                pass
