import time
import numpy as np
import os
import shutil
import errno
import re
import cPickle as pickle
from collections import defaultdict
from functools32 import lru_cache

from sigvisa import Sigvisa

from sigvisa.database.signal_data import get_fitting_runid, insert_wiggle, ensure_dir_exists, RunNotFoundException

from sigvisa.source.event import get_event
from sigvisa.learn.train_param_common import load_modelid
import sigvisa.utils.geog as geog
from sigvisa.models import DummyModel
from sigvisa.models.distributions import Uniform, Poisson, Gaussian, Exponential
from sigvisa.models.ev_prior import setup_event, event_from_evnodes
from sigvisa.models.ttime import tt_predict, tt_log_p, ArrivalTimeNode
from sigvisa.graph.nodes import Node
from sigvisa.graph.dag import DirectedGraphModel
from sigvisa.graph.graph_utils import extract_sta_node, predict_phases, create_key, get_parent_value, parse_key
from sigvisa.models.signal_model import ObservedSignalNode, update_arrivals
from sigvisa.graph.array_node import ArrayNode
from sigvisa.models.templates.load_by_name import load_template_generator
from sigvisa.database.signal_data import execute_and_return_id
from sigvisa.models.wiggles.wiggle import extract_phase_wiggle
from sigvisa.models.wiggles import load_wiggle_generator, load_wiggle_generator_by_family
from sigvisa.plotting.plot import plot_with_fit
from sigvisa.signals.common import Waveform

from sigvisa.utils.fileutils import clear_directory, mkdir_p

class ModelNotFoundError(Exception):
    pass


MAX_TRAVEL_TIME = 2000.0

@lru_cache(maxsize=1024)
def get_param_model_id(runid, sta, phase, model_type, param,
                       template_shape, basisid=None, chan=None, band=None):
    if runid is None:
        raise ModelNotFoundError("no runid specified, so not loading parameter model.")

    # get a DB modelid for a previously-trained parameter model
    s = Sigvisa()
    cursor = s.dbconn.cursor()
    chan_cond = "and chan='%s'" % chan if chan else ""
    band_cond = "and band='%s'" % band if band else ""
    basisid_cond = "and wiggle_basisid=%d" % (basisid) if basisid is not None else ""

    sql_query = "select modelid, shrinkage_iter from sigvisa_param_model where model_type = '%s' and site='%s' %s %s and phase='%s' and fitting_runid=%d and template_shape='%s' and param='%s' %s" % (model_type, sta, chan_cond, band_cond, phase, runid, template_shape, param, basisid_cond)
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        modelid = sorted(results, key = lambda x : -x[1])[0][0] # use the model with the most shrinkage iterations
    except:
        raise ModelNotFoundError("no model found matching model_type = '%s' and site='%s' %s %s and phase='%s' and fitting_runid=%d and template_shape='%s' and param='%s' %s" % (model_type, sta, chan_cond, band_cond, phase, runid, template_shape, param, basisid_cond))
    finally:
        cursor.close()

    return modelid


class SigvisaGraph(DirectedGraphModel):


    """
    Construct the Sigvisa graphical model.

    """

    def _tm_type(self, param, site=None, wiggle_param=True):

        if wiggle_param:
            try:
                tmtype = self.wiggle_model_type[param]
            except TypeError:
                tmtype = self.wiggle_model_type
        else:
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
                 wiggle_model_type="dummy", wiggle_family="fourier_0.8",
                 wiggle_len_s = 30.0, wiggle_basisids=None,
                 dummy_fallback=False,
                 nm_type="ar",
                 run_name=None, iteration=None, runid = None,
                 phases="auto", base_srate=40.0,
                 assume_envelopes=True, smoothing=None,
                 arrays_joint=False, gpmodel_build_trees=False,
                 absorb_n_phases=False, hack_param_constraint=False, uatemplate_rate=1e-3):
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

        self.template_model_type = template_model_type
        self.template_shape = template_shape
        self.tg = dict()
        if type(self.template_shape) == dict:
            for (phase, ts) in self.template_shape.items():
                self.tg[phase] = load_template_generator(ts)


        self.wiggle_model_type = wiggle_model_type
        self.wiggle_family = wiggle_family
        self.wiggle_len_s = wiggle_len_s
        self.base_srate = base_srate
        self.assume_envelopes = assume_envelopes
        self.smoothing = smoothing
        self.wgs = dict()
        if wiggle_basisids is not None:
            for (phase, basisid) in wiggle_basisids.items():
                wg = load_wiggle_generator(basisid=basisid)
                self.wgs[(phase, wg.srate)] = wg

        self.dummy_fallback = dummy_fallback

        self.nm_type = nm_type
        self.phases = phases

        self.phases_used = set()

        self.runid = runid
        if run_name is not None and iteration is not None:
            cursor = Sigvisa().dbconn.cursor()
            try:
                self.runid = get_fitting_runid(cursor, run_name, iteration, create_if_new = False)
            except RunNotFoundException:
                self.runid=None
            cursor.close()

        self.template_nodes = []
        self.wiggle_nodes = []


        self.station_waves = dict() # (sta) -> list of ObservedSignalNodes
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

    def ev_arriving_phases(self, eid, sta):
        [v for n in self.extended_evnodes.values()]

    def template_generator(self, phase):
        if phase not in self.tg and type(self.template_shape) == str:
            self.tg[phase] = load_template_generator(self.template_shape)
        return self.tg[phase]

    def wiggle_generator(self, phase, srate):
        if (phase, srate) not in self.wgs:
            self.wgs[(phase, srate)] =  load_wiggle_generator_by_family(family_name=self.wiggle_family, len_s=self.wiggle_len_s, srate=srate, envelope=self.assume_envelopes)
        return self.wgs[(phase, srate)]

    def get_wiggle_nodes(self, eid, sta, phase, band, chan):
        wg = self.wiggle_generator(phase=phase, srate=self.base_srate)
        nodes = dict()
        for param in wg.params():
            k, node = get_parent_value(eid=eid, sta=sta, phase=phase, param_name=param, chan=chan, band=band, parent_values=self.nodes_by_key, return_key=True)
            nodes[param]=(k, node)
        return nodes

    def get_wiggle_vals(self, eid, sta, phase, band, chan):
        nodes = self.get_wiggle_nodes(eid, sta, phase, band, chan)
        vals = dict([(p, n.get_value(k)) for (p,(k, n)) in nodes.iteritems()])
        return vals

    def get_template_nodes(self, eid, sta, phase, band, chan):
        tg = self.template_generator(phase)
        nodes = dict()
        for param in tg.params() + ('arrival_time',):
            k, node = get_parent_value(eid=eid, sta=sta, phase=phase, param_name=param, chan=chan, band=band, parent_values=self.nodes_by_key, return_key=True)
            nodes[param]=(k, node)

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

    def get_arrival_nodes_byphase(self, eid, sta, band, chan):
        allnodes = dict()
        for phase in self.phases_used:
            try:
                tmnodes = self.get_arrival_nodes(eid, sta, phase, band, chan)
                allnodes[phase] = tmnodes
            except KeyError:
                continue
        return allnodes

    def get_arrival_nodes(self, eid, sta, phase, band, chan):
        nodes = self.get_template_nodes(eid, sta, phase, band, chan)
        nodes.update(self.get_wiggle_nodes(eid, sta, phase, band, chan))
        return nodes

    def get_arrival_vals(self, eid, sta, phase, band, chan):
        nodes = self.get_arrival_nodes(eid, sta, phase, band, chan)
        vals = dict([(p, n.get_value(k)) for (p,(k, n)) in nodes.iteritems()])
        return vals


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

        assert(len(self.station_waves[sta]) == 1)
        wn  = self.station_waves[sta][0]

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
        lp = n * np.log(self.uatemplate_rate) - (self.uatemplate_rate * wn.valid_len)

        return lp


    def current_log_p_breakdown(self):
        nt_lp = self.ntemplates_log_p()
        ne_lp = self.nevents_log_p()

        ua_peak_offset_lp = 0.0
        ua_coda_height_lp = 0.0
        ua_coda_decay_lp = 0.0
        ua_wiggle_lp = 0.0
        for ((sta, band, chan), tmid_set) in self.uatemplate_ids.items():
            for tmid in tmid_set:
                uanodes = self.uatemplates[tmid]
                ua_peak_offset_lp += uanodes['peak_offset'].log_p()
                ua_coda_height_lp += uanodes['coda_height'].log_p()
                ua_coda_decay_lp += uanodes['coda_decay'].log_p()
                for key in uanodes.keys():
                    if (key != "amp_transfer" and "amp_" in key) or "phase_" in key:
                        ua_wiggle_lp += uanodes[key].log_p()

        ev_prior_lp = 0.0
        ev_tt_lp = 0.0
        ev_amp_transfer_lp = 0.0
        ev_peak_offset_lp = 0.0
        ev_coda_decay_lp = 0.0
        ev_wiggle_lp = 0.0
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
                    ev_tt_lp += node.log_p()
                elif  "amp_transfer" in node.label:
                    ev_amp_transfer_lp += node.log_p()
                elif  "coda_decay" in node.label:
                    ev_coda_decay_lp += node.log_p()
                elif  "peak_offset" in node.label:
                    ev_peak_offset_lp += node.log_p()
                elif "amp_" in node.label or "phase_" in node.label:
                    ev_wiggle_lp += node.log_p()
                else:
                    raise Exception('unexpected node %s' % node.label)

        signal_lp = 0.0
        for (sta_, wave_list) in self.station_waves.items():
            for wn in wave_list:
                signal_lp += wn.log_p()


        print "n_uatemplate: %.1f" % nt_lp
        print "n_event: %.1f" % ne_lp
        print "ev priors: ev %.1f" % (ev_prior_lp)
        print "tt_residual: ev %.1f" % (ev_tt_lp)
        print "ev global cost (n + priors + tt): %.1f" % (ev_prior_lp + ev_tt_lp + ne_lp,)
        print "coda_decay: ev %.1f ua %.1f total %.1f" % (ev_coda_decay_lp, ua_coda_decay_lp, ev_coda_decay_lp+ua_coda_decay_lp)
        print "peak_offset: ev %.1f ua %.1f total %.1f" % (ev_peak_offset_lp, ua_peak_offset_lp, ev_peak_offset_lp+ua_peak_offset_lp)
        print "coda_height: ev %.1f ua %.1f total %.1f" % (ev_amp_transfer_lp, ua_coda_height_lp, ev_amp_transfer_lp+ua_coda_height_lp)
        print "wiggles: ev %.1f ua %.1f total %.1f" % (ev_wiggle_lp, ua_wiggle_lp, ev_wiggle_lp+ua_wiggle_lp)
        ev_total = ev_coda_decay_lp + ev_peak_offset_lp + ev_amp_transfer_lp + ev_wiggle_lp
        ua_total = ua_coda_decay_lp + ua_peak_offset_lp + ua_coda_height_lp + ua_wiggle_lp
        print "total param: ev %.1f ua %.1f total %.1f" % (ev_total, ua_total, ev_total+ua_total)
        ev_total += ev_prior_lp + ev_tt_lp + ne_lp
        ua_total += nt_lp
        print "non signals: ev %.1f ua %.1f total %.1f" % (ev_total, ua_total, ev_total + ua_total)
        print "signals: %.1f" % (signal_lp)
        print "overall: %.1f" % (ev_total + ua_total + signal_lp)
        print "official: %.1f" % self.current_log_p()

    def current_log_p(self, **kwargs):
        lp = super(SigvisaGraph, self).current_log_p(**kwargs)
        lp += self.ntemplates_log_p()
        lp += self.nevents_log_p()
        if np.isnan(lp):
            raise Exception('current_log_p is nan')
        return lp

    def add_node(self, node, template=False, wiggle=False):
        if template:
            self.template_nodes.append(node)
        if wiggle:
            self.wiggle_nodes.append(node)
        super(SigvisaGraph, self).add_node(node)

    def remove_node(self, node):
        if node in self.template_nodes:
            self.template_nodes.remove(node)
        if node in self.wiggle_nodes:
            self.wiggle_nodes.remove(node)
        super(SigvisaGraph, self).remove_node(node)

    def get_event(self, eid):
        return event_from_evnodes(self.evnodes[eid])

    def remove_event(self, eid):

        del self.evnodes[eid]
        for node in self.extended_evnodes[eid]:
            self.remove_node(node)
        del self.extended_evnodes[eid]

        self._topo_sort()

    def add_event(self, ev, basisids=None, tmshapes=None, sample_templates=False, fixed=False, eid=None):
        """

        Add an event node to the graph and connect it to all waves
        during which its signals might arrive.

        basisids: optional dictionary of wiggle basis ids (integers), keyed on phase name.
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
            for phase in predict_phases(ev=ev, sta=site, phases=self.phases):
                print "adding phase", phase, "at site", site
                self.phases_used.add(phase)
                if self.absorb_n_phases:
                    if phase == "Pn":
                        phase = "P"
                    elif phase == "Sn":
                        phase = "S"
                tg = self.template_generator(phase)
                wg = self.wiggle_generator(phase, self.base_srate)
                self.add_event_site_phase(tg, wg, site, phase, evnodes, sample_templates=sample_templates)


        self._topo_sort()
        return evnodes

    def destroy_unassociated_template(self, nodes, nosort=False):
        eid, phase, sta, chan, band, param = parse_key(nodes.values()[0].label)
        uaid = -eid
        del self.uatemplates[uaid]
        self.uatemplate_ids[(sta,chan,band)].remove(uaid)

        for (param, node) in nodes.items():
            self.remove_node(node)
        if not nosort:
            self._topo_sort()

    def create_unassociated_template(self, wave_node, atime, wiggles=True, nosort=False, tmid=None, initial_vals=None, sample_wiggles=False):

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
        wg = self.wiggle_generator(phase=phase, srate=wave_node.srate)

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

        if wiggles:
            for param in wg.params():
                label = create_key(param=param, sta=wave_node.sta,
                                   phase=phase, eid=eid,
                                   chan=wave_node.chan, band=wave_node.band)
                model = wg.unassociated_model(param)
                wnodes[param] = Node(label=label, model=model, children=(wave_node,))
                self.add_node(wnodes[param], wiggle=True)

        for (param, node) in tnodes.items():
            node.tmid = tmid
            if initial_vals is None:
                node.parent_sample()
            else:
                node.set_value(initial_vals[param])

        for (param, node) in wnodes.items():
            if initial_vals is not None and param in initial_vals:
                node.set_value(initial_vals[param])
            else:
                if sample_wiggles:
                    node.parent_sample()
                else:
                    node.parent_predict()

        nodes = tnodes
        nodes.update(wnodes)

        self.uatemplates[tmid] = nodes
        self.uatemplate_ids[(wave_node.sta,wave_node.chan,wave_node.band)].add(tmid)

        if not nosort:
            self._topo_sorted_list = nodes.values() + self._topo_sorted_list
            self._gc_topo_sorted_nodes()
        return nodes

    def load_node_from_modelid(self, modelid, label, **kwargs):
        model = load_modelid(modelid, gpmodel_build_trees=self.gpmodel_build_trees)
        node = Node(model=model, label=label, **kwargs)
        node.modelid = modelid
        return node

    def load_array_node_from_modelid(self, modelid, label, **kwargs):
        model = load_modelid(modelid, gpmodel_build_trees=self.gpmodel_build_trees)
        node = ArrayNode(model=model, label=label, st=self.start_time, **kwargs)
        node.modelid = modelid
        return node

    def setup_site_param_node(self, **kwargs):
        if self.arrays_joint:
            return self.setup_site_param_node_joint(**kwargs)
        else:
            return self.setup_site_param_node_indep(**kwargs)

    def setup_site_param_node_joint(self, param, site, phase, parents, model_type,
                              chan=None, band=None, basisid=None,
                              modelid=None,
                              children=(), low_bound=None,
                              high_bound=None, initial_value=None, **kwargs):

        if not model_type.startswith("dummy") and modelid is None:
            try:
                modelid = get_param_model_id(runid=self.runid, sta=site,
                                             phase=phase, model_type=model_type,
                                             param=param, template_shape=self.template_shape,
                                             chan=chan, band=band, basisid=basisid)
            except ModelNotFoundError:
                if self.dummy_fallback:
                    print "warning: falling back to dummy model for %s, %s, %s phase %s param %s" % (site, chan, band, phase, param)
                    model_type = "dummy"
                else:
                    raise
        label = create_key(param=param, sta="%s_arr" % site,
                           phase=phase, eid=parents[0].eid,
                           chan=chan, band=band)
        if model_type.startswith("dummy"):
            return self.setup_site_param_indep(param=param, site=site, phase=phase, parents=parents, chan=chan, band=band, basisid=basisid, model_type=model_type, children=children, low_bound=low_bound, high_bound=high_bound, initial_value=initial_value, **kwargs)
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
                              chan=None, band=None, basisid=None,
                              modelid=None,
                              children=(), low_bound=None,
                              high_bound=None, initial_value=None, **kwargs):


        # for each station at this site, create a node with the
        # appropriate parameter model.
        nodes = dict()
        for sta in self.site_elements[site]:
            if not model_type.startswith("dummy") and modelid is None:
                try:
                    modelid = get_param_model_id(runid=self.runid, sta=sta,
                                                 phase=phase, model_type=model_type,
                                                 param=param, template_shape=self.template_shape,
                                                 chan=chan, band=band, basisid=basisid)
                except ModelNotFoundError:
                    if self.dummy_fallback:
                        print "warning: falling back to dummy model for %s, %s, %s phase %s param %s" % (site, chan, band, phase, param)
                        model_type = "dummy"
                    else:
                        raise
            label = create_key(param=param, sta=sta,
                               phase=phase, eid=parents[0].eid,
                               chan=chan, band=band)
            my_children = [wn for wn in children if wn.sta==sta]
            if model_type.startswith("dummy"):
                if model_type=="dummyPrior":
                    if "tt_residual" in label:
                        model = Gaussian(mean=0.0, std=1.0)
                    elif "amp_transfer" in label:
                        model = Gaussian(mean=0.0, std=2.0)
                    elif "peak_offset" in label:
                        model = Gaussian(mean=-0.5, std=1.0)
                    elif "decay" in label:
                        model = Gaussian(mean=0.0, std=1.0)
                    else:
                        model = DummyModel(default_value=initial_value)
                else:
                    if "tt_residual" in label:
                        model = Gaussian(mean=0.0, std=10.0)
                    elif "amp" in label:
                        model = Gaussian(mean=0.0, std=0.25)
                    else:
                        model = DummyModel(default_value=initial_value)

                node = Node(label=label, model=model, parents=parents, children=my_children, initial_value=initial_value, low_bound=low_bound, high_bound=high_bound, hack_param_constraint=self.hack_param_constraint)
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

    def add_event_site_phase(self, tg, wg, site, phase, evnodes, sample_templates=False):
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
        for sta in self.site_elements[site]:
            for wave_node in self.station_waves[sta]:
                child_wave_nodes.add(wave_node)

        tt_model_type = self._tm_type(param="tt_residual", site=site, wiggle_param=False)
        tt_residual_node = tg.create_param_node(self, site, phase,
                                                band=None, chan=None, param="tt_residual",
                                                model_type=tt_model_type,
                                                evnodes=evnodes,
                                                low_bound = -15,
                                                high_bound = 15)
        arrival_time_node = self.setup_tt(site, phase, evnodes=evnodes,
                                tt_residual_node=tt_residual_node,
                                children=child_wave_nodes)
        ampt_model_type = self._tm_type(param="amp_transfer", site=site, wiggle_param=False)
        amp_transfer_node = tg.create_param_node(self, site, phase,
                                                 band=None, chan=None, param="amp_transfer",
                                                 model_type=ampt_model_type,
                                                 evnodes=evnodes,
                                                 low_bound=-4.0, high_bound=10.0)

        nodes = dict()
        nodes["arrival_time"] = arrival_time_node

        for band in self.site_bands[site]:
            for chan in self.site_chans[site]:
                for param in tg.params():

                    if param == "coda_height":
                        model_type = None
                    else:
                        model_type = self._tm_type(param, site, wiggle_param=False)
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
                for param in wg.params():
                    model_type = self._tm_type(param, site, wiggle_param=True)
                    nodes[(band, chan, param)] = self.setup_site_param_node(param=param, site=site,
                                                                            model_type=model_type,
                                                                            phase=phase, parents=[evnodes['loc'],],
                                                                            band=band, chan=chan, basisid=wg.basisid,
                                                                            children=child_wave_nodes, wiggle=True)

        for ni in [tt_residual_node, amp_transfer_node] + nodes.values():
            for n in extract_sta_node_list(ni):
                if sample_templates:
                    n.parent_sample()
                    # hacks to deal with Gaussians occasionally being negative
                    if "peak_offset" in n.label:
                        v = n.get_value()
                        if type(v) == float:
                            v = 0.5 if v <= 0 else v
                        else:
                            invalid_offsets = (v <= 0)
                            v[invalid_offsets] = 0.5
                    if "coda_decay" in n.label:
                        v = n.get_value()
                        if type(v) == float:
                            v = 0.01 if v >= 0 else v
                        else:
                            invalid_offsets = (v >= 0)
                            v[invalid_offsets] = -0.01


                else:
                    n.parent_predict()
                self.extended_evnodes[eid].append(n)

    def add_wave(self, wave):
        """
        Add a wave node to the graph. Assume that all waves are added before all events.
        """

        wave_node = ObservedSignalNode(model_waveform=wave, nm_type=self.nm_type, observed=True, label=self._get_wave_label(wave=wave), graph=self)

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

    def init_wiggles_from_template(self):
        wave_node = list(self.leaf_nodes)[0]
        pv = wave_node._parent_values()
        arrivals = update_arrivals(parent_values=pv)
        for arrival in arrivals:
            wg = self.wiggle_generator(phase=arrival[1], srate=wave_node.srate)
            wiggle, st, et = extract_phase_wiggle(arrival=arrival, arrivals=arrivals, wave_node=wave_node)
            if st is None or len(wiggle) < wg.npts:
                continue

            wiggle = wiggle[:wg.npts].filled(1)
            self.set_template(eid=arrival[0], sta=wave_node.sta, phase=arrival[1], band=wave_node.band, chan=wave_node.chan, values=wg.features_from_signal(wiggle))

        ll = self.current_log_p()
        self.optim_log += ("init_wiggles: ll=%.1f\n" % ll)

    def optimize_wiggles(self, optim_params):
        st = time.time()
        self.joint_optimize_nodes(node_list=self.wiggle_nodes, optim_params=optim_params)
        et = time.time()
        ll = self.current_log_p()
        self.optim_log += ("optimize_wiggles: t=%.1fs ll=%.1f\n" % (et-st, ll))
        return ll

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
        for n in self.template_nodes + self.wiggle_nodes:
            n.parent_predict()

        # TODO: optimize

    def prior_sample_event(self, min_mb=3.5, stime=None, etime=None):
        s = Sigvisa()

        stime = self.event_start_time if stime is None else stime
        etime = self.end_time if etime is None else etime

        event_time_dist = Uniform(stime, etime)
        event_mag_dist = Exponential(rate=np.log(10.0), min_value=min_mb)

        origin_time = event_time_dist.sample()
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
            if force_mb is not None:
                ev.mb = force_mb
            self.add_event(ev, sample_templates=True)
            evs.append(ev)
        return evs

    def prior_sample_uatemplates(self, wn, **kwargs):
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

        if dump_path is None:
            assert(dump_dirname is not None)
            dump_path = os.path.join('logs', 'dumps', dump_dirname)

        clear_directory(dump_path)
        print "saving debug dump to %s..." % dump_path


        if pickle_graph:
            with open(os.path.join(dump_path, 'pickle.sg'), 'wb') as f:
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

    def save_wiggle_params(self):
        """
        Saves the current values of the wiggle model parameters for
        all waves in the graph. Assumes template parameters have
        already been saved, so tm.fpid is available at each template
        node.
        """

        s = Sigvisa()
        for wave_node in self.leaf_nodes:
            pv = wave_node._parent_values()
            arrivals = update_arrivals(pv)

            for (eid, phase) in arrivals:
                wg = self.wiggle_generator(phase=phase, srate=wave_node.srate)
                fit_param_nodes = self.get_template_nodes(eid=eid, phase=phase,
                                                              chan=wave_node.chan, band=wave_node.band,
                                                              sta=wave_node.sta)
                fpid = list(fit_param_nodes.values())[0][1].fpid

                v = dict([(p, wave_node.get_parent_value(eid, phase, p, pv)) for p in wg.params()])
                param_blob = wg.encode_params(v)
                p = {"fpid": fpid, 'timestamp': time.time(),  "params": param_blob, 'basisid': wg.basisid}
                wiggleid = insert_wiggle(s.dbconn, p)

            s.dbconn.commit()

    def save_template_params(self, tmpl_optim_param_str,
                             wiggle_optim_param_str,
                             hz, elapsed,
                             runid):
        s = Sigvisa()
        cursor = s.dbconn.cursor()

        fitids = []

        wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")
        run_wiggle_dir = os.path.join(wiggle_dir, "runid_" + str(self.runid))
        ensure_dir_exists(run_wiggle_dir)


        for wave_node in self.leaf_nodes:
            wave = wave_node.mw

            sta = wave['sta']
            chan = wave['chan']
            band = wave['band']

            smooth = 0
            for fstr in wave['filter_str'].split(';'):
                if 'smooth' in fstr:
                    smooth = int(fstr[7:])

            st = wave['stime']
            et = wave['etime']
            event = get_event(evid=wave['evid'])

            pv = wave_node._parent_values()
            arrivals = update_arrivals(pv)

            slon, slat, _, _, _, _, _ = s.earthmodel.site_info(sta, st)
            distance = geog.dist_km((event.lon, event.lat), (slon, slat))
            azimuth = geog.azimuth((slon, slat), (event.lon, event.lat))

            tmpl_optim_param_str = tmpl_optim_param_str.replace("'", "''")
            wiggle_optim_param_str = wiggle_optim_param_str.replace("'", "''")
            optim_log = wiggle_optim_param_str.replace("\n", "\\\\n")

            sql_query = "INSERT INTO sigvisa_coda_fit (runid, evid, sta, chan, band, smooth, tmpl_optim_method, wiggle_optim_method, optim_log, iid, stime, etime, hz, acost, dist, azi, timestamp, elapsed, nmid) values (%d, %d, '%s', '%s', '%s', '%d', '%s', '%s', '%s', %d, %f, %f, %f, %f, %f, %f, %f, %f, %d)" % (runid, event.evid, sta, chan, band, smooth, tmpl_optim_param_str, wiggle_optim_param_str, self.optim_log, 1 if wave_node.nm_type != 'ar' else 0, st, et, hz, wave_node.log_p(), distance, azimuth, time.time(), elapsed, wave_node.nmid)

            fitid = execute_and_return_id(s.dbconn, sql_query, "fitid")

            for (eid, phase) in arrivals:

                fit_param_nodes = self.get_template_nodes(eid=eid, phase=phase, chan=wave_node.chan, band=wave_node.band, sta=wave_node.sta)
                fit_params = dict([(p, n.get_value(k)) for (p,(k, n)) in fit_param_nodes.iteritems()])


                tg = self.template_generator(phase)

                peak_decay = fit_params['peak_decay'] if 'peak_decay' in fit_params else 0.0

                if eid > 0:
                    phase_insert_query = "insert into sigvisa_coda_fit_phase (fitid, phase, template_model, arrival_time, peak_offset, coda_height, coda_decay, amp_transfer, peak_decay) values (%d, '%s', '%s', %f, %f, %f, %f, %f, %f)" % (
                    fitid, phase, tg.model_name(), fit_params['arrival_time'], fit_params['peak_offset'], fit_params['coda_height'], fit_params['coda_decay'], fit_params['amp_transfer'], peak_decay)
                else:
                    phase_insert_query = "insert into sigvisa_coda_fit_phase (fitid, phase, template_model, arrival_time, peak_offset, coda_height, coda_decay, peak_decay) values (%d, '%s', '%s', %f, %f, %f, %f, %f)" % (
                    fitid, phase, tg.model_name(), fit_params['arrival_time'], fit_params['peak_offset'], fit_params['coda_height'], fit_params['coda_decay'], peak_decay)

                fpid = execute_and_return_id(s.dbconn, phase_insert_query, "fpid")
                for (k, n) in fit_param_nodes.values():
                   n.fpid = fpid

                if eid < 0:
                    # don't bother extracting wiggles for unass templates
                    continue

                # extract the empirical wiggle for this template fit
                wiggle, wiggle_st, wiggle_et = extract_phase_wiggle(arrival=(eid, phase),
                                                                    arrivals=arrivals,
                                                                    wave_node=wave_node)
                if len(wiggle) < wave['srate']:
                    print "evid %d phase %s at %s (%s, %s) is not prominent enough to extract a wiggle..." % (event.evid, phase, sta, chan, band)
                    wiggle_fname = "NONE"
                    wiggle_st = -1
                else:
                    wiggle_wave = Waveform(data=wiggle, srate=wave['srate'], stime=wiggle_st, sta=sta, chan=chan, evid=event.evid)
                    wiggle_fname = os.path.join("runid_" + str(self.runid), "%d.wave" % (fpid,))
                    wiggle_wave.dump_to_file(os.path.join(wiggle_dir, wiggle_fname))
                sql_query = "update sigvisa_coda_fit_phase set wiggle_fname='%s', wiggle_stime=%f where fpid=%d" % (wiggle_fname, wiggle_st, fpid,)
                cursor.execute(sql_query)


            fitids.append(fitid)
            s.dbconn.commit()
        cursor.close()

        return fitids
