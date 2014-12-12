import numpy as np
import sys
import os
import traceback
import pickle
import copy

import numdifftools as nd
from collections import defaultdict
from optparse import OptionParser
from sigvisa.database.signal_data import *
from sigvisa.database.dataset import *
import itertools

from sigvisa.models.ttime import tt_predict, tt_predict_grad
from sigvisa.graph.sigvisa_graph import SigvisaGraph, get_param_model_id
from sigvisa.learn.train_param_common import load_modelid
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_segments
from sigvisa.infer.event_birthdeath import sample_template_to_associate, template_association_logodds, associate_template, unassociate_template, sample_deassociation_proposal, template_association_distribution, propose_phase_template, deassociation_logprob
from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.infer.mcmc_basic import get_node_scales, gaussian_propose, gaussian_MH_move, MH_accept
from sigvisa.infer.template_mcmc import preprocess_signal_for_sampling, improve_offset_move, indep_peak_move, get_sorted_arrivals, relevant_nodes_hack
from sigvisa.graph.graph_utils import create_key, parse_key
from sigvisa.plotting.plot import savefig, plot_with_fit
from matplotlib.figure import Figure
from sigvisa.utils.geog import wrap_lonlat

from sigvisa.graph.graph_utils import get_parent_value
from sigvisa.source.event import Event
import sigvisa.source.brune_source as brune

from scipy.optimize import leastsq


from sigvisa.infer.propose_hough import propose_event_from_hough
from sigvisa.infer.propose_lstsqr import ev_lstsqr_dist

fixed_node_cache = dict()
relevant_node_cache = dict()


def ev_move_relevant_nodes(node_list, fixed_nodes):

    # loc: children are basically all the stochastic nodes, and arrival_time
    #      we want the stochastic nodes, and arrival_time's default parent

    # mb: children are coda_height, and that's maybe it? we want amp_transfer

    # time: children are arrival_time. we want tt_residual

    # depth: same as loc

    direct_stochastic_children = [c for n in node_list for c in n.children if not c.deterministic()]
    inlaws = [n.parents[n.default_parent_key()] for n in fixed_nodes]
    return set(node_list + direct_stochastic_children + inlaws)

"""
def set_ev(ev_node, v, fixed_vals, fixed_nodes, params, fixed_atimes=None, ignore_illegal=True):
    for (key, val) in zip(params, v):
        ev_node.set_local_value(key=key, value=val, force_deterministic_consistency=False)

    assert(len(fixed_nodes)==len(fixed_vals))
    for (val, n) in zip(fixed_vals, fixed_nodes):
        try:
            if fixed_atimes is not None \
               and n.label in fixed_atimes \
               and (not fixed_atimes[n.label]) \
               and n.deterministic():
                n.parent_predict()
            else:
                n.set_value(val)
        except ValueError as e:
            # ignore "illegal travel time" messages from phases that are about to disappear
            if ignore_illegal:
                pass
            else:
                raise e
"""


def atime_block_update_dist(sg, eid, old_ev, new_ev):

    fix_atime_probs = {}

    for site in sg.site_elements.keys():
        # TODO: implement multiple bands/chans
        assert (len(list(sg.site_bands[site])) == 1)
        band = list(sg.site_bands[site])[0]

        assert (len(list(sg.site_chans[site])) == 1)
        chan = list(sg.site_chans[site])[0]

        for phase in sg.ev_arriving_phases(eid, site=site):
            for sta in sg.site_elements[site]:
                tmnodes = sg.get_template_nodes(eid, sta, phase, band, chan)
                atime_key, atime_node = tmnodes['arrival_time']
                ttr_key, ttr_node = tmnodes['tt_residual']
                current_atime = atime_node.get_value(key=atime_key)
                current_tt_residual = ttr_node.get_value(key=ttr_key)
                current_pred_atime = current_atime - current_tt_residual

                try:
                    new_pred_atime = new_ev.time + tt_predict(new_ev, sta, phase=phase)
                except ValueError:
                    # if this phase is impossible at the new location,
                    # then we'll be deleting it anyway, so it doesn't
                    # matter what we do with arrival time
                    continue

                atime_diff = new_pred_atime-current_pred_atime
                new_tt_residual = current_tt_residual - atime_diff

                try:
                    wn = sg.get_wave_node_by_atime(sta, band, chan, current_pred_atime)
                    wave_lp1 = wn.log_p()
                    atime_node.set_value(new_pred_atime)
                    wave_lp2 = wn.log_p()
                    atime_node.set_value(current_atime)
                except KeyError:
                    wave_lp1 = 0.0
                    wave_lp2 = 0.0

                ttr_lp1 = ttr_node.model.log_p(new_tt_residual)
                ttr_lp2 = ttr_node.log_p(current_tt_residual)

                lp1 = wave_lp1 + ttr_lp1 # world 1: fix atime, shift tt_residual
                lp2 = wave_lp2 + ttr_lp2 # world 2: shift atime, fix tt_residual
                # we'll propose these worlds proportional to their relative probability
                fix_atime_probs[atime_node.label] = 1.0/(1+np.exp(lp2-lp1))

                #print "%s: current atime %.1f, pred %.1f, wave1 %.1f wave2 %.1f, tt1 %.1f tt2 %.1f, lp1 %.1f lp2 %.1f diff %.1f prob %f" \
                #% (sta, current_pred_atime, new_pred_atime, wave_lp1, wave_lp2, ttr_lp1, ttr_lp2, lp1, lp2, lp1-lp2, fix_atime_probs[(sta, phase)])

    return fix_atime_probs

def get_fixed_nodes(ev_node):

    if ev_node not in fixed_node_cache:
        sorted_children = sorted(ev_node.children, key = lambda n: n.label)
        fixed_nodes = [child for child in sorted_children if child.label.endswith("arrival_time") or child.label.endswith("coda_height")]
        fixed_node_cache[ev_node] = fixed_nodes
    else:
        fixed_nodes = fixed_node_cache[ev_node]

    if len(fixed_nodes) > 0:
        assert(fixed_nodes[0] in ev_node.children)
    return fixed_nodes

def clear_node_caches(sg, eid):
    for ev_node in sg.evnodes[eid].values():
        try:
            del fixed_node_cache[ev_node]
        except KeyError as e:
            pass
        try:
            del relevant_node_cache[ev_node]
        except KeyError:
            pass

def add_phase_template(sg, sta, eid, phase, vals=None, node_lps=None):
    tg = sg.template_generator(phase)
    wg = sg.wiggle_generator(phase, sg.base_srate)



    phase_added = False
    if phase not in sg.ev_arriving_phases(eid, sta=sta):
        s = Sigvisa()
        site = s.get_array_site(sta)
        if node_lps is not None:
            node_lps.register_new_phase_pre(sg, site, phase, eid)
        sg.add_event_site_phase(tg, wg, site, phase, sg.evnodes[eid], sample_templates=True)
        phase_added=True

    tmvals, lp = propose_phase_template(sg, sta, eid, phase)

    s = Sigvisa()
    site = s.get_array_site(sta)
    assert (len(list(sg.site_bands[site])) == 1)
    band = list(sg.site_bands[site])[0]
    assert (len(list(sg.site_chans[site])) == 1)
    chan = list(sg.site_chans[site])[0]
    tmnodes = sg.get_template_nodes(eid, sta, phase, band, chan)
    for p, (k, n) in tmnodes.items():
        if p in tmvals:
            n.set_value(value=tmvals[p], key=k)

    if node_lps is not None:
        assert(phase_added) # otherwise the logic is wrong
        node_lps.register_new_phase_post(sg, site, phase, eid)

    return tmvals, lp


def phases_changed(sg, eid, ev):
    # given a proposed event, return dictionaries mapping sites to
    # sets of phases that need to be birthed or killed.

    old_site_phases = dict()
    for site, stas in sg.site_elements.items():
        # TODO: can we get arriving phases for a site instead of sta?
        old_site_phases[site] = set(sg.ev_arriving_phases(eid, site=site))

    birth_phases = dict()
    death_phases = dict()
    jump_required = False
    for site in old_site_phases.keys():
        new_site_phases = sg.predict_phases_site(ev=ev, site=site)
        birth_phases[site] = new_site_phases - old_site_phases[site]
        death_phases[site] = old_site_phases[site] - new_site_phases
        if len(birth_phases[site]) > 0 or len(death_phases[site]) > 0:
            jump_required = True
    return birth_phases, death_phases, jump_required


class UpdatedNodeLPs(object):
    def __init__(self, old_ev):
        self.nodes_added = set()
        self.nodes_removed = dict()
        self.nodes_changed_old = dict()
        self.nodes_changed_new = dict()
        self.uatemplate_count_delta = defaultdict(int)

        self.old_ev=old_ev

    def dump_debug_info(self):
        print "nodes added:"
        for n in self.nodes_added:
            print "  %s" % (n.label)

        print "nodes changed:"
        for n in self.nodes_changed_old.keys():
            print "  %s: %.2f %.2f" % (n.label, self.nodes_changed_old[n], self.nodes_changed_new[n])

        print "nodes removed:"
        for n, lp in self.nodes_removed.items():
            print "  %s: %.2f" % (n.label, lp)

        print "uatemplate counts:"
        for k, delta in self.uatemplate_count_delta.items():
            print k, delta

    def dump_detected_changes(self, lps_old, lps_new, relevant_nodes):
        rn_labels = [n.label for n in relevant_nodes]
        for nl in lps_old.keys():
            if nl not in lps_new:
                print "removed node: %s %.2f" % (nl, lps_old[nl])
                continue
            if lps_old[nl] != lps_new[nl]:
                if nl in rn_labels:
                    continue
                print "changed node: %s %.2f %.2f" % (nl, lps_old[nl], lps_new[nl])
        for nl in lps_new.keys():
            if nl not in lps_old:
                print "added node: %s %.2f" % (nl, lps_new[nl])

    def register_remove_uatemplate(self, sg, tmid, wn_invariant=True):
        assert(wn_invariant)
        tmnodes = sg.uatemplates[tmid]
        for n in tmnodes.values():
            self.nodes_removed[n] = n.log_p()
        eid, phase, sta, chan, band, param = parse_key(tmnodes.values()[0].label)
        self.uatemplate_count_delta[(sta, chan, band)] -= 1

    def register_new_uatemplate(self, sg, tmid, wn_invariant=True):
        assert(wn_invariant)
        tmnodes = sg.uatemplates[tmid]
        for n in tmnodes.values():
            self.nodes_added.add(n)
        eid, phase, sta, chan, band, param = parse_key(tmnodes.values()[0].label)
        self.uatemplate_count_delta[(sta, chan, band)] += 1

    def __register_phase_helper(self, sg, site, phase, eid, d, f_lp):
        for sta in sg.site_elements[site]:
            for band in sg.site_bands[site]:
                for chan in sg.site_chans[site]:
                    nodes = sg.get_arrival_nodes(eid, sta, phase, band, chan)
                    for (k, n) in nodes.values():
                        if not n.deterministic():
                            if isinstance(d, dict):
                                d[n] = f_lp(n)
                            else:
                                d.add(n)

    def _get_phase_wns(self, sg, site, phase, eid):
        wns = []
        for sta in sg.site_elements[site]:
            for band in sg.site_bands[site]:
                for chan in sg.site_chans[site]:
                    wn = sg.get_arrival_wn(sta, eid, phase, band, chan, revert_to_atime=True)
                    wns.append(wn)

        return wns

    def register_new_phase_pre(self, sg, site, phase, eid):
        for wn in self._get_phase_wns(sg, site, phase, eid):
            if wn not in self.nodes_changed_old:
                self.nodes_changed_old[wn] = wn.log_p()

    def register_new_phase_post(self, sg, site, phase, eid):
        self.__register_phase_helper(sg, site, phase, eid, self.nodes_added, lambda n : n.log_p())
        for wn in self._get_phase_wns(sg, site, phase, eid):
            self.nodes_changed_new[wn] = wn.log_p()

    def register_phase_changed_oldvals(self, sg, site, phase, eid, wn_invariant=False):
        self.__register_phase_helper(sg, site, phase, eid, self.nodes_changed_old, self.tmnode_lp_under_old_ev)
        if not wn_invariant:
            for wn in self._get_phase_wns(sg, site, phase, eid):
                if wn not in self.nodes_changed_old:
                    self.nodes_changed_old[wn] = wn.log_p()

    def register_phase_changed_newvals(self, sg, site, phase, eid, wn_invariant=False):
        self.__register_phase_helper(sg, site, phase, eid, self.nodes_changed_new, lambda n : n.log_p())
        if not wn_invariant:
            for wn in self._get_phase_wns(sg, site, phase, eid):
                self.nodes_changed_new[wn] = wn.log_p()

    def register_phase_removed_pre(self, sg, site, phase, eid, wn_invariant=False):
        self.__register_phase_helper(sg, site, phase, eid, self.nodes_removed, self.tmnode_lp_under_old_ev)
        if wn_invariant:
            return
        self._wns_for_removed_phase = set()
        for wn in self._get_phase_wns(sg, site, phase, eid):
            self._wns_for_removed_phase.add(wn)
            if wn not in self.nodes_changed_old:
                self.nodes_changed_old[wn] = wn.log_p()

    def register_phase_removed_post(self, sg, site, phase, eid):
        for wn in self._wns_for_removed_phase:
            self.nodes_changed_new[wn] = wn.log_p()
        del self._wns_for_removed_phase

    def tmnode_lp_under_old_ev(self, n):
        pv = copy.copy(n._parent_values())
        for k in pv.keys():
            if "lon" in k:
                pv[k] = self.old_ev.lon
            elif "lat" in k:
                pv[k] = self.old_ev.lat
            elif "depth" in k:
                pv[k] = self.old_ev.depth
            elif "time" in k:
                pv[k] = self.old_ev.time
            elif "mb" in k:
                pv[k] = self.old_ev.mb
            elif "natural_source" in k:
                pv[k] = self.old_ev.natural_source
            elif "prefix" in k:
                continue
            else:
                raise KeyError("unrecognized parent key %s for node %s" % (k, n.label))

        if "tt_residual" in n.label:
            atime_children = [nn for nn in n.children if "arrival_time" in nn.label]
            assert(len(atime_children)==1)
            n_atime = atime_children[0]
            pv2 = copy.copy(n_atime._parent_values())
            pv2.update(pv)
            ttr = n_atime.invert(n_atime.get_value(), n.single_key, parent_values=pv2)
            lp = n.log_p(v=ttr, parent_values=pv2)
            # residual should reflect predicted ttime from old ev, not current ev
        elif "amp_transfer" in n.label:
            amp_children = [nn for nn in n.children if "coda_height" in nn.label]
            assert(len(amp_children)==1)
            n_amp = amp_children[0]
            pv2 = copy.copy(n_amp._parent_values())
            pv2.update(pv)
            amp_transfer = n_amp.invert(n_amp.get_value(), n.single_key, parent_values=pv2)
            lp = n.log_p(v=amp_transfer, parent_values=pv2)
        else:
            lp = n.log_p(parent_values=pv)
        return lp

    def update_lp_old(self, sg, relevant_nodes):
        lp_delta = 0
        # retroactively include any nodes that were since deleted
        for (n, lp) in self.nodes_removed.items():
            if n not in relevant_nodes:
                lp_delta += lp
        return lp_delta

    def update_relevant_nodes_for_lpnew(self, relevant_nodes):
        for n in self.nodes_added:
            relevant_nodes.add(n)
        for n in self.nodes_removed.keys():
            if n in relevant_nodes:
                relevant_nodes.remove(n)

    def update_lp_new(self, sg, relevant_nodes):
        lp_delta = 0

        # update any nodes that were changed
        for n in self.nodes_changed_old.keys():
            if n in relevant_nodes:
                continue
            lp_delta += self.nodes_changed_new[n] - self.nodes_changed_old[n]

        # also account for the uatemplate Poisson process prior
        for k, delta in self.uatemplate_count_delta.items():
            sta, chan, band = k
            new_count = len(sg.uatemplate_ids[k])
            old_count = new_count - delta
            old_lp = sg.ntemplates_sta_log_p(sta, n=old_count)
            new_lp = sg.ntemplates_sta_log_p(sta, n=new_count)
            lp_delta += new_lp-old_lp

        return lp_delta

    def update(self, b):
        self.old_ev = None
        self.nodes_added = self.nodes_added | b.nodes_added

        b.nodes_removed.update(self.nodes_removed)
        self.nodes_removed = b.nodes_removed

        b.nodes_changed_old.update(self.nodes_changed_old)
        self.nodes_changed_old = b.nodes_changed_old

        self.nodes_changed_new.update(b.nodes_changed_new)
        self.uatemplate_count_delta = defaultdict(int, [(k, self.uatemplate_count_delta[k] + b.uatemplate_count_delta[k])  for k in set(self.uatemplate_count_delta.keys()) | set(b.uatemplate_count_delta.keys())])

    def __add__(self, b):
        n = UpdatedNodeLPs(None)
        n.nodes_added = self.nodes_added | b.nodes_added

        n.nodes_removed.update(self.nodes_removed).update(b.nodes_removed)
        n.nodes_changed_old.update(self.nodes_changed_old).update(b.nodes_changed_old)
        n.nodes_changed_new.update(self.nodes_changed_new).update(b.nodes_changed_new)

        for (k, d) in self.uatemplate_count_delta.items():
            n.uatemplate_count_delta[k] += d
        for (k, d) in b.uatemplate_count_delta.items():
            n.uatemplate_count_delta[k] += d
        return n


def ev_phasejump(sg, eid, new_ev, params_changed, adaptive_blocking=False, birth_phases=None, death_phases=None, jump_required=None):
    # given proposed new event parameters, propose to birth/kill
    # phases as needed to generate the proper set of phases for the
    # new location.
    # sets the graph to the newly proposed state, and returns:
    #     - log_qforward: logp of the proposed templates
    #     - log_qbackward: probability of proposing the reverse
    #                      changes, if told to move from the new
    #                      ev back to the original ev
    #     - revert_changes: method to reverse the changes and move the
    #                       chain back to the old state (old ev and
    #                       old templates)
    #     - jump_required: whether this move added/removed nodes from the graph

    def deterministic_phase_swap(sg, eid, birth_phases, death_phases, inverse_fns, phase1, phase2):
        if phase1 in birth_phases and phase2 in death_phases:
            rename_phase(sg, eid, phase1, phase2)
            birth_phases.remove(phase1)
            death_phases.remove(phase2)
            inverse_fns.append( lambda : rename_phase(sg, eid, phase2, phase1  ) )
        elif phase2 in birth_phases and phase1 in death_phases:
            rename_phase(sg, eid, phase2, phase1)
            birth_phases.remove(phase2)
            death_phases.remove(phase1)
            inverse_fns.append( lambda : rename_phase(sg, eid, phase1, phase2  ) )

    move_logprob = 0
    reverse_logprob = 0
    old_ev = sg.get_event(eid)

    if birth_phases is None:
        birth_phases, death_phases, jump_required = phases_changed(sg, eid, new_ev)

    if adaptive_blocking:
        atime_fix_probs = atime_block_update_dist(sg, eid, old_ev, new_ev)
        fixed_atime_block = dict([(k, np.random.rand() < p) for (k, p) in atime_fix_probs.items()])
        fixed_atime_block_lp = np.sum([np.log(atime_fix_probs[k] if fix else 1-atime_fix_probs[k]) for (k, fix) in fixed_atime_block.items()])
        preserve_templates= [nl for (nl, fix) in fixed_atime_block.items() if  fix]
        move_logprob += fixed_atime_block_lp
    else:
        fixed_atime_block = None
        fixed_atime_block_lp = 0.0
        preserve_templates=True

    node_lps = UpdatedNodeLPs(old_ev)
    new_site_phases = dict()
    forward_fns = []
    inverse_fns = [lambda :     sg.set_event(eid, old_ev, params_changed=params_changed, preserve_templates=preserve_templates, illegal_phase_action="ignore"),]
    associations = []
    deassociations = []
    tmid_i = 0
    tmids = []
    sg.set_event(eid, new_ev, params_changed=params_changed, preserve_templates=preserve_templates, node_lps=node_lps, illegal_phase_action="ignore")
    s = Sigvisa()
    for site in birth_phases.keys():
        # TODO: implement multiple bands/chans
        assert (len(list(sg.site_bands[site])) == 1)
        band = list(sg.site_bands[site])[0]

        assert (len(list(sg.site_chans[site])) == 1)
        chan = list(sg.site_chans[site])[0]

        deterministic_phase_swap(sg, eid, birth_phases[site], death_phases[site], inverse_fns, "P", "Pn")

        # the set of phases generated by each event is
        # deterministic. so whenever an event move generates a new
        # phase at some station, our proposal must include that phase.
        # as with an event birth move, we sample whether to associate
        # an existing unass template, or create a new template from
        # scratch.
        for sta in sg.site_elements[site]:
            for phase in birth_phases[site]:
                tmid, assoc_logprob = sample_template_to_associate(sg, sta, eid, phase)
                if tmid is not None:
                    # associate an unass. template
                    forward_fns.append(lambda sta=sta,phase=phase,tmid=tmid: associate_template(sg, sta, tmid, eid, phase, create_phase_arrival=True, node_lps=node_lps))
                    inverse_fns.append(lambda sta=sta,phase=phase: unassociate_template(sg, sta, eid, phase, remove_event_phase=True))
                    associations.append((sta, phase, True))
                    print "proposing to associate %d to %d %s at %s" % (tmid, eid, phase, sta),
                else:
                    # propose a new template from scratch
                    forward_fns.append( lambda sta=sta,phase=phase,eid=eid: add_phase_template(sg, sta, eid, phase, node_lps=node_lps)[1] )
                    inverse_fns.append(lambda eid=eid,sta=sta,phase=phase: sg.delete_event_phase(eid, sta, phase))
                    associations.append((sta, phase, False))
                    print "proposing new template for %d %s at %s" % (eid, phase, sta),

                move_logprob += assoc_logprob

            # similarly, for every phase that is no longer generated from
            # the new location, we must either delete or de-associate the
            # corresponding template.
            for phase in death_phases[site]:
                deassociate, deassociate_logprob = sample_deassociation_proposal(sg, sta, eid, phase)
                deassociations.append((sta, phase, deassociate, tmid_i))
                if deassociate:
                    # deassociation will produce a new uatemplated
                    # with incrementing tmid. We keep track of this
                    # tmid (kind of a hack) to ensure that we
                    # reassociate the same template if the move gets
                    # rejected.
                    forward_fns.append(lambda sta=sta,phase=phase: tmids.append(unassociate_template(sg, sta, eid, phase, remove_event_phase=True, node_lps=node_lps)))
                    inverse_fns.append(lambda sta=sta,phase=phase,tmid_i=tmid_i: associate_template(sg, sta, tmids[tmid_i], eid, phase, create_phase_arrival=True))
                    tmid_i += 1
                    print "proposing to deassociate %s for %d at %s (lp %.1f)" % (phase, eid, sta, deassociate_logprob),
                else:
                    template_param_array = sg.get_arrival_vals(eid, sta, phase, band, chan)

                    forward_fns.append(lambda eid=eid,site=site,phase=phase: node_lps.register_phase_removed_pre(sg, site, phase, eid))
                    forward_fns.append(lambda eid=eid,sta=sta,phase=phase: sg.delete_event_phase(eid, sta, phase))
                    forward_fns.append(lambda eid=eid,site=site,phase=phase: node_lps.register_phase_removed_post(sg, site, phase, eid))
                    inverse_fns.append( lambda sta=sta,phase=phase,eid=eid: add_phase_template(sg, sta, eid, phase) )
                    inverse_fns.append(lambda sta=sta,phase=phase,band=band,chan=chan,template_param_array=template_param_array : sg.set_template(eid,sta, phase, band, chan, template_param_array))
                    tmp = propose_phase_template(sg, sta, eid, phase, template_param_array, fix_result=True, ev=old_ev)
                    reverse_logprob += tmp
                    print "proposing to delete %s for %d at %s (lp %f)"% (phase, eid, sta, deassociate_logprob),

                move_logprob += deassociate_logprob

    for fn in forward_fns:
        x = fn()
        if x is not None:
            move_logprob += x
    sg._topo_sort()
    clear_node_caches(sg, eid)


    if adaptive_blocking:
         atime_fix_probs_reverse = atime_block_update_dist(sg, eid, new_ev, old_ev)
         fixed_atime_block_reverse_lp = np.sum([np.log(atime_fix_probs_reverse[k] if fix else 1-atime_fix_probs_reverse[k]) for (k, fix) in fixed_atime_block.items()])
         reverse_logprob += fixed_atime_block_reverse_lp

    # revert the event to the old location, temporarily, so that we
    # can compute probabilities for the reverse move
    if jump_required:
        sg.set_event(eid, old_ev, params_changed=params_changed, preserve_templates=preserve_templates, illegal_phase_action="ignore")
        for (sta, phase, associated) in associations:
            dl = deassociation_logprob(sg, sta, eid, phase, deletion_prob=not associated)
            reverse_logprob += dl
            print "rl +=", dl, "to associate", sta, phase

        for (sta, phase, deassociate, tmid_i) in deassociations:
            c = template_association_distribution(sg, sta, eid, phase)
            if deassociate:
                tmid = tmids[tmid_i]
                tmp = np.log(c[tmid])
                reverse_logprob += tmp
                print "rl +=", tmp, "to deassociate", sta, phase, tmid
            else:
                tmp = np.log(c[None])
                print "rl +=", tmp, "to delete", sta, phase
                reverse_logprob += tmp
        sg.set_event(eid, new_ev, params_changed=params_changed, preserve_templates=preserve_templates, illegal_phase_action="ignore")

    def revert_move():
        for fn in inverse_fns:
            fn()
        sg._topo_sort()
        clear_node_caches(sg, eid)


    return move_logprob, reverse_logprob, revert_move, jump_required, node_lps



def ev_move_full(sg, ev_node, std, params, adaptive_blocking=False, debug_probs=False):
    # jointly propose a new event location along with new tt_residual values,
    # such that the event arrival times remain constant.

    def update_ev(old_ev, params, new_v):
        new_ev = copy.copy(old_ev)
        try:
            i = params.index("lon")
            new_ev.lon = new_v[i]
        except ValueError:
            pass
        try:
            i = params.index("lat")
            new_ev.lat = new_v[i]
        except ValueError:
            pass
        try:
            i = params.index("depth")
            new_ev.depth = new_v[i]
        except ValueError:
            pass
        try:
            i = params.index("time")
            new_ev.time = new_v[i]
        except ValueError:
            pass
        try:
            i = params.index("mb")
            new_ev.mb = new_v[i]
        except ValueError:
            pass
        return new_ev


    d = len(params)
    # get the current values of the params we're updating
    current_v = np.zeros((d,))
    for i in range(d):
        current_v[i] = ev_node.get_local_value(params[i])

    # find the nodes whose values should be held fixed even as the event moves
    fixed_nodes = get_fixed_nodes(ev_node)
    fixed_vals = [n.get_value() for n in fixed_nodes]

    if ev_node not in relevant_node_cache:
        node_list = [ev_node,]
        relevant_nodes = ev_move_relevant_nodes(node_list, fixed_nodes)
        relevant_node_cache[ev_node] = (node_list, relevant_nodes)
    else:
        (node_list, relevant_nodes) = relevant_node_cache[ev_node]

    # propose a new set of param values
    gsample = np.random.normal(0, std, d)
    move = gsample * std
    new_v = current_v + move

    if params[0] == "depth":
        if new_v[0] < 0:
            new_v[0] = 0.0
        if new_v[0] > 700:
            new_v[0] = 700.0

    if "lon" in params:
        new_v[0], new_v[1] = wrap_lonlat(new_v[0], new_v[1])

    eid = int(ev_node.label.split(';')[0])
    old_ev = sg.get_event(eid)
    new_ev = update_ev(old_ev, params, new_v)

    if debug_probs:
        lp_old_full = sg.current_log_p()
        lps_old = dict([(n.label, n.log_p()) for n in sg.all_nodes.values() if not n.deterministic()])

    lp_old = sg.joint_logprob(node_list=node_list, relevant_nodes=relevant_nodes, values=None)
    move_logprob, reverse_logprob, revert_move, jump_required, node_lps = ev_phasejump(sg, eid, new_ev, params, adaptive_blocking=adaptive_blocking)
    lp_old += node_lps.update_lp_old(sg, relevant_nodes)


    node_lps.update_relevant_nodes_for_lpnew(relevant_nodes)
    lp_new = sg.joint_logprob(node_list=node_list, relevant_nodes=relevant_nodes, values=None)
    lp_new += node_lps.update_lp_new(sg, relevant_nodes)

    if debug_probs:
        lp_new_full = sg.current_log_p()
        lps_new = dict([(n.label, n.log_p()) for n in sg.all_nodes.values() if not n.deterministic()])
        print "updates"
        node_lps.dump_debug_info()
        print "actual changes:"
        node_lps.dump_detected_changes(lps_old, lps_new, relevant_nodes)
        assert( np.abs( (lp_new-lp_old) - (lp_new_full-lp_old_full) ) < 1e-8)

    u = np.random.rand()
    move_accepted = (lp_new + reverse_logprob) - (lp_old+move_logprob)  > np.log(u)

    if move_accepted:
        return True
    else:
        revert_move()
        return False

def ev_lonlat_density(frame=None, fname="ev_viz.png"):

    d = np.load("ev_vals.npz")
    latlons = d['evloc']
    lonlats = np.array([(a,b) for (b,a) in latlons])

    if frame is not None:
        if frame > len(lonlats):
            raise ValueError("no more frames!")
        lonlats_plot = lonlats[:frame]
    else:
        lonlats_plot = lonlats

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from sigvisa.plotting.event_heatmap import EventHeatmap

    f = Figure((11,8))
    ax = f.add_subplot(111)
    hm = EventHeatmap(f=None, autobounds=lonlats, autobounds_quantile=0.9995, calc=False)
    hm.init_bmap(axes=ax)
    hm.plot_earth(y_fontsize=16, x_fontsize=16)

    ev = get_event(evid=5393637)

    baseline_alpha = 0.008
    alpha_fade_time = 500
    if frame is not None:
        alpha = np.ones((frame,)) * baseline_alpha
        t = min(frame,alpha_fade_time)
        alpha[-t:] = np.linspace(baseline_alpha, 0.2, alpha_fade_time)[-t:]
    else:
        alpha = baseline_alpha

    #hm.plot_locations(X, marker=".", ms=6, mfc="red", mec="none", mew=0, alpha=0.2)
    scplot = hm.plot_locations(lonlats_plot, marker=".", ms=8, mfc="red", mew=0, mec="none", alpha=alpha)
    hm.plot_locations(np.array(((ev.lon, ev.lat),)), marker="x", ms=5, mfc="blue", mec="blue", mew=3, alpha=1.0)
    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig(fname, bbox_inches="tight", dpi=300, transparent=True, )

def ev_lonlat_frames():
    for i in range(40, 10000, 40):
        ev_lonlat_density(frame=i, fname='ev_viz_step%06d.png' % i)


def propose_event_lsqr_prob(sg, eid, **kwargs):
    z, C = ev_lstsqr_dist(sg, eid, **kwargs)
    rv = scipy.stats.multivariate_normal(z, C)
    old_ev = sg.get_event(eid)
    old_vals = np.array([old_ev.lon, old_ev.lat, old_ev.depth, old_ev.time])
    old_lp = rv.logpdf(old_vals)

    print "backward lp", old_lp

    return old_lp

def propose_event_lsqr(sg, eid, **kwargs):
    z, C = ev_lstsqr_dist(sg, eid, **kwargs)
    rv = scipy.stats.multivariate_normal(z, C)
    proposed_vals = rv.rvs(1)
    lon, lat, depth, time = proposed_vals
    proposal_lp = rv.logpdf(proposed_vals)

    # this breaks Gaussianity, technically we should be using a
    # circular (von Mises?) distribution. but hopefully it doesn't
    # matter.
    lon, lat = wrap_lonlat(lon, lat)

    # this definitely breaks Gaussianity, we should be explicitly truncating the distribution
    if depth > 700:
        depth = 700
    elif depth < 0:
        depth = 0

    old_ev = sg.get_event(eid)
    new_ev = copy.copy(old_ev)
    new_ev.lon = lon
    new_ev.lat = lat
    new_ev.depth=depth
    new_ev.time=time

    move_logprob, reverse_logprob, revert_move, jump_required, node_lps = ev_phasejump(sg, eid, new_ev, params_changed=['lon', 'lat', 'depth', 'time'])

    move_logprob += proposal_lp

    return move_logprob, reverse_logprob, revert_move, jump_required, node_lps


def sample_uniform_pair_to_swap(sg, wn, adjacency_decay=0.8):
    sorted_arrs = get_sorted_arrivals(wn)
    n = len(sorted_arrs)

    # if we sample adjacency=1, swap an adjacent pair
    # adjacency=2 => swap a pair separated by another template
    # etc.
    adjacency = np.random.geometric(adjacency_decay)
    adjacency_prob = adjacency_decay * (1-adjacency_decay)**(adjacency-1)
    if adjacency > n-1:
        return None, None, 1.0

    # then just propose a pair to swap uniformly at random
    first_idx = np.random.choice(np.arange(n-adjacency))
    second_idx = first_idx + adjacency
    choice_prob = 1.0/(n-adjacency)
    return sorted_arrs[first_idx], sorted_arrs[second_idx], adjacency_prob*choice_prob



def swap_params(t1nodes, t2nodes):
    for (p, (k1, n1)) in t1nodes.items():
        if p=="amp_transfer" or p=="tt_residual":
            continue
        k2, n2 = t2nodes[p]
        v1 = n1.get_value(key=k1)
        v2 = n2.get_value(key=k2)
        n1.set_value(key=k1, value=v2)
        n2.set_value(key=k2, value=v1)

        if p == "arrival_time":
            atime1, atime2 = v1, v2
    return atime1, atime2

def swap_association_move(sg, wave_node, repropose_events=False, debug_probs=False, stas=None):

    # sample from all pairs of adjacent templates in which not both are uatemplates
    arr1, arr2, pair_prob = sample_uniform_pair_to_swap(sg, wave_node)
    if arr1 is None:
        return False

    # get all relevant nodes for the arrivals we sampled
    t1nodes = sg.get_template_nodes(eid=arr1[1], phase=arr1[2], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)
    t2nodes = sg.get_template_nodes(eid=arr2[1], phase=arr2[2], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)
    rn = set(relevant_nodes_hack(t1nodes) + relevant_nodes_hack(t2nodes))
    if repropose_events:
        if arr1[1] > 0:
            evnodes = set([n for n in sg.extended_evnodes[arr1[1]] if not n.deterministic()])
            rn = rn.union(evnodes)
        if arr2[1] != arr1[1] and arr2[1] > 0:
            evnodes = [n for n in sg.extended_evnodes[arr2[1]] if not n.deterministic()]
            rn = rn.union(evnodes)

    if debug_probs:
        lp_old_full = sg.current_log_p()
        lps_old = dict([(n.label, n.log_p()) for n in sg.all_nodes.values() if not n.deterministic()])
    lp_old = sg.joint_logprob_keys(rn)

    # swap proposal is symmetric, but we still need to track
    # probabilities of the event proposals.
    log_qforward = 0
    log_qbackward = 0

    if repropose_events:
        if arr1[1] > 0:
            log_qbackward += propose_event_lsqr_prob(sg, eid=arr1[1], stas=stas)
        if (arr2[1] != arr1[1]) and arr2[1] > 0:
            log_qbackward += propose_event_lsqr_prob(sg, eid=arr2[1], stas=stas)



    # switch their parameters
    atime1, atime2 = swap_params(t1nodes, t2nodes)

    revert_fns = []

    lp_old_delta1 = 0
    lp_old_delta2 = 0
    lp_new_delta1 = 0
    lp_new_delta2 = 0
    node_lps1 = None
    node_lps2 = None
    if repropose_events:
        if arr1[1] > 0:
            proposal_lp, old_lp, revert_move, jump_required, node_lps1 = propose_event_lsqr(sg, eid=arr1[1], stas=stas)
            log_qforward += proposal_lp
            revert_fns.append(revert_move)
            lp_old_delta1 = node_lps1.update_lp_old(sg, rn)
            lp_new_delta1 = node_lps1.update_lp_new(sg, rn)
            node_lps1.update_relevant_nodes_for_lpnew(rn)
        if (arr2[1] != arr1[1]) and arr2[1] > 0:
            proposal_lp, old_lp, revert_move, jump_required, node_lps2 = propose_event_lsqr(sg, eid=arr2[1], stas=stas)
            log_qforward += proposal_lp
            revert_fns.append(revert_move)
            lp_old_delta2 = node_lps2.update_lp_old(sg, rn)
            lp_new_delta2 = node_lps2.update_lp_new(sg, rn)
            node_lps2.update_relevant_nodes_for_lpnew(rn)

    lp_new = sg.joint_logprob_keys(rn)
    lp_new += lp_new_delta1 + lp_new_delta2
    lp_old += lp_old_delta1 + lp_old_delta2

    if debug_probs:
        lp_new_full = sg.current_log_p()
        lps_new = dict([(n.label, n.log_p()) for n in sg.all_nodes.values() if not n.deterministic()])

        if node_lps1 is not None:
            print "updates from ev1 proposal"
            node_lps1.dump_debug_info()
        if node_lps2 is not None:
            print "updates from ev2 proposal"
            node_lps2.dump_debug_info()
        print "actual changes:"
        node_lps1.dump_detected_changes(lps_old, lps_new, rn)
        assert( np.abs( (lp_new-lp_old) - (lp_new_full-lp_old_full) ) < 1e-8)


    u = np.random.rand()

    def revert_all():
        for fn in revert_fns:
            fn()
        atime1, atime2 = swap_params(t1nodes, t2nodes)

    #return lp_new, lp_old, log_qbackward, log_qforward, arr1, arr2, revert_all


    if (lp_new + log_qbackward) - (lp_old + log_qforward) > np.log(u):
        return True
    else:
        revert_all()
        return False


def run_event_MH(sg, evnodes, wn_list, burnin=0, skip=40, steps=10000):

    n_accepted = dict()
    n_tried = dict()
    moves = ( 'indep_peak', 'peak_offset', 'tt_residual', 'amp_transfer', 'coda_decay', 'evloc', 'evloc_big', 'evtime', 'evdepth', 'evmb')
    for move in moves:
        n_accepted[move] = 0
        n_tried[move] = 0

    stds = {'peak_offset': .1, 'tt_residual': .1, 'amp_transfer': .1, 'coda_decay': 0.01, 'evloc': 0.01, 'evloc_big': 0.5, 'evtime': 2.0, "evmb": 0.5, "evdepth": 5.0}

    templates = dict()
    params_over_time = defaultdict(list)

    for wn in wn_list:
        wave_env = wn.get_value() if wn.env else wn.get_wave().filter('env').data
        wn.cdf = preprocess_signal_for_sampling(wave_env)

        arrivals = wn.arrivals()
        eid, phase = list(arrivals)[0]
        templates[wn.sta] = dict([(param, node) for (param, (key, node)) in sg.get_template_nodes(eid=eid, phase=phase, sta=wn.sta, band=wn.band, chan=wn.chan).items()])

    for step in range(steps):
        for wn in wn_list:
            arrivals = wn.arrivals()
            eid, phase = list(arrivals)[0]

            wg = sg.wiggle_generator(phase=phase, srate=wn.srate)
            tmnodes = templates[wn.sta]

            n_accepted['peak_offset'] += improve_offset_move(sg, arrival_node=tmnodes["tt_residual"],
                                                           offset_node=tmnodes["peak_offset"],
                                                             wave_node=wn, std=stds['peak_offset'])
            n_tried["peak_offset"] += 1

            for param in ("tt_residual","amp_transfer","coda_decay"):
                n = tmnodes[param]
                n_accepted[param] += gaussian_MH_move(sg, node_list=(n,), relevant_nodes=(n, wn), std=stds[param])
                n_tried[param] += 1

            for (param, n) in tmnodes.items():
                params_over_time["%s_%s" % (wn.sta, param)].append(n.get_value())

        n_accepted["evloc"] += ev_move(sg, evnodes['loc'], std=stds['evloc'], params=('lon', 'lat'))
        n_tried["evloc"] += 1

        n_accepted["evloc_big"] += ev_move(sg, evnodes['loc'], std=stds['evloc_big'], params=('lon', 'lat'))
        n_tried["evloc_big"] += 1

        n_accepted["evtime"] += ev_move(sg, evnodes['time'], std=stds['evtime'], params=("time",))
        n_tried["evtime"] += 1

        print evnodes['loc'].get_local_value(key="depth")

        n_accepted["evdepth"] += ev_move(sg, evnodes['loc'], std=stds['evdepth'], params=("depth",))
        n_tried["evdepth"] += 1

        print evnodes['loc'].get_local_value(key="depth")

        n_accepted["evmb"] += ev_move(sg, evnodes['mb'], std=stds['evmb'], params=("mb",))
        n_tried["evmb"] += 1

        params_over_time["evloc"].append( evnodes['loc'].get_mutable_values())
        params_over_time["evtime"].append( evnodes['time'].get_mutable_values())
        params_over_time["evdepth"].append( evnodes['loc'].get_mutable_values())
        params_over_time["evmb"].append( evnodes['mb'].get_mutable_values())

        if step > 0 and ((step % skip == 0) or (step < 15)):
            lp = sg.current_log_p()
            print "step %d: lp %.2f, %d templates, accepted " % (step, lp, len(arrivals)),
            for move in moves:
                if (move == "birth") or (move == "death"):
                    print "%s: %d, " % (move, n_accepted[move]),
                else:
                    accepted_percent = float(n_accepted[move]) / n_tried[move] *100 if n_tried[move] > 0 else 0
                    print "%s: %d%%, " % (move, accepted_percent),
            print
            print " ev loc", evnodes['loc'].get_mutable_values()
            #for wn in wn_list:
            #    plot_with_fit("ev_%s_step%06d.png" % (wn.sta, step), wn)

        if step % 200 == 10:
            np.savez('ev_vals.npz', **params_over_time)
    """
    for (param, vals) in params_over_time.items():
        fig = Figure(figsize=(8, 5), dpi=144)
        axes = fig.add_subplot(111)
        axes.set_xlabel("Steps", fontsize=8)
        axes.set_ylabel(param, fontsize=8)
        axes.plot(vals)
        savefig("mcmc_unass_%s.png" % param, fig)

    np.savez('mcmc_unass_vals.npz', **params_over_time)
    """


def main():

    parser = OptionParser()
    register_svgraph_cmdline(parser)
    register_svgraph_event_based_signal_cmdline(parser)
    (options, args) = parser.parse_args()

    sg = setup_svgraph_from_cmdline(options, args)

    evnodes = load_event_based_signals_from_cmdline(sg, options, args)

    key_prefix = "%d;" % (evnodes['mb'].eid)

    evnodes['natural_source'].fix_value(key = key_prefix + "natural_source")
    evnodes['lon'].set_value(key = key_prefix + "lon", value=124.3)
    evnodes['lat'].set_value(key = key_prefix + "lat", value=44.5)
    evnodes['depth'].set_value(key = key_prefix + "depth", value = 10.0)
    evnodes['time'].set_value(key = key_prefix + "time", value=ev_true.time+5.0)
    evnodes['mb'].set_value(key = key_prefix + "mb", value=3.0)

    np.random.seed(1)
    run_event_MH(sg, evnodes, wave_nodes)

    #print "atime", sg.get_value(key=create_key(param="arrival_time", eid=en.eid, sta="FIA3", phase="P"))
    print ll

    #plot_with_fit("unass.png", wn)

if __name__ == "__main__":
    try:
        #sample_template()
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
