import numpy as np
import sys
import os
import traceback
import pickle
import copy

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_event_station_chan
from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.models.signal_model import extract_arrival_from_key
from sigvisa.infer.mcmc_basic import get_node_scales, gaussian_propose, gaussian_MH_move, MH_accept
from sigvisa.graph.graph_utils import create_key
from sigvisa.graph.dag import get_relevant_nodes
from sigvisa.plotting.plot import savefig, plot_with_fit, plot_waveform
from matplotlib.figure import Figure

import scipy.weave as weave
from scipy.weave import converters

MERGE_MAX_ATIME_DIFFERENCE_S = 50

#######################################################################

"""

Methods for sampling an arrival time independent of the current
arrival time, with probability proportional to the height of the
envelope at each point.

"""

def preprocess_signal_for_sampling(wave_env):

    d = wave_env**2
    """
    # sample locations where the envelope is increasing, relative to how fast it's increasing
    grad = np.gradient(wave_env)
    incr = (grad > 0)
    d = grad**2
    d[~incr] = max(np.min(d), 1e-3)
    """
    s = np.sum(d)
    normalized_env = d/s
    cdf = np.concatenate([np.array((0,)), np.cumsum(normalized_env)])
    return cdf

def peak_log_p(cdf, stime, srate, peak_time):
    # compute the probability that sample_peak_time_from_signal would
    # have sampled the current atime. formally this should be 0 for
    # all peak times that don't line up with an integer index, but we
    # just force an integer and hope that's okay.

    # we add one here since we created cdf with an initial
    # 0, so that the subtraction below works properly
    idx = np.floor((peak_time - stime) * srate) + 1

    """"in principle, we shouldn't be allowed to kill things outside of
    the signal window, but in practice it helps a lot.
    """
    #if (idx < 1) or (idx >= len(cdf)): return np.float('-inf')
    if (idx < 1) or (idx >= len(cdf)): return np.log(1.0/len(cdf))

    #return np.log(1.0/len(cdf))
    return np.log(cdf[idx] - cdf[idx-1])

def get_signal_diff_positive_part(wave_node, arrival_set):
    value = wave_node.get_value().data
    pred_signal = wave_node.assem_signal(arrivals=arrival_set, include_wiggles=False)

    npts = wave_node.npts
    signal_diff_pos = wave_node.signal_diff
    code = """
for(int i=0; i < npts; ++i) {
double v = fabs(value(i)) - fabs(pred_signal(i));
signal_diff_pos(i) = v > 0 ? v : 0;
}
"""
    weave.inline(code,['npts', 'signal_diff_pos', 'value', 'pred_signal'],type_converters = converters.blitz,verbose=2,compiler='gcc')
    return signal_diff_pos

def get_current_conditional_cdf(wave_node, arrival_set):
    signal_diff_pos = get_signal_diff_positive_part(wave_node, arrival_set)
    return preprocess_signal_for_sampling(signal_diff_pos)


def sample_peak_time_from_signal(cdf, stime, srate, return_lp=False):
    u = np.random.rand()
    idx = np.searchsorted(cdf, u)
    peak_time = stime + float(idx-1)/srate
    #peak_time = u * (len(cdf)-1)/float(srate) + stime

    if return_lp:
        return peak_time, np.log(cdf[idx]-cdf[idx-1])
        #return peak_time, np.log(1.0/len(cdf))
    return peak_time

def indep_peak_move(sg, arrival_node, offset_node, wave_node):
    arr = extract_arrival_from_key(arrival_node.label, wave_node.r)
    other_arrs = wave_node.arrivals() - set(arr)

    current_atime = arrival_node.get_value()
    peak_offset = offset_node.get_value()

    cdf = get_current_conditional_cdf(wave_node, arrival_set=other_arrs)
    proposed_peak_time, proposal_lp =  sample_peak_time_from_signal(cdf, wave_node.st,
                                                                  wave_node.srate,
                                                                  return_lp=True)
    backward_propose_lp = peak_log_p(cdf, wave_node.st,
                                     wave_node.srate,
                                     peak_time = current_atime + peak_offset)

    proposed_arrival_time = proposed_peak_time - peak_offset
    return MH_accept(sg, oldvalues = (current_atime,),
                     newvalues = (proposed_arrival_time,),
                     log_qforward = proposal_lp,
                     log_qbackward = backward_propose_lp,
                     node_list = (arrival_node,),
                     relevant_nodes = (arrival_node, wave_node))

######################################################################

def indep_offset_move(sg, arrival_node, offset_node, wave_node):
    current_offset = offset_node.get_value()
    atime = arrival_node.get_value()
    proposed_offset = np.random.rand() * 40
    new_atime = atime + (current_offset - proposed_offset)
    accepted = MH_accept(sg=sg, oldvalues=(atime, current_offset),
                         newvalues = (new_atime, proposed_offset),
                         node_list = (arrival_node, offset_node),
                         relevant_nodes=(arrival_node, offset_node, wave_node))
    return accepted

def improve_offset_move(sg, arrival_node, offset_node, wave_node, **kwargs):
    """
    Update the peak_offset while leaving the peak time constant, i.e.,
    adjust the arrival time to compensate for the change in offset.
    """
    current_offset = offset_node.get_value()
    atime = arrival_node.get_value()
    proposed_offset = gaussian_propose(sg, node_list=(offset_node,), values=(current_offset,), **kwargs)[0]
    new_atime = atime + (current_offset - proposed_offset)
    accepted = MH_accept(sg=sg, oldvalues=(atime, current_offset),
                         newvalues = (new_atime, proposed_offset),
                         node_list = (arrival_node, offset_node),
                         relevant_nodes=(arrival_node, offset_node, wave_node))
    return accepted
#######################################################################

def get_sorted_arrivals(wave_node):
    # choose a template at random to try to split
    arrivals = wave_node.arrivals()
    arr_params = []
    for (eid, phase) in arrivals:
        arr_params.append((wave_node.get_template_params_for_arrival(eid, phase)[0], eid, phase))
    sorted_arrs = sorted(arr_params, key = lambda x : x[0]['arrival_time'])
    return sorted_arrs

def try_split(sg, wave_node):
    sorted_arrs = get_sorted_arrivals(wave_node)
    n = len(sorted_arrs)
    k = np.random.randint(0, n)
    arr_to_split = sorted_arrs[k]
    atime = arr_to_split[0]['arrival_time']

    # the arrival times for the split templates are sampled such that
    # the two new templates are still adjacent (no other template
    # arrives between them). This makes the move reversible, since the
    # merge move combines adjacent templates.
    atime_diff = MERGE_MAX_ATIME_DIFFERENCE_S/2.0
    if k < n-1:
        atime_diff = min(atime_diff, sorted_arrs[k+1][0]['arrival_time'] - atime)
    if k > 0:
        atime_diff = min(atime_diff, atime - sorted_arrs[k-1][0]['arrival_time'])

    # TODO: figure out what to do with wiggles
    tnodes = sg.get_template_nodes(eid=arr_to_split[1], phase=arr_to_split[2], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)
    tmpl = dict([(p, n) for (p, (k, n)) in tnodes.items()])
    return split_move(sg, tmpl, wave_node, atime_diff)


def split_move(sg, tmpl, wave_node, atime_diff):

    lp_old = sg.current_log_p()

    arrival_time, peak_offset, coda_height, coda_decay = tmpl['arrival_time'].get_value(), tmpl['peak_offset'].get_value(), tmpl['coda_height'].get_value(), tmpl['coda_decay'].get_value()

    new_tmpl = sg.create_unassociated_template(wave_node, atime=arrival_time, nosort=True)
    sg._topo_sorted_list = new_tmpl.values() + sg._topo_sorted_list
    sg._gc_topo_sorted_nodes()

    # WLOG give the earlier arrival time to the current template, and
    # the later one to the new template we create. For all other params,
    # decide at random which template gets which.
    eps_atime = np.random.rand()*atime_diff
    tmpl['arrival_time'].set_value(arrival_time - eps_atime)
    new_tmpl['arrival_time'].set_value(arrival_time + eps_atime)

    new_offset = np.random.rand() * peak_offset  * 2
    new_tmpl['peak_offset'].set_value(new_offset)
    tmpl['peak_offset'].set_value(2*peak_offset - new_offset)

    new_decay = np.random.rand() * coda_decay  * 2
    new_tmpl['coda_decay'].set_value(new_decay)
    tmpl['coda_decay'].set_value(2*coda_decay - new_decay)

    new_logheight = np.log(np.random.rand()) + coda_height
    new_tmpl['coda_height'].set_value(new_logheight)
    tmpl['coda_height'].set_value( np.log(np.exp(coda_height) - np.exp(new_logheight)) )

    lp_new = sg.current_log_p()

    log_qforward = -np.log(2*np.abs(coda_decay)) - np.log(2*peak_offset) - np.log(atime_diff) - np.log(np.exp(coda_height))
    jacobian_determinant = 2.07944 ## log(8)
    log_qbackward = 0 ## merge move is deterministic

    u = np.random.rand()
    if (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant > np.log(u):
        print "split template %d from %d: %.1f + %.5f - (%.1f + %.5f) + %f = %.1f vs %.1f" % (new_tmpl["arrival_time"].unassociated_templateid, tmpl["arrival_time"].unassociated_templateid, lp_new, log_qbackward, lp_old, log_qforward, jacobian_determinant, (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant, np.log(u))
        return new_tmpl
    else:
        sg.destroy_unassociated_template(new_tmpl, nosort=True)

        tmpl['arrival_time'].set_value(arrival_time)
        tmpl['peak_offset'].set_value(peak_offset)
        tmpl['coda_decay'].set_value(coda_decay)
        tmpl['coda_height'].set_value(coda_height)

        # WARNING: this assumes the list hasn't been re-sorted by any
        # of our intermediate calls.
        sg._topo_sorted_list = sg._topo_sorted_list[len(new_tmpl):]
        sg._gc_topo_sorted_nodes()
        sg.next_uatemplateid -= 1
        #lp = sg.current_log_p()
        #assert(np.abs(lp - lp_old) < 1e-10)
        return False

def try_merge(sg, wave_node):
    sorted_arrs = get_sorted_arrivals(wave_node)
    n = len(sorted_arrs)
    if n < 2:
        return False
    k = np.random.randint(0, n-1)
    arr1 = sorted_arrs[k]
    arr2 = sorted_arrs[k+1]
    if arr2[0]['arrival_time'] - arr1[0]['arrival_time'] > MERGE_MAX_ATIME_DIFFERENCE_S:
        return False

    tnodes1 = sg.get_template_nodes(eid=arr1[1], phase=arr1[2], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)
    tmpl1 = dict([(param, node) for (param, (key, node)) in tnodes1.items()])

    tnodes2 = sg.get_template_nodes(eid=arr2[1], phase=arr2[2], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)
    tmpl2 = dict([(param, node) for (param, (key, node)) in tnodes2.items()])

    merged_atime = (tmpl1['arrival_time'].get_value() + tmpl2['arrival_time'].get_value())/2.0
    post_merge_atime_diff = MERGE_MAX_ATIME_DIFFERENCE_S
    if k+2 < n:
        post_merge_atime_diff = min(post_merge_atime_diff, sorted_arrs[k+2][0]['arrival_time'] - merged_atime)
    if k > 0:
        post_merge_atime_diff = min(post_merge_atime_diff, merged_atime - sorted_arrs[k-1][0]['arrival_time'])

    return merge_move(sg, tmpl1, tmpl2, wave_node, post_merge_atime_diff)

def merge_move(sg, tmpl1_nodes, tmpl2_nodes, wave_node, post_merge_atime_diff):

    lp_old = sg.current_log_p()
    orig_topo_sorted = copy.copy(sg._topo_sorted_list)

    tmpl1_values = dict((k, n.get_value()) for (k,n) in tmpl1_nodes.items())
    tmpl2_values = dict((k, n.get_value()) for (k,n) in tmpl2_nodes.items())

    merged_atime = (tmpl1_values['arrival_time'] + tmpl2_values['arrival_time'])/2.0
    merged_offset = (tmpl1_values['peak_offset'] + tmpl2_values['peak_offset'])/2.0
    merged_decay = (tmpl1_values['coda_decay'] + tmpl2_values['coda_decay'])/2.0
    merged_amp = np.log(np.exp(tmpl1_values['coda_height']) + np.exp(tmpl2_values['coda_height']))

    for (param, node) in tmpl2_nodes.items():
        sg.remove_node(node)
        sg._topo_sorted_list[node._topo_sorted_list_index] = None

    tmpl1_nodes['arrival_time'].set_value(merged_atime)
    tmpl1_nodes['peak_offset'].set_value(merged_offset)
    tmpl1_nodes['coda_decay'].set_value(merged_decay)
    tmpl1_nodes['coda_height'].set_value(merged_amp)

    lp_new = sg.current_log_p()

    log_qforward = 0
    jacobian_determinant = -2.07944 ## log(1/8)
    log_qbackward = -np.log(2*np.abs(merged_decay)) - np.log(2*merged_offset) - np.log(post_merge_atime_diff/2.0) - np.log(np.exp(merged_amp))

    u = np.random.rand()
    if (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant > np.log(u):
        print "merged template %d into %d: %.1f + %.5f - (%.1f + %.5f) + %f = %.1f vs %.1f" % (tmpl1_nodes["arrival_time"].unassociated_templateid, tmpl2_nodes["arrival_time"].unassociated_templateid, lp_new, log_qbackward, lp_old, log_qforward, jacobian_determinant, (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant, np.log(u))

        uaid = tmpl2_nodes['arrival_time'].unassociated_templateid
        del sg.uatemplates[uaid]
        sg.uatemplate_ids[(wave_node.sta,wave_node.chan,wave_node.band)].remove(uaid)

        return True
    else:
        for (param, node) in tmpl2_nodes.items():
            sg.add_node(node)
            node.addChild(wave_node)
        for (param, node) in tmpl1_nodes.items():
            node.set_value(tmpl1_values[param])
        wave_node.arrivals()
        sg._topo_sorted_list = orig_topo_sorted
        sg._gc_topo_sorted_nodes()
        #lp = sg.current_log_p()
        #assert(np.abs(lp - lp_old) < 1e-10)
        return False




#######################################################################



def birth_move(sg, wave_node, dummy=False, **kwargs):
    lp_old = sg.current_log_p()
    ntlp_old = sg.ntemplates_log_p()

    cdf = get_current_conditional_cdf(wave_node, arrival_set=wave_node.arrivals())
    peak_time, proposal_lp =  sample_peak_time_from_signal(cdf, wave_node.st,
                                                           wave_node.srate,
                                                           return_lp=True)

    tmpl = sg.create_unassociated_template(wave_node, peak_time, nosort=True, **kwargs)
    sg._topo_sorted_list = tmpl.values() + sg._topo_sorted_list
    sg._gc_topo_sorted_nodes()
    tmpl["arrival_time"].set_value(peak_time - tmpl["peak_offset"].get_value())


    lp_new = sg.current_log_p()
    ntlp_new = sg.ntemplates_log_p()

    # probability of this birth move is the product of probabilities
    # of all sampled params (including arrival time)
    log_qforward = proposal_lp
    for (key, node) in tmpl.items():
        if key == "arrival_time": continue
        log_qforward += node.log_p()

    # reverse (death) probability is just the probability of killing a
    # random template
    ntemplates = len([1 for (eid, phase) in wave_node.arrivals() if eid < 0])
    log_qbackward = 0 #np.log(1.0/ntemplates)

    u = np.random.rand()
    move_accepted = (lp_new + log_qbackward) - (lp_old + log_qforward) > np.log(u)
    if move_accepted or dummy:
        print "birth template %d: %.1f + %.1f - (%.1f + %.1f) = %.1f vs %.1f" % (tmpl["arrival_time"].unassociated_templateid, lp_new, log_qbackward, lp_old, log_qforward, (lp_new + log_qbackward) - (lp_old + log_qforward), np.log(u))
    if move_accepted and not dummy:
        return tmpl
    else:
        sg.destroy_unassociated_template(tmpl, nosort=True)

        # WARNING: this assumes the list hasn't been re-sorted by any
        # of our intermediate calls.
        sg._topo_sorted_list = sg._topo_sorted_list[len(tmpl):]
        sg._gc_topo_sorted_nodes()
        sg.next_uatemplateid -= 1
        #lp = sg.current_log_p()
        #assert(np.abs(lp - lp_old) < 1e-10)
        return False


def death_move(sg, wave_node, dummy=False):
    templates = [(eid, phase) for (eid, phase) in wave_node.arrivals() if eid < 0]
    if len(templates) < 1:
        return False

    u0 = np.random.rand()
    for i in range(len(templates)):
        if u0 <= float(i+1)/len(templates):
            tmpl_to_destroy = templates[i]
            break

    lp_old = sg.current_log_p()
    orig_topo_sorted = copy.copy(sg._topo_sorted_list)
    log_qforward = 0 #np.log(1.0/len(templates))

    tnodes = sg.get_template_nodes(eid=tmpl_to_destroy[0], phase=tmpl_to_destroy[1], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)
    wnodes = sg.get_wiggle_nodes(eid=tmpl_to_destroy[0], phase=tmpl_to_destroy[1], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)

    current_peak = tnodes['arrival_time'][1].get_value() + tnodes['peak_offset'][1].get_value()
    log_qbackward = 0
    for (param, (label, node)) in tnodes.items():
        if param != "arrival_time":
            log_qbackward += node.log_p()
        sg.remove_node(node)
        sg._topo_sorted_list[node._topo_sorted_list_index] = None
    for (param, (label, node)) in wnodes.items():
        log_qbackward += node.log_p()
        sg.remove_node(node)
        sg._topo_sorted_list[node._topo_sorted_list_index] = None

    arrs = wave_node.arrivals()
    cdf = get_current_conditional_cdf(wave_node, arrival_set=arrs)
    log_qbackward += peak_log_p(cdf, wave_node.st,
                                wave_node.srate,
                                peak_time = current_peak)


    lp_new = sg.current_log_p()

    u = np.random.rand()
    move_accepted = (lp_new + log_qbackward) - (lp_old + log_qforward) > np.log(u)
    if move_accepted or dummy:
        print "death of template %d: %.1f + %.1f - (%.1f + %.1f) = %.1f vs %.1f" % (tnodes["arrival_time"][1].unassociated_templateid, lp_new, log_qbackward, lp_old, log_qforward, (lp_new + log_qbackward) - (lp_old + log_qforward), np.log(u))
    if move_accepted and not dummy:

        uaid = -tmpl_to_destroy[0]
        del sg.uatemplates[uaid]
        sg.uatemplate_ids[(wave_node.sta,wave_node.chan,wave_node.band)].remove(uaid)


        return True
    else:
        for (param, (label, node)) in tnodes.items() + wnodes.items():
            sg.add_node(node)
            node.addChild(wave_node)
        wave_node.arrivals()
        sg._topo_sorted_list = orig_topo_sorted
        sg._gc_topo_sorted_nodes()
        #lp = sg.current_log_p()
        #assert(np.abs(lp - lp_old) < 1e-10)
        return False


#####################################################################

def run_open_world_MH(sg, wns, burnin=0, skip=40, steps=10000, wiggles=False):
    n_accepted = dict()
    moves = ('birth', 'death', 'split', 'merge', 'indep_peak', 'peak_offset', 'arrival_time', 'coda_height', 'coda_decay', 'wiggle_amp', 'wiggle_phase')

    for move in moves:
        n_accepted[move] = 0
    n_attempted = 0

    stds = {'peak_offset': .1, 'arrival_time': .1, 'coda_height': .02, 'coda_decay': 0.05, 'wiggle_amp': .25, 'wiggle_phase': .5}

    params_over_time = dict()

    for step in range(steps):
        for wn in wns:
            new_nodes = birth_move(sg, wn, wiggles=wiggles)
            if new_nodes:
                tmplid = new_nodes['arrival_time'].unassociated_templateid
                for param in new_nodes.keys():
                    params_over_time["%d_%s" % (tmplid, param)] = [np.float('nan')] * step
                n_accepted['birth'] += 1

            uaids = sg.uatemplate_ids[(wn.sta,wn.chan,wn.band)]
            if len(uaids) >= 1:
                n_accepted['death'] += death_move(sg, wn)


            arrivals = wn.arrivals()
            if len(arrivals) >= 1:
                split_nodes = try_split(sg, wn)
                if split_nodes:
                    tmplid = split_nodes['arrival_time'].unassociated_templateid
                    for param in split_nodes.keys():
                        params_over_time["%d_%s" % (tmplid, param)] = [np.float('nan')] * step
                    n_accepted['split'] += 1

            arrivals = wn.arrivals()
            if len(arrivals) >= 1:
                n_accepted['merge'] += try_merge(sg, wn)


            arrivals = wn.arrivals()
            for (eid, phase) in arrivals:
                l3 = len(arrivals)
                wg = sg.wiggle_generator(phase=phase, srate=wn.srate)

                if eid < 0:
                    uaid = -eid
                    tmnodes = sg.uatemplates[uaid]
                else:
                    tmnodes = sg.get_template_nodes(eid=eid, sta=wn.sta, phase=phase, band=wn.band, chan=wn.chan)

                n_accepted['indep_peak'] += indep_peak_move(sg, arrival_node=tmnodes["arrival_time"],
                                                             offset_node=tmnodes["peak_offset"],
                                                             wave_node=wn)
                n_accepted['peak_offset'] += improve_offset_move(sg, arrival_node=tmnodes["arrival_time"],
                                                               offset_node=tmnodes["peak_offset"],
                                                                 wave_node=wn, std=stds['peak_offset'])
                for param in ("arrival_time", "coda_height", "coda_decay"):
                    n = tmnodes[param]
                    n_accepted[param] += gaussian_MH_move(sg, node_list=(n,), relevant_nodes=(n, wn), std=stds[param])
                n_attempted += 1

                if wiggles:
                    for param in wg.params():
                        n = tmnodes[param]
                        if param.startswith("amp"):
                            phase_wraparound = False
                            move = 'wiggle_amp'
                        else:
                            phase_wraparound = True
                            move = 'wiggle_phase'
                        n_accepted[move] += float(gaussian_MH_move(sg, node_list=(n,), relevant_nodes=(n, wn), std=stds[move], phase_wraparound=phase_wraparound)) / (wg.dimension()/2.0)
                for (param, n) in tmnodes.items():
                    params_over_time["%d_%s" % (tmplid, param)].append(n.get_value())

            if step > 0 and ((step % skip == 0) or (step < 15)):
                lp = sg.current_log_p()

                print "step %d, %s: lp %.2f, %d templates, accepted " % (step, wn.sta, lp, len(arrivals)),
                for move in moves:
                    if move in ("split", "merge", "birth", "death"):
                        print "%s: %d, " % (move, n_accepted[move]),
                    else:
                        accepted_percent = (float(n_accepted[move]) / n_attempted *100.0 if n_accepted[move] > 0 else 0)
                        print "%s: %d%%, " % (move, accepted_percent),
                print
                plot_with_fit("unass_%s_step%06d.png" % (wn.sta, step), wn)
                #signal_diff_pos = get_signal_diff_positive_part(wn, wn.arrivals())
                #w = wn.get_wave()
                #w.data = signal_diff_pos
                #savefig(fname="unass_diff%06d.png" % step, fig=plot_waveform(w))


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


def test_moves(sg, wn):
    stds = {'peak_offset': .1, 'arrival_time': .1, 'coda_height': .02, 'coda_decay': 0.05, 'wiggle_amp': .25, 'wiggle_phase': .5}
    #tmnodes = sg.create_unassociated_template(wave_node=wn, atime=1239915963.891)
    #tmnodes["peak_offset"].set_value(4.804)
    #tmnodes["coda_height"].set_value(-1.786)
    #tmnodes["coda_decay"].set_value(-0.030)

    print sg.current_log_p()

    births = 0
    deaths = 0
    merges = 0
    splits = 0
    for i in range(1000):
        new_nodes = birth_move(sg, wn)
        if new_nodes:
            births += 1
        deaths += death_move(sg, wn)
        split_nodes = try_split(sg, wn)
        if split_nodes:
            splits += 1
        merges += try_merge(sg, wn)

        if (i % 10) == 0:
            print "%d templates:" % len(wn.arrivals()), births, "births, ", deaths, "deaths, ", merges, "merges", splits, 'splits'

    print sg.current_log_p()

    plot_with_fit("tmpl.png", wn)

    death_move(sg, wn, dummy=True)
    death_move(sg, wn, dummy=True)
    death_move(sg, wn, dummy=True)
    death_move(sg, wn, dummy=True)
    death_move(sg, wn, dummy=True)
    death_move(sg, wn, dummy=True)
    death_move(sg, wn, dummy=True)
    death_move(sg, wn, dummy=True)
    death_move(sg, wn, dummy=True)
    death_move(sg, wn, dummy=True)


def main():

    """
    s = Sigvisa()
    cursor = s.dbconn.cursor()
    wave = load_event_station_chan(evid=5326226, sta="FIA3", chan="SHZ", cursor=cursor).filter("%s;env" % "freq_2.0_3.0").filter('hz_5.0')
    cursor.close()
    """
    sg = SigvisaGraph(template_model_type="dummy", template_shape="paired_exp",
                      wiggle_model_type="dummy", wiggle_family="dummy",
                      phases="leb", nm_type = "ar", wiggle_len_s = 60.0,
                      assume_envelopes=True)

    wave = Waveform(data = np.load("sampled_wave.npy"),
                    srate=5.0, stime=1239915900.0,
                    sta="FIA3", chan="SHZ", arrivals=np.array(()),
                    filter_str="freq_2.0_3.0;env;hz_5.0")
    wn = sg.add_wave(wave)
    env = wn.get_value().data if wn.env else wn.get_wave().filter("env").data.data
    wn.cdf = preprocess_signal_for_sampling(env)
    #sg.create_unassociated_template(wave_node=wn, atime=1239915940.253)
    #sg.create_unassociated_template(wave_node=wn, atime=1239915969.623)

    for fname in os.listdir('.'):
        if fname.startswith("unass_step") or fname.startswith("mcmc_unass"):
            os.remove(fname)

    np.random.seed(0)
    #test_moves(sg, wn)
    run_open_world_MH(sg, wn, wiggles=True)
    #print "atime", sg.get_value(key=create_key(param="arrival_time", eid=en.eid, sta="FIA3", phase="P"))


    ll = wn.log_p()
    print ll

    plot_with_fit("unass.png", wn)

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
