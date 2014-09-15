import numpy as np
import sys
import os
import traceback
import pickle
import copy

from collections import defaultdict
from optparse import OptionParser
from sigvisa.database.signal_data import *
from sigvisa.database.dataset import *
import itertools

from sigvisa.models.ttime import tt_predict
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_segments
from sigvisa.infer.event_birthdeath import sample_template_to_associate, template_association_logodds, associate_template, unassociate_template, sample_deassociation_proposal, template_association_distribution, phase_template_proposal_logp, deassociation_prob, propose_phase_template
from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.infer.mcmc_basic import get_node_scales, gaussian_propose, gaussian_MH_move, MH_accept
from sigvisa.infer.template_mcmc import preprocess_signal_for_sampling, improve_offset_move, indep_peak_move
from sigvisa.graph.graph_utils import create_key
from sigvisa.plotting.plot import savefig, plot_with_fit
from matplotlib.figure import Figure


from sigvisa.infer.propose import propose_event_from_hough

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

def set_ev(ev_node, v, fixed_vals, fixed_nodes, params):
    for (key, val) in zip(params, v):
        ev_node.set_local_value(key=key, value=val, force_deterministic_consistency=False)


    for (val, n) in zip(fixed_vals, fixed_nodes):
        try:
            n.set_value(val)
        except ValueError:
            # ignore "illegal travel time" messages from phases that are about to disappear
            pass

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

def add_phase_template(sg, sta, eid, phase):
    tg = sg.template_generator(phase)
    wg = sg.wiggle_generator(phase, sg.base_srate)
    if phase not in sg.ev_arriving_phases(eid, sta=sta):
        s = Sigvisa()
        site = s.get_array_site(sta)
        sg.add_event_site_phase(tg, wg, site, phase, sg.evnodes[eid], sample_templates=True)

    tmvals, lp = propose_phase_template(sg, sta, eid, phase)
    return tmvals, lp

def ev_phasejump_move(sg, eid, ev_node, current_v, new_v, params, fixed_vals, fixed_nodes, birth_phases, death_phases):

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


    """
    it's complicated to keep track of which nodes are in the markov
    blanket, so instead we'll be lazy and just use the whole graph, since
    these moves should be pretty rare. might have to fix this if it's a
    bottleneck, or if we end up doing parallel inference where we don't have the whole
    graph.
    """
    lp_old = sg.current_log_p()
    set_ev(ev_node, new_v, fixed_vals, fixed_nodes, params)

    new_site_phases = dict()
    forward_fns = []
    inverse_fns = [lambda : set_ev(ev_node, current_v, fixed_vals, fixed_nodes, params),]
    associations = []
    deassociations = []
    move_logprob = 0
    reverse_logprob = 0
    tmid_i = 0
    tmids = []
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
            #print "proposing phases at", sta, site
            for phase in birth_phases[site]:
                tmid, assoc_logprob = sample_template_to_associate(sg, sta, eid, phase)

                if tmid is not None:
                    # associate an unass. template
                    forward_fns.append(lambda sta=sta,phase=phase,tmid=tmid: associate_template(sg, sta, tmid, eid, phase, create_phase_arrival=True))
                    inverse_fns.append(lambda sta=sta,phase=phase: unassociate_template(sg, sta, eid, phase, remove_event_phase=True))
                    associations.append((sta, phase, True))
                    print "proposing to associate %d to %d %s at %s" % (tmid, eid, phase, sta),
                else:
                    # propose a new template from scratch
                    forward_fns.append( lambda sta=sta,phase=phase,eid=eid: add_phase_template(sg, sta, eid, phase)[1] )
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
                    forward_fns.append(lambda sta=sta,phase=phase: tmids.append(unassociate_template(sg, sta, eid, phase, remove_event_phase=True)))
                    inverse_fns.append(lambda sta=sta,phase=phase,tmid_i=tmid_i: associate_template(sg, sta, tmids[tmid_i], eid, phase, create_phase_arrival=True))
                    tmid_i += 1
                    print "proposing to deassociate %s for %d at %s (lp %.1f)" % (phase, eid, sta, deassociate_logprob),
                else:
                    template_param_array = sg.get_arrival_vals(eid, sta, phase, band, chan)
                    inverse_fns.append(lambda sta=sta,phase=phase,band=band,chan=chan,template_param_array=template_param_array : sg.set_template(eid,sta, phase, band, chan, template_param_array))
                    tmp = phase_template_proposal_logp(sg, sta, eid, phase, template_param_array)
                    reverse_logprob += tmp
                    print "proposing to delete %s for %d at %s (lp %f)"% (phase, eid, sta, deassociate_logprob),
                move_logprob += deassociate_logprob

    for fn in forward_fns:
        x = fn()
        if x is not None:
            move_logprob += x
    sg._topo_sort()
    clear_node_caches(sg, eid)
    fixed_nodes = get_fixed_nodes(ev_node)
    lp_new = sg.current_log_p()

    # revert the event to the old location, temporarily, so that we
    # can compute probabilities for the reverse move
    set_ev(ev_node, current_v, fixed_vals, fixed_nodes, params)
    for (sta, phase, associated) in associations:
        reverse_logprob += np.log(deassociation_prob(sg, sta, eid, phase, deletion_prob=not associated))
    for (sta, phase, deassociate, tmid_i) in deassociations:
        c = template_association_distribution(sg, sta, eid, phase)
        if deassociate:
            tmid = tmids[tmid_i]
            tmp = np.log(c[tmid])
            reverse_logprob += tmp
        else:
            tmp = np.log(c[None])
            reverse_logprob += tmp

    u = np.random.rand()
    move_accepted = (lp_new + reverse_logprob) - (lp_old + move_logprob)  > np.log(u)

    if move_accepted:
        set_ev(ev_node, new_v, fixed_vals, fixed_nodes, params)
        print "move accepted"
        return True
    else:
        print "move rejected",
        for fn in inverse_fns:
            fn()
        sg._topo_sort()
        clear_node_caches(sg, eid)
        print "changes reverted"
        return False


def ev_move_full(sg, ev_node, std, params):
    # jointly propose a new event location along with new tt_residual values,
    # such that the event arrival times remain constant.

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
        if new_v[0] < -180:
            new_v[0] += 360
        if new_v[0] > 180:
            new_v[0] -= 360
        if new_v[1] < -90:
            new_v[1] = -180 - new_v[1]
        if new_v[1] > 90:
            new_v[1] = 180 - new_v[1]


    eid = int(ev_node.label.split(';')[0])
    old_site_phases = dict()
    for site, stas in sg.site_elements.items():
        # TODO: can we get arriving phases for a site instead of sta?
        old_site_phases[site] = set(sg.ev_arriving_phases(eid, site=site))

    lp_old = sg.joint_logprob(node_list=node_list, relevant_nodes=relevant_nodes, values=None)

    set_ev(ev_node, new_v, fixed_vals, fixed_nodes, params)

    # if the predicted phases in the new location are different from
    # in the old location, we have to do a reversible jump move to
    # birth the new phases / destroy the no-longer-feasible phases.
    ev = sg.get_event(eid)
    birth_phases = dict()
    death_phases = dict()
    phases_changed = False
    for site in old_site_phases.keys():
        new_site_phases = sg.predict_phases_site(ev=ev, site=site)
        birth_phases[site] = new_site_phases - old_site_phases[site]
        death_phases[site] = old_site_phases[site] - new_site_phases
        if len(birth_phases[site]) > 0 or len(death_phases[site]) > 0:
            phases_changed = True
    if phases_changed:
        set_ev(ev_node, current_v, fixed_vals, fixed_nodes, params)
        return ev_phasejump_move(sg, eid, ev_node, current_v, new_v, params, fixed_vals, fixed_nodes, birth_phases, death_phases)

    lp_new = sg.joint_logprob(node_list=node_list, relevant_nodes=relevant_nodes, values=None)

    u = np.random.rand()
    move_accepted = lp_new - lp_old  > np.log(u)
    if move_accepted:
        return True
    else:
        set_ev(ev_node, current_v, fixed_vals, fixed_nodes, params)
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
