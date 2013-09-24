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
from sigvisa.graph.sigvisa_graph import SigvisaGraph, predict_phases
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_segments
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

def ev_move(sg, ev_node, std, params):
    # jointly propose a new event location along with new tt_residual values,
    # such that the event arrival times remain constant.

    d = len(params)

    def set_ev(ev_node, v, fixed_vals, fixed_nodes):
        for (key, val) in zip(params, v):
            ev_node.set_local_value(key=key, value=val)
        for (val, n) in zip(fixed_vals, fixed_nodes):
            n.set_value(val)

    current_v = np.zeros((d,))
    for i in range(d):
        current_v[i] = ev_node.get_local_value(params[i])

    if ev_node not in fixed_node_cache:
        sorted_children = sorted(ev_node.children, key = lambda n: n.label)
        fixed_nodes = [child for child in sorted_children if child.label.endswith("arrival_time") or child.label.endswith("coda_height")]
        fixed_node_cache[ev_node] = fixed_nodes
    else:
        fixed_nodes = fixed_node_cache[ev_node]
    fixed_vals = [n.get_value() for n in fixed_nodes]

    if ev_node not in relevant_node_cache:
        node_list = [ev_node,]
        relevant_nodes = ev_move_relevant_nodes(node_list, fixed_nodes)
        relevant_node_cache[ev_node] = (node_list, relevant_nodes)
    else:
        (node_list, relevant_nodes) = relevant_node_cache[ev_node]

    gsample = np.random.normal(0, std, d)
    move = gsample * std
    new_v = current_v + move

    if params[0] == "depth":
        if new_v[0] < 0:
            new_v[0] = 0.0
        if new_v[0] > 700:
            new_v[0] = 700.0

    lp_old = sg.joint_logprob(node_list=node_list, relevant_nodes=relevant_nodes, values=None)
    set_ev(ev_node, new_v, fixed_vals, fixed_nodes)
    lp_new = sg.joint_logprob(node_list=node_list, relevant_nodes=relevant_nodes, values=None)

    u = np.random.rand()
    if lp_new - lp_old > np.log(u):
        return True
    else:
        set_ev(ev_node, current_v, fixed_vals, fixed_nodes)
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
