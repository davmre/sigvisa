import numpy as np
import sys
import os
import traceback
import pickle
import copy

from collections import defaultdict
from optparse import OptionParser

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.graph.load_sigvisa_graph import register_svgraph_cmdline, register_svgraph_signal_cmdline, setup_svgraph_from_cmdline, load_signals_from_cmdline
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_event_station_chan
from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.models.signal_model import extract_arrival_from_key
from sigvisa.infer.mcmc_basic import get_node_scales, gaussian_propose, gaussian_MH_move, MH_accept
from sigvisa.infer.event_birthdeath import ev_birth_move, ev_death_move
from sigvisa.infer.event_mcmc import ev_move
from sigvisa.infer.template_mcmc import try_split, try_merge, birth_move, death_move, indep_peak_move, improve_offset_move
from sigvisa.graph.graph_utils import create_key
from sigvisa.graph.dag import get_relevant_nodes
from sigvisa.plotting.plot import savefig, plot_with_fit, plot_waveform
from matplotlib.figure import Figure

def do_template_moves(sg, wn, tmnodes, tg, wg, template_moves_gaussian, n_attempted, n_accepted):
    for param in tg.params() + ('arrival_time',):
        try:
            k, n = tmnodes[param]
        except TypeError:
            n = tmnodes[param]
            k = n.single_key

        # here we re-implement get_relevant_nodes from sigvisa.graph.dag, with a few shortcuts
        relevant_nodes = [wn,]
        if n.deterministic():
            parent = n.parents[n.default_parent_key()]
            relevant_nodes.append(parent)
            # technically we should also add the children of parent, but for template moves this will never matter
        else:
            relevant_nodes.append(n)

        try:
            n_accepted[param] += gaussian_MH_move(sg, keys=(k,), node_list=(n,), relevant_nodes=relevant_nodes, std=template_moves_gaussian[param])
            n_attempted[param] += 1
        except KeyError:
            continue

    for param in wg.params():
        try:
            k, n = tmnodes[param]
        except ValueError:
            n = tmnodes[param]
            k = n.single_key

        if param.startswith("amp"):
            phase_wraparound = False
            move = 'wiggle_amp'
        else:
            phase_wraparound = True
            move = 'wiggle_phase'

        n_attempted[move] += 1
        n_accepted[move] += gaussian_MH_move(sg, keys=(k,), node_list=(n,), relevant_nodes=(n, wn), std=template_moves_gaussian[move], phase_wraparound=phase_wraparound)

def log_mcmc(sg, step, n_accepted, n_attempted):
    lp = sg.current_log_p()

    print "step %d: lp %.2f, accepted " % (step, lp),
    for key in sorted(n_accepted.keys()):
        print "%s: %.3f%%, " % (key, float(n_accepted[key])/n_attempted[key]),
    print
        #plot_with_fit("unass_%s_step%06d.png" % (wn.sta, step), wn)
        #signal_diff_pos = get_signal_diff_positive_part(wn, wn.arrivals())
        #w = wn.get_wave()
        #w.data = signal_diff_pos
        #savefig(fname="unass_diff%06d.png" % step, fig=plot_waveform(w))


def run_open_world_MH(sg, burnin=0, skip=40, steps=10000):
    global_moves = {'event_birth': ev_birth_move,
                    'event_death': ev_death_move}
    event_moves_gaussian = {'loc': ('loc', ('lon', 'lat'), 0.01),
                            'loc_big': ('loc', ('lon', 'lat'), 0.5),
                            'time': ('time', ('time',), 2.0),
                            'mb': ('mb', ('mb',), 0.5),
                            'depth': ('depth', ('depth',), 5.0)}
    event_moves_special = {}
    sta_moves = {'tmpl_birth': birth_move,
                 'tmpl_death': death_move,
                 'tmpl_split': try_split,
                 'tmpl_merge': try_merge}
    template_moves_special = {'indep_peak': indep_peak_move,
                              'peak_offset': improve_offset_move}
    template_moves_gaussian = {'arrival_time': .1,
                               'coda_height': .02,
                               'coda_decay': .05,
                               'wiggle_amp': .25,
                               'wiggle_phase': .5}

    n_accepted = defaultdict(int)
    n_attempted = defaultdict(int)

    params_over_time = dict()

    for step in range(steps):

        # moves to adjust existing events
        for (eid, evnodes) in sg.evnodes.items():

            for (move_name, (node_name, params, std)) in event_moves_gaussian.items():
                n_attempted[move_name] += 1
                n_accepted[move_name] += ev_move(sg, evnodes[node_name], std=std, params=params)

            for (move_name, fn) in event_moves_special.items():
                n_attempted[move_name] += 1
                n_accepted[move_name] += fn(sg, eid)

        for (site, elements) in sg.site_elements.items():
            for sta in elements:
                assert(len(sg.station_waves[sta]) == 1)
                wn = list(sg.station_waves[sta])[0]

                # moves to birth/death/split/merge new unassociated templates
                for (move_name, fn) in sta_moves.items():
                    n_attempted[move_name] += 1
                    n_accepted[move_name] += fn(sg, wn)

                # moves to adjust existing unass. templates
                tg = sg.template_generator('UA')
                wg = sg.wiggle_generator('UA', wn.srate)
                for tmid in sg.uatemplate_ids[(sta, wn.chan, wn.band)]:
                    tmnodes = sg.uatemplates[tmid]

                    # special moves only for unass templates
                    for (move_name, fn) in template_moves_special.iteritems():
                        n_attempted[move_name] += 1
                        n_accepted[move_name] = fn(sg, wn, tmnodes)

                    # also do basic wiggling-around of all template params
                    do_template_moves(sg, wn, tmnodes, tg, wg, template_moves_gaussian, n_attempted, n_accepted)

                # also wiggle every event arrival at this station
                for (eid,evnodes) in sg.evnodes.iteritems():
                    for phase in sg.phases:
                        tg = sg.template_generator(phase)
                        wg = sg.wiggle_generator(phase, wn.srate)

                        tmnodes = sg.get_template_nodes(eid, sta, phase, wn.band, wn.chan)
                        do_template_moves(sg, wn, tmnodes, tg, wg, template_moves_gaussian, n_attempted, n_accepted)

        for (move, fn) in global_moves.items():
            u = np.random.rand()
            if u < .8: continue

            n_attempted[move] += 1
            n_accepted[move] += fn(sg)

        if step > 0 and ((step % skip == 0) or (step < 15)):
            log_mcmc(sg, step, n_accepted, n_attempted)


def main():

    # to locate DPRK event:
    # python infer/run_mcmc.py -s ILAR,WRA,FINES,YKA,JNU,ASAR,NVAR,AKASG,STKA -r dprk_limited  --template_model_types=gp_lld --phases=P --start_time=1243212800 --end_time=1243215800

    parser = OptionParser()

    parser.add_option("--steps", dest="steps", default=100, type="int",
                      help="MCMC steps to take (1000)")
    parser.add_option("--burnin", dest="burnin", default=0, type="int",
                      help="burnin steps (0)")
    parser.add_option("--skip", dest="skip", default=10, type="int",
                      help="how often to print/save MCMC state, in steps (10)")


    register_svgraph_cmdline(parser)
    register_svgraph_signal_cmdline(parser)
    (options, args) = parser.parse_args()

    sg = setup_svgraph_from_cmdline(options, args)
    load_signals_from_cmdline(sg, options, args)

    np.random.seed(0)
    run_open_world_MH(sg, burnin=options.burnin, skip=options.skip, steps=options.steps)
    sg.debug_dump("mcmc_run")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
