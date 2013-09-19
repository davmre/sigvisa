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

def run_open_world_MH(sg, wns, burnin=0, skip=40, steps=10000):
    global_moves = {'event_birth': ev_birth_move,
                    'event_death': ev_death_move}
    event_moves_gaussian = {'loc': (('lon', 'lat'), 0.01),
                            'loc_big': (('lon', 'lat'), 0.5),
                            'time': (('time',), 2.0),
                            'mb': (('mb',), 0.5),
                            'depth': (('depth',), 5.0)}
    event_moves_special = {}
    sta_moves = {'tmpl_birth': try_birth,
                 'tmpl_death': try_death,
                 'tmpl_split': try_split,
                 'tmpl_merge': try_merge}
    template_moves_special = {'indep_peak': indep_peak_move,
                              'peak_offset': improve_offset_move}
    template_moves_gaussian = {'arrival_time': .1,
                               'coda_height': .02,
                               'coda_decay': .05,
                               'wiggle_amp': .25,
                               'wiggle_phase': .5}

    n_accepted = defaultdict(0)
    n_attempted = defaultdict(0)

    params_over_time = dict()

    for step in range(steps):

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

                for (move_name, fn) in sta_moves:
                    n_attempted[move_name] += 1
                    n_accepted[move_name] += fn(sg, wn)

                tg = sg.template_generator('UA')
                wg = sg.wiggle_generator('UA', wn.srate)
                for tmid in sg.uatemplate_ids[sta]:
                    tmnodes = sg.uatemplates[tmid]

                    for (move_name, fn) in template_moves_special:
                        n_attempted[move_name] += 1
                        n_accepted[move_name] = fn(sg, wave_nodes, tmnodes)

                    for param in tg.params():
                        n_attempted[param] += 1
                        n = tmnodes[param]
                        n_accepted[param] = gaussian_MH_move(sg, node_list=(n,), relevant_nodes=(n, wn), std=template_moves_gaussian[param])

                    for param in wg.params():
                       n = tmnodes[param]
                       if param.startswith("amp"):
                           phase_wraparound = False
                           move = 'wiggle_amp'
                       else:
                           phase_wraparound = True
                           move = 'wiggle_phase'
                        n_attempted[move] += 1
                        n_accepted[move] += gaussian_MH_move(sg, node_list=(n,), relevant_nodes=(n, wn), std=stds[move], phase_wraparound=phase_wraparound)



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
