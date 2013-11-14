import numpy as np
import sys
import os
import traceback
import pickle
import copy
import time

from collections import defaultdict
from optparse import OptionParser

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.graph.load_sigvisa_graph import register_svgraph_cmdline, register_svgraph_signal_cmdline, setup_svgraph_from_cmdline, load_signals_from_cmdline
from sigvisa import Sigvisa
from sigvisa.infer.mcmc_basic import get_node_scales, gaussian_propose, gaussian_MH_move, MH_accept
from sigvisa.infer.event_birthdeath import ev_birth_move, ev_death_move, set_hough_options
from sigvisa.infer.event_mcmc import ev_move
from sigvisa.infer.template_mcmc import split_move, merge_move, birth_move, death_move, indep_peak_move, improve_offset_move, improve_atime_move, swap_association_move
from sigvisa.plotting.plot import plot_with_fit, plot_with_fit_shapes
from sigvisa.utils.fileutils import clear_directory, mkdir_p, next_unused_int_in_dir

global_stds = {'coda_height': .5,
            'coda_decay': .1,
            'wiggle_amp': .1,
            'wiggle_phase': .1,
            'peak_offset': 0.5,
            'arrival_time': 7.0,
            'evloc': 0.20,
            'evloc_big': 0.8,
            'evtime': 2.0,
            'evmb': 0.8,
            'evdepth': 8.0}

def do_template_moves(sg, wn, tmnodes, tg, wg, stds, n_attempted, n_accepted, move_times, step):

    for param in tg.params():
        k, n = tmnodes[param]

        # here we re-implement get_relevant_nodes from sigvisa.graph.dag, with a few shortcuts
        relevant_nodes = [wn,]
        if n.deterministic():
            parent = n.parents[n.default_parent_key()]
            relevant_nodes.append(parent)
            # technically we should also add the children of parent, but for template moves this will never matter
        else:
            relevant_nodes.append(n)

        try:
            run_move(move_name=param, fn=gaussian_MH_move,
                     step=step, n_accepted=n_accepted,
                     n_attempted=n_attempted, move_times=move_times,
                     sg=sg, keys=(k,), node_list=(n,),
                     relevant_nodes=relevant_nodes, std=stds[param])
        except KeyError:
            continue

    for param in wg.params():
        k, n = tmnodes[param]

        if param.startswith("amp"):
            phase_wraparound = False
            move = 'wiggle_amp'
        else:
            phase_wraparound = True
            move = 'wiggle_phase'

        run_move(move_name=move, fn=gaussian_MH_move, step=step,
                 n_accepted=n_accepted, n_attempted=n_attempted,
                 move_times=move_times,
                 sg=sg, keys=(k,), node_list=(n,), relevant_nodes=(n, wn),
                 std=stds[move], phase_wraparound=phase_wraparound)

def run_move(move_name, fn, step, n_accepted, n_attempted, move_times, move_prob=None, **kwargs):

    if move_prob is not None:
        u = np.random.rand()
        if u > move_prob:
            return

    n_attempted[move_name] += 1
    t0  = time.time()
    n_accepted[move_name] += fn(**kwargs)
    t1 = time.time()
    move_times[move_name].append((step, t1-t0))

#################################################################
def print_mcmc_acceptances(sg, step, n_accepted, n_attempted):
    lp = sg.current_log_p()

    print "step %d: lp %.2f, accepted " % (step, lp),
    for key in sorted(n_accepted.keys()):
        print "%s: %.3f%%, " % (key, float(n_accepted[key])/n_attempted[key]),
    print ", uatemplates: ", len(sg.uatemplates),
    print ", events: ", len(sg.evnodes)

def setup_mcmc_logging(run_dir=None):
    if run_dir is None:
        base_path = os.path.join("logs", "mcmc")
        mkdir_p(base_path)
        run_dir = os.path.join(base_path, "%05d" % next_unused_int_in_dir(base_path))
    mkdir_p(run_dir)

    log_handles = dict()
    log_handles['dir'] = run_dir
    return log_handles

def cleanup_mcmc(log_handles):
    for v in log_handles.values():
        if type(v) == file:
            v.close()

def log_mcmc(sg, step, n_accepted, n_attempted, move_times, log_handles, dumpsteps=False, dump_interval=500):
    run_dir = log_handles['dir']

    if 'lp' not in log_handles:
        log_handles['lp'] = open(os.path.join(run_dir, 'lp.txt'), 'a')
    lp = sg.current_log_p()
    log_handles['lp'].write('%f\n' % lp)

    if (step % dump_interval == dump_interval-1):
        sg.debug_dump(dump_path = os.path.join(run_dir, 'step_%06d' % step), pickle_only=True)
        for f in log_handles.values():
            if type(f) == file:
                f.flush()


    for (eid, evnodes) in sg.evnodes.items():
        if eid not in log_handles:
            log_handles[eid] = open(os.path.join(run_dir, 'ev_%05d.txt' % eid), 'a')
        evlon = evnodes['loc'].get_local_value('lon')
        evlat = evnodes['loc'].get_local_value('lat')
        evdepth = evnodes['loc'].get_local_value('depth')
        evtime = evnodes['time'].get_local_value('time')
        evmb = evnodes['mb'].get_local_value('mb')
        evsource = evnodes['natural_source'].get_local_value('natural_source')
        log_handles[eid].write('%06d\t%3.4f\t%3.4f\t%4.4f\t%10.2f\t%2.3f\t%d\n' % (step, evlon, evlat, evdepth, evtime, evmb, evsource))
        for (sta,wns) in sg.station_waves.items():
            for wn in wns:
                for phase in sg.phases:
                    lbl = "%d_%s_%s" % (eid, wn.label, phase)
                    if lbl not in log_handles:
                        mkdir_p(os.path.join(run_dir, 'ev_%05d' % eid))
                        log_handles[lbl] = open(os.path.join(run_dir, 'ev_%05d' % eid, "tmpl_%s" % lbl), 'a')
                    tmvals = sg.get_template_vals(eid, sta, phase, wn.band, wn.chan)
                    log_handles[lbl].write('%06d %f %f %f %f\n' % (step,
                                                                   tmvals['arrival_time'],
                                                                   tmvals['peak_offset'],
                                                                   tmvals['coda_height'],
                                                                   tmvals['coda_decay']))

    for move_name in move_times.keys():
        if move_name not in log_handles:
            log_handles[move_name] = open(os.path.join(run_dir, 'move_%s_times.txt' % move_name), 'a')
        for (step, t) in move_times[move_name]:
            log_handles[move_name].write('%d %f\n' % (step, t));
        del move_times[move_name]

    if dumpsteps:
        # dump images for each station at each step
        print_mcmc_acceptances(sg, step, n_accepted, n_attempted)
        for (sta, waves) in sg.station_waves.items():
            for wn in waves:
                plot_with_fit_shapes(os.path.join(run_dir, "%s_step%06d.png" % (wn.label, step)), wn)


############################################################################

def run_open_world_MH(sg, skip=40, steps=10000,
                      enable_event_openworld=True,
                      enable_event_moves=True,
                      enable_template_openworld=True,
                      enable_template_moves=True,
                      run_dir=None,
                      dumpsteps=False):
    global_moves = {'event_birth': ev_birth_move,
                    'event_death': ev_death_move} if enable_event_openworld else {}
    event_moves_gaussian = {'evloc': ('loc', ('lon', 'lat')),
                            'evloc_big': ('loc', ('lon', 'lat')),
                            'evtime': ('time', ('time',)),
                            'evmb': ('mb', ('mb',)),
                            'evdepth': ('depth', ('depth',))} if enable_event_moves else {}
    event_moves_special = {}
    sta_moves = {'tmpl_birth': birth_move,
                 'tmpl_death': death_move,
                 'tmpl_split': split_move,
                 'tmpl_merge': merge_move,
                 'swap_association': swap_association_move} if enable_template_openworld else {}

    template_moves_special = {'indep_peak': indep_peak_move,
                              'peak_offset': improve_offset_move,
                              'arrival_time': improve_atime_move} if enable_template_moves else {}

    stds = global_stds
    tmpl_openworld_move_probability = .10
    ev_openworld_move_probability = .05

    n_accepted = defaultdict(int)
    n_attempted = defaultdict(int)
    move_times = defaultdict(list)

    params_over_time = dict()

    log_handles = setup_mcmc_logging(run_dir=run_dir)
    run_dir = log_handles['dir']

    for step in range(steps):

        # moves to adjust existing events
        for (eid, evnodes) in sg.evnodes.items():

            for (move_name, (node_name, params)) in event_moves_gaussian.items():
                run_move(move_name=move_name, fn=ev_move, step=step, n_attempted=n_attempted,
                         n_accepted=n_accepted, move_times=move_times,
                         sg=sg, ev_node=evnodes[node_name], std=stds[move_name], params=params)

            for (move_name, fn) in event_moves_special.items():
                n_attempted[move_name] += 1
                n_accepted[move_name] += fn(sg, eid)
                run_move(move_name=move_name, fn=fn, step=step, n_attempted=n_attempted,
                         n_accepted=n_accepted, move_times=move_times,
                         sg=sg, eid=eid)

        for (site, elements) in sg.site_elements.items():
            for sta in elements:
                assert(len(sg.station_waves[sta]) == 1)
                wn = list(sg.station_waves[sta])[0]

                # moves to birth/death/split/merge new unassociated templates
                for (move_name, fn) in sta_moves.items():
                    run_move(move_name=move_name, fn=fn, step=step, n_attempted=n_attempted,
                             n_accepted=n_accepted, move_times=move_times,
                             move_prob=tmpl_openworld_move_probability,
                             sg=sg, wave_node=wn)

                # moves to adjust existing unass. templates
                tg = sg.template_generator('UA')
                wg = sg.wiggle_generator('UA', wn.srate)
                for tmid in sg.uatemplate_ids[(sta, wn.chan, wn.band)]:
                    tmnodes = dict([(p, (n.single_key, n)) for (p, n) in sg.uatemplates[tmid].items()])

                    # special moves
                    for (move_name, fn) in template_moves_special.iteritems():
                        try:
                            run_move(move_name=move_name, fn=fn, step=step, n_attempted=n_attempted,
                                     n_accepted=n_accepted, move_times=move_times,
                                     sg=sg, wave_node=wn, tmnodes=tmnodes, std=stds[move_name])
                        except KeyError as e:
                            # some moves don't have a std param
                            run_move(move_name=move_name, fn=fn, step=step, n_attempted=n_attempted,
                                     n_accepted=n_accepted, move_times=move_times,
                                     sg=sg, wave_node=wn, tmnodes=tmnodes)


                    # also do basic wiggling-around of all template params
                    if enable_template_moves:
                        do_template_moves(sg, wn, tmnodes, tg, wg, stds,
                                          n_attempted, n_accepted, move_times, step)

                # also wiggle every event arrival at this station
                for (eid,evnodes) in sg.evnodes.iteritems():
                    for phase in sg.phases:
                        tg = sg.template_generator(phase)
                        wg = sg.wiggle_generator(phase, wn.srate)

                        tmnodes = sg.get_arrival_nodes(eid, sta, phase, wn.band, wn.chan)

                        for (move_name, fn) in template_moves_special.iteritems():
                            run_move(move_name=move_name, fn=fn, step=step, n_attempted=n_attempted,
                                     n_accepted=n_accepted, move_times=move_times,
                                     sg=sg, wave_node=wn, tmnodes=tmnodes)

                        if enable_template_moves:
                            do_template_moves(sg, wn, tmnodes, tg, wg, stds,
                                              n_attempted, n_accepted, move_times, step)

        for (move, fn) in global_moves.items():

            run_move(move_name=move, fn=fn, step=step, n_attempted=n_attempted,
                     n_accepted=n_accepted, move_times=move_times,
                     move_prob=ev_openworld_move_probability,
                     sg=sg, log_to_run_dir=run_dir)


        log_mcmc(sg, step, n_accepted, n_attempted, move_times, log_handles, dumpsteps, dump_interval=skip)
        if step > 0 and ((step % 10 == 0) or (step < 15)):
            print_mcmc_acceptances(sg, step, n_accepted, n_attempted)


    cleanup_mcmc(log_handles)

def main():

    # to locate DPRK event:
    # python infer/run_mcmc.py -s ILAR,WRA,FINES,YKA,JNU,ASAR,NVAR,AKASG,STKA -r dprk_limited  --template_model_types=gp_lld --phases=P --start_time=1243212800 --end_time=1243215800

    parser = OptionParser()

    parser.add_option("--dprk", dest="dprk", default=False, action="store_true",
                      help="initialize with 2009 dprk event (False)")
    parser.add_option("--steps", dest="steps", default=1000, type="int",
                      help="MCMC steps to take (1000)")
    parser.add_option("--skip", dest="skip", default=500, type="int",
                      help="how often to print/save MCMC state, in steps (500)")
    parser.add_option("--startfrom", dest="startfrom", default=None, type="str",
                      help="file name of pickled graph from previous run. starts a new run, initialized with the state from that graph. (None)")
    parser.add_option("--run_dir", dest="run_dir", default=None, type="str",
                      help="directory to save results  (auto)")


    register_svgraph_cmdline(parser)
    register_svgraph_signal_cmdline(parser)
    (options, args) = parser.parse_args()

    if options.startfrom is None:
        sg = setup_svgraph_from_cmdline(options, args)
        load_signals_from_cmdline(sg, options, args)
    else:
        with open(options.startfrom, 'rb') as f:
            sg = pickle.load(f)
            #sg.next_eid = 200
        sg.current_log_p_breakdown()

    if options.dprk:
        from sigvisa.source.event import get_event
        ev = get_event(evid=5393637)
        ev.natural_source=True
        sg.add_event(ev)

    set_hough_options({'bin_width_deg': 1.0, 'time_tick_s': 10, 'smoothbins': True})

    np.random.seed(0)
    run_open_world_MH(sg, skip=options.skip, steps=options.steps)

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
