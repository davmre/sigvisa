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
from sigvisa.infer.mcmc_logger import MCMCLogger
from sigvisa.infer.template_mcmc import split_move, merge_move, birth_move, death_move, swap_association_move
from sigvisa.infer.arrival_time_moves import indep_peak_move, improve_offset_move, improve_atime_move
from sigvisa.infer.autoregressive_mcmc import wiggle_param_step, latent_arrival_block_gibbs
from sigvisa.plotting.plot import plot_with_fit, plot_with_fit_shapes
from sigvisa.utils.fileutils import clear_directory, mkdir_p, next_unused_int_in_dir

global_stds = {'coda_height': .7,
            'coda_decay': .5,
            'wiggle_amp': .1,
            'wiggle_phase': .1,
            'peak_offset': 1.0,
            'arrival_time': 9.0,
            'evloc': 0.15,
            'evloc_big': 0.4,
            'evtime': 1.0,
            'evmb': 0.2,
            'evdepth': 8.0}

def do_template_moves(sg, wn, tmnodes, tg, wg, stds, n_attempted, n_accepted, move_times, step):

    latent_key, latent = tmnodes['latent_arrival']

    for param in tg.params():
        k, n = tmnodes[param]

        # here we re-implement get_relevant_nodes from sigvisa.graph.dag, with a few shortcuts
        relevant_nodes = [latent,]
        if n.deterministic():
            parent = n.parents[n.default_parent_key()]
            relevant_nodes.append(parent)
            # technically we should also add the children of parent, but for template moves this will never matter
        else:
            relevant_nodes.append(n)

        if param == 'arrival_time':
            relevant_nodes.append(wn)

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
                 sg=sg, keys=(k,), node_list=(n,), relevant_nodes=(n, latent),
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



############################################################################

def run_open_world_MH(sg, steps=10000,
                      enable_event_openworld=True,
                      enable_event_moves=True,
                      enable_template_openworld=True,
                      enable_template_moves=True,
                      logger=None,
                      disable_moves=[],
                      enable_moves = [],
                      extra_moves=[],
                      start_step=0):
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
                              'arrival_time': improve_atime_move,
                              'latent_arrival': latent_arrival_block_gibbs} if enable_template_moves else {}

    optional_moves = {'ar_wiggle': (template_moves_special, wiggle_param_step)}

    # allow the caller to disable specific moves by name
    for move in disable_moves:
        for move_type in (global_moves, event_moves_gaussian, event_moves_special, sta_moves, template_moves_special):
            try:
                del move_type[move]
            except KeyError:
                continue

    # also, some optional moves are not enabled by default
    for move in enable_moves:
        move_set, fn = optional_moves[move]
        move_set[move] = fn


    stds = global_stds
    tmpl_openworld_move_probability = .10
    ev_openworld_move_probability = .05

    n_accepted = defaultdict(int)
    n_attempted = defaultdict(int)
    move_times = defaultdict(list)

    params_over_time = dict()

    if logger is None:
        logger = MCMCLogger()
    run_dir = logger.run_dir


    for step in range(start_step, steps):

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

                # also adjust every event arrival at this station
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


        logger.log(sg, step, n_accepted, n_attempted, move_times)





def get_last_savepoint(run_dir):
    last_step = 0
    last_fname = None
    for fname in os.listdir(run_dir):
        if fname.startswith('step'):
            fname_step = int(fname.split('_')[1])
            if fname_step > last_step:
                last_step = fname_step
                last_fname = os.path.join(run_dir, fname, 'pickle.sg')
    return last_fname, last_step

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
    parser.add_option("--no_template_openworld", dest="no_template_openworld", default=False, action="store_true",
                      help="disable template birth/death/merge/split moves")
    parser.add_option("--no_event_openworld", dest="no_event_openworld", default=False, action="store_true",
                      help="disable event birth/death moves")
    parser.add_option("--run_dir", dest="run_dir", default=None, type="str",
                      help="directory to save results  (auto)")
    parser.add_option("--preset", dest="preset", default=None, type="str", help="options are 'localize' (default None)")
    parser.add_option("--disable_moves", dest="disable_moves", default='', type="str", help="comma-separated list of specific MCMC move names to disable")

    register_svgraph_cmdline(parser)
    register_svgraph_signal_cmdline(parser)
    (options, args) = parser.parse_args()

    disable_moves = options.disable_moves.split(',')

    if options.preset == "localize":
        options.initialize_leb="yes"
        options.no_event_openworld=True
        options.no_template_openworld=True
        disable_moves.append('indep_peak')

    if options.dprk:
        options.initialize_leb="no"
        options.start_time = 1243212800
        options.end_time = 1243214600


    resume_fname = None
    start_step = 0
    if options.run_dir is None:
        run_dir = setup_mcmc_logging()['dir']
    else:
        run_dir = options.run_dir
        if os.path.exists(run_dir):
            resume_fname, start_step = get_last_savepoint(run_dir)
            if resume_fname is None:
                raise Exception("specified run dir %s already exists, but I couldn't find any savepoints!" % run_dir)
        else:
            mkdir_p(run_dir)

    if resume_fname is None:
        sg = setup_svgraph_from_cmdline(options, args)
        init_evs = load_signals_from_cmdline(sg, options, args)
        with open(os.path.join(run_dir, "events.pkl"), 'wb') as f:
            pickle.dump(init_evs, f)
    else:
        print "resuming from", resume_fname
        with open(resume_fname, 'rb') as f:
            sg = pickle.load(f)
        sg.current_log_p_breakdown()

    if options.dprk:
        from sigvisa.source.event import get_event
        ev = get_event(evid=5393637)
        ev.natural_source=True
        sg.add_event(ev)
        init_evs = [ev,]
        with open(os.path.join(run_dir, "events.pkl"), 'wb') as f:
            pickle.dump(init_evs, f)

    set_hough_options({'bin_width_deg': 1.0, 'time_tick_s': 10, 'smoothbins': True})


    sys.setrecursionlimit(20000)
    np.random.seed(0)

    logger = MCMCLogger(run_dir=run_dir, write_template_vals=True, dump_interval=options.skip)
    run_open_world_MH(sg, steps=options.steps,
                      enable_event_openworld= not options.no_event_openworld,
                      enable_event_moves=True,
                      enable_template_openworld= not options.no_template_openworld,
                      enable_template_moves=True,
                      disable_moves=disable_moves,
                      logger=logger,
                      start_step = start_step)

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
