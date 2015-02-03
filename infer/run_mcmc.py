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
from sigvisa.infer.event_swap import swap_events_move_lstsqr, repropose_event_move_lstsqr, swap_threeway_lstsqr
from sigvisa.infer.event_birthdeath import ev_birth_move_hough, ev_death_move_hough, ev_birth_move_lstsqr, ev_death_move_lstsqr, set_hough_options
from sigvisa.infer.event_mcmc import ev_move_full, swap_association_move
from sigvisa.infer.mcmc_logger import MCMCLogger
from sigvisa.infer.template_mcmc import split_move, merge_move, optimizing_birth_move, death_move_for_optimizing_birth, indep_peak_move, improve_offset_move_gaussian, improve_atime_move, hamiltonian_template_move, hamiltonian_move_reparameterized
from sigvisa.plotting.plot import plot_with_fit, plot_with_fit_shapes
from sigvisa.utils.fileutils import clear_directory, mkdir_p, next_unused_int_in_dir

global_stds = {'coda_height': .7,
               'coda_height_small': .1,
               'coda_decay': .5,
               'peak_decay': 0.5,
               'peak_offset': 1.0,
               'improve_offset_move_gaussian': 0.5,
               'arrival_time_big': 9.0,
               'arrival_time': 0.5,
               'evloc': 0.15,
               'evloc_big': 0.4,
               'evtime': 1.0,
               'evmb': 0.2,
               'evdepth': 8.0}


extra_move_args = {'hamiltonian_reversing': {'log_eps_mean': 6.5,
                                             'log_eps_std': 1.5,
                                             'epsL': 0.15,
                                             'reverse_block_size': 5,
                                             'reverse_block_min_std': 0.5,
                                             'reverse_block_max_std': 1000,}}

"""
extra_move_args = {'hamiltonian_reversing': {'log_eps_mean': 5,
                                             'log_eps_std': 0.1,
                                             'L_min': 5,
                                             'L_max': 100,
                                             'reverse_block_size': 1,
                                             'reverse_block_min_std': 0.5,
                                             'reverse_block_max_std': 1000,}}
"""

def do_template_moves(sg, wn, tmnodes, tg, stds,
                      n_attempted=None, n_accepted=None,
                      move_times=None, step=None, proxy_lps=None):


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
                     relevant_nodes=relevant_nodes, std=stds[param], proxy_lps=proxy_lps)
        except KeyError:
            continue


def run_move(move_name, fn, step=None, n_accepted=None, n_attempted=None, move_times=None, move_prob=None, cyclic=False, **kwargs):

    if move_prob is not None:
        if cyclic:
            interval = int(1.0/move_prob)
            if step % interval != 0:
                return
        else:
            u = np.random.rand()
            if u > move_prob:
                return

    if n_attempted is not None:
        n_attempted[move_name] += 1

    if move_name in extra_move_args:
        kwargs.update(extra_move_args[move_name])

    t0  = time.time()
    accepted = fn(**kwargs)
    t1 = time.time()

    if n_accepted is not None:
        n_accepted[move_name] += accepted
    if move_times is not None:
        move_times[move_name].append((step, t1-t0))

###########################################################################


############################################################################

def single_template_MH(sg, wn, tmnodes, phase, steps=1000, rw=True, hamiltonian=False, window_lps=None, sorted_params=None):
    #tmnodes = dict([(p, (n.single_key, n)) for (p, n) in sg.uatemplates[tmid].items()])

    template_moves_special = {'peak_offset': improve_offset_move_gaussian,
                              'arrival_time_big': improve_atime_move,
                              'arrival_time': improve_atime_move}
    if hamiltonian:
        template_moves_special['hamiltonian_reversing'] = hamiltonian_template_move

    stds = global_stds

    n_accepted = defaultdict(int)
    n_attempted = defaultdict(int)

    vals = []
    if sorted_params is None:
        sorted_params = sorted([p for (p, (k, n)) in tmnodes.items() if not n.deterministic()])

    for step in range(steps):
        # special template moves
        for (move_name, fn) in template_moves_special.iteritems():
            run_move(move_name=move_name, fn=fn,
                     sg=sg, wave_node=wn, tmnodes=tmnodes,
                     n_attempted=n_attempted, n_accepted=n_accepted,
                     std=stds[move_name] if move_name in stds else None,
                     window_lps=window_lps)

        if rw:
            # also do basic wiggling-around of all template params
            tg = sg.template_generator(phase)

            proxy_lps = wn.window_lps_to_proxy_lps(window_lps)

            do_template_moves(sg, wn, tmnodes, tg, stds,
                              n_attempted=n_attempted, n_accepted=n_accepted,
                              proxy_lps=proxy_lps)


        v = np.array([tmnodes[p][1].get_value(tmnodes[p][0]) for p in sorted_params])
        vals.append(v)

        #if step % 50 == 0:
        #    print "step %d: lp %f, %s" % (step, sg.current_log_p(), v)

    return sorted_params, np.array(vals)

def run_open_world_MH(sg, steps=10000,
                      enable_event_openworld=True,
                      enable_event_moves=True,
                      enable_template_openworld=True,
                      enable_template_moves=True,
                      template_move_type="hamiltonian", # can be "hamiltonian", "rw" (i.e. random-walk MH), or "both"
                      logger=None,
                      disable_moves=[],
                      start_step=0,
                      cyclic_template_moves=False,
                      use_proxy_lp=False,
                      template_openworld_custom=None):


    if enable_event_openworld:
        global_moves = {'event_swap': swap_events_move_lstsqr,
                        'event_repropose': repropose_event_move_lstsqr,
                        'event_threeway_swap': swap_threeway_lstsqr,
                        'event_birth_hough': ev_birth_move_hough,
                        'event_death_hough': ev_death_move_hough,
                        'event_birth_lstsqr': ev_birth_move_lstsqr,
                        'event_death_lstsqr': ev_death_move_lstsqr}
    else:
        if enable_template_openworld:
            # swap moves leave behind uatemplates, so only allow them
            # if we have uatemplate birth/death moves
            global_moves = {'event_swap': swap_events_move_lstsqr,
                            'event_repropose': repropose_event_move_lstsqr}
        else:
            global_moves = {}

    event_moves_gaussian = {'evloc': ('loc', ('lon', 'lat')),
                            'evloc_big': ('loc', ('lon', 'lat')),
                            'evtime': ('time', ('time',)),
                            'evmb': ('mb', ('mb',)),
                            'evdepth': ('depth', ('depth',))} if enable_event_moves else {}
    event_moves_special = {}

    if template_openworld_custom is not None:
        sta_moves = template_openworld_custom
    else:
        if enable_template_openworld:
            sta_moves = {'tmpl_birth': optimizing_birth_move,
                         'tmpl_death': death_move_for_optimizing_birth,
                         'tmpl_split': split_move,
                         'tmpl_merge': merge_move,
                         'swap_association': swap_association_move}
        else:
            sta_moves = {'swap_association': swap_association_move}

    template_moves_special = {'indep_peak': indep_peak_move,
                              'peak_offset': improve_offset_move_gaussian,
                              'arrival_time': improve_atime_move,} if enable_template_moves else {}
    if template_move_type in ("hamiltonian", "both"):
        template_moves_special['hamiltonian_reversing'] = hamiltonian_template_move


    # allow the caller to disable specific moves by name
    for move in disable_moves:
        for move_type in (global_moves, event_moves_gaussian, event_moves_special, sta_moves, template_moves_special):
            try:
                del move_type[move]
            except KeyError:
                continue

    stds = global_stds
    tmpl_openworld_move_probability = 0.05
    ev_openworld_move_probability = .05

    move_probs = defaultdict(lambda : 0.05)
    move_probs["swap_association"] = 0.2

    n_accepted = defaultdict(int)
    n_attempted = defaultdict(int)
    move_times = defaultdict(list)

    params_over_time = dict()

    if logger is None:
        logger = MCMCLogger()

    if logger != False:
        run_dir = logger.run_dir
        logger.start()
    else:
        run_dir = "/dev/null"

    for step in range(start_step, steps):

        # moves to adjust existing events
        for (eid, evnodes) in sg.evnodes.items():

            for (move_name, (node_name, params)) in event_moves_gaussian.items():
                run_move(move_name=move_name, fn=ev_move_full, step=step,
                         n_attempted=n_attempted,
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
                    move_prob = move_probs[move_name]
                    run_move(move_name=move_name, fn=fn, step=step, n_attempted=n_attempted,
                             n_accepted=n_accepted, move_times=move_times,
                             move_prob=move_probs, cyclic=cyclic_template_moves,
                             sg=sg, wave_node=wn)

                # moves to adjust existing unass. templates
                tg = sg.template_generator('UA')
                for tmid in sg.uatemplate_ids[(sta, wn.chan, wn.band)]:
                    tmnodes = dict([(p, (n.single_key, n)) for (p, n) in sg.uatemplates[tmid].items()])

                    window_lps = None
                    if use_proxy_lp:
                        window_lps = wn.cache_latent_signal_for_template_optimization(-tmid, "UA", force_bounds=False)

                    # special template moves
                    for (move_name, fn) in template_moves_special.iteritems():
                        run_move(move_name=move_name, fn=fn, step=step, n_attempted=n_attempted,
                                 n_accepted=n_accepted, move_times=move_times,
                                 sg=sg, wave_node=wn, tmnodes=tmnodes,
                                 std=stds[move_name] if move_name in stds else None,
                                 window_lps = window_lps)


                    # also do basic wiggling-around of all template params
                    if enable_template_moves and template_move_type in ("rw", "both"):

                        proxy_lps = None
                        if use_proxy_lp:
                            proxy_lps = wn.window_lps_to_proxy_lps(window_lps)

                        do_template_moves(sg, wn, tmnodes, tg, stds,
                                          n_attempted, n_accepted, move_times, step,
                                          proxy_lps=proxy_lps)

                # also adjust every event arrival at this station
                for (eid,evnodes) in sg.evnodes.iteritems():

                    nodes_by_phase = sg.get_arrival_nodes_byphase(eid, sta, wn.band, wn.chan)
                    for (phase, tmnodes) in nodes_by_phase.items():
                        tg = sg.template_generator(phase)

                        for (move_name, fn) in template_moves_special.iteritems():
                            run_move(move_name=move_name, fn=fn, step=step, n_attempted=n_attempted,
                                     n_accepted=n_accepted, move_times=move_times,
                                     sg=sg, wave_node=wn, tmnodes=tmnodes)

                        if enable_template_moves and template_move_type in ("rw", "both"):
                            do_template_moves(sg, wn, tmnodes, tg, stds,
                                              n_attempted, n_accepted, move_times, step)

        for (move, fn) in global_moves.items():

            run_move(move_name=move, fn=fn, step=step, n_attempted=n_attempted,
                     n_accepted=n_accepted, move_times=move_times,
                     move_prob=ev_openworld_move_probability,
                     sg=sg, log_to_run_dir=run_dir)

        if logger != False:
            logger.log(sg, step, n_accepted, n_attempted, move_times)

        """"
        atnodes = [n for n in sg.extended_evnodes[1] if "arrival_time" in n.label]
        for n in atnodes:
            at1 = n.get_value()
            n.parent_predict()
            at2 = n.get_value()
            assert(np.abs(at1-at2) < 1e-8)
        """


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
    parser.add_option("--seed", dest="seed", default=0, type="int",
                      help="random seed")
    parser.add_option("--no_template_openworld", dest="no_template_openworld", default=False, action="store_true",
                      help="disable template birth/death/merge/split moves")
    parser.add_option("--no_event_openworld", dest="no_event_openworld", default=False, action="store_true",
                      help="disable event birth/death moves")
    parser.add_option("--run_dir", dest="run_dir", default=None, type="str",
                      help="directory to save results  (auto)")
    parser.add_option("--preset", dest="preset", default=None, type="str", help="options are 'localize' (default None)")
    parser.add_option("--template_move_type", dest="template_move_type", default="rw", type="str", help="options are 'hamiltonian' (default), 'rw', or 'both'")
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

    if options.preset == "localize_ua":
        options.initialize_leb="yes"
        options.no_event_openworld=True
        options.no_template_openworld=False

    if options.dprk:
        options.initialize_leb="no"
        options.start_time = 1243212800
        options.end_time = 1243214600


    resume_fname = None
    start_step = 0
    if options.run_dir is not None:
        if os.path.exists(options.run_dir):
            resume_fname, start_step = get_last_savepoint(options.run_dir)
            if resume_fname is None:
                raise Exception("specified run dir %s already exists, but I couldn't find any savepoints!" % options.run_dir)
    logger = MCMCLogger(run_dir=options.run_dir, write_template_vals=True, dump_interval=options.skip)
    run_dir=logger.run_dir
    print "MCMC logging to %s" %(run_dir)

    with open(os.path.join(run_dir, 'cmd.txt'), 'w') as f:
        f.write(" ".join(sys.argv))

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

        if options.preset == "location":
            sg.add_event(ev)
            init_evs = [ev,]
            with open(os.path.join(run_dir, "events.pkl"), 'wb') as f:
                pickle.dump(init_evs, f)

    set_hough_options({'bin_width_deg': 1.0, 'time_tick_s': 10, 'smoothbins': True})


    sys.setrecursionlimit(20000)
    np.random.seed(options.seed)


    run_open_world_MH(sg, steps=options.steps,
                      enable_event_openworld= not options.no_event_openworld,
                      enable_event_moves=True,
                      enable_template_openworld= not options.no_template_openworld,
                      enable_template_moves=True,
                      disable_moves=disable_moves,
                      logger=logger,
                      start_step = start_step,
                      template_move_type = options.template_move_type)

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
