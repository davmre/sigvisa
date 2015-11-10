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
from sigvisa.infer.autoregressive_mcmc import arnoise_params_rw_move, arnoise_mean_rw_move, arnoise_std_rw_move
from sigvisa.infer.template_xc import atime_xc_move, constpeak_atime_xc_move, adjpeak_atime_xc_move
from sigvisa.infer.mcmc_basic import get_node_scales, gaussian_propose, gaussian_MH_move, MH_accept, mh_accept_lp
from sigvisa.infer.event_swap import swap_events_move_hough, repropose_event_move_hough, swap_threeway_hough
from sigvisa.infer.event_birthdeath import ev_birth_move_hough, ev_birth_move_hough_oes, ev_birth_move_hough_offset, ev_birth_move_hough_oes_offset, ev_death_move_hough, ev_death_move_hough_oes, ev_death_move_hough_offset, ev_death_move_hough_oes_offset, ev_birth_move_lstsqr, ev_death_move_lstsqr, set_hough_options, ev_birth_move_correlation, ev_death_move_correlation
from sigvisa.infer.event_mcmc import ev_move_full, swap_association_move
from sigvisa.infer.mcmc_logger import MCMCLogger
from sigvisa.infer.template_mcmc import split_move, merge_move, optimizing_birth_move, death_move_for_optimizing_birth, indep_peak_move, improve_offset_move_gaussian, improve_atime_move, hamiltonian_template_move, hamiltonian_move_reparameterized
from sigvisa.plotting.plot import plot_with_fit, plot_with_fit_shapes
from sigvisa.utils.fileutils import clear_directory, mkdir_p, next_unused_int_in_dir

global_stds = {'coda_height': .7,
               'coda_height_small': .1,
               'coda_decay': .3,
               'peak_decay': 0.3,
               'peak_offset': 0.3,
               'mult_wiggle_std': 0.1,
               'improve_offset_move_gaussian': 0.5,
               'arrival_time_big': 4.0,
               'arrival_time': 0.3,
               'evloc': 0.15,
               'evloc_big': 0.4,
               'evtime': 1.0,
               'evmb': 0.2,
               'evdepth': 8.0,
               'signal_var': 0.4,
               'noise_var': 0.4,
               'noise_var_small': 0.05,
               'depth_lscale': 15.0,
               'horiz_lscale': 15.0}


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

def do_gp_hparam_moves(sg, stds, n_attempted=None, n_accepted=None,
                      move_times=None, step=None):

    from numpy.linalg.linalg import LinAlgError

    """
    hacky state of affairs, as I understand it:
    - changing signal_var and noise_var will subtly change the overall logp in a way
      that's not accounted for by these moves, because it will affect the numeric
      properties of the calculation of
    """

    def hparam_move(node, std):
        v1 = node.get_value()
        v2 = v1 + np.random.randn()*std

        if v2 < 0:
            return False

        #print "running hparam move on node %s with children %s" % (node.label, [n.param for n in node.child_jgps])

        def lp(v):
            node.set_value(v)
            ll = node.log_p()
            if np.isfinite(ll):
                for jgp in node.child_jgps:
                    ll += jgp.log_likelihood()
            return ll

        return mh_accept_lp(sg, lp, v1, v2)

    for sta in sg._joint_gpmodels.keys():
        for hnodes in sg._jointgp_hparam_nodes.values():
            for (hparam, n) in hnodes.items():
                try:
                    run_move(move_name=hparam, fn=hparam_move,
                             step=step, n_accepted=n_accepted,
                             n_attempted=n_attempted, move_times=move_times,
                             node=n, std=stds[hparam])
                    if hparam=="noise_var":
                        run_move(move_name=hparam, fn=hparam_move,
                                 step=step, n_accepted=n_accepted,
                                 n_attempted=n_attempted, move_times=move_times,
                                 node=n, std=stds["noise_var_small"])
                except KeyError:
                    continue

def do_template_moves(sg, wn, tmnodes, tg, stds,
                      n_attempted=None, n_accepted=None,
                      move_times=None, step=None, proxy_lps=None):


    for param in tg.params(env=wn.is_env):

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

        if param=="coda_height":
            run_move(move_name="coda_height_small", fn=gaussian_MH_move,
                     step=step, n_accepted=n_accepted,
                     n_attempted=n_attempted, move_times=move_times,
                     sg=sg, keys=(k,), node_list=(n,),
                     relevant_nodes=relevant_nodes, std=stds["coda_height_small"], proxy_lps=proxy_lps)


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

def single_template_MH(sg, wn, tmnodes, steps=1000, rw=True, hamiltonian=False, window_lps=None, sorted_params=None):
    #tmnodes = dict([(p, (n.single_key, n)) for (p, n) in sg.uatemplates[tmid].items()])

    template_moves_special = {'peak_offset': improve_offset_move_gaussian,
                              'arrival_time_big': improve_atime_move,
                              'arrival_time': improve_atime_move}
    if hamiltonian:
        template_moves_special['hamiltonian_reversing'] = hamiltonian_template_move

    stds = global_stds

    n_accepted = defaultdict(int)
    n_attempted = defaultdict(int)

    if len(tmnodes) == 0:
        return None, np.array(tmnodes)

    vals = []
    if sorted_params is None:
        sorted_params = sorted([p for (p, (k, n)) in tmnodes[0][1].items() if not n.deterministic()])

    for step in range(steps):
        for phase, phase_tmnodes in tmnodes:
            # special template moves
            for (move_name, fn) in template_moves_special.iteritems():
                run_move(move_name=move_name, fn=fn,
                         sg=sg, wn=wn, tmnodes=phase_tmnodes,
                         n_attempted=n_attempted, n_accepted=n_accepted,
                         std=stds[move_name] if move_name in stds else None,
                         window_lps=window_lps)

            if rw:
                # also do basic wiggling-around of all template params
                tg = sg.template_generator(phase)

                proxy_lps = wn.window_lps_to_proxy_lps(window_lps)

                do_template_moves(sg, wn, phase_tmnodes, tg, stds,
                                  n_attempted=n_attempted, n_accepted=n_accepted,
                                  proxy_lps=proxy_lps)


        v = np.array([tmn[p][1].get_value(tmn[p][0]) for (phase, tmn) in tmnodes for p in sorted_params])
        vals.append(v)

        #if step % 50 == 0:
        #    print "step %d: lp %f, %s" % (step, sg.current_log_p(), v)

    return sorted_params, np.array(vals)

def run_open_world_MH(sg, steps=10000,
                      enable_event_openworld=True,
                      enable_event_moves=True,
                      enable_template_openworld=True,
                      enable_template_moves=True,
                      enable_hparam_moves=True,
                      template_move_type="rw", # can be "hamiltonian", "rw" (i.e. random-walk MH), or "both"
                      logger=None,
                      disable_moves=[],
                      start_step=0,
                      cyclic_template_moves=False,
                      use_proxy_lp=False,
                      template_openworld_custom=None,
                      stop_condition=None):


    if enable_event_openworld:
        hough_rate = 0.1
        correlation_rate = 0.0
        global_moves = {'event_swap': (swap_events_move_hough, 0.00),
                        'event_repropose': (repropose_event_move_hough, 0.00),
                        'event_threeway_swap': (swap_threeway_hough, 0.00),
                        'event_birth_hough': (ev_birth_move_hough, hough_rate),
                        'event_birth_hough_offset': (ev_birth_move_hough_offset, hough_rate),
                        'event_birth_hough_oes': (ev_birth_move_hough_oes, hough_rate),
                        'event_birth_hough_oes_offset': (ev_birth_move_hough_oes_offset, hough_rate),
                        'event_death_hough': (ev_death_move_hough, hough_rate),
                        'event_death_hough_offset': (ev_death_move_hough_offset, hough_rate),
                        'event_death_hough_oes': (ev_death_move_hough_oes, hough_rate),
                        'event_death_hough_oes_offset': (ev_death_move_hough_oes_offset, hough_rate),
                        'event_birth_correlation': (ev_birth_move_correlation, correlation_rate),
                        'event_death_correlation': (ev_death_move_correlation, correlation_rate),
        }
                        #'event_birth_lstsqr': ev_birth_move_lstsqr,
                        #'event_death_lstsqr': ev_death_move_lstsqr        }
    else:
        if enable_template_openworld:
            # swap moves leave behind uatemplates, so only allow them
            # if we have uatemplate birth/death moves
            global_moves = {'event_swap': (swap_events_move_hough, 0.05),
                            'event_repropose': (repropose_event_move_hough, 0.05)}
            global_moves = {}
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
            sta_moves = {'tmpl_birth': (optimizing_birth_move, 0.5),
                         'tmpl_death': (death_move_for_optimizing_birth, 0.5),
                         'tmpl_split': (split_move, 0.0),
                         'tmpl_merge': (merge_move, 0.0),
                         'swap_association': (swap_association_move, 0.2),
                         'arnoise_mean': (arnoise_mean_rw_move, 0.2),
                         'arnoise_std': (arnoise_std_rw_move, 0.2),
                         'arnoise_params': (arnoise_params_rw_move, 0.2)
            }

        else:
            sta_moves = {'swap_association': (swap_association_move, 0.2),
                         'arnoise_mean': (arnoise_mean_rw_move, 0.2),
                         'arnoise_std': (arnoise_std_rw_move, 0.2),
                         'arnoise_params': (arnoise_params_rw_move, 0.2)
            }


    template_moves_special = {'indep_peak': (indep_peak_move, 1.0),
                              'peak_offset': (improve_offset_move_gaussian, 1.0),
                              'arrival_time': (improve_atime_move, 1.0),
                              'atime_xc': (atime_xc_move, 0.1),
                              'arrival_time_big': (improve_atime_move, 1.0),
                              #'constpeak_atime_xc': constpeak_atime_xc_move,
                              #'adjpeak_atime_xc': adjpeak_atime_xc_move,
                             } if enable_template_moves else {}
    if template_move_type in ("hamiltonian", "both"):
        template_moves_special['hamiltonian_reversing'] = (hamiltonian_template_move, 1.0)

    # allow the caller to disable specific moves by name
    for move in disable_moves:
        for move_type in (global_moves, event_moves_gaussian, event_moves_special, sta_moves, template_moves_special):
            try:
                del move_type[move]
            except KeyError:
                continue

    stds = global_stds

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

    try:
        np.random.seed(sg.seed)
    except AttributeError:
        pass

    # we don't actually need the LP, but this will
    # cause all relevant messages to be passed.
    # HACK ALERT, should have a separate method
    # to do this.
    if start_step == 0:
        init_lp = sg.current_log_p()

    for step in range(start_step, start_step+steps):

        # moves to adjust existing events
        for (eid, evnodes) in sg.evnodes.items():

            if sg.event_is_fixed(eid):
                #print "skipping fixed eid %d" % (eid,)
                continue

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
                for wn in sg.station_waves[sta]:

                    # moves to birth/death/split/merge new unassociated templates
                    for (move_name, (fn, move_prob)) in sta_moves.items():
                        run_move(move_name=move_name, fn=fn, step=step, n_attempted=n_attempted,
                                 n_accepted=n_accepted, move_times=move_times,
                                 move_prob=move_prob, cyclic=cyclic_template_moves,
                                 sg=sg, wn=wn)

                    for (eid, phase) in wn.arrivals():

                        if eid in sg.fully_fixed_events:
                            continue

                        tg = sg.template_generator(phase)
                        if eid < 0:
                            tmid = -eid
                            tmnodes = dict([(p, (n.single_key, n)) for (p, n) in sg.uatemplates[tmid].items()])
                        else:
                            tmnodes = sg.get_template_nodes(eid, sta, phase, wn.band, wn.chan)

                        window_lps = None
                        if use_proxy_lp:
                            window_lps = wn.cache_latent_env_for_template_optimization(eid, phase, force_bounds=False)

                        # special template moves
                        for (move_name, (fn, move_prob)) in template_moves_special.iteritems():
                            run_move(move_name=move_name, fn=fn, step=step, n_attempted=n_attempted,
                                     n_accepted=n_accepted, move_times=move_times,
                                     sg=sg, wn=wn, tmnodes=tmnodes,
                                     eid=eid, phase=phase, move_prob=move_prob,
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

        for (move, (fn, prob)) in global_moves.items():

            run_move(move_name=move, fn=fn, step=step, n_attempted=n_attempted,
                     n_accepted=n_accepted, move_times=move_times,
                     move_prob=prob,
                     sg=sg, log_to_run_dir=run_dir)

        if sg.jointgp and enable_hparam_moves:
            do_gp_hparam_moves(sg, stds, step=step, n_attempted=n_attempted,
                     n_accepted=n_accepted, move_times=move_times)


        seed = np.random.randint(2**31)
        sg.seed = seed
        np.random.seed(sg.seed)

        if logger != False:
            logger.log(sg, step, n_accepted, n_attempted, move_times)

        if stop_condition is not None:
            if stop_condition(logger):
                logger.dump(sg)
                return

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
