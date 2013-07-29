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
from sigvisa.source.event import Event
from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.infer.mcmc_basic import get_node_scales, gaussian_propose, gaussian_MH_move, MH_accept
from sigvisa.graph.graph_utils import create_key
from sigvisa.graph.dag import get_relevant_nodes
from sigvisa.plotting.plot import savefig, plot_with_fit
from matplotlib.figure import Figure



def run_event_MH(sg, en, wn_list, burnin=0, skip=40, steps=10000, wiggles=False):

    n_accepted = dict()
    moves = ('birth', 'death', 'indep_peak', 'peak_offset', 'arrival_time', 'coda_height', 'coda_decay', 'wiggle_amp', 'wiggle_phase')
    for move in moves:
        n_accepted[move] = 0

    stds = {'peak_offset': .1, 'arrival_time': .1, 'coda_height': .02, 'coda_decay': 0.05, 'wiggle_amp': .25, 'wiggle_phase': .5}


    for wn in wn_list:
        wave_env = wn.get_value() if wn.env else wn.get_wave().filter('env').data
        wn.cdf = preprocess_signal_for_sampling(wave_env)

    templates = dict()
    params_over_time = dict()

    for step in range(steps):

        for wn in wn_list:

            arrivals = wn.arrivals()
            for (eid, phase) in arrivals:
                wg = sg.wiggle_generator(phase=phase, srate=wn.srate)
                tmplid = -eid
                tmnodes = templates[tmplid]

                n_accepted['indep_peak'] += indep_peak_move(sg, arrival_node=tmnodes["arrival_time"],
                                                             offset_node=tmnodes["peak_offset"],
                                                             wave_node=wn)
                n_accepted['peak_offset'] += improve_offset_move(sg, arrival_node=tmnodes["arrival_time"],
                                                               offset_node=tmnodes["peak_offset"],
                                                                 wave_node=wn, std=stds['peak_offset'])
                for param in ("arrival_time", "coda_height", "coda_decay"):
                    n = tmnodes[param]
                    n_accepted[param] += gaussian_MH_move(sg, node_list=(n,), relevant_nodes=(n, wn), std=stds[param])

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

            print "step %d: lp %.2f, %d templates, accepted " % (step, lp, len(arrivals)),
            for move in moves:
                if (move == "birth") or (move == "death"):
                    print "%s: %d, " % (move, n_accepted[move]),
                else:
                    accepted_percent = float(n_accepted[move]) / (step * len(templates)) * 100 if (step * len(templates)) > 0 else 0
                    print "%s: %d%%, " % (move, accepted_percent),
            print
            plot_with_fit("unass_step%06d.png" % step, wn)


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

    parser.add_option("-e", "--evid", dest="evid", default=None, type="int", help="event ID to locate")
    parser.add_option("-s", "--sites", dest="sites", default=None, type="str",
                      help="comma-separated list of stations with which to locate the event")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str",
                      help="name of training run specifying the set of models to use")
    parser.add_option(
        "--template_shape", dest="template_shape", default="paired_exp", type="str", help="template model type (paired_exp)")
    parser.add_option(
        "-m", "--model", dest="model", default=None, type="str", help="name of training run specifying the set of models to use")
    parser.add_option(
        "--phases", dest="phases", default="auto", help="comma-separated list of phases to include in predicted templates (auto)")
    parser.add_option(
        "--template_model_types", dest="tm_types", default="tt_residual:constant_gaussian,peak_offset:constant_gaussian,amp_transfer:constant_gaussian,coda_decay:constant_gaussian",
        help="comma-separated list of param:model_type mappings (peak_offset:constant_gaussian,coda_height:constant_gaussian,coda_decay:constant_gaussian)")
    parser.add_option("--wiggle_model_type", dest="wm_type", default="dummy", help = "")
    parser.add_option("--wiggle_family", dest="wiggle_family", default="fourier_0.8", help = "")
    parser.add_option("--hz", dest="hz", default=5, type=float, help="downsample signals to a given sampling rate, in hz (5)")
    parser.add_option("--dummy_fallback", dest="dummy_fallback", default=False, action="store_true",
                      help="fall back to a dummy model instead of throwing an error if no model for the parameter exists in the database (False)")
    parser.add_option("--chans", dest="chans", default="BHZ,SHZ", type="str",
                      help="comma-separated list of channel names to use for inference (BHZ)")
    parser.add_option("--bands", dest="bands", default="freq_2.0_3.0", type="str",
                      help="comma-separated list of band names to use for inference (freq_2.0_3.0)")
    parser.add_option("--nm_type", dest="nm_type", default="ar", type="str",
                      help="type of noise model to use (ar)")


    (options, args) = parser.parse_args()

    evid = options.evid
    sites = options.sites.split(',')

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    # train / load coda models
    run_name = options.run_name
    iters = np.array(sorted(list(read_fitting_run_iterations(cursor, run_name))))
    run_iter, runid = iters[-1, :]

    tm_types = {}
    if ',' in options.tm_types:
        for p in options.tm_types.split(','):
            (param, model_type) = p.strip().split(':')
            tm_types[param] = model_type
    else:
        tm_types = options.tm_types

    if options.phases in ("auto", "leb"):
        phases = options.phases
    else:
        phases = options.phases.split(',')

    if options.bands == "all":
        bands = s.bands
    else:
        bands = options.bands.split(',')

    if options.chans == "all":
        chans = s.chans
    else:
        chans = options.chans.split(',')

    ev_true = get_event(evid=evid)

    # inference is based on segments from all specified stations,
    # starting at the min predicted arrival time (for the true event)
    # minus 60s, and ending at the max predicted arrival time plus
    # 240s
    statimes = [ev_true.time + tt_predict(event=ev_true, sta=sta, phase=phase) for (sta, phase) in itertools.product(sites, s.phases)]
    stime = np.min(statimes) - 60
    etime = np.max(statimes) + 240
    segments = load_segments(cursor, sites, stime, etime, chans = chans)
    segments = [seg.with_filter('env;hz_%.3f' % options.hz) for seg in segments]

    sg = SigvisaGraph(template_shape = options.template_shape, template_model_type = tm_types,
                      wiggle_family = options.wiggle_family, wiggle_model_type = options.wm_type,
                      dummy_fallback = options.dummy_fallback, nm_type = options.nm_type,
                      runid=runid, phases=phases, gpmodel_build_trees=False)


    wave_nodes = []
    for seg in segments:
        for band in bands:
            filtered_seg = seg.with_filter(band)
            for chan in filtered_seg.get_chans():
                wave = filtered_seg[chan]
                wave_nodes.append(sg.add_wave(wave))
    ev_node = sg.add_event(ev_true)

    for fname in os.listdir('.'):
        if fname.startswith("unass_step") or fname.startswith("mcmc_unass"):
            os.remove(fname)

    np.random.seed(0)
    run_event_MH(sg, wn, wiggles=True)
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
