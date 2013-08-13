import numpy as np
import sys
import os
import traceback
import pickle
import copy

from optparse import OptionParser
from sigvisa.database.signal_data import *
from sigvisa.database.dataset import *
import itertools

from sigvisa.models.ttime import tt_predict
from sigvisa.graph.sigvisa_graph import SigvisaGraph, predict_phases
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_segments
from sigvisa.source.event import Event
from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.infer.mcmc_basic import get_node_scales, gaussian_propose, gaussian_MH_move, MH_accept
from sigvisa.infer.template_mcmc import preprocess_signal_for_sampling, indep_offset_move, improve_offset_move, indep_peak_move
from sigvisa.graph.graph_utils import create_key
from sigvisa.graph.dag import get_relevant_nodes
from sigvisa.plotting.plot import savefig, plot_with_fit_hack_for_DTRA
from matplotlib.figure import Figure

#def visualize_ev_locations(lonlats)

def regen_station_templates(sg, ev_node, wn_list, skip=40, steps=10000):

    moves = ( 'indep_peak', 'peak_offset', 'tt_residual', 'amp_transfer', 'coda_decay', 'evloc', 'evloc_big')
    templates = dict()

    params_over_time = np.load('ev_vals.npz')

    for wn in wn_list:

        arrivals = wn.arrivals()
        eid, phase = list(arrivals)[0]
        templates[wn.sta] = dict([(param, node) for (param, (key, node)) in sg.get_template_nodes(eid=eid, phase=phase, sta=wn.sta, band=wn.band, chan=wn.chan).items()])


    for step in range(0, steps, skip):
        for wn in wn_list:
            arrivals = wn.arrivals()
            eid, phase = list(arrivals)[0]

            wg = sg.wiggle_generator(phase=phase, srate=wn.srate)
            tmnodes = templates[wn.sta]

            for (param, n) in tmnodes.items():
                n.set_value(params_over_time["%s_%s" % (wn.sta, param)][step])

        #latlon = params_over_time["evloc"][step]
        #ev_node.set_local_value(key="lat", value=latlon[0])
        #ev_node.set_local_value(key="lon", value=latlon[1])

        for wn in wn_list:
            plot_with_fit_hack_for_DTRA("ev_%s_step%06d.png" % (wn.sta, step), wn, show_template=False)

def ev_move(sg, ev_node, std, param):
    # jointly propose a new event location along with new tt_residual values,
    # such that the event arrival times remain constant.

    def set_ev(ev_node, v, atimes, atime_nodes):
        ev_node.set_local_value(key=param, value=v)
        for (at, atn) in zip(atimes, atime_nodes):
            atn.set_value(at)

    current_v = ev_node.get_local_value(param)

    sorted_children = sorted(ev_node.children, key = lambda n: n.label)
    atime_nodes = [child for child in sorted_children if child.label.endswith("arrival_time")]
    ttr_nodes = [child for child in sorted_children if child.label.endswith("tt_residual")]
    current_atimes = [atn.get_value() for atn in atime_nodes]

    gsample = np.random.normal(0, std, 1)
    move = gsample * std
    new_v = current_v + move

    node_list, relevant_nodes = get_relevant_nodes([ev_node,] + ttr_nodes)

    lp_old = sg.joint_prob(node_list=node_list, relevant_nodes=relevant_nodes, values=None)

    set_ev(ev_node, new_v, current_atimes, atime_nodes)
    lp_new = sg.joint_prob(node_list=node_list, relevant_nodes=relevant_nodes, values=None)

    u = np.random.rand()
    if lp_new - lp_old > np.log(u):
        return True
    else:
        set_ev(ev_node, current_v, current_atimes, atime_nodes)
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

def ev_lonlat_move(sg, ev_node, std):
    # jointly propose a new event location along with new tt_residual values,
    # such that the event arrival times remain constant.

    def set_ev_loc(ev_node, lat, lon, atimes, atime_nodes):
        ev_node.set_local_value(key="lat", value=lat)
        ev_node.set_local_value(key="lon", value=lon)
        for (at, atn) in zip(atimes, atime_nodes):
            atn.set_value(at)

    current_lon = ev_node.get_local_value("lon")
    current_lat = ev_node.get_local_value("lat")
    current_latlon = np.array((current_lat, current_lon))

    sorted_children = sorted(ev_node.children, key = lambda n: n.label)
    atime_nodes = [child for child in sorted_children if child.label.endswith("arrival_time")]
    ttr_nodes = [child for child in sorted_children if child.label.endswith("tt_residual")]
    current_atimes = [atn.get_value() for atn in atime_nodes]

    gsample = np.random.normal(0, std, 2)
    move = gsample * std
    new_latlon = current_latlon + move

    node_list, relevant_nodes = get_relevant_nodes([ev_node,] + ttr_nodes)

    lp_old = sg.joint_prob(node_list=node_list, relevant_nodes=relevant_nodes, values=None)

    set_ev_loc(ev_node, new_latlon[0], new_latlon[1], current_atimes, atime_nodes)
    lp_new = sg.joint_prob(node_list=node_list, relevant_nodes=relevant_nodes, values=None)

    u = np.random.rand()
    if lp_new - lp_old > np.log(u):
        return True
    else:
        set_ev_loc(ev_node, current_latlon[0], current_latlon[1], current_atimes, atime_nodes)
        return False



def run_event_MH(sg, ev_node, wn_list, burnin=0, skip=40, steps=10000):

    n_accepted = dict()
    n_tried = dict()
    moves = ( 'indep_peak', 'peak_offset', 'tt_residual', 'amp_transfer', 'coda_decay', 'evloc', 'evloc_big', 'evtime', 'evdepth', 'evmb')
    for move in moves:
        n_accepted[move] = 0
        n_tried[move] = 0

    stds = {'peak_offset': .1, 'tt_residual': .1, 'amp_transfer': .1, 'coda_decay': 0.01, 'evloc': 0.01, 'evloc_big': 0.5, 'evtime': 2.0, "evmb": 0.5, "evdepth": 5.0}

    templates = dict()
    params_over_time = dict()

    for wn in wn_list:
        wave_env = wn.get_value() if wn.env else wn.get_wave().filter('env').data
        wn.cdf = preprocess_signal_for_sampling(wave_env)

        arrivals = wn.arrivals()
        eid, phase = list(arrivals)[0]
        templates[wn.sta] = dict([(param, node) for (param, (key, node)) in sg.get_template_nodes(eid=eid, phase=phase, sta=wn.sta, band=wn.band, chan=wn.chan).items()])

        for param in templates[wn.sta].keys():
            params_over_time["%s_%s" % (wn.sta, param)] = []
    params_over_time["evloc"] = []


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

        n_accepted["evloc"] += ev_lonlat_move(sg, ev_node, std=stds['evloc'])
        n_tried["evloc"] += 1

        n_accepted["evloc_big"] += ev_lonlat_move(sg, ev_node, std=stds['evloc_big'])
        n_tried["evloc_big"] += 1

        n_accepted["evtime"] += ev_move(sg, ev_node, std=stds['evtime'], param="time")
        n_tried["evtime"] += 1

        n_accepted["evdepth"] += ev_move(sg, ev_node, std=stds['evdepth'], param="depth")
        n_tried["evdepth"] += 1

        n_accepted["evmb"] += ev_move(sg, ev_node, std=stds['evmb'], param="mb")
        n_tried["evmb"] += 1

        params_over_time["evloc"].append( ev_node.get_mutable_values())
        params_over_time["evtime"].append( ev_node.get_mutable_values())
        params_over_time["evdepth"].append( ev_node.get_mutable_values())
        params_over_time["evmb"].append( ev_node.get_mutable_values())

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
            print " ev loc", ev_node.get_mutable_values()
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
    ev_node.fix_value()
    ev_node.unfix_value(key = "%d;lon" % ev_node.eid)
    ev_node.unfix_value(key = "%d;lat" % ev_node.eid)
    ev_node.unfix_value(key = "%d;time" % ev_node.eid)
    ev_node.unfix_value(key = "%d;depth" % ev_node.eid)
    ev_node.unfix_value(key = "%d;mb" % ev_node.eid)
    ev_node.set_value(key = "%d;lon" % ev_node.eid, value=124.3)
    ev_node.set_value(key = "%d;lat" % ev_node.eid, value=44.5)
    ev_node.set_value(key = "%d;time" % ev_node.eid, value=ev_true.time + 5)
    ev_node.set_value(key = "%d;depth" % ev_node.eid, value=10)
    ev_node.set_value(key = "%d;mb" % ev_node.eid, value=3.0)

    #for fname in os.listdir('.'):
    #    if fname.startswith("unass_step") or fname.startswith("mcmc_unass"):
    #        os.remove(fname)

    np.random.seed(0)
    run_event_MH(sg, ev_node, wave_nodes)
    #regen_station_templates(sg, ev_node, wave_nodes)
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
