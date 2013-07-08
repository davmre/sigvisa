import numpy as np
import sys
import os
import traceback
import pickle
import copy

import scipy.stats

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_event_station_chan
from sigvisa.source.event import Event
from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.models.signal_model import extract_arrival_from_key
from sigvisa.infer.mcmc_basic import get_node_scales, gaussian_propose, gaussian_MH_move, MH_accept
from sigvisa.graph.graph_utils import create_key
from sigvisa.graph.dag import get_relevant_nodes
from sigvisa.plotting.plot import savefig, plot_with_fit, plot_waveform, bounds_without_outliers
from matplotlib.figure import Figure

from sigvisa.infer.template_mcmc import birth_move, death_move, indep_peak_move, improve_offset_move, get_signal_diff_positive_part

BASE_DIR = os.path.join(os.getenv("SIGVISA_HOME"), "demos", "template_mcmc_070513")

def run_MH(sg, wn, burnin=0, skip=40, steps=10000, wiggles=False, name=None, open_world=False):
    n_accepted = dict()
    moves = ('birth', 'death', 'indep_peak', 'peak_offset', 'arrival_time', 'coda_height', 'coda_decay', 'wiggle_amp', 'wiggle_phase')

    for move in moves:
        n_accepted[move] = 0

    stds = {'peak_offset': .1, 'arrival_time': .1, 'coda_height': .02, 'coda_decay': 0.05, 'wiggle_amp': .25, 'wiggle_phase': .5}


    templates = dict()
    params_over_time = dict()
    params_over_time["overall_logp"] = []

    ymin, ymax = bounds_without_outliers(data=wn.get_value())

    if not open_world:
        arrivals = wn.arrivals()
        for (eid, phase) in arrivals:
            tmplid = -eid
            templates[tmplid] = dict([(param, node) for (param, (key, node)) in sg.get_template_nodes(eid=eid, phase=phase, sta=wn.sta, band=wn.band, chan=wn.chan).items()])
            if wiggles:
                wiggle_nodes =  dict([(param, node) for (param, (key, node)) in sg.get_wiggle_nodes(eid=eid, phase=phase, sta=wn.sta, band=wn.band, chan=wn.chan).items()])
                templates[tmplid].update(wiggle_nodes)

            for param in templates[tmplid].keys():
                params_over_time["%d_%s" % (tmplid, param)] = []

    for step in range(steps):
        if open_world:
            new_nodes = birth_move(sg, wn, wiggles=wiggles)
            if new_nodes:
                tmplid = new_nodes['arrival_time'].unassociated_templateid
                templates[tmplid] = new_nodes
                for param in new_nodes.keys():
                    params_over_time["%d_%s" % (tmplid, param)] = [np.float('nan')] * step
                n_accepted['birth'] += 1

            arrivals = wn.arrivals()
            if len(arrivals) >= 1:
                n_accepted['death'] += death_move(sg, wn)


        """
        split_nodes = try_split(sg, wn)
        if split_nodes:
            tmplid = split_nodes['arrival_time'].unassociated_templateid
            templates[tmplid] = split_nodes
            for param in split_nodes.keys():
                params_over_time["%d_%s" % (tmplid, param)] = [np.float('nan')] * step
            n_accepted['split'] += 1
        arrivals = wn.arrivals()
        if len(arrivals) >= 1:
            n_accepted['merge'] += try_merge(sg, wn)
        """

        for (eid, phase) in arrivals:
            l3 = len(arrivals)
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
        params_over_time["overall_logp"].append(sg.current_log_p())

        if step > 0 and ((step % skip == 0) or (step < 15)):
            lp = sg.current_log_p()

            print "step %d: lp %.2f, %d templates, accepted " % (step, lp, len(arrivals)),
            for move in moves:
                if move in ("split", "merge", "birth", "death"):
                    print "%s: %d, " % (move, n_accepted[move]),
                else:
                    accepted_percent = float(n_accepted[move]) / (step * len(templates)) * 100 if (step * len(templates)) > 0 else 0
                    print "%s: %d%%, " % (move, accepted_percent),
            print
            plot_with_fit(os.path.join(BASE_DIR, "%sstep%06d.png" % (name, step)), wn, title="step %d of %d" % (step, steps), ymin=ymin, ymax=ymax)
            #signal_diff_pos = get_signal_diff_positive_part(wn, wn.arrivals())
            #w = wn.get_wave()
            #w.data = signal_diff_pos
            #savefig(fname=os.path.join(BASE_DIR, "%sdiff%06d.png" % (name, step)), fig=plot_waveform(w))


    np.savez(os.path.join(BASE_DIR, '%smcmc_vals.npz' %name) , **params_over_time)


def trace_plot(param, vals):
    fig = Figure(figsize=(8, 5), dpi=144)
    axes = fig.add_subplot(111)
    axes.set_xlabel("Steps", fontsize=8)
    axes.set_ylabel(param, fontsize=8)
    axes.plot(vals)
    return fig

def autocorr_plot(param, vals):
    fig = Figure(figsize=(8, 5), dpi=144)
    axes = fig.add_subplot(111)
    axes.set_title("%s autocorrelation" % param)
    axes.set_xlabel("Lag", fontsize=8)
    axes.set_ylabel("autocorrelation", fontsize=8)
    axes.acorr((vals - np.mean(vals))/np.std(vals), maxlags=500)
    return fig

def running_mean_plot(param, vals):
    fig = Figure(figsize=(8, 5), dpi=144)
    axes = fig.add_subplot(111)
    axes.set_title("%s running mean" % param)
    axes.set_xlabel("Steps", fontsize=8)
    axes.set_ylabel(param, fontsize=8)

    running_sum = np.cumsum(vals)
    running_mean = running_sum / np.cumsum(np.isfinite(vals))
    axes.plot(running_mean)
    return fig

def posterior_kde_plot(param, vals):
    kernel = scipy.stats.gaussian_kde(vals)
    x = np.linspace(np.min(vals), np.max(vals), 100)
    y = kernel(x)

    fig = Figure(figsize=(8, 5), dpi=144)
    axes = fig.add_subplot(111)
    axes.set_title("%s posterior density (%d samples)" % (param, len(vals)))
    axes.set_xlabel(param, fontsize=8)
    axes.set_ylabel("density", fontsize=8)
    axes.plot(x, y)
    return fig

def visualize_mcmc(name, burnin=100):
    import shlex
    #import shutil
    import subprocess

    visual_dir = BASE_DIR

    page = "<html><head><title>MCMC run '%s'</title></head><body><h1>MCMC run '%s'</h1>\n" % (name, name)

    cmd_str = "convert %s/%sstep*.png -delay 100 -layers Optimize %s" % (BASE_DIR, name, os.path.join(visual_dir, "%sstep_anim.gif" % (name)))
    result = subprocess.call(shlex.split(cmd_str),
                             shell=False)
    page += "<table><tr><td><b>true</b></td><td><b>sampled</b></td></tr><tr><td><img src='%sstep_anim.gif' width=100%%></td><td><img src='%ssampled.png' width=100%%></td></tr></table>" % (name, name)

    with open(os.path.join(BASE_DIR, "%ssampled_templates.pkl" %name), 'rb') as f:
        t = pickle.load(f)
    page += "<hr><b>True param vals</b>:<br><table><tr><td><b>arrival_time</b></td><td><b>peak_offset</b></td><td><b>coda_height</b></td><td><b>coda_decay</b></td></tr>"
    for template in t:
        page += "<tr>"
        for param in ("arrival_time", "peak_offset", "coda_height", "coda_decay"):
            page += "<td>%.3f</td>" % template[param].get_value()
        page += "</tr>\n"
    page += "</table>"

    page += "<hr><table><tr align=center><td><b>param</b></td><td><b>trace</b></td><td><b>autocorr</b></td><td><b>running mean</b></td><td><b>posterior</b></td></tr>"

    v = np.load(os.path.join(BASE_DIR, "%smcmc_vals.npz" % name))
    for key in sorted(v.keys()):
        page += "<tr><td>%s</td>" % key

        trace_fig = trace_plot(param=key, vals = v[key][burnin:])
        trace_fig_name = "%s%s_trace.png" % (name, key)
        savefig(os.path.join(visual_dir, trace_fig_name), trace_fig)
        page += "<td><a href='%s'><img src='%s' width=100%%></a></td>" % (trace_fig_name, trace_fig_name)

        autocorr_fig = autocorr_plot(param=key, vals = v[key][burnin:])
        autocorr_fig_name = "%s%s_autocorr.png" % (name, key)
        savefig(os.path.join(visual_dir, autocorr_fig_name), autocorr_fig)
        page += "<td><a href='%s'><img src='%s' width=100%%></a></td>" % (autocorr_fig_name, autocorr_fig_name)

        running_mean_fig = running_mean_plot(param=key, vals=v[key][burnin:])
        running_mean_fig_name = "%s%s_running_mean.png" % (name, key)
        savefig(os.path.join(visual_dir, running_mean_fig_name), running_mean_fig)
        page += "<td><a href='%s'><img src='%s' width=100%%></a></td>" % (running_mean_fig_name, running_mean_fig_name)

        posterior_kde_fig = posterior_kde_plot(param=key, vals=v[key][burnin:])
        posterior_kde_fig_name = "%s%s_posterior.png" % (name, key)
        savefig(os.path.join(visual_dir, posterior_kde_fig_name), posterior_kde_fig)
        page += "<td><a href='%s'><img src='%s' width=100%%></a></td>" % (posterior_kde_fig_name, posterior_kde_fig_name)
        page += "</tr>\n"
    page += "</table></body></html>"
    with open(os.path.join(visual_dir, "%srun.html" % name), 'w') as f:
        f.write(page)


def main(name="", seed=None):
    if len(name) > 0:
        name += "_"

    """
    s = Sigvisa()
    cursor = s.dbconn.cursor()
    wave = load_event_station_chan(evid=5326226, sta="FIA3", chan="SHZ", cursor=cursor).filter("%s;env" % "freq_2.0_3.0").filter('hz_5.0')
    cursor.close()
    """
    sg = SigvisaGraph(template_model_type="dummy", template_shape="paired_exp",
                      wiggle_model_type="dummy", wiggle_family="dummy",
                      phases="leb", nm_type = "ar", wiggle_len_s = 60.0,
                      assume_envelopes=True)

    wave = Waveform(data = np.load(os.path.join(BASE_DIR, "%ssampled_wave.npy" % name)),
                    srate=5.0, stime=1239915900.0,
                    sta="FIA3", chan="SHZ", arrivals=np.array(()),
                    filter_str="freq_2.0_3.0;env;hz_5.0")
    wn = sg.add_wave(wave)
    env = wn.get_value().data if wn.env else wn.get_wave().filter("env").data.data
    sg.create_unassociated_template(wave_node=wn, atime=1239915940.253)
    sg.create_unassociated_template(wave_node=wn, atime=1239915969.623)

    for fname in os.listdir('.'):
        if fname.startswith("%sstep" % name) or fname.startswith("%smcmc" %name):
            os.remove(fname)

    if seed is not None:
        np.random.seed(seed)
    #test_moves(sg, wn)
    run_MH(sg, wn, wiggles=True, name=name, open_world=False)
    #print "atime", sg.get_value(key=create_key(param="arrival_time", eid=en.eid, sta="FIA3", phase="P"))

    ll = wn.log_p()
    print ll

    plot_with_fit("unass.png", wn)

if __name__ == "__main__":
    try:
        #sample_template()
        main(name="nowiggle", seed=0)
        visualize_mcmc(name="nowiggle_")
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
