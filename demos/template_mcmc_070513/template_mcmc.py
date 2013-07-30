import numpy as np
import numpy.ma as ma
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

from sigvisa.infer.template_mcmc import birth_move, death_move, indep_peak_move, improve_offset_move, get_signal_diff_positive_part, try_split, try_merge, get_sorted_arrivals, MERGE_MAX_ATIME_DIFFERENCE_S

BASE_DIR = os.path.join(os.getenv("SIGVISA_HOME"), "demos", "template_mcmc_070513")


#######################################################################

def try_split2(sg, wave_node):
    sorted_arrs = get_sorted_arrivals(wave_node)
    n = len(sorted_arrs)
    if n < 1:
        return False

    k = np.random.randint(0, n)
    arr_to_split = sorted_arrs[k]
    atime = arr_to_split[0]['arrival_time']

    # the arrival times for the split templates are sampled such that
    # the two new templates are still adjacent (no other template
    # arrives between them). This makes the move reversible, since the
    # merge move combines adjacent templates.
    if k > 0:
        prev_atime = sorted_arrs[k-1][0]['arrival_time']
    else:
        prev_atime = wave_node.st
    if k < n-1:
        next_atime = sorted_arrs[k+1][0]['arrival_time']
    else:
        next_atime = wave_node.et
    prev_atime = max(prev_atime, atime - MERGE_MAX_ATIME_DIFFERENCE_S)
    next_atime = min(next_atime, atime + MERGE_MAX_ATIME_DIFFERENCE_S)
    atime_window = next_atime - prev_atime

    # TODO: figure out what to do with wiggles
    tnodes = sg.get_template_nodes(eid=arr_to_split[1], phase=arr_to_split[2], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)
    tmpl = dict([(p, n) for (p, (k, n)) in tnodes.items()])
    return split_move2(sg, tmpl, wave_node, prev_atime, atime_window)


def split_move2(sg, tmpl, wave_node, prev_atime, atime_window):

    lp_old = sg.current_log_p()

    arrival_time, peak_offset, coda_height, coda_decay = tmpl['arrival_time'].get_value(), tmpl['peak_offset'].get_value(), tmpl['coda_height'].get_value(), tmpl['coda_decay'].get_value()

    new_tmpl = sg.create_unassociated_template(wave_node, atime=arrival_time, nosort=True)
    sg._topo_sorted_list = new_tmpl.values() + sg._topo_sorted_list
    sg._gc_topo_sorted_nodes()

    # WLOG give the earlier arrival time to the current template, and
    # the later one to the new template we create. For all other params,
    # decide at random which template gets which.
    eps_atime = np.random.rand()*atime_window
    new_tmpl['arrival_time'].set_value(prev_atime + eps_atime)

    new_offset = tmpl['peak_offset'].model.sample()
    new_tmpl['peak_offset'].set_value(new_offset)

    new_decay = tmpl['coda_decay'].model.sample()
    new_tmpl['coda_decay'].set_value(new_decay)

    new_logheight = np.log(np.random.rand()) + coda_height
    new_tmpl['coda_height'].set_value(new_logheight)
    tmpl['coda_height'].set_value( np.log(np.exp(coda_height) - np.exp(new_logheight)) )

    lp_new = sg.current_log_p()

    log_qforward = tmpl['peak_offset'].model.log_p(new_offset) + tmpl['coda_decay'].model.log_p(new_decay) - np.log(atime_window) - np.log(np.exp(coda_height))
    jacobian_determinant = 0
    log_qbackward = 0 ## merge move is deterministic

    u = np.random.rand()
    if (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant > np.log(u):
        print "split template %d from %d: %.1f + %.5f - (%.1f + %.5f) + %f = %.1f vs %.1f" % (new_tmpl["arrival_time"].unassociated_templateid, tmpl["arrival_time"].unassociated_templateid, lp_new, log_qbackward, lp_old, log_qforward, jacobian_determinant, (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant, np.log(u))
        return new_tmpl
    else:
        sg.destroy_unassociated_template(new_tmpl, nosort=True)

        tmpl['arrival_time'].set_value(arrival_time)
        tmpl['peak_offset'].set_value(peak_offset)
        tmpl['coda_decay'].set_value(coda_decay)
        tmpl['coda_height'].set_value(coda_height)

        # WARNING: this assumes the list hasn't been re-sorted by any
        # of our intermediate calls.
        sg._topo_sorted_list = sg._topo_sorted_list[len(new_tmpl):]
        sg._gc_topo_sorted_nodes()
        sg.next_uatemplateid -= 1
        #lp = sg.current_log_p()
        #assert(np.abs(lp - lp_old) < 1e-10)
        return False

def try_merge2(sg, wave_node):
    sorted_arrs = get_sorted_arrivals(wave_node)
    n = len(sorted_arrs)
    if n < 2:
        return False
    k = np.random.randint(0, n-1)
    arr1 = sorted_arrs[k]
    arr2 = sorted_arrs[k+1]
    if arr2[0]['arrival_time'] - arr1[0]['arrival_time'] > MERGE_MAX_ATIME_DIFFERENCE_S:
        return False

    tnodes1 = sg.get_template_nodes(eid=arr1[1], phase=arr1[2], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)
    tmpl1 = dict([(param, node) for (param, (key, node)) in tnodes1.items()])

    tnodes2 = sg.get_template_nodes(eid=arr2[1], phase=arr2[2], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)
    tmpl2 = dict([(param, node) for (param, (key, node)) in tnodes2.items()])

    merged_atime = tmpl1['arrival_time'].get_value()
    post_merge_atime_diff = MERGE_MAX_ATIME_DIFFERENCE_S
    if k+2 < n:
        next_atime = sorted_arrs[k+2][0]['arrival_time']
    else:
        next_atime = wave_node.et
    next_atime = min(next_atime, merged_atime + MERGE_MAX_ATIME_DIFFERENCE_S)
    if k > 0:
        prev_atime = sorted_arrs[k-1][0]['arrival_time']
    else:
        prev_atime = wave_node.st
    prev_atime = max(prev_atime, merged_atime - MERGE_MAX_ATIME_DIFFERENCE_S)
    atime_window = next_atime - prev_atime

    return merge_move2(sg, tmpl1, tmpl2, wave_node, atime_window)

def merge_move2(sg, tmpl1_nodes, tmpl2_nodes, wave_node, atime_window):

    lp_old = sg.current_log_p()
    orig_topo_sorted = copy.copy(sg._topo_sorted_list)

    tmpl1_values = dict((k, n.get_value()) for (k,n) in tmpl1_nodes.items())
    tmpl2_values = dict((k, n.get_value()) for (k,n) in tmpl2_nodes.items())

    lost_atime = tmpl2_values['arrival_time']
    lost_offset = tmpl2_values['peak_offset']
    lost_decay = tmpl2_values['coda_decay']

    merged_amp = np.log(np.exp(tmpl1_values['coda_height']) + np.exp(tmpl2_values['coda_height']))

    for (param, node) in tmpl2_nodes.items():
        sg.remove_node(node)
        sg._topo_sorted_list[node._topo_sorted_list_index] = None

    #tmpl1_nodes['arrival_time'].set_value(merged_atime)
    #tmpl1_nodes['peak_offset'].set_value(merged_offset)
    #tmpl1_nodes['coda_decay'].set_value(merged_decay)
    tmpl1_nodes['coda_height'].set_value(merged_amp)

    lp_new = sg.current_log_p()

    log_qforward = 0
    jacobian_determinant = 0
    log_qbackward = tmpl2_nodes['peak_offset'].model.log_p(lost_offset) + tmpl2_nodes['coda_decay'].model.log_p(lost_decay) - np.log(atime_window) - np.log(np.exp(merged_amp))

    u = np.random.rand()
    if (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant > np.log(u):
        print "merged template %d into %d: %.1f + %.5f - (%.1f + %.5f) + %f = %.1f vs %.1f" % (tmpl1_nodes["arrival_time"].unassociated_templateid, tmpl2_nodes["arrival_time"].unassociated_templateid, lp_new, log_qbackward, lp_old, log_qforward, jacobian_determinant, (lp_new + log_qbackward) - (lp_old + log_qforward) + jacobian_determinant, np.log(u))
        return True
    else:
        for (param, node) in tmpl2_nodes.items():
            sg.add_node(node)
            node.addChild(wave_node)
        for (param, node) in tmpl1_nodes.items():
            node.set_value(tmpl1_values[param])
        wave_node.arrivals()
        sg._topo_sorted_list = orig_topo_sorted
        sg._gc_topo_sorted_nodes()
        #lp = sg.current_log_p()
        #assert(np.abs(lp - lp_old) < 1e-10)
        return False




#######################################################################

def run_MH(sg, wn, burnin=0, skip=40, steps=10000, wiggles=False, name=None, open_world=False, splitmerge=0):
    n_accepted = dict()
    moves = ('birth', 'death', 'split', 'merge', 'indep_peak', 'peak_offset', 'arrival_time', 'coda_height', 'coda_decay', 'wiggle_amp', 'wiggle_phase')

    for move in moves:
        n_accepted[move] = 0

    stds = {'peak_offset': .1, 'arrival_time': .1, 'coda_height': .02, 'coda_decay': 0.05, 'wiggle_amp': .25, 'wiggle_phase': .5}


    templates = dict()
    params_over_time = dict()
    params_over_time["overall_logp"] = []
    params_over_time["ntemplates"] = []

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


        if splitmerge == 1:
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

        if splitmerge == 2:
            split_nodes = try_split2(sg, wn)
            if split_nodes:
                tmplid = split_nodes['arrival_time'].unassociated_templateid
                templates[tmplid] = split_nodes
                for param in split_nodes.keys():
                    params_over_time["%d_%s" % (tmplid, param)] = [np.float('nan')] * step
                n_accepted['split'] += 1
            arrivals = wn.arrivals()
            if len(arrivals) >= 1:
                n_accepted['merge'] += try_merge2(sg, wn)


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
        params_over_time["ntemplates"].append(len(wn.arrivals()))

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
    vals = vals[np.isfinite(vals)]
    axes.acorr((vals - np.mean(vals))/np.std(vals), maxlags=min(500, int(len(vals)/1.5)))
    return fig

def running_mean_plot(param, vals):
    fig = Figure(figsize=(8, 5), dpi=144)
    axes = fig.add_subplot(111)
    axes.set_title("%s running mean" % param)
    axes.set_xlabel("Steps", fontsize=8)
    axes.set_ylabel(param, fontsize=8)

    vals[~np.isfinite(vals)] = 0
    running_sum = np.cumsum(vals)
    running_mean = running_sum / np.cumsum(np.isfinite(vals))
    axes.plot(running_mean)
    return fig

def posterior_kde_plot(param, vals):
    finite = np.isfinite(vals)
    kernel = scipy.stats.gaussian_kde(vals[finite])
    x = np.linspace(np.min(vals[finite]), np.max(vals[finite]), 100)
    y = kernel(x)

    fig = Figure(figsize=(8, 5), dpi=144)
    axes = fig.add_subplot(111)
    axes.set_title("%s posterior density (%d samples)" % (param, np.sum(finite)))
    axes.set_xlabel(param, fontsize=8)
    axes.set_ylabel("density", fontsize=8)
    axes.plot(x, y)
    return fig

def visualize_mcmc(name, burnin=100, generate_step_gif=True, min_lifetime_steps=100, nowiggle=False):
    import shlex
    #import shutil
    import subprocess

    visual_dir = BASE_DIR
    page = "<html><head><title>MCMC run '%s'</title></head><body><h1>MCMC run '%s'</h1>\n" % (name, name)

    # generated animated gif showing MCMC steps
    anim_fname = "%sstep_anim.gif" % (name)
    if generate_step_gif:
        cmd_str = "convert %s/%sstep*.png -delay 100 -layers Optimize %s" % (BASE_DIR, name, os.path.join(visual_dir, anim_fname))
        result = subprocess.call(shlex.split(cmd_str),
                             shell=False)

    page += "<table><tr><td><b>true</b></td><td><b>sampled</b></td></tr><tr><td><img src='%ssampled.png' width=100%%></td><td><a href='%sstep000001.png'><img src='%s' width=100%%></a></td></tr></table>" % (name, name, anim_fname)

    # load true templates
    try:
        with open(os.path.join(BASE_DIR, "%ssampled_templates.pkl" %name), 'rb') as f:
            t = pickle.load(f)
        page += "<hr><b>True param vals</b>:<br><table><tr><td><b>arrival_time</b></td><td><b>peak_offset</b></td><td><b>coda_height</b></td><td><b>coda_decay</b></td></tr>"
        for template in sorted(t, key = lambda tmpl: tmpl['arrival_time'].get_value()):
            page += "<tr>"
            for param in ("arrival_time", "peak_offset", "coda_height", "coda_decay"):
                page += "<td>%.3f</td>" % template[param].get_value()
            page += "</tr>\n"
        page += "</table>"
    except IOError:
        t = []

    try:
        with open(os.path.join(BASE_DIR, "%ssampled_logp.txt" % name), 'r') as f:
            true_logp = float(f.read())
    except IOError:
        true_logp = 0


    page += "<hr><table><tr align=center><td><b>param</b></td><td><b>trace</b></td><td><b>autocorr</b></td><td><b>running mean</b></td><td><b>posterior</b></td><td><b>true val</b></td></tr>"

    # try to find a correspondence between true and inferred templates
    v = np.load(os.path.join(BASE_DIR, "%smcmc_vals.npz" % name))
    template_map = dict()
    for atkey in v.keys():
        if not atkey.endswith("arrival_time"): continue
        if np.sum(np.isfinite(v[atkey])) < min_lifetime_steps: continue
        mcmc_template_num = int(atkey.split("_")[0])
        closest_true_template = None
        closest_time_diff = 8
        for true_template in t:
            t_atime = true_template['arrival_time'].get_value()
            atime_diff = np.abs(scipy.stats.nanmedian(v[atkey]) - t_atime)
            if atime_diff < closest_time_diff:
                closest_true_template = true_template
                closest_time_diff = atime_diff
        template_map[mcmc_template_num] = closest_true_template

    for key in sorted(v.keys()):

        if nowiggle and ("amp" in key) or ("phase" in key): continue
        if np.sum(np.isfinite(v[key])) < min_lifetime_steps: continue

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

        try:
            posterior_kde_fig = posterior_kde_plot(param=key, vals=v[key][burnin:])
            posterior_kde_fig_name = "%s%s_posterior.png" % (name, key)
            savefig(os.path.join(visual_dir, posterior_kde_fig_name), posterior_kde_fig)
            page += "<td><a href='%s'><img src='%s' width=100%%></a></td>" % (posterior_kde_fig_name, posterior_kde_fig_name)
        except np.linalg.LinAlgError:
            pass

        if key == "overall_logp":
            page += "<td>%.3f</td>" % true_logp
        elif key == "ntemplates":
            page += "<td>%d</td>" % len(t)
        else:
            mcmc_template_num = int(key.split("_")[0])
            param = "_".join(key.split("_")[1:])
            if template_map[mcmc_template_num] is not None:
                page += "<td>%.3f</td>" % template_map[mcmc_template_num][param].get_value()
            else:
                page += "<td>??</td>"

        page += "</tr>\n"
    page += "</table></body></html>"
    with open(os.path.join(visual_dir, "%srun.html" % name), 'w') as f:
        f.write(page)


def main(name="", seed=None, env=True, wiggles=True, num_init_templates=2, evid=None, **kwargs):
    if len(name) > 0:
        name += "_"

    if env:
        if wiggles:
            wiggle_family = "fourier_0.8"
        else:
            wiggle_family = "dummy"
    else:
        wiggle_family = "fourier_2.5"

    if seed is not None:
        np.random.seed(seed)


    """
    """
    sg = SigvisaGraph(template_model_type="dummy", template_shape="paired_exp",
                      wiggle_model_type="dummy", wiggle_family=wiggle_family,
                      phases="leb", nm_type = "ar", wiggle_len_s = 60.0,
                      assume_envelopes=env)

    if evid is None:
        wave = Waveform(data = np.load(os.path.join(BASE_DIR, "%ssampled_wave.npy" % name)),
                        srate=5.0, stime=1239915900.0,
                        sta="FIA3", chan="SHZ", arrivals=np.array(()),
                        filter_str="freq_2.0_3.0;env;hz_5.0")
    else:
        s = Sigvisa()
        cursor = s.dbconn.cursor()
        wave = load_event_station_chan(evid=evid, sta="FIA3", chan="SHZ", cursor=cursor).filter("%s;env" % "freq_2.0_3.0").filter('hz_5.0')
        cursor.close()
        fig = plot_waveform(wave)
        savefig(os.path.join(BASE_DIR, "%ssampled.png" % name), fig)

    wn = sg.add_wave(wave)

    env = wn.get_value().data if wn.env else wn.get_wave().filter("env").data.data

    for i in range(num_init_templates):
        sg.create_unassociated_template(wave_node=wn, atime=wn.st + np.random.rand() * (wn.et - wn.st - 5))


    for fname in os.listdir('.'):
        if fname.startswith("%sstep" % name) or fname.startswith("%smcmc" %name):
            os.remove(fname)

    #test_moves(sg, wn)
    run_MH(sg, wn, wiggles=wiggles, name=name, open_world=(num_init_templates==0), **kwargs)
    #print "atime", sg.get_value(key=create_key(param="arrival_time", eid=en.eid, sta="FIA3", phase="P"))

    ll = wn.log_p()
    print ll

if __name__ == "__main__":
    try:
        #main(name="nowiggle", seed=0, wiggles=False, env=True, steps=10000, num_init_templates=2)
        #main(name="wiggle", seed=0, wiggles=True, env=True, steps=2000)
        #main(name="wiggle", seed=1, wiggles=True, env=True, steps=2000, num_init_templates=2)
        #main(name="wiggle_noenv", seed=1, wiggles=True, env=False, steps=1000, num_init_templates=2)

        #main(name="nowiggle_birthdeath", seed=0, wiggles=False, env=True, steps=4000, num_init_templates=0)
        #main(name="nowiggle_splitmerge1", seed=0, wiggles=False, env=True, steps=10000, num_init_templates=0, splitmerge=2)
        #main(name="nowiggle_splitmerge2", seed=0, wiggles=False, env=True, steps=10000, num_init_templates=0, splitmerge=2)

        #main(name="nowiggle_splitmerge2_5326226", seed=0, wiggles=False, env=True, steps=10000, num_init_templates=0, splitmerge=2, evid=5326226)
        #main(name="nowiggle_5326226", seed=0, wiggles=False, env=True, steps=10000, num_init_templates=1, evid=5326226)

        #visualize_mcmc(name="wiggle_", nowiggle=True)
        #visualize_mcmc(name="nowiggle_")
        #visualize_mcmc(name="nowiggle_birthdeath_")
        #visualize_mcmc(name="nowiggle_splitmerge1_")
        #visualize_mcmc(name="nowiggle_splitmerge2_")
        #visualize_mcmc(name="wiggle_noenv_", nowiggle=True)
        #visualize_mcmc(name="nowiggle_splitmerge2_5326226_", nowiggle=True, burnin=5, min_lifetime_steps=1000)
        visualize_mcmc(name="nowiggle_5326226_", nowiggle=True, burnin=500, min_lifetime_steps=1000)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
