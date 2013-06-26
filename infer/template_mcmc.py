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


#######################################################################

"""

Methods for sampling an arrival time independent of the current
arrival time, with probability proportional to the height of the
envelope at each point.

"""

def preprocess_signal_for_sampling(wave_env):

    d = wave_env**2

    """
    # sample locations where the envelope is increasing, relative to how fast it's increasing
    grad = np.gradient(wave_env)
    incr = (grad > 0)
    d = grad**2
    d[~incr] = max(np.min(d), 1e-3)
    """

    s = np.sum(d)
    normalized_env = d/s
    cdf = np.concatenate([np.array((0,)), np.cumsum(normalized_env)])
    return cdf

def atime_log_p(wave_node, arrival_node, offset_node):
    atime = arrival_node.get_value()
    offset = offset_node.get_value()
    peak = atime + offset
    return peak_log_p(wave_node.cdf, wave_node.st, wave_node.srate, peak)

def peak_log_p(cdf, stime, srate, peak_time):
    # compute the probability that sample_peak_time_from_signal would
    # have sampled the current atime. formally this should be 0 for
    # all peak times that don't line up with an integer index, but we
    # just force an integer and hope that's okay.

    # we add one here since we created cdf with an initial
    # 0, so that the subtraction below works properly
    idx = np.round((peak_time - stime) * srate) + 1

    if (idx < 1) or (idx >= len(cdf)): return np.float('-inf')
    return np.log(cdf[idx] - cdf[idx-1])

def sample_peak_time_from_signal(cdf, stime, srate, return_lp=False):
    u = np.random.rand()
    idx = np.searchsorted(cdf, u)
    peak_time = stime + float(idx-1)/srate
    if return_lp:
        return peak_time, np.log(cdf[idx]-cdf[idx-1])
    return peak_time

def indep_peak_move(sg, arrival_node, offset_node, wave_node):
    current_atime = arrival_node.get_value()
    peak_offset = offset_node.get_value()
    proposed_peak_time = sample_peak_time_from_signal(wave_node.cdf, wave_node.st, wave_node.srate)
    proposed_arrival_time = proposed_peak_time - peak_offset
    return MH_accept(sg, oldvalues = (current_atime,),
                     newvalues = (proposed_arrival_time,),
                     node_list = (arrival_node,),
                     relevant_nodes = (arrival_node, wave_node))


######################################################################

def indep_offset_move(sg, arrival_node, offset_node, wave_node):
    current_offset = offset_node.get_value()
    atime = arrival_node.get_value()
    proposed_offset = np.random.rand() * 40
    new_atime = atime + (current_offset - proposed_offset)
    accepted = MH_accept(sg=sg, oldvalues=(atime, current_offset),
                         newvalues = (new_atime, proposed_offset),
                         node_list = (arrival_node, offset_node),
                         relevant_nodes=(arrival_node, offset_node, wave_node))
    return accepted

def improve_offset_move(sg, arrival_node, offset_node, wave_node, **kwargs):
    """
    Update the peak_offset while leaving the peak time constant, i.e.,
    adjust the arrival time to compensate for the change in offset.
    """
    current_offset = offset_node.get_value()
    atime = arrival_node.get_value()
    proposed_offset = gaussian_propose(sg, node_list=(offset_node,), values=(current_offset,), **kwargs)[0]
    new_atime = atime + (current_offset - proposed_offset)
    accepted = MH_accept(sg=sg, oldvalues=(atime, current_offset),
                         newvalues = (new_atime, proposed_offset),
                         node_list = (arrival_node, offset_node),
                         relevant_nodes=(arrival_node, offset_node, wave_node))
    return accepted

#######################################################################



def birth_move(sg, wave_node, **kwargs):
    lp_old = sg.current_log_p()

    peak_time, lp = sample_peak_time_from_signal(wave_node.cdf, wave_node.st, wave_node.srate, return_lp=True)
    tmpl = sg.create_unassociated_template(wave_node, peak_time, nosort=True, **kwargs)
    sg._topo_sorted_list = tmpl.values() + sg._topo_sorted_list
    sg._gc_topo_sorted_nodes()
    tmpl["arrival_time"].set_value(peak_time - tmpl["peak_offset"].get_value())


    lp_new = sg.current_log_p()

    # probability of this birth move is the product of probabilities
    # of all sampled params (including arrival time)
    log_qforward = lp
    for (key, node) in tmpl.items():
        if key == "arrival_time": continue
        log_qforward += node.log_p()

    # reverse (death) probability is just the probability of killing a
    # random template
    ntemplates = len([1 for (eid, phase) in wave_node.arrivals() if eid < 0])
    log_qbackward = np.log(1.0/ntemplates)

    u = np.random.rand()
    if (lp_new + log_qbackward) - (lp_old + log_qforward) > np.log(u):
        print "birth template %d: %.1f + %.1f - (%.1f + %.1f) = %.1f vs %.1f" % (tmpl["arrival_time"].unassociated_templateid, lp_new, log_qbackward, lp_old, log_qforward, (lp_new + log_qbackward) - (lp_old + log_qforward), np.log(u))
        return tmpl
    else:
        sg.destroy_unassociated_template(tmpl, nosort=True)

        # WARNING: this assumes the list hasn't been re-sorted by any
        # of our intermediate calls.
        sg._topo_sorted_list = sg._topo_sorted_list[len(tmpl):]
        sg._gc_topo_sorted_nodes()
        sg.next_uatemplateid -= 1
        #lp = sg.current_log_p()
        #assert(np.abs(lp - lp_old) < 1e-10)
        return False


def death_move(sg, wave_node):
    templates = [(eid, phase) for (eid, phase) in wave_node.arrivals() if eid < 0]
    u0 = np.random.rand()
    for i in range(len(templates)):
        if u0 <= float(i+1)/len(templates):
            tmpl_to_destroy = templates[i]
            break

    lp_old = sg.current_log_p()
    orig_topo_sorted = copy.copy(sg._topo_sorted_list)
    log_qforward = np.log(1.0/len(templates))

    tnodes = sg.get_template_nodes(eid=tmpl_to_destroy[0], phase=tmpl_to_destroy[1], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)
    wnodes = sg.get_wiggle_nodes(eid=tmpl_to_destroy[0], phase=tmpl_to_destroy[1], sta=wave_node.sta, band=wave_node.band, chan=wave_node.chan)

    log_qbackward = atime_log_p(wave_node, tnodes['arrival_time'][1], tnodes['peak_offset'][1])
    for (param, (label, node)) in tnodes.items():
        if param != "arrival_time":
            log_qbackward += node.log_p()
        sg.remove_node(node)
        sg._topo_sorted_list[node._topo_sorted_list_index] = None
    for (param, (label, node)) in wnodes.items():
        log_qbackward += node.log_p()
        sg.remove_node(node)
        sg._topo_sorted_list[node._topo_sorted_list_index] = None

    lp_new = sg.current_log_p()

    u = np.random.rand()
    if (lp_new + log_qbackward) - (lp_old + log_qforward) > np.log(u):
        print "death of template %d: %.1f + %.1f - (%.1f + %.1f) = %.1f vs %.1f" % (tnodes["arrival_time"][1].unassociated_templateid, lp_new, log_qbackward, lp_old, log_qforward, (lp_new + log_qbackward) - (lp_old + log_qforward), np.log(u))
        return True
    else:
        for (param, (label, node)) in tnodes.items() + wnodes.items():
            sg.add_node(node)
            node.addChild(wave_node)
        wave_node.arrivals()
        sg._topo_sorted_list = orig_topo_sorted
        sg._gc_topo_sorted_nodes()
        #lp = sg.current_log_p()
        #assert(np.abs(lp - lp_old) < 1e-10)
        return False


#####################################################################

def run_open_world_MH(sg, wn, burnin=0, skip=40, steps=10000, wiggles=False):
    n_accepted = dict()
    moves = ('birth', 'death', 'indep_peak', 'peak_offset', 'arrival_time', 'coda_height', 'coda_decay', 'wiggle_amp', 'wiggle_phase')
    for move in moves:
        n_accepted[move] = 0

    stds = {'peak_offset': .1, 'arrival_time': .1, 'coda_height': .02, 'coda_decay': 0.05, 'wiggle_amp': .25, 'wiggle_phase': .5}

    wave_env = wn.get_value() if wn.env else wn.get_wave().filter('env').data
    wn.cdf = preprocess_signal_for_sampling(wave_env)

    templates = dict()
    params_over_time = dict()

    for step in range(steps):
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

    wave = Waveform(data = np.load("sampled_wave.npy"),
                    srate=5.0, stime=1239915900.0,
                    sta="FIA3", chan="SHZ", arrivals=np.array(()),
                    filter_str="freq_2.0_3.0;env;hz_5.0")
    wn = sg.add_wave(wave)
    #sg.create_unassociated_template(wave_node=wn, atime=1239915940.253)
    #sg.create_unassociated_template(wave_node=wn, atime=1239915969.623)

    for fname in os.listdir('.'):
        if fname.startswith("unass_step") or fname.startswith("mcmc_unass"):
            os.remove(fname)

    np.random.seed(0)
    run_open_world_MH(sg, wn, wiggles=True)
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
