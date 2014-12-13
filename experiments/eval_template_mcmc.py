
import numpy as np
import sys
import os
import traceback


import pickle
import copy

from sigvisa import Sigvisa

from sigvisa.graph.sigvisa_graph import SigvisaGraph




from sigvisa.infer.run_mcmc import run_open_world_MH
from sigvisa.infer.mcmc_logger import MCMCLogger

from sigvisa.infer.template_mcmc import *
from sigvisa.plotting.plot import plot_with_fit, plot_with_fit_shapes, plot_pred_atimes
from sigvisa.signals.common import Waveform
from sigvisa.utils.fileutils import clear_directory, mkdir_p

from sigvisa.experiments.one_station_templates_test import sample_template


import time
from matplotlib.backends.backend_agg import FigureCanvasAgg

def do_plot(wn, ax=None):
    if ax is None:
      f = plt.figure(figsize=(15,5))
      ax=f.add_subplot(111)
    shape_colors = plot_with_fit_shapes(fname=None, wn=wn, axes=ax, plot_wave=True)
    atimes = dict([("%d_%s" % (eid, phase), wn.get_template_params_for_arrival(eid=eid, phase=phase)[0]['arrival_time']) for (eid, phase) in wn.arrivals()])
    colors = dict([("%d_%s" % (eid, phase), shape_colors[eid]) for (eid, phase) in wn.arrivals()])
    plot_pred_atimes(dict(atimes), wn.get_wave(), axes=ax, color=colors, alpha=1.0, bottom_rel=-0.1, top_rel=0.0)
    return ax

from sigvisa.results.mwmatching import maxWeightMatching


def match_templates(wn, gold_templates, max_delta_atime=10):

    edges = []
    n_gold = len(gold_templates)
    arrs = sorted(wn.arrivals())
    for (i, arr) in enumerate(arrs):
        guess_vals, _ = wn.get_template_params_for_arrival(*arr)
        atime_guess = guess_vals['arrival_time']
        amp_guess = guess_vals['coda_height']

        for j in range(n_gold):
            atime_gold = gold_templates[j]['arrival_time'].get_value()
            amp_gold = gold_templates[j]['coda_height'].get_value()

            if np.abs(atime_gold-atime_guess) < max_delta_atime:
                wt = -np.abs(atime_gold-atime_guess) - np.abs(amp_gold-amp_guess)
                edges.append((j, i+n_gold, wt))

    mat = maxWeightMatching(edges, maxcardinality=True)

    indices = []
    for i in range(len(arrs)):

        # maxWeightMatching only knows about templates that have edges
        # to other templates, so if 'mat' has fewer entries than the
        # total number of (gold+guess) templates, it's because the
        # last few guessed templates had no edges, thus no matches.
        if n_gold + i >= len(mat):
            break

        if mat[n_gold+i] >= 0:
            assert(mat[n_gold+i] < n_gold)
            indices.append((i, mat[n_gold+i]))
    return indices

def avg_dicts(ds):
    ds = [d for d in ds if isinstance(d, dict)]
    keys = ds[0].keys()
    n = float(len(ds))
    acc = dict([(k, 0.0) for k in keys])
    for d in ds:
        for k in keys:
            acc[k] += d[k]/n
    return acc

def analyze_template_matching(indices, wn, gold_templates):
    n_gold = len(gold_templates)
    arrs = sorted(wn.arrivals())
    n_guess = len(arrs)

    n_matches = float(len(indices))

    if n_gold > 0:
        recall = n_matches/n_gold
    else:
        recall = 1.0

    if n_guess > 0:
        precision = n_matches/n_guess
    else:
        precision = 1.0

    errs = []
    for (guess_idx, gold_idx) in indices:
        guess_vals, tg = wn.get_template_params_for_arrival(*arrs[guess_idx])

        err_dict = dict()
        for param in tg.params() + ('arrival_time',):
            guess_val = guess_vals[param]
            gold_val = gold_templates[gold_idx][param].get_value()
            err_dict[param] = np.abs(guess_val-gold_val)
        errs.append(err_dict)

    if len(errs) > 0:
        mean_errs = avg_dicts(errs)
    else:
        mean_errs = None
    return recall, precision, mean_errs, errs

def evaluate_inferred_templates(wn, gold_templates):
    indices = match_templates(wn, gold_templates)
    recall, precision, mean_errs, errs = analyze_template_matching(indices, wn, gold_templates)
    print "recall: %f, precision: %f\nmean absolute errors: %s" % (recall, precision, mean_errs)


def initialize_templates(sg, wn, tries = 20):
    successes = 0
    for i in range(tries):
        accepted = optimizing_birth_move(sg, wn)
        successes += 1 if accepted else 0
    print "initialized %d templates at %s" % (successes, wn.label)


def regen_run_page(out_dir, seeds=50):
    outfile = open(os.path.join(out_dir, "run.html"), 'w')
    outfile.write("<html><body>")

    total_recall = 0
    total_precision = 0
    all_mean_errs = []

    actual_seeds = 0
    for seed in range(seeds):

        try:
            with open(os.path.join(out_dir, "%d.txt" % seed), 'r') as f:
                results = f.read()
        except IOError:
            continue
        actual_seeds += 1

        outfile.write("<h2>seed %d</h2><pre>%s</pre><table><tr><td>True</td><td>Inferred</td></tr><tr><td><img src=\"%d_true.png\" width=600></td><td><img src=\"%d.png\" width=600></td></tr></table>\n\n" % (seed, results, seed, seed))

        result_lines = results.split("\n")

        recall = float(result_lines[4].split(" ")[1])
        precision = float(result_lines[5].split(" ")[1])
        total_recall += recall
        total_precision += precision

        errs = [eval(l) for l in result_lines[8:] if l.startswith("{")]
        mean_errs = avg_dicts(errs) if len(errs) > 0 else None
        all_mean_errs.append(mean_errs)

    outfile.write("<h2>Summary</h2><tt>mean recall: %f, mean precision: %f</tt><p>\n" % (total_recall/float(actual_seeds), total_precision/float(actual_seeds)))

    mean_mean_errs = avg_dicts(all_mean_errs)

    outfile.write("overall mean errs:<p><pre>%s</pre>\n" % mean_mean_errs)
    outfile.write("</body></html>")
    outfile.close()


def eval_template_move_combo(new_merge=True, old_merge=True,
                             new_birth=True, old_birth=True,
                             template_move_type="rw", seeds=50,
                             steps=1000, nm_type='l1', start_at=0,
                             smart_init=False):

    if not old_birth and not new_birth:
        print "can't do inference without a birth move!"
        return

    moves = {}
    run_attrs = [template_move_type, nm_type, str(steps)]
    if new_merge:
        moves['tmpl_merge_opt'] = merge_move
        moves['tmpl_split_opt'] = split_move
        run_attrs.append('newmerge')
    if old_merge:
        moves['tmpl_merge'] = merge_move_old
        moves['tmpl_split'] = split_move_old
        run_attrs.append('oldmerge')
    if new_birth:
        moves['tmpl_birth_opt'] = optimizing_birth_move
        moves['tmpl_death_opt'] = death_move_for_optimizing_birth
        run_attrs.append('newbirth')
    if old_birth:
        moves['tmpl_birth_old'] = birth_move
        moves['tmpl_death_old'] = death_move
        run_attrs.append('oldbirth')
    if smart_init:
        run_attrs.append('smartinit')

    s = Sigvisa()
    out_dir = os.path.join(s.homedir, 'experiments', 'template_mcmc', "_".join(run_attrs))
    mkdir_p(out_dir)

    for seed in range(start_at, seeds):
        wave, gold_templates, sg, wn = sample_template(seed=seed, hardcoded=False, nm_type=nm_type,
                                                  srate=1.0, rate=5e-3, return_graph=True)
        fig = Figure(figsize=(15, 5))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        do_plot(wn, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "%d_true.png" % seed))

        np.random.seed(seed)
        sg = SigvisaGraph(template_model_type="dummy", template_shape="lin_polyexp",
                              wiggle_model_type="dummy", wiggle_family="dummy",
                              phases="leb", nm_type = nm_type, wiggle_len_s = 60.0)
        wn = sg.add_wave(wave)
        sg.uatemplate_rate=5e-3

        tg = sg.template_generator('UA')
        tg.hack_force_mean = np.log(wn.nm.c * 10)

        if smart_init:
            initialize_templates(sg, wn)

        t0 = time.time()
        run_open_world_MH(sg, steps=steps, enable_event_openworld=False, enable_event_moves=False,
                          template_move_type=template_move_type, template_openworld_custom=moves,
                          logger=False)
        t1 = time.time()

        indices = match_templates(wn, gold_templates)
        recall, precision, mean_errs, errs = analyze_template_matching(indices, wn, gold_templates)


        fig = Figure(figsize=(15, 5))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        do_plot(wn, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "%d.png" % seed))

        results = "logp %f\ntime %f s\n%d true templates\n%d inferred\nrecall %f\nprecision %f\n\nerrs\n" % (sg.current_log_p(), t1-t0, len(gold_templates), len(wn.arrivals()), recall, precision)
        for err in errs:
            results += str(err) + "\n"

        with open(os.path.join(out_dir, "%d.txt" % seed), 'w') as f:
            f.write(results)

        regen_run_page(out_dir, seeds=seed+1)

def main():
    print sys.argv

    def tobool(s):
        return s.lower().startswith('t')

    if sys.argv[1] == "regen":
        regen_run_page(sys.argv[2])
        return

    eval_template_move_combo(new_merge=tobool(sys.argv[1]),
                             new_birth=tobool(sys.argv[2]),
                             old_merge = tobool(sys.argv[3]),
                             old_birth=tobool(sys.argv[4]),
                             template_move_type=sys.argv[5], seeds=int(sys.argv[6]),
                             steps=int(sys.argv[7]),
                             nm_type=sys.argv[8],
                             start_at=int(sys.argv[9]),
                             smart_init=tobool(sys.argv[10]))



if __name__ == "__main__":
    main()
