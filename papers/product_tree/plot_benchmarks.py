from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import os
import numpy as np
import re

class NoResultsError(Exception):
    pass

def extract_results(txt, prefix, require_times=True):
    for line in txt:

        if line.startswith(prefix):
            if (not require_times) or "times" in line:
                d = dict()
                for w in ("mean", "std", "min", "10th", "50th", "90th", "max"):
                    d[w] = float(re.search(r"%s ([\.\d]+)" % w, line).group(1))
                return d
    raise NoResultsError("could not find line with prefix %s" % prefix)

def read_results(benchmarks_dir):

    dense = dict()
    sparse = dict()
    hybrid = dict()
    tree = dict()

    for subdir in os.listdir(benchmarks_dir):
        (se, pts, n) = subdir.split("_")
        scale = float(pts[:4])
        n = int(n)

        try:
            with  open(os.path.join(benchmarks_dir, subdir, "results.txt"), 'r') as f:
                txt = f.readlines()
        except:
            print "skipping", subdir
            continue
        
        if scale not in dense:
            dense[scale] = dict()
            sparse[scale] = dict()
            hybrid[scale] = dict()
            tree[scale] = dict()

        try:
            dense[scale][n] = extract_results(txt, "dense covar")
        except NoResultsError:
            pass

        try:
            sparse[scale][n] = extract_results(txt, "sparse covar")
        except NoResultsError:
            pass

        try:
            hybrid[scale][n] = extract_results(txt, "sparse covar spkernel")
        except NoResultsError:
            pass

        try:
            tree[scale][n] = extract_results(txt, "best tree")
        except NoResultsError:
            pass
        
    return dense, sparse, hybrid, tree

markers = {0.25: "D", 1.0: "s", 5.0: "o"}
def plot_method(ax, arr, name, linestyle):
    for scale in arr.keys():
        keys, vals = zip(*arr[scale].items())
        keys = np.array(keys)
        perm = keys.argsort()
        keys = keys[perm]
        means = np.array([d['mean'] for d in vals])[perm] * 1000
        stds = np.array([d['std'] for d in vals])[perm] * 1000
        tens = np.array([d['10th'] for d in vals])[perm] * 1000
        nineties = np.array([d['10th'] for d in vals])[perm] * 1000

        errs = [means-tens, nineties-means]

        strscale = ".25" if scale==0.25 else "%.1f" % scale
        ax.plot(keys, means, marker=markers[scale], linestyle=linestyle, c='black', label="%s, $v=%s$" % (name, strscale))
#        ax.errorbar(keys/1000, means, yerr=(nineties-means), marker=marker, linestyle=linestyle[scale], c='black', label="%s ($v=%s$)" % (name, strscale), ecolor='r', elinewidth=5)


dense, sparse, hybrid, tree = read_results(os.path.join(os.getenv("SIGVISA_HOME"), "papers", "product_tree", "benchmarks"))
fig = Figure((12,9))
ax = fig.add_subplot(1,1,1)

plot_method(ax, dense, "Dense", linestyle="-")
plot_method(ax, sparse, "Sparse", linestyle="--")
plot_method(ax, hybrid, "Hybrid", linestyle="-.")
plot_method(ax, tree, "Tree", linestyle=":")
ax.set_xlabel("$n$ ", fontsize=20)
ax.set_ylabel("time (ms)", fontsize=20)

ax.set_ylim((0, 4))
ax.set_xlim((1000, 220000))
ax.set_xscale("log")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, handlelength=3, loc='center left', bbox_to_anchor=(1, 0.5))


canvas = FigureCanvasAgg(fig)
canvas.draw()
fig.savefig("synthetic_log.png", bbox_inches='tight')
