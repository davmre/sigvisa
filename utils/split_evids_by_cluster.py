import numpy as np


from optparse import OptionParser

from sigvisa.treegp.gp import GP, GPCov

from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.treegp.cover_tree import VectorTree
import pyublas


def main():
    
    parser = OptionParser()

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    parser.add_option("--evid_file", dest="evid_file", default=None, type="str", help="file containing evids to cluster")
    parser.add_option("--clusters_file", dest="clusters_file", default=None, type="str", help="file containing cluster centers")
    parser.add_option("--out_prefix", dest="out_prefix", default="cluster_evids", type="str", help="prefix for output evid files")
    parser.add_option("--dummy", dest="dummy", default=False, action="store_true", help="don't actually write any files, just print cluster sizes")

    (options, args) = parser.parse_args()

    evids = np.array([int(evid) for evid in np.loadtxt(options.evid_file)])
    evs = [get_event(evid) for evid in evids]
    X = np.array([(ev.lon, ev.lat, ev.depth) for ev in evs])

    cluster_centers = np.loadtxt(options.clusters_file)

    cluster_metric = GPCov(wfn_str="se", dfn_str="lld", dfn_params=(1.0, 1.0), wfn_params=(1.0,))
    cluster_tree = VectorTree(cluster_centers, 1, *cluster_metric.tree_params())

    cluster_distances = cluster_tree.kernel_matrix(pyublas.why_not(X), 
                                                        pyublas.why_not(cluster_centers), True)

    cluster_assignments = np.argmin(cluster_distances, axis=1)
    
    n_clusters = len(cluster_centers)
    for i in range(n_clusters):
        idxs = cluster_assignments == i
        cluster_evids = evids[idxs]
        out_fname = options.out_prefix + "_%03d" % i
        distances= cluster_distances[idxs, i]
        mind, maxd, meand = np.min(distances), np.max(distances), np.mean(distances)
        if not options.dummy:
            np.savetxt(out_fname, cluster_evids, fmt="%d")
        print "wrote", len(cluster_evids), "events to", out_fname, "distance to center min %.1f max %.1f mean %.f" % (mind, maxd, meand)

    
if __name__ == "__main__":
    main()
