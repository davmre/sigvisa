import numpy as np
import sklearn.cluster


from optparse import OptionParser

from sigvisa import Sigvisa
from sigvisa.source.event import get_event


def cluster_evids(evids, 
                  target_cluster_size=50, 
                  depth_scaling=100):

    evs = [get_event(evid) for evid in evids]
    X = np.array([(ev.lon, ev.lat, ev.depth/depth_scaling) for ev in evs])

    n = X.shape[0]
    n_clusters = (n / target_cluster_size) * 2

    km = sklearn.cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, 
                                max_iter=300, tol=0.0001, precompute_distances='auto', 
                                verbose=0, random_state=None, copy_x=True, n_jobs=1)
    r = km.fit(X)

    assignments = r.predict(X)
    print "%d clusters of sizes %s" % (n_clusters, [np.sum(assignments==i) for i in range(n_clusters)])

    centers = r.cluster_centers_
    centers[:, 2] *= depth_scaling

    return centers

def main():
    
    parser = OptionParser()

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    parser.add_option("--evid_file", dest="evid_file", default=None, type="str", help="file containing evids to cluster")
    parser.add_option("--evid_runid", dest="evid_runid", default=None, type="int", help="cluster all evids with fits in this runid...")
    parser.add_option("--origin_type", dest="origin_type", default="isc", type="str", help="")
    parser.add_option("--origin_stime", dest="origin_stime", default=None, type="float", help="")
    parser.add_option("--origin_etime", dest="origin_etime", default=None, type="float", help="")
    parser.add_option("-o", "--outfile", dest="outfile", default=None, type="str", help="file to save cluster centers to")
    parser.add_option("--target_cluster_size", dest="target_cluster_size", default=50, type="int", help="set the number of clusters heuristically to target this cluster size")

    (options, args) = parser.parse_args()

    if options.evid_file is not None:
        evids = np.array([int(evid) for evid in np.loadtxt(options.evid_file)])
    elif options.evid_runid is not None:
        s = Sigvisa()
        r = s.sql("select evid from sigvisa_coda_fit where runid=%d" % options.evid_runid)
        evids = list(set([int(rr[0]) for rr in r]))
    else:
        s = Sigvisa()
        r = s.sql("select evid from %s_origin where time between %f and %f" % (options.origin_type, options.origin_stime, options.origin_etime))
        evids = list(set([int(rr[0]) for rr in r]))

    centers = cluster_evids(evids, target_cluster_size=options.target_cluster_size)
    np.savetxt(options.outfile, centers)
    

if __name__ == "__main__":
    main()
