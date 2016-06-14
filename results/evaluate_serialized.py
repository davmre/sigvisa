
import numpy as np
from optparse import OptionParser

from sigvisa.results.compare import f1_and_error, find_matched, find_unmatched, find_matching
from sigvisa.graph.serialization import load_serialized_from_file
from sigvisa import Sigvisa
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.graph.region import Region

def get_bulletin(stime, etime, origin_type="isc"):

    region_cond = "and lon between -126 and -100 and lat between 33 and 49"
    query = "select lon, lat, depth, time, mb from %s_origin where time between %f and %f %s order by time" % (origin_type, stime, etime, region_cond)
    s = Sigvisa()
    isc_bulletin = np.array(s.sql(query))
    
    return isc_bulletin

def fmt_ev(evarr):
    lon, lat, depth, time, mb = evarr[:5]
    return "%6.3f %6.3f %4.1f %.1f %3.1f " % (lon, lat, depth, time, mb)

def print_bulletin(bulletin):
    sb = sorted(bulletin, key = lambda b : b[3])
    for evinfo in sb:
        (lon, lat, depth, time, mb) = evinfo[:5]
        print "%6.3f %6.3f %4.1f %.1f %3.1f " % (lon, lat, depth, time, mb)


def print_bulletin_matching(bulletin, ground_bulletin, indices):
    matching_map = dict([(j, i) for (i,j) in indices])
    #sb = sorted(bulletin, key = lambda b : b[3])
    for j, evinfo in enumerate(bulletin):
        (lon, lat, depth, time, mb) = evinfo[:5]
        print "%6.3f %6.3f %4.1f %.1f %3.1f " % (lon, lat, depth, time, mb),
        if j in matching_map:
            print " matches %s" % fmt_ev(ground_bulletin[matching_map[j]])
        else:
            print

def simulate_prior(nevs, stime, etime):
    region = Region(lons=(-126.0, 100.0), lats=(33.0, 49.0), times=(stime, etime))
    sg = SigvisaGraph(inference_region=region)
    prior_evs = sg.prior_sample_events(n_events=nevs, stime=stime, etime=etime)
    prior_bulletin = np.array([(ev.lon, ev.lat, ev.depth, ev.time, ev.mb) for ev in prior_evs])
    return prior_bulletin



def main():
    parser = OptionParser()

    parser.add_option("--serialized", dest="serialized", default=None, type="str",
                      help="tgz file with serialized events")
    parser.add_option("--stime", dest="stime", default=None, type="float",
                      help="")
    parser.add_option("--etime", dest="etime", default=None, type="float",
                      help="")
    parser.add_option("--max_delta_deg", dest="max_delta_deg", default=1.0, type="float",
                      help="")
    parser.add_option("--max_delta_time", dest="max_delta_time", default=50.0, type="float",
                      help="")

    (options, args) = parser.parse_args()
    
    assert(options.stime is not None)
    assert(options.etime is not None)

    isc_bulletin = get_bulletin(options.stime, options.etime, origin_type="isc")
    print "true (ISC) bulletin: %d events" % (len(isc_bulletin))
    print_bulletin(isc_bulletin)
    print

    leb_bulletin = get_bulletin(options.stime, options.etime, origin_type="leb")
    print "LEB: %d events" % (len(leb_bulletin),)
    f, p, r, err = f1_and_error(isc_bulletin, leb_bulletin, max_delta_deg=options.max_delta_deg)
    indices = find_matching(isc_bulletin, leb_bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=options.max_delta_time)
    print_bulletin_matching(leb_bulletin, isc_bulletin, indices)
    print "f1", f
    print "precision", p
    print "recall", r
    print "mean location err %.1fkm" % err[0]
    print


    sel3_bulletin = get_bulletin(options.stime, options.etime, origin_type="sel3")
    print "sel3: %d events" % (len(sel3_bulletin),)
    f, p, r, err = f1_and_error(isc_bulletin, sel3_bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=options.max_delta_time)
    indices = find_matching(isc_bulletin, sel3_bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=options.max_delta_time)
    print_bulletin_matching(sel3_bulletin, isc_bulletin, indices)
    print "f1", f
    print "precision", p
    print "recall", r
    print "mean location err %.1fkm" % err[0]
    print


    visa_bulletin = get_bulletin(options.stime, options.etime, origin_type="visa")
    print "netvisa: %d events" % len(visa_bulletin)
    f, p, r, err = f1_and_error(isc_bulletin, visa_bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=options.max_delta_time)
    indices = find_matching(isc_bulletin, visa_bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=options.max_delta_time)
    print_bulletin_matching(visa_bulletin, isc_bulletin, indices)
    print "f1", f
    print "precision", p
    print "recall", r
    print "mean location err %.1fkm" % err[0]
    print

    if options.serialized is None:
        return

    evdicts, uadicts_by_sta = load_serialized_from_file(options.serialized)
    inferred_bulletin = np.array(sorted([(d["lon"], d["lat"], d["depth"], d["time"], d["mb"], d['score'] if 'score' in d else 0.0) for d in evdicts], key = lambda x: x[3]))
    nevs = len(inferred_bulletin)

    """
    print "prior (10 simulations of %d events)" % nevs
    fs = []
    ps = []
    rs = []
    errs = []
    for i in range(10):
        prior_bulletin = simulate_prior(nevs, options.stime, options.etime)
        f, p, r, err = f1_and_error(isc_bulletin, prior_bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=50.0)
        fs.append("%.3f" % f)
        ps.append("%.3f" % p)
        rs.append("%.3f" % r)
        errs.append("(%.1f %.1f)" % err)
    print "f1", fs
    print "precision", ps
    print "recall", rs
    print "err", errs
    print
    """

    print "sigvisa inferred %d events" % (nevs,)
    indices = find_matching(isc_bulletin, inferred_bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=options.max_delta_time)
    print_bulletin_matching(inferred_bulletin, isc_bulletin, indices)
    f, p, r, err = f1_and_error(isc_bulletin, inferred_bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=options.max_delta_time)
    print "f1", f
    print "precision", p
    print "recall", r
    print "mean location err %.1fkm" % err[0]
    print

    precs, recalls = precision_recall_curve(inferred_bulletin, isc_bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=options.max_delta_time)
    np.savetxt("sigvisa_precisions.txt", precs)
    np.savetxt("sigvisa_recalls.txt", recalls)
    plot_precision_recall("prec_recall.png", precs, recalls)


def plot_precision_recall(fname, precs, recalls):

    from matplotlib.figure import Figure
    from sigvisa.plotting.plot import savefig

    fig = Figure(figsize=(8, 5), dpi=300)
    axes = fig.add_subplot(111)

    axes.plot(precs, recalls)
    axes.set_xlabel("Precision")
    axes.set_ylabel("Recall")
    axes.set_xlim((0, 100))
    axes.set_ylim((0, 100))

    savefig(fname, fig, bbox_inches="tight", dpi=300)
    print "saved plot to", fname

def precision_recall_curve(guess, gold, **kwargs):
    guess_by_score = np.asarray(sorted(guess, key = lambda x : -x[5]))
    precs = []
    recalls = []

    # for each prefix of the high scoring events
    for i in range(len(guess)):
        guess_thresholded = guess_by_score[:i]
        f, p, r, err = f1_and_error(gold, guess_thresholded, **kwargs)
        precs.append(p)
        recalls.append(r)

        if i % 50 == 0:
            print "%d events precision %.1f recall %.1f" % (i, p, r)

    return precs, recalls

if __name__ == "__main__":
    main()
