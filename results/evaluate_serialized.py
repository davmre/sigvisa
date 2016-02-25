
import numpy as np
from optparse import OptionParser

from sigvisa.results.compare import f1_and_error, find_matched, find_unmatched, find_matching
from sigvisa.graph.serialization import load_serialized_from_file
from sigvisa import Sigvisa
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.graph.region import Region

def get_bulletin(stime, etime, origin_type="isc"):

    region_cond = "and lon between -126 and -100 and lat between 33 and 49"
    query = "select lon, lat, depth, time, mb from %s_origin where time between %f and %f and mb > 0 %s order by time" % (origin_type, stime, etime, region_cond)
    s = Sigvisa()
    isc_bulletin = np.array(s.sql(query))
    
    return isc_bulletin

def fmt_ev(evarr):
    lon, lat, depth, time, mb = evarr
    return "%6.3f %6.3f %4.1f %.1f %3.1f " % (lon, lat, depth, time, mb)

def print_bulletin(bulletin):
    sb = sorted(bulletin, key = lambda b : b[4])
    for (lon, lat, depth, time, mb) in sb:
        print "%6.3f %6.3f %4.1f %.1f %3.1f " % (lon, lat, depth, time, mb)

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
    
    assert(options.serialized is not None)
    assert(options.stime is not None)
    assert(options.etime is not None)

    print "true bulletin"
    isc_bulletin = get_bulletin(options.stime, options.etime, origin_type="isc")
    print_bulletin(isc_bulletin)
    print

    print "LEB"
    leb_bulletin = get_bulletin(options.stime, options.etime, origin_type="leb")
    f, p, r, err = f1_and_error(isc_bulletin, leb_bulletin, max_delta_deg=options.max_delta_deg)
    print_bulletin(leb_bulletin)
    print "f1", f
    print "precision", p
    print "recall", r
    print "err", err
    print

    print "sel3"
    sel3_bulletin = get_bulletin(options.stime, options.etime, origin_type="sel3")
    f, p, r, err = f1_and_error(isc_bulletin, sel3_bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=50.0)
    print "f1", f
    print "precision", p
    print "recall", r
    print "err", err
    print

    evdicts, uadicts_by_sta = load_serialized_from_file(options.serialized)
    inferred_bulletin = np.array([(d["lon"], d["lat"], d["depth"], d["time"], d["mb"]) for d in evdicts])
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

    print "inferred"
    f, p, r, err = f1_and_error(isc_bulletin, inferred_bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=50.0)
    print "f1", f
    print "precision", p
    print "recall", r
    print "err", err
    print

    

    indices = find_matching(isc_bulletin, inferred_bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=50.0)
    for (i,j) in indices:
        print "matched isc", fmt_ev(isc_bulletin[i])
        print "with inferred", fmt_ev(inferred_bulletin[j])



if __name__ == "__main__":
    main()
