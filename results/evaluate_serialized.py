
import numpy as np
from optparse import OptionParser

from sigvisa.results.compare import f1_and_error
from sigvisa.graph.serialization import load_serialized_from_file
from sigvisa import Sigvisa

def get_bulletin(stime, etime, origin_type="isc"):

    region_cond = "and lon between -126 and -100 and lat between 33 and 49"
    query = "select lon, lat, depth, mb, time from %s_origin where time between %f and %f and mb > 0 %s" % (origin_type, stime, etime, region_cond)
    s = Sigvisa()
    isc_bulletin = np.array(s.sql(query))
    
    return isc_bulletin


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

    (options, args) = parser.parse_args()
    
    assert(options.serialized is not None)
    assert(options.stime is not None)
    assert(options.etime is not None)

    isc_bulletin = get_bulletin(options.stime, options.etime, origin_type="isc")

    print "LEB"
    leb_bulletin = get_bulletin(options.stime, options.etime, origin_type="leb")
    f, p, r, err = f1_and_error(isc_bulletin, leb_bulletin, max_delta_deg=options.max_delta_deg)
    print "f1", f
    print "precision", p
    print "recall", r
    print "err", err
    print

    print "sel3"
    sel3_bulletin = get_bulletin(options.stime, options.etime, origin_type="sel3")
    f, p, r, err = f1_and_error(isc_bulletin, sel3_bulletin, max_delta_deg=options.max_delta_deg)
    print "f1", f
    print "precision", p
    print "recall", r
    print "err", err
    print

    print "inferred"
    evdicts, uadicts_by_sta = load_serialized_from_file(options.serialized)
    inferred_bulletin = np.array([(d["lon"], d["lat"], d["depth"], d["mb"], d["time"]) for d in evdicts])

    f, p, r, err = f1_and_error(isc_bulletin, inferred_bulletin, max_delta_deg=options.max_delta_deg)
    print "f1", f
    print "precision", p
    print "recall", r
    print "err", err

if __name__ == "__main__":
    main()
