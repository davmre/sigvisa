
import numpy as np
from optparse import OptionParser

from sigvisa.results.compare import f1_and_error
from sigvisa.graph.serialization import load_serialized_from_file
from sigvisa import Sigvisa

def true_bulletin(stime, etime):

    query = "select lon, lat, depth, mb, time from isc_origin where time between %f and %f and mb > 0" % (stime, etime)
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

    (options, args) = parser.parse_args()
    
    assert(options.serialized is not None)
    assert(options.stime is not None)
    assert(options.etime is not None)

    isc_bulletin = true_bulletin(options.stime, options.etime)
    evdicts, uadicts_by_sta = load_serialized_from_file(options.serialized)

    inferred_bulletin = [(d["lon"], d["lat"], d["depth"], d["mb"], d["time"]) for d in evdicts]

    f, p, r, err = f1_and_error(isc_bulletin, inferred_bulletin)
    print "f1", f
    print "precision", p
    print "recall", r
    print "err", err

if __name__ == "__main__":
    main()
