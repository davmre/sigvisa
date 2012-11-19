import database.db
from database.dataset import *
import utils.geog
import sys
import itertools
import time, calendar
import parsedatetime.parsedatetime as pdt
from optparse import OptionParser

from sigvisa import *

def main():
    parser = OptionParser()

    parser.add_option("-s", "--sta", dest="sta", default=None, type="str", help="name of station for which to fit templates")
    parser.add_option("-e", "--evid_file", dest="evid_file", default=None, type="int", help="event ID")
    parser.add_option("-o", "--outfile", dest="outfile", default="", type="str", help="save potential doublets to this file")
    parser.add_option("-p", "--phases", dest="phases", default="", type="str", help="comma-separated list of phases to require")
    parser.add_option("--min_mb", dest="min_mb", default=5, type="float", help="exclude all events with mb less than this value (0)")
    parser.add_option("--max_mb", dest="max_mb", default=10, type="float", help="exclude all events with mb greater than this value (10)")
    parser.add_option("--start_time", dest="start_time", default=None, type="str", help="exclude all events with time less than this value")
    parser.add_option("--end_time", dest="end_time", default=None, type="str", help="exclude all events with time greater than this value")

    (options, args) = parser.parse_args()

    s = Sigvisa()

    if options.sta is None:
        raise Exception("must specify a station name (use -s)!")
    else:
        sta = options.sta

    if not options.evid_file:

        p = pdt.Calendar()
        st = p.parse(options.start_time)[0]
        et = p.parse(options.end_time)[0]

        print "start time:",  time.asctime(st)
        print "end time:", time.asctime(et)
        st = calendar.timegm(st)
        et = calendar.timegm(et)

        phases = [ p for p in options.phases.split(',') if p != ""]
        evids = read_evids_detected_at_station(s.cursor, sta, st, et, phases, min_mb = 0, max_mb = 99999)
    else:
        f = open(options.evid_file, 'r')
        evids = [int(line) for line in f]
        f.close()

    print "loading", len(evids), "events..."
    events = [read_event(s.cursor, evid=evid) for evid in evids]
    print "---------------------------------------------------------"



    evpairs = itertools.combinations(events, 2)
    p = [(utils.geog.dist_km((e1[EV_LON_COL], e1[EV_LAT_COL]), (e2[EV_LON_COL], e2[EV_LAT_COL])), e1[EV_DEPTH_COL], e2[EV_DEPTH_COL], e1[EV_EVID_COL], e2[EV_EVID_COL], e1[EV_ORID_COL], e2[EV_ORID_COL]) for (e1, e2) in evpairs if e1[3]!=e2[3]]

    # filter for pairs with short distances and similar depths
    p = [pair for pair in p if pair[0] < 25 and abs(pair[2]-pair[1]) <= 40]
    p.sort()

    for pair in p:
        print "evids %d, %d are at distance %.3fkm and have depths (%.3fkm, %.3fkm) (orid %d %d)" % (pair[3], pair[4], pair[0], pair[1], pair[2], pair[5], pair[6])

    if options.outfile:
        f = open(options.outfile, 'w')
        for pair in p:
            f.write("%d, %d, %.3f\n" % (pair[3], pair[4], pair[0]))
        f.close()
        print "wrote pairs to", options.outfile



if __name__ == "__main__":
    main()
