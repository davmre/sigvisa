import database.db
from database.dataset import *
import time
import learn
import sys
import numpy as np

from optparse import OptionParser

from priors.coda_decay.coda_decay_common import *


def print_event_set(cursor, evids, siteids, stas, sites):

    print "evid\tmb\tdepth\tsta\tdistance\tphase\tttime\tamp\tsnr"
    for evid in evids:
        sta_cond = "(" + " or ".join(["l.sta='%s'" % sta for sta in stas]) + ")"
        sql_query = "select lebo.lon, lebo.lat, lebo.depth, lebo.mb, leba.phase, l.sta, l.amp, lebo.time, l.time, l.snr from leb_origin lebo, leb_assoc leba, leb_arrival l where lebo.evid=%d and %s and lebo.orid=leba.orid and leba.arid=l.arid" % (evid, sta_cond)
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        for r in rows:
            siteid = siteids[stas.index(r[5])]
            sll = sites[siteid-1][0:2]
            dist = utils.geog.dist_km(r[0:2], sll)
            ttime = r[8] - r[7]
            print "%d\t%.2f\t%.1f\t%s\t%.1f\t\t%s\t%.2f\t%.2f\t%.2f" % (evid, r[3], r[2], r[5], dist, r[4], ttime, r[6], r[9])


def main():

    parser = OptionParser()

    parser.add_option("-s", "--siteids", dest="siteids", default=None, type="str", help="comma-separated list of station siteids (default is no stations)")
    parser.add_option("-r", "--runids", dest="runids", default=None, type="str", help="only consider arrivals having fits from this set of runids (default is to consider all arrivals)")
    parser.add_option("--min_mb", dest="min_mb", default=5, type="float", help="exclude all events with mb less than this value (5)")
    parser.add_option("--max_mb", dest="max_mb", default=10, type="float", help="exclude all events with mb greater than this value (10)")
    parser.add_option("--start_time", dest="start_time", default=0, type="float", help="exclude all events with time less than this value (0)")
    parser.add_option("--end_time", dest="end_time", default=1237680000, type="float", help="exclude all events with time greater than this value (1237680000)")


    (options, args) = parser.parse_args()

    if options.siteids is not None:
        siteids = [int(s) for s in options.siteids.split(',')]
    else:
        siteids = []


    # first want  events which are detected at all stations - print the evid, magnitude, station list (all phases at each station, each with amp/snr)

    cursor = db.connect().cursor()
    sites = read_sites(cursor)
    """

    stations_cond = " and ".join(["exists (select ll.arid from leb_arrival ll, leb_origin llebo, leb_assoc lleba %s, static_siteid lsid where llebo.evid=lebo.evid and llebo.orid=lleba.orid and ll.arid=lleba.arid and ll.sta=lsid.sta and lsid.id=%d %s)" % (runids_source, siteid, runids_cond) for siteid in siteids])



    sql_query = "select lebo.evid, lebo.mb from leb_origin lebo where lebo.mb > %f and lebo.mb < %f and lebo.time > %f and lebo.time < %f and %s" % (options.min_mb, options.max_mb, options.start_time, options.end_time, stations_cond)

    print sql_query
    cursor.execute(sql_query)
    a = cursor.fetchall()
    print a
    """

    if options.runids is not None:
        runids = [int(s) for s in options.runids.split(',')]
        runids_source = ", sigvisa_coda_fits fit"
        runids_cond = "and fit.arid = leba.arid and (" + " or ".join(["fit.runid = %d" % runid for runid in runids]) +  ")"
    else:
        runids = []
        runids_source = ""
        runids_cond = ""


    evids = dict()
    stas = []
    overlap = None
    for siteid in siteids:
        sta = siteid_to_sta(siteid, cursor)
        stas.append(sta)
        sql_query = "select distinct lebo.evid from leb_origin lebo, leb_assoc leba %s where lebo.mb > %f and lebo.mb < %f and lebo.time > %f and lebo.time < %f and lebo.orid=leba.orid and leba.sta='%s' %s" % (runids_source, options.min_mb, options.max_mb, options.start_time, options.end_time, sta, runids_cond)
        cursor.execute(sql_query)
        evids[sta] = set([r[0] for r in cursor.fetchall()])

        overlap = evids[sta] if overlap is None else overlap.intersection(evids[sta])

    print "events detected at all stations"
    print overlap
#    print_event_set(cursor, overlap, siteids, stas, sites)
    print


    overlaps_except = dict()
    for sta in stas:
        overlaps_except[sta] = None
        for sta2 in stas:
            if sta2 != sta:
                overlaps_except[sta] = evids[sta2] if overlaps_except[sta] is None else overlaps_except[sta].intersection(evids[sta2])
        print "events detected everywhere except %s:" % sta
        print overlaps_except[sta]
        print_event_set(cursor, overlaps_except[sta], siteids, stas, sites)


if __name__ == "__main__":
    main()
