import database.db
from database.dataset import *
import time
import learn
import sys
import numpy as np

from collections import defaultdict

from optparse import OptionParser

from priors.coda_decay.coda_decay_common import *


def print_event_set(cursor, evids, siteids, stas, sites, runids_source, runids_cond, exclude_sta=""):

    print "evid\tmb\tdepth\tsta\tdistance\tphase\tttime\tamp\tsnr"
    for evid in evids:
        sta_cond = "(" + " or ".join(["l.sta='%s'" % sta for sta in stas]) + ")"
        sql_query = "select distinct lebo.lon, lebo.lat, lebo.depth, lebo.mb, leba.phase, l.sta, l.amp, lebo.time, l.time, l.snr from leb_origin lebo, leb_assoc leba, leb_arrival l %s where lebo.evid=%d and %s and lebo.orid=leba.orid and leba.arid=l.arid %s" % (runids_source, evid, sta_cond, runids_cond)
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        for r in rows:
            if r[5] == exclude_sta:
                import pdb
                pdb.set_trace()
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

    cursor, sigmodel, earthmodel, sites, dbconn = sigvisa_util.init_sigmodel()

    if options.siteids is not None:
        siteids = [int(s) for s in options.siteids.split(',')]
    else:
        siteids = []
    
    if options.runids is not None:
        runids = [int(s) for s in options.runids.split(',')]
        runids_source = ", sigvisa_coda_fits fit"
        runids_cond = "and fit.arid = leba.arid and (" + " or ".join(["fit.runid = %d" % runid for runid in runids]) +  ")"
    else:
        runids = []
        runids_source = ""
        runids_cond = ""

    # first find events detected at all stations
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
    print_event_set(cursor, overlap, siteids, stas, sites, runids_source, runids_cond)
    print

    # next, find events which are missed by one station

    # the first step is to find events that each station could
    # plausibly have detected (i.e., events for which we know the
    # station has waveform data)


    all_evids = reduce(lambda a,b : a.union(b), evids.values())

   
    for sta in stas:
        siteid = siteids[stas.index(sta)]
        potential_examples = all_evids

        for sta2 in stas:
            if sta2 != sta:
                potential_examples= potential_examples.intersection(evids[sta2])


        print "checking %d possible examples at sta %s" % (len(potential_examples), sta)
        confirmed_examples = [evid for evid in potential_examples if sigvisa_util.has_trace(cursor, siteid=siteid, sta=sta, evid=evid, earthmodel=earthmodel)]
                
        print "events detected everywhere except %s:" % sta
        print_event_set(cursor, confirmed_examples, siteids, stas, sites, runids_source, runids_cond, exclude_sta=sta)


if __name__ == "__main__":
    main()
