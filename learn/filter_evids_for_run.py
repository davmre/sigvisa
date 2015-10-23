import os
import errno
import sys
import time
import traceback
import numpy as np
import scipy

from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *
from sigvisa.database import db

from optparse import OptionParser

from sigvisa import *
from sigvisa.utils.interaction import query_yes_no
from sigvisa.source.event import Event
from sigvisa.models.ttime import tt_predict

def has_signal_data_for_event(cursor, sta, evid, chans=None):

    ev = Event(evid=evid)
    atime = tt_predict(ev, sta, 'P') + ev.time
    stime = atime - 60
    etime = atime + 1200

    return has_signal_data_for_window(cursor, sta, stime, etime, chans)

def main():
    parser = OptionParser()

    parser.add_option("-s", "--stations", dest="stations", default=None, type="str",
                      help="comma-separated list of station names for which to fit templates")
    parser.add_option("-o", dest="output", default=None, type="str",
                      help="write fitting commands to a file (e.g. for later execution using GNU parallel) rather than executing them directly")
    parser.add_option(
        "--min_mb", dest="min_mb", default=0.0, type="float", help="exclude all events with mb less than this value (0)")
    parser.add_option(
        "--max_mb", dest="max_mb", default=10.0, type="float", help="exclude all events with mb greater than this value (10)")
    parser.add_option(
        "--min_snr", dest="min_snr", default=5, type="float", help="exclude all events with snr less than this value (0)")
    parser.add_option("--max_snr", dest="max_snr", default=float('inf'), type="float",
                      help="exclude all events with snr greater than this value (inf)")
    parser.add_option("--all_combinations", dest="all_combinations", action="store_true", default=False,
                      help="output all pairs of events and stations, even those where LEB does not register a detection")
    parser.add_option("--dataset", dest="dataset", default="training", type="str",
                      help="use the start and end time of the given dataset (training)")
    parser.add_option("--start_time", dest="start_time", default=None, type="float",
                      help="exclude all events with time less than this value (0)")
    parser.add_option("--end_time", dest="end_time", default=None, type="float",
                      help="exclude all events with time greater than this value (1237680000)")
    parser.add_option("--require_phases", dest="require_phases", default='P', type="str",
                      help="exclude all ev/sta pairs not having arrivals for *all* phases in the specificed (comma-separated) list (P)")
    parser.add_option("--only_phases", dest="only_phases", default=None, type="str",
                      help="exclude all ev/sta pairs having arrivals for any phase *not* in the specified list (None)")
    parser.add_option("--array_refsta_only", dest="array_refsta_only", default=False, action="store_true",
                      help="don't explode array stations into their individual elements (False)")


    (options, args) = parser.parse_args()

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    if options.start_time is None or options.end_time is None:
        (st, et) = read_timerange(cursor, options.dataset, hours=None, skip=0)

    st = options.start_time if options.start_time is not None else st
    et = options.end_time if options.end_time is not None else et

    print st, et

    required_phase_list = options.require_phases.split(',') if options.only_phases else []
    only_phase_list = options.only_phases.split(',') if options.only_phases else []
    if options.output is None:
        raise Exception("must specify an output file")

    print "writing to", options.output, '...'
    with open(options.output, 'w') as f:
        evids = dict()
        for sta in options.stations.split(','):
            # want to select all events, with certain properties, which have a P or S phase detected at this station
            evids[sta] = read_evids_detected_at_station(s.dbconn, sta, st, et, phases=required_phase_list,
                                                        min_mb=options.min_mb, max_mb=options.max_mb,
                                                        min_snr=options.min_snr, max_snr=options.max_snr,
                                                        only_phases=only_phase_list, time_filter_direct=True)

        if options.all_combinations:
            all_evids = set()
            for sta in evids.keys():
                all_evids.update(set(evids[sta]))
            for sta in evids.keys():
                if s.is_array_station(sta) and not options.array_refsta_only:
                    elements = [el for el in s.get_array_elements(sta) if has_signal_data_for_event(cursor, el, evid)]
                else:
                    elements = [sta,]
                for element in elements:
                    for evid in all_evids:
                        f.write('%s %d\n' % (element, evid))
        else:
            for sta in evids.keys():
                if s.is_array_station(sta) and not options.array_refsta_only:
                    elements = [el for el in s.get_array_elements(sta) if has_signal_data_for_event(cursor, el, evid)]
                else:
                    elements = [sta,]
                for element in elements:
                    for evid in evids[sta]:
                        f.write('%s %d\n' % (element, evid))
            
    print "done."


if __name__ == "__main__":
    main()
