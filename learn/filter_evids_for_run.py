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


def main():
    parser = OptionParser()

    parser.add_option("-s", "--stations", dest="stations", default=None, type="str",
                      help="comma-separated list of station names for which to fit templates")
    parser.add_option("-o", dest="output", default=None, type="str",
                      help="write fitting commands to a file (e.g. for later execution using GNU parallel) rather than executing them directly")
    parser.add_option(
        "--min_mb", dest="min_mb", default=5, type="float", help="exclude all events with mb less than this value (0)")
    parser.add_option(
        "--max_mb", dest="max_mb", default=10, type="float", help="exclude all events with mb greater than this value (10)")
    parser.add_option(
        "--min_snr", dest="min_snr", default=5, type="float", help="exclude all events with snr less than this value (0)")
    parser.add_option("--max_snr", dest="max_snr", default=float('inf'), type="float",
                      help="exclude all events with snr greater than this value (inf)")
    parser.add_option("--dataset", dest="dataset", default="training", type="str",
                      help="use the start and end time of the given dataset (training)")
    parser.add_option("--start_time", dest="start_time", default=None, type="float",
                      help="exclude all events with time less than this value (0)")
    parser.add_option("--end_time", dest="end_time", default=None, type="float",
                      help="exclude all events with time greater than this value (1237680000)")

    (options, args) = parser.parse_args()

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    if options.start_time is None or options.end_time is None:
        (st, et) = read_timerange(cursor, options.dataset, hours=None, skip=0)

    st = options.start_time if options.start_time is not None else st
    et = options.end_time if options.end_time is not None else et

    if options.output is None:
        raise Exception("must specify an output file")

    print "writing to", options.output, '...'
    with open(options.output, 'w') as f:
        for sta in options.stations.split(','):
            # want to select all events, with certain properties, which have a P or S phase detected at this station
            evids = read_evids_detected_at_station(cursor, sta, st, et, s.P_phases + s.S_phases, min_mb=options.min_mb,
                                                   max_mb=options.max_mb, min_snr=options.min_snr, max_snr=options.max_snr)
            for evid in evids:
                f.write('%s %d\n' % (sta, evid))

    print "done."

if __name__ == "__main__":
    main()
