import sigvisa.database.db
from sigvisa.database.dataset import *
import sigvisa.utils.geog
import sys
import itertools
import time
import calendar

try:
    import parsedatetime.parsedatetime as pdt
except:
    import parsedatetime as pdt

from optparse import OptionParser

from sigvisa.source.event import get_event
from sigvisa import *

from sigvisa.explore.doublets.xcorr_pairs import extract_phase_window


def get_first_arrivals(events, sta):
    s = Sigvisa()
    cursor = s.dbconn.cursor()

    arriving_events = []
    evids = []
    first_arriving_times = []
    first_arriving_phases = []
    for event in events:
        dets = read_event_detections(cursor, event.evid, stations=[sta])
        phase = ""
        i = 0
        for det in dets:
            phaseid_minus_1 = det[DET_PHASE_COL]
            phase = s.phasenames[phaseid_minus_1]
            if phase in s.P_phases:
                break
        if phase not in s.P_phases:
            continue

        arriving_events.append(event)
        evids.append(event.evid)
        first_arriving_phases.append(phase)
        first_arriving_times.append(det[DET_TIME_COL])
    arrival_dict = dict(zip(evids, zip(first_arriving_times, first_arriving_phases)))

    return arriving_events, arrival_dict


def main():
    parser = OptionParser()

    parser.add_option("-s", "--sta", dest="sta", default=None, type="str", help="name of station for which to fit templates")
    parser.add_option("-c", "--chan", dest="chan", default="BHZ", type="str", help="name of channel for which to fit templates")
    parser.add_option("-e", "--evid_file", dest="evid_file", default=None, type="int", help="event ID")
    parser.add_option("-o", "--outfile", dest="outfile", default="", type="str", help="save potential doublets to this file")
    parser.add_option("-p", "--phases", dest="phases", default="", type="str", help="comma-separated list of phases to require")
    parser.add_option(
        "--min_mb", dest="min_mb", default=1, type="float", help="exclude all events with mb less than this value (0)")
    parser.add_option(
        "--max_mb", dest="max_mb", default=99, type="float", help="exclude all events with mb greater than this value (10)")
    parser.add_option(
        "--start_time", dest="start_time", default=None, type="str", help="exclude all events with time less than this value")
    parser.add_option(
        "--end_time", dest="end_time", default=None, type="str", help="exclude all events with time greater than this value")

    (options, args) = parser.parse_args()

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    if options.sta is None:
        raise Exception("must specify a station name (use -s)!")
    else:
        sta = options.sta

    chan = options.chan

    if not options.evid_file:

        p = pdt.Calendar()

        try:
            st = float(options.start_time)
            et = float(options.end_time)

        except:

            st = p.parse(options.start_time)[0]
            et = p.parse(options.end_time)[0]

            print "start time:", time.asctime(st)
            print "end time:", time.asctime(et)
            st = calendar.timegm(st)
            et = calendar.timegm(et)

        phases = [p for p in options.phases.split(',') if p != ""]
        evids = read_evids_detected_at_station(cursor, sta, st, et, phases, min_mb=options.min_mb, max_mb=options.max_mb)
    else:
        f = open(options.evid_file, 'r')
        evids = [int(line) for line in f]
        f.close()

    print "loading", len(evids), "events..."
    events = [get_event(evid) for evid in evids]
    print "---------------------------------------------------------"

    arriving_events, arrival_dict = get_first_arrivals(events, sta)

    for ev in arriving_events:
        atime, phase = arrival_dict[ev.evid]
        d = extract_phase_window(sta, chan, phase, atime, window_len=10, filter_str="freq_0.8_3.5", evid=ev.evid, cache=True)
        print "extracted", ev


if __name__ == "__main__":
    main()
