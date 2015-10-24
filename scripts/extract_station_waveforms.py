import sigvisa.database.db
from sigvisa.database.dataset import *
from sigvisa.explore.doublets.xcorr_pairs import extract_phase_window
import sigvisa.utils.geog
import sys
import itertools
import time
import calendar
import cPickle as pickle

try:
    import parsedatetime.parsedatetime as pdt
except:
    import parsedatetime as pdt

from optparse import OptionParser

from sigvisa.source.event import get_event
from sigvisa import *


def get_first_arrivals(events, sta):
    s = Sigvisa()
    arriving_events = []
    evids = []
    first_arriving_times = []
    first_arriving_phases = []
    cursor = s.dbconn.cursor()
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

    if options.sta is None:
        raise Exception("must specify a station name (use -s)!")
    else:
        sta = options.sta

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
	cursor = s.dbconn
        evids = read_evids_detected_at_station(cursor, sta, st, et, phases, min_mb=options.min_mb, max_mb=options.max_mb, min_snr=2, time_filter_direct=True)
    else:
        f = open(options.evid_file, 'r')
        evids = [int(line) for line in f]
        f.close()

    print "loading", len(evids), "events..."
    events = [get_event(evid) for evid in evids]
    print "---------------------------------------------------------"

    arriving_events, arrival_dict = get_first_arrivals(events, sta)
    arriving_signals = []
    for i, ev in enumerate(arriving_events):
        
        if (i % 1000 == 0 and i > 1):
           with open(options.outfile + '_%d' % i, 'wb') as f:
                pickle.dump(arriving_signals, f) 
           arriving_signals = []
           print 'saved to', options.outfile + '_%d' % i

        time, phase = arrival_dict[ev.evid]
        window_len=20.0
	filter_str="freq_0.8_4.5;hz_20"
        try:
            d = extract_phase_window(sta, 'auto', phase, time, window_len, filter_str, ev.evid, leadin_s=5.0)
            arriving_signals.append((ev, d))
        except Exception as e:
            print e
            continue

        
    with open(options.outfile + '_final', 'wb') as f:
         pickle.dump(arriving_signals, f)

if __name__ == "__main__":
    main()
