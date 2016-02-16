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

    (options, args) = parser.parse_args()

    s = Sigvisa()

    if options.sta is None:
        raise Exception("must specify a station name (use -s)!")
    else:
        sta = options.sta

    # csv format:
    # lon, lat, depth, time, mb, id
    evs = np.loadtxt(options.evfile)

    phases = options.phases.split(",")

    arriving_signals = []
    for i, (lon, lat, depth, time, mb, eid) in enumerate(evs):
        ev = Event(lon=lon, lat=lat, depth=depth, time=time, mb=mb)
        phases = predict_phases_sta(ev, sta, phases)

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
