from sigvisa import *
from sigvisa.database.dataset import *

def register_arrival_options(parser):
    parser.add_option("-s", "--stations", dest="stations", default=None, type="str", help="comma-separated list of station names for which to fit templates")
    parser.add_option("--min_mb", dest="min_mb", default=5, type="float", help="exclude all events with mb less than this value (0)")
    parser.add_option("--max_mb", dest="max_mb", default=10, type="float", help="exclude all events with mb greater than this value (10)")
    parser.add_option("--start_time", dest="start_time", default=None, type="float", help="exclude all events with time less than this value (0)")
    parser.add_option("--end_time", dest="end_time", default=None, type="float", help="exclude all events with time greater than this value (1237680000)")
# TODO:L implement this #    parser.add_option("--require_p_s", dest="require_p_s", default=False, action="store_true", help="only plot events with both P and S detected")
    parser.add_option("--save_to_file", dest="save_to_file", default=None, type="str", help="filename to save selected event/station pairs to")
    parser.add_option("--load_from_file", dest="load_from_file", default=None, type="str", help="filename to load  event/station pairs from")

def arrivals_from_options(options):

    if options.load_from_file is not None:
        return load_arrivals_from_file(options.load_from_file)

    s = Sigvisa()
    st = options.start_time
    et = options.end_time

    arrivals = []

    for sta in options.stations.split(','):
        evids = read_evids_detected_at_station(s. cursor, sta, st, et, s.P_phases+s.S_phases, min_mb = options.min_mb, max_mb = options.max_mb)
        for evid in evids:
            arrivals.append((sta, evid[0]))

    if options.save_to_file is not None:
        print "saving to", options.save_to_file
        save_arrivals_to_file(arrivals, options.save_to_file, options)
    else:
        print "not saving"
    return arrivals

def save_arrivals_to_file(arrivals, fname, options):
    with open(fname, 'w') as f:
        for (sta, evid) in arrivals:
            f.write("%s %d\n" % (sta, evid))
        f.write("########\n" + repr(vars(options)))

def load_arrivals_from_file(fname):
    arrivals = []
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith("########"):
                break
            (sta, evid) = line.split(" ")
            arrivals.append((sta, int(evid)))

    return arrivals
