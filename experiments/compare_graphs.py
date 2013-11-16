import numpy as np
import sys
import os
import traceback
import pickle

from optparse import OptionParser
from sigvisa.results.compare import f1_and_error

from sigvisa.models.ev_prior import event_from_evnodes

def extract_graph_events(sg):
    events = []

    for(eid, evnodes) in sg.evnodes.items():
        ev = event_from_evnodes(evnodes)
        ev_vec = [ev.lon, ev.lat, ev.depth, ev.mb, ev.time]
        events.append(ev_vec)
    return np.array(events)

def main():
    parser = OptionParser()

    parser.add_option("--gold", dest="gold", default=None, type="str",
                      help="pickled graph file containing true events")
    parser.add_option("--guess", dest="guess", default=None, type="str",
                      help="pickled graph containing inference results")


    (options, args) = parser.parse_args()

    with open(options.gold, 'rb') as f:
        sg_gold = pickle.load(f)
    with open(options.guess, 'rb') as f:
        sg_guess = pickle.load(f)

    events_gold = extract_graph_events(sg_gold)
    events_guess = extract_graph_events(sg_guess)

    f, p, r, err = f1_and_error(events_gold, events_guess)
    print "F1=%.2f, Prec=%.2f, Recall=%.2f, Avg Error = %.2f+-%.2f" % (f, p, r, err[0], err[1])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
