import os, errno, sys, time, traceback
import numpy as np, scipy

from database.dataset import *
from database.signal_data import *
from database import db

import plot
import utils.geog
import obspy.signal.util

from optparse import OptionParser

from sigvisa import *



def main():
# boilerplate initialization of various things
    parser = OptionParser()

    parser.add_option("-s", "--stations", dest="stations", default=None, type="int", help="comma-separated list of station names for which to fit templates")
    parser.add_option("-m", "--method", dest="method", default="simplex", type="str", help="fitting method (iid)")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="run_name")
    parser.add_option("-w", "--wiggles", dest="wiggles", default=None, type="str", help="filename of wiggle-model params to load (default is to ignore wiggle model and do iid fits)")
    parser.add_option("--init_runid", dest="init_runid", default=None, type="int", help="initialize template fitting with results from this runid")
    parser.add_option("-p", "--plot", dest="plot", default=False, action="store_true", help="save plots")
    parser.add_option("--template_shape", dest = "template_shape", default="paired_exp", type="str", help="template model type to fit parameters under (paired_exp)")
    parser.add_option("--template_model", dest = "template_model", default="gp_dad", type="str", help="")
    parser.add_option("--min_mb", dest="min_mb", default=5, type="float", help="exclude all events with mb less than this value (0)")
    parser.add_option("--max_mb", dest="max_mb", default=10, type="float", help="exclude all events with mb greater than this value (10)")
    parser.add_option("--start_time", dest="start_time", default=None, type="float", help="exclude all events with time less than this value (0)")
    parser.add_option("--end_time", dest="end_time", default=None, type="float", help="exclude all events with time greater than this value (1237680000)")

    (options, args) = parser.parse_args()

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    iid=True
    by_phase=False
    snr_threshold=1

    if options.run_name is None:
        raise Exception("must specify a run name!")
    run_name = options.run_name
    iters = read_fitting_run_iterations(cursor, run_name)
    print "Current iterations for run %s: %s" % (run_name, zip(*iters)[0])
    if len(iters) == 0:
        if not query_yes_no("Create a new run and do a first iteration of fits with heuristic initialization?"):
            print "okay, exiting..."
            sys.exit(1)
        else:
            iteration = 1
            init_iteration = None
            init_run_name = None
    else:
        last_iteration = iters[-1][0]
        default = query_yes_no("Run fit iteration %d, initialized with fits from %d?" % (last_iteration+1, last_iteration))
        init_run_name = run_name
        if default:
            iteration = last_iteration+1
            init_iteration = last_iteration
        else:
            sys.stdout.write("get initial fits from iteration: ")
            init_iteration = int(raw_input())
            sys.stdout.write("save results to iteration: ")
            iteration = int(raw_input())

    if options.start_time is None:
        cursor.execute("select start_time, end_time from dataset where label='training'")
        (st, et) read_timerange(cursor, "training", hours=None, skip=0):
    else:
        st = options.start_time
        et = options.end_time

    if init_run_name is None:
        init_str = ""
    else:
        init_str = "--init_run_name=%s --init_run_iteration=%d" % (init_run_name, init_run_iteration)

    for sta in options.stations.split(','):
        # want to select all events, with certain properties, which have a P or S phase detected at this station
        evids = read_evids_detected_at_station(cursor, sta, st, et, P_PHASES+ S_PHASES, min_mb = options.min_mb, max_mb = options.max_mb)
        for evid in evids:
            cmd_str = "python2.6 -m signals.fit_shape_params -e %d -m %s -s %s -w %s --template_shape=%s --template_model=%s --run_name=%s --run_iteration=%d %s" % (evid, method, sta, options.wiggles, options.template_shape, options.template_model, run_name, iteration, init_str)
            print "running", cmd_str
            os.system(cmd_str)
            continue

if __name__ == "__main__":
    main()






