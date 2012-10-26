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
    parser.add_option("-r", "--runid", dest="runid", default=None, type="int", help="runid")
    parser.add_option("-w", "--wiggles", dest="wiggles", default=None, type="str", help="filename of wiggle-model params to load (default is to ignore wiggle model and do iid fits)")
    parser.add_option("--init_runid", dest="init_runid", default=None, type="int", help="initialize template fitting with results from this runid")
    parser.add_option("-p", "--plot", dest="plot", default=False, action="store_true", help="save plots")
    parser.add_option("--template_shape", dest = "template_shape", default="paired_exp", type="str", help="template model type to fit parameters under (paired_exp)")
    parser.add_option("--template_run_name", dest = "template_run_name", default=None, type="str", help="name of previously trained template model to load (None)")
    parser.add_option("--template_model", dest = "template_model", default="gp_dad", type="str", help="")
    parser.add_option("--min_mb", dest="min_mb", default=5, type="float", help="exclude all events with mb less than this value (0)")
    parser.add_option("--max_mb", dest="max_mb", default=10, type="float", help="exclude all events with mb greater than this value (10)")
    parser.add_option("--start_time", dest="start_time", default=None, type="float", help="exclude all events with time less than this value (0)")
    parser.add_option("--end_time", dest="end_time", default=None, type="float", help="exclude all events with time greater than this value (1237680000)")

    (options, args) = parser.parse_args()

    s = Sigvisa()
    cursor = s.cursor

    runid = options.runid

    iid=True
    by_phase=False
    snr_threshold=1

    if options.start_time is None:
        cursor.execute("select start_time, end_time from dataset where label='training'")
        (st, et) read_timerange(cursor, "training", hours=None, skip=0):
    else:
        st = options.start_time
        et = options.end_time

    # want to select all events, with certain properties, which have a P or S phase detected at this station
    phase_condition = "(" + " or ".join(["leba.phase='%s'" % (pn) for pn in S_PHASES + P_PHASES]) + ")"
    ev_condition = "and l.time between %f and %f and lebo.mb between %f and %f and l.snr > 5" % (st, et, options.min_mb, options.max_mb)
    sql_query="SELECT distinct lebo.evid FROM leb_arrival l, leb_origin lebo, leb_assoc leba, dataset d where leba.arid=l.arid and lebo.orid=leba.orid and %s and l.dta=%s %s" % (phase_condition, sta, ev_condition)
    cursor.execute(sql_query)
    evids = cursor.fetchall()

    if runid is None:
        cursor.execute("select max(runid) from sigvisa_coda_fits")
        runid, = cursor.fetchone()
        if runid is None:
            runid=0
        else:
            runid = int(runid)+1

    for evid in evids:

        cmd_str = "python2.6 -m signals.fit_shape_params -r %d -e %d -m %s -s %d --template_shape=%s --template_model=%s" % (runid, evid, method, siteid, options.template_shape, options.template_model)
        if options.template_run_name is not None:
            cmd_str += " --template_run_name=%s" % options.template_run_name

        print "running", cmd_str
        os.system(cmd_str)
        continue

if __name__ == "__main__":
    main()






