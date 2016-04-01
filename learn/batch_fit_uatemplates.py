import os
import errno
import sys
import time
import traceback
import re
import hashlib
import numpy as np
import scipy

from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *
from sigvisa.database import db

import sigvisa.utils.geog
import obspy.signal.util

from sigvisa import *
from sigvisa.utils.interaction import query_yes_no

import sys
import subprocess
import multiprocessing
import shlex
import os
import shutil
import re

from sigvisa.learn.batch_fit_from_evids import PassThroughOptionParser, run_fit_and_rename_output

from optparse import (OptionParser,BadOptionError,AmbiguousOptionError)

def run_uatemplate_fit_and_remane_output(args):
    return run_fit_and_rename_output(args, basename="uatemplates")

def main():
    parser = PassThroughOptionParser()

    parser.add_option("--runid", dest="runid", default=None, type="int", help="runid from which to load fits")
    parser.add_option("--processes", dest="processes", default=-1, type="int", help="number of concurrent processes to run (default: one per cpu)")
    parser.add_option("--dummy", dest="dummy", default=False, action="store_true", help="don't actually do any fitting; just print out the commands to run")

    (options, args) = parser.parse_args()

    extra_option_string = ""
    for a in args:
        if a.startswith("--"):
            extra_option_string += " " + a
        else:
            if "'" in a:
                # hack to escape args that contain quotes (eg tmpl_optim_params)
                extra_option_string += "=\"" + a + "\""
            else:
                extra_option_string += "=" + a

    s = Sigvisa()

    sql_query = "select fitid from sigvisa_coda_fit where runid=%d" % options.runid
    fitids = s.sql(sql_query)
    cmds = []
    for fitid in fitids:

        cmd_str = "python -m learn.fit_uatemplates_for_hough_tuning --fitid=%d %s" % (fitid[0],
                extra_option_string)
        cmds.append((cmd_str, options.runid))


    if options.dummy:
        for cmd in cmds:
            print cmd[0]
    else:
        if options.processes > 0:
            count = options.processes
        else:
            count = multiprocessing.cpu_count()
        print "starting thread pool with %d concurrent processes..." % count
        pool = multiprocessing.Pool(processes=count)
        r = pool.map_async(run_uatemplate_fit_and_remane_output, cmds)
        r.wait()

if __name__ == "__main__":
    main()
