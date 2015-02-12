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


from optparse import (OptionParser,BadOptionError,AmbiguousOptionError)

class PassThroughOptionParser(OptionParser):
    """
    An unknown option pass-through implementation of OptionParser.

    When unknown arguments are encountered, bundle with largs and try again,
    until rargs is depleted.

    sys.exit(status) will still be called if a known argument is passed
    incorrectly (e.g. missing arguments or bad argument types, etc.)
    """
    def _process_args(self, largs, rargs, values):
        while rargs:
            try:
                OptionParser._process_args(self,largs,rargs,values)
            except (BadOptionError,AmbiguousOptionError), e:
                largs.append(e.opt_str)

def run_fit_and_rename_output(args):
    """

    Runs the provided fitting command, and blocks until it
    completes. Saves the output to a temporary file, and upon
    completion moves it to a new directory according to whether the
    fitting run completed successfully.

    """
    cmd, runid = args

    paramhash = hashlib.sha1(str(cmd)).hexdigest()[:8]

    print "about to run", cmd

    base_output = os.path.join(os.getenv("SIGVISA_HOME"), "logs", "fitting")
    tmp_output_dir = os.path.join(base_output, "temp")
    tmp_output_file = os.path.join(tmp_output_dir, "run%d_%s.log" % (runid, paramhash))
    ensure_dir_exists(tmp_output_dir)
    print "running fit, tmp output to %s" % tmp_output_file

    result = work(cmd, tmp_output_file)

    with open(tmp_output_file, 'r') as f:
        lines = f.readlines()

        fitid = None
        for line in lines[::-1]:
            m = re.match(r"fit id (\d+) completed successfully.", line.strip())
            if m is not None:
                fitid = m.group(1)
                success_output_dir = os.path.join(base_output, "successful")
                success_output_file = os.path.join(success_output_dir, "run%d_fit%d.log" % (runid, int(fitid)))
                ensure_dir_exists(success_output_dir)
                print "fit succeeded: moving %s to %s" % (tmp_output_file, success_output_file)
                shutil.move(tmp_output_file, success_output_file)
                break

        if fitid is None:
            failed_output_dir = os.path.join(base_output, "failed")
            failed_output_file = os.path.join(failed_output_dir, "run%d_%s.log" % (runid, paramhash))
            ensure_dir_exists(failed_output_dir)
            print "fit failed: moving %s to %s" % (tmp_output_file, failed_output_file)
            shutil.move(tmp_output_file, failed_output_file)

    return result


def work(cmd, output_file_str):
    on_posix = 'posix' in sys.builtin_module_names
    with open(output_file_str, 'w') as output_file:
        output_file.write(cmd + "\n\n")
        output_file.flush()
        result = subprocess.call(shlex.split(cmd),
                                 shell=False,
                                 close_fds=on_posix,
                                 bufsize=-1,
                                 stdin=open(os.devnull, 'rb'),
                                 stdout=output_file,
                                 stderr=subprocess.STDOUT)
    return result


def main():
    parser = PassThroughOptionParser()

    parser.add_option("-e", "--evidfile", dest="evidfile", default=None, type="str", help="file of 'evid, sta' pairs to fit")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="run_name")
    parser.add_option("-i", "--run_iter", dest="run_iter", default=None, type="int", help="run_iter")
    parser.add_option("-w", "--fit_wiggles", dest="fit_wiggles", default=False, action="store_true",
                      help="find/save wiggle fits as well")
    parser.add_option("--init_runid", dest="init_runid", default=None, type="int",
                      help="initialize template fitting with results from this runid")
    parser.add_option("--template_shape", dest="template_shape", default=None, type="str",
                      help="template model type to fit parameters under (paired_exp)")
    parser.add_option("--template_model", dest="template_model", default=None, type="str", help="")
    parser.add_option("--optim_params", dest="optim_params", default=None, type="str", help="fitting parameters to use")
    parser.add_option("--processes", dest="processes", default=-1, type="int", help="number of concurrent processes to run (default: one per cpu)")
    parser.add_option("--dummy", dest="dummy", default=False, action="store_true", help="don't actually do any fitting; just print out the commands to run")
    parser.add_option("--per_station_limit", dest="per_station_limit", default=None, type="int", help="max number of evids to fit per station (none)")


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
    cursor = s.dbconn.cursor()

    if options.evidfile is None:
        raise Exception("must specify an evid file!")

    if options.run_name is None:
        run_name = options.evidfile + "_test"
        iteration = 1
        init_iteration = None
        init_run_name = None
        init_str = ""
    else:
        run_name = options.run_name

        iteration = options.run_iter
        """        if iteration is None:
            iters = read_fitting_run_iterations(cursor, run_name)
            if len(iters) == 0:
                iteration = 1
            else:
                last_iteration = iters[-1][0]
                iteration = last_iteration+1
                init_iteration = None
                init_run_name = None
                init_str = ""
        else:
            last_iteration = iters[-1][0]
            init_run_name = run_name
            iteration = last_iteration+1
            init_iteration = last_iteration
            init_str = "--init_run_name=%s --init_run_iteration=%d" % (init_run_name, init_run_iteration)
            """
        init_str = ""

    runid = get_fitting_runid(cursor, run_name, iteration)
    cursor.close()
    s.dbconn.commit()

    print "hello", runid
    cmds = []

    sta_evids = {}
    with open(options.evidfile, 'r') as f:
        for line in f:
            (sta, evid) = [i.strip() for i in line.split(' ')]

            if not sta in sta_evids:
                sta_evids[sta] = set()
            sta_evids[sta].add(evid)

            if options.per_station_limit and len(sta_evids[sta]) > options.per_station_limit:
                continue

            cmd_str = "python -m learn.fit_shape_params_mcmc -e %d -s %s %s %s %s  --run_name=%s --run_iteration=%d %s %s %s" % (
                int(evid), sta, "--fit_wiggles" if options.fit_wiggles else "",
                "--template_shape=%s" % options.template_shape if options.template_shape else "",
                "--template_model=\"%s\"" % options.template_model if options.template_model else "",
                run_name, iteration, init_str,
                "--optim_params=\"%s\" " % options.optim_params if options.optim_params is not None else "",
                extra_option_string)

#            run_fit_and_rename_output((cmd_str, runid))
#            sys.exit(1)
            cmds.append((cmd_str, runid))
#            print cmd_str

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
        r = pool.map_async(run_fit_and_rename_output, cmds)
        r.wait()

if __name__ == "__main__":
    main()
