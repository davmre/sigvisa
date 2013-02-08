import os, errno, sys, time, traceback, re, hashlib
import numpy as np, scipy

from database.dataset import *
from database.signal_data import *
from database import db

import utils.geog
import obspy.signal.util

from optparse import OptionParser

from sigvisa import *
from utils.interaction import query_yes_no

import sys, subprocess, multiprocessing, shlex, os, shutil, re


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
        last_line = lines[-1].strip()
        m = re.match(r"fit id (\d+) completed successfully.", last_line)
        if m is not None:
            fitid = m.group(1)
            success_output_dir = os.path.join(base_output, "successful")
            success_output_file = os.path.join(success_output_dir, "run%d_fit%d.log" % (runid, int(fitid)))
            ensure_dir_exists(success_output_dir)
            print "fit succeeded: moving %s to %s" % (tmp_output_file, success_output_file)
            shutil.move(tmp_output_file, success_output_file)
        else:
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
                                 stdin=open(os.devnull,'rb'),
                                 stdout=output_file,
                                 stderr=subprocess.STDOUT)
    return result

def main():
    parser = OptionParser()

    parser.add_option("-e", "--evidfile", dest="evidfile", default=None, type="str", help="file of 'evid, sta' pairs to fit")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="run_name")
    parser.add_option("-i", "--run_iter", dest="run_iter", default=None, type="int", help="run_iter")
    parser.add_option("-w", "--wiggles", dest="wiggles", default=None, type="str", help="filename of wiggle-model params to load (default is to ignore wiggle model and do iid fits)")
    parser.add_option("--init_runid", dest="init_runid", default=None, type="int", help="initialize template fitting with results from this runid")
    parser.add_option("--template_shape", dest = "template_shape", default="paired_exp", type="str", help="template model type to fit parameters under (paired_exp)")
    parser.add_option("--template_model", dest = "template_model", default="gp_dad", type="str", help="")
    parser.add_option("--optim_params", dest = "optim_params", default=None, type="str", help="fitting parameters to use")

    (options, args) = parser.parse_args()

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

    print "hello", runid
    cmds = []

    with open(options.evidfile, 'r') as f:
        for line in f:
            (sta, evid) = [i.strip() for i in line.split(' ')]

            cmd_str = "/vdec/software/site/usr/bin/python2.6 -m learn.fit_shape_params -e %d -s %s %s --template_shape=%s --template_model=%s --run_name=%s --run_iteration=%d %s %s" % (int(evid), sta, "-w %s" % options.wiggles if options.wiggles else "", options.template_shape, options.template_model, run_name, iteration, init_str, "--optim_params=\"%s\" " % options.optim_params if options.optim_params is not None else "")

#            run_fit_and_rename_output((cmd_str, runid))
#            sys.exit(1)
            cmds.append((cmd_str, runid))
#            print cmd_str

    count = multiprocessing.cpu_count()
    print "starting thread pool with %d concurrent processes..." % count
    pool = multiprocessing.Pool(processes=count)
    r = pool.map_async(run_fit_and_rename_output, cmds)
    r.wait()

if __name__ == "__main__":
    main()
