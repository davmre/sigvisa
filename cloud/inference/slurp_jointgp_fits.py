import os
import subprocess32 as subprocess
import time
import numpy as np
import shutil
import uuid
import cPickle as pickle

from optparse import OptionParser

from sigvisa.utils.fileutils import mkdir_p
from sigvisa.learn.extract_fits_from_jointgp import extract_jointgp_fits


def sync_jobs(jobfile, dummy=False):
    with open(jobfile, "rb") as f:
        jobs = pickle.load(f)


    local_syncdir = "/home/dmoore/python/sigvisa/cloud/jgp_fits/"
    local_dirs = []
    mkdir_p(local_syncdir)
    for (jobid, cmd, sta, evidfile, host, pid) in jobs:
        serial_dir = os.path.join("/home/sigvisa/python/sigvisa/logs/mcmc/", "%s.1.1.1" % jobid)
        if not dummy:
            subprocess.call(["rsync", "-avz", "-e", "ssh", "vagrant@%s:%s" % (host, serial_dir), local_syncdir])

        local_dir = os.path.join(local_syncdir, "%s.1.1.1" % jobid)
        if os.path.exists(local_dir):
            local_dirs.append(local_dir)

    return local_dirs



def main():
    
    parser = OptionParser()

    parser.add_option("--run_name", dest="run_name", default=None, type="str",
                      help="")
    parser.add_option("--run_iteration", dest="run_iteration", default=0, type="int",
                      help="")
    parser.add_option("--burnin", dest="burnin", default=50, type="int",
                      help="")
    parser.add_option("--jobfile", dest="jobfile", default="", type="str",
                      help="file in which to record job progress")


    (options, args) = parser.parse_args()

    local_dirs = sync_jobs(options.jobfile, dummy=False)
    #local_dirs = ["/home/dmoore/python/sigvisa/cloud/jgp_fits/nvar_kampos_01124.1.1.1/",]

    print "not extracting"
    return

    for local_dir in local_dirs:
        steps = sorted([d for d in os.listdir(local_dir) if d.startswith("step_")])
        if len(steps) == 0:
            print "no steps found in", local_dir
            continue

        print "local dir", local_dir
        sg_fname = os.path.join(local_dir, steps[-1], "pickle.sg")
        try:
            extract_jointgp_fits(sg_fname=sg_fname, 
                                 run_name = options.run_name, 
                                 run_iter = options.run_iteration, 
                                 burnin=options.burnin,
                                 delete_existing=True)
        except Exception as e:
            print e
            continue



if __name__ == "__main__":
    main()
