import os
import subprocess32 as subprocess
import time
import numpy as np
import shutil

from optparse import OptionParser

from sigvisa.utils.fileutils import mkdir_p
from sigvisa.cloud.inference.remote_job_management import JobManager, blocking_run_job, put_to_host
from sigvisa.cloud.inference.merge_results import summarize_results
from sigvisa.graph.serialization import load_serialized_from_file, save_serialized_to_file, merge_serializations, extract_time_period

from fabric.api import env

env.user = 'vagrant'
env.key_filename = '/home/dmoore/.ssh/fabric_is_terrible.key'


def parallel_jointgp_alignment(label, clusters_evids, stas, runid, nnodes=2, ncpus=4):

    infer_script = "learn/jointgp_for_region.py  --runid=1 "
    infer_script = "/bin/bash /home/sigvisa/python/sigvisa/cloud/infer.sh " + infer_script


    s_homedir =  os.getenv("SIGVISA_HOME")

    hostnames = ["sigvisa%d.cloudapp.net" % (k+1) for k in range(nnodes)]

    remote_sigvisa_home = "/home/sigvisa/python/sigvisa"
    for hostname in hostnames:
        for evidfile in clusters_evids:
            put_to_host(hostname, evidfile, remote_sigvisa_home, use_sudo=True)

    log_prefix = lambda jobid : "/home/sigvisa/python/sigvisa/logs/mcmc/%s" % jobid
    jm = JobManager(hostnames, ncpus, log_prefix)

    jobs = []

    try:
        for sta in stas:
            for evidfile in clusters_evids:
                fname = os.path.basename(evidfile)
                cmd = "%s --sta=%s --evidfile=%s" % (infer_script, sta, fname)
                print cmd
                jobid = blocking_run_job(jm, cmd, sudo=True, user="sigvisa")
                jobs.append((jobid, cmd, sta, evidfile))

    except KeyboardInterrupt as e:
        yn = raw_input("Kill currently running jobs (y/n)?: ")
        for (jobid, cmd, sta, evidfile) in jobs:
            host, pid = jm.hosts_by_job[jobid]
            print "jobid", jobid, "host", host, "pid", pid
            if yn.lower().startswith("y"):
                jm.kill_job(jobid)
        raise e

        

def main():
    
    parser = OptionParser()

    parser.add_option("--nnodes", dest="nnodes", default=0, type="int",
                      help="number of cloud nodes to execute on")
    parser.add_option("--ncpus", dest="ncpus", default=4, type="int",
                      help="number of cpus per node")
    parser.add_option("--runid", dest="runid", default=11, type="int",
                      help="runid from which to load param models for envelopes")
    parser.add_option("--runid_raw", dest="runid_raw", default=12, type="int",
                      help="runid from which to load param models for raw signals")
    parser.add_option("--label", dest="label", default="", type="str",
                      help="arbitrary label given to this inference run")
    parser.add_option("--cluster_dir", dest="cluster_dir", default="", type="str",
                      help="directory containing files listing evids for each cluster")
    parser.add_option("--stations", dest="stations", default="", type="str",
                      help="comma-separated list of stations")


    (options, args) = parser.parse_args()

    infer_script = " ".join(args)

    assert(len(options.label) > 0)
    assert(len(options.stations) > 0)


    cevids = []
    bdir = "train_clusters"
    for cf in os.listdir(options.cluster_dir):
        cevids.append(os.path.join(options.cluster_dir, cf))
    
    stas = options.stations.split(",")

    parallel_jointgp_alignment(options.label, cevids, stas, nnodes=options.nnodes, ncpus=options.ncpus)


if __name__ == "__main__":
    main()
