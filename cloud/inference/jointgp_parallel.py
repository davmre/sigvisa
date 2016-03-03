import os
import subprocess32 as subprocess
import time
import numpy as np
import shutil
import uuid
import cPickle as pickle

from optparse import OptionParser

from sigvisa.utils.fileutils import mkdir_p
from sigvisa.cloud.inference.remote_job_management import JobManager, blocking_run_job, put_to_host
from sigvisa.cloud.inference.merge_results import summarize_results
from sigvisa.graph.serialization import load_serialized_from_file, save_serialized_to_file, merge_serializations, extract_time_period

from fabric.api import env

env.user = 'vagrant'
env.key_filename = '/home/dmoore/.ssh/fabric_is_terrible.key'

def dump_jobs(jobs, jobfile):
    with open(jobfile, "wb") as f:
        pickle.dump(jobs, f)

def parallel_jointgp_alignment(label, clusters_evids, stas, infer_script, 
                               old_jobfile=None, 
                               jobfile=None, nnodes=2, ncpus=4):

    infer_script = "/bin/bash /home/sigvisa/python/sigvisa/cloud/infer.sh " + infer_script


    s_homedir =  os.getenv("SIGVISA_HOME")

    hostnames = ["sigvisa%d.cloudapp.net" % (k+1) for k in range(nnodes)]

    remote_sigvisa_home = "/home/sigvisa/python/sigvisa"
    #for hostname in hostnames:
    #    for evidfile in clusters_evids:
    #        put_to_host(hostname, evidfile, remote_sigvisa_home, use_sudo=True)

    log_prefix = lambda jobid : "/home/sigvisa/python/sigvisa/logs/mcmc/%s" % jobid
    jm = JobManager(hostnames, ncpus, log_prefix)

    jobs = []

    if old_jobfile is not None:
        with open(old_jobfile, "rb") as f:
            oldjobs = pickle.load(f)
    else:
        oldjobs = None

    try:
        for sta in stas:
            for evidfile in clusters_evids:
                fname = os.path.basename(evidfile)
                cmd = "%s --sta=%s --evidfile=%s " % (infer_script, sta, fname)
                print cmd

                if oldjobs is not None and len([jb2 for (jb2, cmd2, sta2, evidfile2, host2, pid2) in oldjobs if cmd2==cmd]) > 0:
                    print "skipping because we've already run this command"
                    continue
                else:
                    print "not skipping"


                jobid = blocking_run_job(jm, cmd, sudo=True, user="sigvisa")

                host, pid = jm.hosts_by_job[jobid]
                jobs.append((jobid, cmd, sta, evidfile, host, pid))
                if jobfile is not None:
                    dump_jobs(jobs, jobfile)

    except KeyboardInterrupt as e:
        yn = raw_input("Kill currently running jobs (y/n)?: ")
        for (jobid, cmd, sta, evidfile, host, pid) in jobs:
            try:
                host, pid = jm.hosts_by_job[jobid]
                print "jobid", jobid, "host", host, "pid", pid
                if yn.lower().startswith("y"):
                    jm.kill_job(jobid)
            except Exception as ee:
                print "exception:", ee
        raise e

        

def main():
    
    parser = OptionParser()

    parser.add_option("--nnodes", dest="nnodes", default=0, type="int",
                      help="number of cloud nodes to execute on")
    parser.add_option("--ncpus", dest="ncpus", default=4, type="int",
                      help="number of cpus per node")
    parser.add_option("--label", dest="label", default="", type="str",
                      help="arbitrary label given to this inference run")
    parser.add_option("--cluster_dir", dest="cluster_dir", default="", type="str",
                      help="directory containing files listing evids for each cluster")
    parser.add_option("--stations", dest="stations", default="", type="str",
                      help="comma-separated list of stations")
    parser.add_option("--jobfile", dest="jobfile", default="", type="str",
                      help="file in which to record job progress")
    parser.add_option("--old_jobfile", dest="old_jobfile", default=None, type="str",
                      help="file from which to read job progress")


    (options, args) = parser.parse_args()

    infer_script = " ".join(args)

    assert(len(options.label) > 0)
    assert(len(options.stations) > 0)
    assert(len(options.jobfile) > 0)
    assert(len(infer_script) > 0)

    cevids = []
    bdir = "train_clusters"
    for cf in os.listdir(options.cluster_dir):
        cevids.append(os.path.join(options.cluster_dir, cf))
    
    stas = options.stations.split(",")

    parallel_jointgp_alignment(options.label, cevids, stas, infer_script, 
                               jobfile=options.jobfile,
                               old_jobfile=options.old_jobfile,
                               nnodes=options.nnodes, ncpus=options.ncpus)


if __name__ == "__main__":
    main()
