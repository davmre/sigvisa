import os
import subprocess32 as subprocess
import time
import numpy as np

from optparse import OptionParser

from sigvisa.utils.fileutils import mkdir_p
from sigvisa.cloud.inference.remote_job_management import JobManager
from sigvisa.graph.serialization import load_serialized_from_file, save_serialized_to_file, merge_serializations, extract_time_period

from fabric.api import env

env.user = 'vagrant'
env.key_filename = '/home/dmoore/.ssh/fabric_is_terrible.key'


def sync_serializations(hostname, jobid, local_dir):
    serial_dir = os.path.join("/home/sigvisa/python/sigvisa/logs/mcmc/", "%s" % jobid, "serialized")
    subprocess.call(["rsync", "-avz", "-e", "ssh", "vagrant@%s:%s" % (hostname, serial_dir), local_dir])
    
def last_serialization(dirname):
    fname = sorted(os.path.listdir(dirname))[:-1]
    return os.path.join(dirname, fname)

def summarize_results(jobdir, jobs):
    serialized_periods = []
    for (jobid, cmd, stime, etime) in jobids:
        serial_dir = os.path.join("/home/sigvisa/python/sigvisa/logs/mcmc/", "%s" % jobid, "serialized")
        final_state_file = last_serialization(serial_dir)
        evdicts, uadicts_by_sta = load_serialized_from_file(final_state_file)
        serialized_periods.append((evdicts, uadicts_by_sta, stime, etime))
    return merge_serializations(*serialized_periods)

def parallel_inference(infer_script, label, nnodes, stime, etime, 
                       block_s=3600, ncpus=4, 
                       inference_s=3600.0, sync_s=60):

    infer_script = "/bin/bash /home/sigvisa/python/sigvisa/cloud/infer.sh " + infer_script

    s_homedir =  os.getenv("SIGVISA_HOME")
    jobdir = os.path.join(s_homedir, "cloud", "remote_jobs", label)
    mkdir_p(jobdir)

    hostnames = ["sigvisa%d.cloudapp.net" % (k+1) for k in range(nnodes)]

    total_len = etime-stime
    nblocks = int(np.floor(total_len / float(block_s)))

    stimes = np.array([stime + k * block_s for k in range(nblocks)], dtype=np.float)
    etimes = stimes + block_s
    etimes[-1] = etime

    log_prefix = lambda jobid : "/home/sigvisa/python/sigvisa/logs/mcmc/%s" % jobid
    jm = JobManager(hostnames, ncpus, log_prefix)

    try:
        jobs = []
        for (block_stime, block_etime) in zip(stimes, etimes):
            cmd = "%s --stime=%f --etime=%f" % (infer_script, block_stime, block_etime)
            jobid = jm.run_job(cmd, sudo=True, user="sigvisa")
            jobs.append((jobid, cmd, block_stime, block_etime))

        t0 = time.time()
        t = time.time()
        while t - t0 < inference_s:

            for (jobid, cmd, block_stime, block_etime) in jobs:
                local_dir = os.path.join(jobdir, "%s" % jobid)
                mkdir_p(local_dir)

                hostname, _ = jm.hosts_by_job[jobid]
                sync_serializations(hostname, jobid, local_dir)

            time.sleep(sync_s)        
            t = time.time()
    except KeyboardInterrupt as e:
        yn = raw_input("Kill currently running jobs (y/n)?: ")
        for (jobid, cmd, block_stime, block_etime) in jobs:
            host, pid = jm.hosts_by_job[jobid]
            print "jobid", jobid, "host", host, "pid", pid
            if yn.lower().startswith("y"):
                jm.kill_job(jobid)
        raise e

    for (jobid, cmd, block_stime, block_etime) in jobs:
        print "killing job", jobid
        jm.kill_job(jobid)

    evdicts, uadicts_by_sta = summarize_results(jobdir, jobs)

    save_serialized_to_file(os.path.join(jobdir, "merged"), evdicts, uadicts_by_sta)
        

def main():
    
    parser = OptionParser()

    parser.add_option("--nnodes", dest="nnodes", default=0, type="int",
                      help="number of cloud nodes to execute on")
    parser.add_option("--ncpus", dest="ncpus", default=4, type="int",
                      help="number of cpus per node")
    parser.add_option("--block_s", dest="block_s", default=3600.0, type="float",
                      help="length in seconds of each inference block")
    parser.add_option("--inference_s", dest="inference_s", default=3600.0, type="float",
                      help="time to run the remote inference processes")
    parser.add_option("--label", dest="label", default="", type="str",
                      help="arbitrary label given to this inference run")
    parser.add_option("--stime", dest="stime", default=None, type="float",
                      help="start time")
    parser.add_option("--etime", dest="etime", default=None, type="float",
                      help="end time")
    parser.add_option("--sync_s", dest="sync_s", default=60.0, type="float",
                      help="interval between syncing serialized inference states")

    (options, args) = parser.parse_args()

    infer_script = " ".join(args)

    assert(len(options.label) > 0)
    assert(len(infer_script) > 0)

    parallel_inference(infer_script=infer_script, 
                       label=options.label,
                       nnodes=options.nnodes, 
                       stime=options.stime, 
                       etime=options.etime,
                       block_s = options.block_s, 
                       ncpus = options.ncpus,
                       inference_s=options.inference_s, 
                       sync_s=options.sync_s)

if __name__ == "__main__":
    main()
