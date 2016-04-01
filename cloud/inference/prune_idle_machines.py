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
from sigvisa.cloud.inference.remote_job_management import running_processes

from collections import defaultdict


from fabric.api import env

env.user = 'vagrant'
env.key_filename = '/home/dmoore/.ssh/fabric_is_terrible.key'


def get_idle_hosts(jobfile):
    with open(jobfile, "rb") as f:
        jobs = pickle.load(f)


    pids_by_host = defaultdict(set)
    for (jobid, cmd, sta, evidfile, host, pid) in jobs:
        pids_by_host[host].add(pid)

    idle_hosts = []
    for host, pids in pids_by_host.items():
        try:
            running_pids = running_processes(host)
        except:
            print "no response from", host, "skipping"
            continue

        alive=False
        for pid in pids:
            if pid in running_pids:
                alive = True
                break
        if not alive:
            idle_hosts.append(host)

    return idle_hosts

def main():
    
    parser = OptionParser()

    parser.add_option("--jobfile", dest="jobfile", default="", type="str",
                      help="file in which to record job progress")

    (options, args) = parser.parse_args()

    idle_hosts = get_idle_hosts(options.jobfile)

    for host in idle_hosts:
        print host
    print "done"

if __name__ == "__main__":
    main()
