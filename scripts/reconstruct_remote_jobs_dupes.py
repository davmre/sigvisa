import numpy as np
import os
import time
import re
import cPickle as pickle
import socket
from collections import defaultdict

import signal


def pids_for_cmd(sta, cluster_fname):
    pids = []
    for dirname in os.listdir('/proc'):
        try:
            with open('/proc/{}/cmdline'.format(dirname), mode='rb') as fd:
                content = fd.read()
        except Exception as e:
            continue

        if content.startswith("python") and sta in content and cluster_fname in content:
            pid = int(dirname)
            with open('/proc/{}/stat'.format(dirname), mode="r") as fd:
                etime = float(fd.read().split()[21])
            pids.append((pid, etime))
    return pids


jobdir = "/home/sigvisa/python/sigvisa/logs/mcmc"

jobs = []

hostname = socket.gethostname() + ".cloudapp.net"

jobs_dict = defaultdict(list)

for run in os.listdir(jobdir):
    if run.endswith( ".1.1.1"):
        continue

    if run.endswith("killed"):
        continue

    cmdfile = os.path.join(jobdir, run, "cmd.txt")
    dtime = os.path.getmtime( cmdfile  )
    try:
        with open(cmdfile, "r") as f:
            cmd = f.read()
    except:
        continue

    full_cmd = "/bin/bash /home/sigvisa/python/sigvisa/cloud/infer.sh " + cmd

    cmd_parser = re.compile(r".+--sta=(\w+).+--evidfile=(\w+)")
    m = cmd_parser.match(cmd)
    sta, cluster_fname = m.groups()

    jobs_dict[(full_cmd, sta, cluster_fname)].append((run, dtime))


for (full_cmd, sta, cluster_fname), jobs_list in jobs_dict.items():


    pids = pids_for_cmd(sta, cluster_fname)

    # match processes and directories from most recent to least recent
    sorted_pids = sorted(pids, key = lambda (pid, etime) : etime, reverse=True)
    sorted_jobs = sorted(jobs_list, key = lambda (run, dtime): dtime, reverse=True)

    if len(sorted_pids) == len(sorted_jobs):
        pid, etime = sorted_pids[-1]
    else:
        pid = None

    print sorted_jobs
    print sorted_pids

    for dupe_pid, etime in sorted_pids[:-1]:

        print "killing dupe process", dupe_pid
        os.kill(dupe_pid, signal.SIGTERM)


    for dupe_run, dtime in sorted_jobs[:-1]:
        os.rename(os.path.join(jobdir, dupe_run), os.path.join(jobdir, dupe_run + "_killed"))

    full_cluster_fname = os.path.join("train_clusters", cluster_fname)
    run, dtime = sorted_jobs[-1]
    jobs.append((run, full_cmd, sta, full_cluster_fname, hostname, pid))


print jobs
with open("jobs_%s.pkl" % hostname, "wb") as f:
    pickle.dump(jobs, f)
