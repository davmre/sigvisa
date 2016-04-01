import numpy as np
import os
import re
import cPickle as pickle
import socket


jobdir = "/home/sigvisa/python/sigvisa/logs/mcmc"

jobs = []

hostname = socket.gethostname() + ".cloudapp.net"

for run in os.listdir(jobdir):
    if run.endswith( ".1.1.1"):
        continue

    cmdfile = os.path.join(jobdir, run, "cmd.txt")

    try:
        with open(cmdfile, "r") as f:
            cmd = f.read()
    except:
        continue

    full_cmd = "/bin/bash /home/sigvisa/python/sigvisa/cloud/infer.sh " + cmd

    cmd_parser = re.compile(r".+--sta=(\w+).+--evidfile=(\w+)")
    m = cmd_parser.match(cmd)
    sta, cluster_fname = m.groups()
    full_cluster_fname = os.path.join("train_clusters", cluster_fname)



    pid = None
    for dirname in os.listdir('/proc'):
        try:
            with open('/proc/{}/cmdline'.format(dirname), mode='rb') as fd:
                content = fd.read()
            if content.startswith("python") and sta in content and cluster_fname in content:
                pid = int(dirname)
                break
        except Exception:
            continue

    jobs.append((run, full_cmd, sta, full_cluster_fname, hostname, pid))

print jobs
with open("jobs_%s.pkl" % hostname, "wb") as f:
    pickle.dump(jobs, f)
