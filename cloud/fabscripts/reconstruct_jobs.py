from fabric.api import *

# push param models from local install/mysql to remote install. 

from sigvisa.database.db import connect
from sigvisa.utils.fileutils import mkdir_p
import cPickle as pickle
import csv
import tarfile
import os

import logging ; logging.basicConfig(level=logging.DEBUG)


#env.hosts = open('fabric_hosts', 'r').readlines()
env.hosts = ['sigvisa%d.cloudapp.net' % i for i in range(1, 61)]


#env.use_ssh_config = True
env.user = 'vagrant'
env.key_filename = '/home/dmoore/.ssh/fabric_is_terrible.key'

remote_sigvisa_home = "/home/sigvisa/python/sigvisa"


def reconstruct_jobs():

    jobfile = "/home/dmoore/python/sigvisa/jobs_reconstructed_new"
    deadfile = "/home/dmoore/python/sigvisa/dead_machines"

    tmp_jobdir = "/home/dmoore/python/sigvisa/cloud/tmpjobs"
    mkdir_p(tmp_jobdir)

    try:
        put("/home/dmoore/python/sigvisa/scripts/reconstruct_remote_jobs_dupes.py", remote_sigvisa_home, use_sudo=True)
    except:
        return

    sudo("source /home/sigvisa/.virtualenvs/sigvisa/bin/activate && cd %s && python reconstruct_remote_jobs_dupes.py" % (remote_sigvisa_home), user="sigvisa")

    remote_hostname = env.host_string
    remote_fname = "jobs_%s.pkl" % remote_hostname
    print remote_hostname
    print remote_fname
    get(os.path.join(remote_sigvisa_home, remote_fname), tmp_jobdir)

    with open(os.path.join(tmp_jobdir, remote_fname), "rb") as f:
        remote_jobs = pickle.load(f)

    machine_alive = False
    for (_, _, _, _, _, pid) in remote_jobs:
        if pid is not None:
            machine_alive = True
    
    if not machine_alive:
        with open(deadfile, 'a') as f:
            f.write(remote_hostname + "\n")

    try:
        with open(jobfile, "rb") as f:
            jobs = pickle.load(f)
    except:
        jobs = []

    jobs.extend(remote_jobs)
    with open(jobfile, "wb") as f:
        jobs = pickle.dump(jobs, f)
    
