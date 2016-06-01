from fabric.api import *

# push param models from local install/mysql to remote install. 

from sigvisa.database.db import connect
import csv
import tarfile
import os
import sys

import logging ; logging.basicConfig(level=logging.DEBUG)


#env.hosts = open('fabric_hosts', 'r').readlines()
env.hosts = ['sigvisa%d.cloudapp.net' % i for i in range(1, 1+1)]

#env.use_ssh_config = True
env.user = 'vagrant'
env.key_filename = '/home/dmoore/.ssh/fabric_is_terrible.key'


remote_sigvisa_home = "/home/sigvisa/python/sigvisa"


def get_cache_files():
    sigvisa_home = os.getenv("SIGVISA_HOME")
    cachedir = os.path.join(sigvisa_home, "db_cache")

    fnames = []
    for fname in os.listdir(cachedir):
        if fname.startswith("history"):
            fnames.append(os.path.join(cachedir, fname))
        elif fname.startswith("ev_prior_grid"):
            fnames.append(os.path.join(cachedir, fname))

    return fnames


# this is the "main" method to drive all the rest
def push_cache():
    fnames = get_cache_files()
    remote_cache_dir = os.path.join(remote_sigvisa_home, "db_cache")
    for fname in fnames:
        put(fname, remote_cache_dir, use_sudo=True)

