import os
import numpy as np
import shutil

mcmcdir=  "logs/mcmc"
for d in os.listdir(mcmcdir):
    ddir = os.path.join(mcmcdir, d)
    try:
        contents = os.listdir(ddir)
    except:
        continue
    if len(contents)==0:
        print "deleting empty dir", ddir
        shutil.rmtree(ddir)
    if "cmd.txt" not in contents and "lp.txt" not in contents: continue
    steps = [int(s[5:]) for s in contents if "step" in s]
    max_step = np.max(steps) if len(steps) > 0 else -1
    if max_step < 10:
        print "deleting dir %s with %d steps" % (ddir, max_step)
        shutil.rmtree(ddir)
    else:
        print "keeping dir %s with %d steps" % (ddir, max_step)
