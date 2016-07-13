import os
import sys
import time
import numpy as np

from sigvisa.utils.fileutils import mkdir_p

from sortedcontainers import SortedSet
from frozendict import frozendict

from optparse import OptionParser

from sigvisa.graph.serialization import load_serialized_from_file, save_serialized_to_file, merge_serializations, extract_time_period
from sigvisa.results.evaluate_serialized import print_bulletin

"""
Given the time of an event, figure out which inference run and subdir it came from, so I can inspect the aligned signal. 

Example cmdline:
python results/explain_event.py 1204132558 cloud/remote_jobs/test_omg_seed1/ cloud/remote_jobs/test_omg_seed2/ cloud/remote_jobs/test_omg_seed3/ cloud/remote_jobs/test_highfreq_seed11/ cloud/remote_jobs/test_highfreq_seed12/

"""

def main():

    tolerance_s = 50.0
    
    evtime = float(sys.argv[1])
    jobdirs = sys.argv[2:]

    
    subdirs = []
    best_score = -1
    best_dir = None
    for jobdir in jobdirs:

        evdicts, uadicts_by_sta = load_serialized_from_file(os.path.join(jobdir, "merged.tgz"))
        bulletin = np.array(sorted([(d["lon"], d["lat"], d["depth"], d["time"], d["mb"], d['score'] if 'score' in d else 0.0) for d in evdicts], key = lambda x: x[3]))
        candidates = np.array([ev for ev in bulletin if evtime-tolerance_s < ev[3] < evtime+tolerance_s ])

        with open(os.path.join(jobdir, "jobs.txt"), "r") as f:
            jobs = eval(f.read())
        for (jobid, cmd, (stime, etime)) in jobs:
            if stime < evtime < etime:
                subdir = os.path.join(jobdir, jobid)
                subdirs.append(subdir)
                break

        print subdir
        print_bulletin(candidates)
        if len(candidates) > 0:
            score = np.max(candidates[:, 5])
        else:
            score = 0.0

        if score > best_score:
            best_score = score
            best_dir = subdir
            
    print "best dir", best_dir
    target = os.path.join("/home/dmoore/python/sigvisa/logs/mcmc/", os.path.basename(best_dir))
    try:
        os.symlink(os.path.abspath(best_dir), target)
        print "created symlink", target
    except Exception as e:
        print "could not symlink %s: %s" % (target, e)

if __name__ == "__main__":
    main()

