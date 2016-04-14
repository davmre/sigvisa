import numpy as np
import cPickle as pickle

dead_machines_file = "dead_machines"
jobfile = "jobs_reconstructed"

with open(jobfile, "rb") as f:
    jobs = pickle.load(f)

with open(dead_machines_file, "r") as f:
    dead_machines = [l.strip() for l in f.readlines()]

jobs_dead = []
jobs_alive = []
for job in jobs:
    (_, _, _, _, hostname, _) = job
    if hostname in dead_machines:
        jobs_dead.append(job)
    else:
        jobs_alive.append(job)

print "saving", len(jobs_dead), "jobs on dead machines"
with open(jobfile+"_dead", "wb") as f:
    pickle.dump(jobs_dead, f)

print "saving", len(jobs_alive), "jobs on living machines"
with open(jobfile+"_alive", "wb") as f:
    pickle.dump(jobs_alive, f)
