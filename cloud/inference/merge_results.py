import os
import time
import numpy as np


from optparse import OptionParser

from sigvisa.graph.serialization import load_serialized_from_file, save_serialized_to_file, merge_serializations, extract_time_period


def last_serialization(dirname):
    fname = sorted(os.listdir(dirname))[-1]
    return os.path.join(dirname, fname)

def summarize_results(jobdir, jobs):
    serialized_periods = []
    s_homedir =  os.getenv("SIGVISA_HOME")
    for (jobid, cmd, (stime, etime)) in jobs:
        serial_dir = os.path.join(jobdir, jobid, "serialized")
        final_state_file = last_serialization(serial_dir)
        evdicts, uadicts_by_sta = load_serialized_from_file(final_state_file)
        serialized_periods.append((evdicts, uadicts_by_sta, stime, etime))
    return merge_serializations(*serialized_periods)

def main():
    
    parser = OptionParser()

    parser.add_option("--jobdir", dest="jobdir", default=None, type="str",
                      help="number of cloud nodes to execute on")

    (options, args) = parser.parse_args()

    jobdir = options.jobdir

    with open(os.path.join(jobdir, "jobs.txt"), "r") as f:
        jobs = eval(f.read())

    evdicts, uadicts_by_sta = summarize_results(jobdir, jobs)

    save_serialized_to_file(os.path.join(jobdir, "merged"), evdicts, uadicts_by_sta)
    

if __name__ == "__main__":
    main()
