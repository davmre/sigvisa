import os
import time
import numpy as np


from optparse import OptionParser

from sigvisa.graph.serialization import load_serialized_from_file, save_serialized_to_file, merge_serializations, extract_time_period

def load_evscores(dirname, evdicts):

    def match_evdict(evdicts, evd2):
        for evd in evdicts:
            if np.abs(evd["time"] - evd2["time"]) > 5:
                continue
            if np.abs(evd["lon"] - evd2["lon"]) > 1e-1:
                continue
            if np.abs(evd["lat"] - evd2["lat"]) > 1e-1:
                continue
            if np.abs(evd["depth"] - evd2["depth"]) > 5:
                continue
            if np.abs(evd["mb"] - evd2["mb"]) > 0.5:
                continue
            return evd
        raise Exception("could not match ev dict %s" % (repr(evd2)))

    scorefile = os.path.join(dirname, 'ev_scores.repr')
    if not os.path.exists(scorefile):
        print "could not find ev score file %s, skipping ev scores" % scorefile
        return

    with open(scorefile, 'r') as f:
        score_dict = eval(f.read(), {'nan': np.nan})
    
    for eid, (score, evd2) in score_dict.items():        
        if np.isnan(score):
            score = 0.0

        evd = match_evdict(evdicts, evd2)
        evd["score"] = score

    print "loaded ev scores from", scorefile

def last_serialization(dirname):
    fname = sorted(os.listdir(dirname))[-1]
    return os.path.join(dirname, fname)

def summarize_results(jobdir, jobs):
    serialized_periods = []
    s_homedir =  os.getenv("SIGVISA_HOME")
    for (jobid, cmd, (stime, etime)) in jobs:
        sgdir = os.path.join(jobdir, jobid)
        serial_dir = os.path.join(sgdir, "serialized")
        final_state_file = last_serialization(serial_dir)
        evdicts, uadicts_by_sta = load_serialized_from_file(final_state_file)

        load_evscores(sgdir, evdicts)

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
