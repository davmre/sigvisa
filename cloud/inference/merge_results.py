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

        try:
            evd = match_evdict(evdicts, evd2)
            evd["score"] = score
        except Exception as e:
            print e
            continue

    for evd in evdicts:
        if "score" not in evd:
            print "WARNING: assigning default score 0 to event %s from %s" % ({"lon": evd["lon"], "lat": evd["lat"], "time": evd["time"]}, scorefile)
            evd["score"] = 0.0

    print "loaded ev scores from", scorefile

def last_serialization(dirname):
    fname = sorted(os.listdir(dirname))[-1]
    return os.path.join(dirname, fname)

def consolidate_time_blocks(time_blocks):
    # given a list of (stime, etime) blocks, merge any that are directly adjacent
    consolidated = []

    n = len(time_blocks)
    time_blocks = sorted(time_blocks)
    running_stime = None
    for i in range(n):
        block_stime, block_etime = time_blocks[i]

        if i < n-1:
            next_stime, next_etime = time_blocks[i+1]
            if next_stime < block_etime + 10:
                running_stime = block_stime if running_stime is None else running_stime
            else:
                stime = running_stime if running_stime is not None else block_stime
                consolidated.append((stime, block_etime))
                running_stime = None
        else:
            stime = running_stime if running_stime is not None else block_stime
            consolidated.append((stime, block_etime))
    return consolidated

def load_dead_events(sgdir, min_steps=10):

    import re
    m = re.compile(r"proposing to kill eid (\d+): evid (.+), loc (.+) W (.+) N, depth (.+)km, time (.+), mb (.+), (.+) source\nacceptance lp (.+) \(lp_old (.+) lp_new (.+) log_qforward (.+) log_qbackward (.+)\)")

    def entry_to_evdict(re_match):
        g = re_match.groups()
        d = {'eid': int(g[0]),
             'lon': -float(g[2]),
             'lat': float(g[3]),
             'depth': float(g[4]),
             'time': float(g[5]),
             'mb': float(g[6]),
             'score': -float(g[8])}
        return d

    def steps_alive(eid):
        try:
            with open(os.path.join(sgdir, "ev_%05d" % eid, 'trajectory.txt'), 'r') as f:
                lines = f.readlines()
        except IOError:
            return 0

        return len(lines)

    def load_death_log(logfile):
        with open(logfile, 'r') as f:
            log = f.read()
        entries = log.split("\n\n")
        matches = [m.match(entry) for entry in entries]
        evdicts = [entry_to_evdict(match) for match in matches if match is not None]
        return evdicts

    try:
        prior_deaths = load_death_log(os.path.join(sgdir, 'prior_deaths.txt'))
    except:
        prior_deaths = []
    try:
        hough_deaths = load_death_log(os.path.join(sgdir, 'hough_deaths.txt'))
    except:
        hough_deaths = []

    deaths = prior_deaths + hough_deaths
    actual_deaths = [evd for evd in deaths if evd['score'] < 0]
    meaningful_deaths = [evd for evd in actual_deaths if steps_alive(evd['eid']) > min_steps]
    return meaningful_deaths

def summarize_results(jobdir, jobs, include_dead=False):

    time_blocks = []
    serialized_periods = []
    s_homedir =  os.getenv("SIGVISA_HOME")
    for (jobid, cmd, (stime, etime)) in jobs:
        sgdir = os.path.join(jobdir, jobid)
        serial_dir = os.path.join(sgdir, "serialized")

        try:
            final_state_file = last_serialization(serial_dir)
        except Exception as e:
            print "ERROR on dir %s: %s" % (sgdir, e)
            continue
            
        evdicts, uadicts_by_sta = load_serialized_from_file(final_state_file)
        load_evscores(sgdir, evdicts)

        if include_dead:
            dead_evdicts = load_dead_events(sgdir)
            evdicts += dead_evdicts

        serialized_periods.append((evdicts, uadicts_by_sta, stime, etime))
        time_blocks.append((stime, etime))

    evdicts, uadicts_by_sta = merge_serializations(*serialized_periods)
    time_periods = consolidate_time_blocks(time_blocks)
    return evdicts, uadicts_by_sta, time_periods

def main():
    
    parser = OptionParser()

    parser.add_option("--jobdir", dest="jobdir", default=None, type="str",
                      help="number of cloud nodes to execute on")
    parser.add_option("--include_dead", dest="include_dead", default=False, action="store_true",
                      help="include events that have been killed")

    (options, args) = parser.parse_args()

    jobdir = options.jobdir

    with open(os.path.join(jobdir, "jobs.txt"), "r") as f:
        jobs = eval(f.read())

    evdicts, uadicts_by_sta, time_periods = summarize_results(jobdir, jobs, include_dead=options.include_dead)
    save_serialized_to_file(os.path.join(jobdir, "merged_dead" if options.include_dead else "merged"), evdicts, uadicts_by_sta)
    np.savetxt(os.path.join(jobdir, "times.txt"), time_periods)

if __name__ == "__main__":
    main()
