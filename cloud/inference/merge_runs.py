import os
import time
import numpy as np

from sigvisa.utils.fileutils import mkdir_p

from sortedcontainers import SortedSet
from frozendict import frozendict

from optparse import OptionParser

from sigvisa.graph.serialization import load_serialized_from_file, save_serialized_to_file, merge_serializations, extract_time_period

def merge_times(timeblocks_list):
    # ensure all runs are over the same time period
    times = timeblocks_list[0]
    #for t in timeblocks_list[1:]:
    #    assert(t.shape == times.shape)
    #    assert( np.all(t == times) )
    return times

def merge_scored_events(evdicts_list, dupe_threshold_deg=1.0, 
                        dupe_threshold_time=50, score_rule='max'):
    
    keys = ("lon", "lat", "time", "depth", "mb", "score")

    gold_events = SortedSet(key = lambda d : d['time'])

    def find_duplicate(evd):
        for evd2 in gold_events.irange_key(evd['time'] - dupe_threshold_time, evd['time']+dupe_threshold_time):
            if np.abs(evd2['lon'] - evd['lon']) < dupe_threshold_deg and np.abs(evd2['lat'] - evd['lat']) < dupe_threshold_deg:
                return evd2
        return None

    def merge_evs(evd1, evd2, rule='max'):
        master = evd1 if evd1['loc_score'] > evd2['loc_score'] else evd2

        if rule=='sum':
            master = dict(master)
            master['score'] = evd1['score'] + evd2['score']
            master = frozendict(master)
        elif rule=='logsum':
            master = dict(master)
            master['score'] = np.logaddexp(evd1['score'], evd2['score'])
            master = frozendict(master)
        elif rule=='l2':
            master = dict(master)
            master['score'] = np.sqrt(evd1['score']**2 + evd2['score']**2)
            master = frozendict(master)
        elif rule=='max':
            master = dict(master)
            master['score'] = max(evd1['score'], evd2['score'])
            master = frozendict(master)

        print "merging events %s %s" % (evd1, evd2)
        return master

    for evdicts in evdicts_list:
        for evd in evdicts:

            # preprocess the dicts a bit
            tmp = {key: evd[key] for key in keys if key in evd}
            if 'score' not in tmp:
                print "warning: no score for", tmp
                tmp['score'] = 0.0
            tmp['loc_score'] = tmp['score']

            sub_evd = frozendict(tmp)
            dupe = find_duplicate(sub_evd)
            if dupe is None:
                gold_events.add(sub_evd)
            else:
                gold_events.remove(dupe)
                combined = merge_evs(dupe, sub_evd, rule=score_rule)
                gold_events.add(combined)

    unfrozen = [dict(evd) for evd in gold_events]
    return unfrozen

def main():
    
    parser = OptionParser()

    parser.add_option("--outdir", dest="outdir", default=None, type="str",
                      help="name of directory to write results")
    parser.add_option("--scoring_rule", dest="scoring_rule", default="sum", type="str",
                      help="how to combine scores of matching events (sum, logsum, l2, max)")

    (options, args) = parser.parse_args()

    evdicts_list = []
    times_list = []
    for indir in args:
        serialized = os.path.join(indir, "merged.tgz")
        evdicts, uadicts_by_sta = load_serialized_from_file(serialized)
        evdicts_list.append(evdicts)

        try:
            timefile = os.path.join(indir, 'times.txt')
            time_blocks = np.loadtxt(timefile).reshape((-1, 2))
            times_list.append(time_blocks)
        except:
            pass

    merged_times = merge_times(times_list)
    mkdir_p(options.outdir)
    np.savetxt(os.path.join(options.outdir, "times.txt"), merged_times)

    merged_evdicts = merge_scored_events(evdicts_list, score_rule=options.scoring_rule)
    save_serialized_to_file(os.path.join(options.outdir, 'merged'), 
                            merged_evdicts, 
                            uadicts_by_sta = {})


if __name__ == "__main__":
    main()
