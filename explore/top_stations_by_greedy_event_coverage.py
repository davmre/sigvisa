import numpy as np

from collections import defaultdict
from sigvisa import Sigvisa

"""
Find a set of stations that jointly detect as many events as possible. That is, if you could only have N stations to detect global events, which ones would you want? 

Solving this in general is a complicated combinatorial optimization problem. It becomes especially complicated when you require that each event has >3 detections, ideally from different directions (small azimuthal gap) so that it can be well localized.  

Instead we do a simple greedy approximation. 

"""

def sta_evid_list(sta):
    evids = s.sql("select lebo.evid from leb_origin lebo, leb_assoc leba, leb_arrival l where l.arid=leba.arid and leba.orid=lebo.orid and l.time between 1238889600 and 1245456000 and l.sta='%s'" % sta)
    return set([int(evid[0]) for evid in evids])

s = Sigvisa()

stas = [sta[0] for sta in s.sql("select sta from static_siteid")]
nstas = len(stas)

evids_detected = {sta: sta_evid_list(sta) for sta in stas}
print "loaded evids from database"

greedy_stas = []
greedy_evs_covered = defaultdict(int)
greedy_coverage_1 = [0,]
greedy_coverage_3 = [0,]
greedy_coverage_4 = [0,]

while len(greedy_stas) < nstas:

    sta_scores = {}
    
    for sta, evids in evids_detected.items():
        # score for a new station is
        #   # of first detections
        # + # of second detections
        # + # of third detections

        first_detections = len([evid for evid in evids if greedy_evs_covered[evid] == 0])
        second_detections = len([evid for evid in evids if greedy_evs_covered[evid] == 1])
        third_detections = len([evid for evid in evids if greedy_evs_covered[evid] == 2])
        fourth_detections = len([evid for evid in evids if greedy_evs_covered[evid] == 3])
        score = first_detections + second_detections + third_detections + fourth_detections
    
        sta_scores[sta] = score

    best_sta = max(sta_scores, key=sta_scores.get)
    greedy_stas.append(best_sta)
    evids = evids_detected[best_sta]
    
    first_detections = len([evid for evid in evids if greedy_evs_covered[evid] == 0])
    third_detections = len([evid for evid in evids if greedy_evs_covered[evid] == 2])
    fourth_detections = len([evid for evid in evids if greedy_evs_covered[evid] == 3])
    for evid in evids:
        greedy_evs_covered[evid] += 1
        
    total_1 = greedy_coverage_1[-1] + first_detections
    total_3 = greedy_coverage_3[-1] + third_detections
    total_4 = greedy_coverage_4[-1] + fourth_detections
    greedy_coverage_1.append(total_1)
    greedy_coverage_3.append(total_3)
    greedy_coverage_4.append(total_4)
    del evids_detected[best_sta]

    print "%s score %d total coverage single %d triple %d quad %d" % (best_sta, sta_scores[best_sta], total_1, total_3, total_4)

print "finished"
print "stas", greedy_stas
print "single coverage", greedy_coverage_1[1:]
print "triple coverage", greedy_coverage_3[1:]
print "quad coverage", greedy_coverage_4[1:]

print "top 30"
print "stas", greedy_stas[:30]
print "quad coverage", greedy_coverage_4[1:31]
