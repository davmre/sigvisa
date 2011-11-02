import database.db
from database.dataset import *
import utils.geog

cursor = database.db.connect().cursor()
detections, arid2num = read_detections(cursor, 1237680000, 1237680000 + 168*3600, arrival_table="leb_arrival", noarrays=False)

last_det = dict()

overlaps = 0
for det in detections:
    site = det[0]
    time = det[2]

    if site in last_det:
        gap = time - last_det[site]
        if gap < 5:
            print " arrival %d at siteid %d occured %f seconds after previous at %f : phase %s" % (det[1], site, gap, last_det[site], det[DET_PHASE_COL])
            overlaps = overlaps+1
    last_det[site] = time

print "total overlaps: ", overlaps, " out of ", len(detections), " detections"
    
