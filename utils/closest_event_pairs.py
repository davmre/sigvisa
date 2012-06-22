import database.db
from database.dataset import *
import utils.geog
import sys
import itertools

if len(sys.argv) < 2:
    print "not enough arguments (need station name): ", sys.argv
    sys.exit(1)
else:
    sta = sys.argv[1]

cursor = database.db.connect().cursor()
sites = read_sites(cursor)
cursor.execute("select distinct lebo.lon, lebo.lat, lebo.depth, lebo.evid from leb_origin lebo, leb_assoc leba, leb_arrival l, sigvisa_coda_fits fit where leba.arid=fit.arid and leba.orid=lebo.orid and l.arid=leba.arid and l.sta='%s'" % (sta))
events = cursor.fetchall()

evpairs = itertools.combinations(events, 2)

p = [(utils.geog.dist_km((e1[0], e1[1]), (e2[0], e2[1])), e1[2], e2[2], e1[3], e2[3]) for (e1, e2) in evpairs]
p.sort()
for pair in p[:40]:
    print "evids %d, %d are at distance %.3fkm and have depths (%.3fkm, %.3fkm)" % (pair[3], pair[4], pair[0], pair[1], pair[2])

