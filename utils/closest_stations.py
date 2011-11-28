import database.db
from database.dataset import *
import utils.geog
import sys

if len(sys.argv) < 2:
    print "not enough arguments (need leb event ID): ", sys.argv
    sys.exit(1)

cursor = database.db.connect().cursor()
sites = read_sites(cursor)
cursor.execute("select lon, lat, depth, time, mb, orid from leb_origin "
                   "where orid=%s" % (sys.argv[1]))
ev = cursor.fetchone()
print ev
print "occurs", (float(ev[EV_TIME_COL]) - 1237680000)/3600.0, "hours in."
ssites = utils.geog.stations_by_distance(ev[EV_LON_COL],ev[EV_LAT_COL],sites)

for site in ssites:
    print site[0]+1, site[1][0]
