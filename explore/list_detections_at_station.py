import sigvisa.database.db
from sigvisa.database.dataset import *
import time
import learn
import sys
import numpy as np

siteid = int(sys.argv[1])
start_time = float(sys.argv[2])
end_time = start_time + float(sys.argv[3])*3600

cursor = database.db.connect().cursor()

sites = read_sites(cursor)
site_up = read_uptime(cursor, start_time, end_time)
phasenames, phasetimedef = read_phases(cursor)
earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)

cursor.execute("select og.orid, og.time, og.lon, og.lat, og.depth, ar.arid, ar.time, ar.iphase, ar.amp, ar.azimuth, ar.slow from leb_arrival ar, leb_assoc ac, leb_origin og, static_siteid ss  where ss.id=%d and ss.sta=ar.sta and ar.time between %f and %f and ar.arid=ac.arid and og.orid=ac.orid" % (siteid, start_time, end_time))

print "og.orid, og.time, og.lon, og.lat, og.depth, ar.arid, ar.time, ar.phase, ar.amp, ar.azimuth, ar.slow"
res = cursor.fetchall()
for r in res:
    expected_time = earthmodel.ArrivalTime(r[2], r[3], r[4], r[1], np.where(phasenames==r[7])[0][0], siteid)
    print r, expected_time, "off by", r[6]-expected_time
