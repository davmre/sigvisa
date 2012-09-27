import database.db
from database.dataset import *
import time
import learn
import sys
import numpy as np

evid = int(sys.argv[1])

cursor = database.db.connect().cursor()

sites = read_sites(cursor)
phasenames, phasetimedef = read_phases(cursor)
earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)

cursor.execute("select og.time, og.lon, og.lat, og.depth, ss.id, ar.sta, ar.arid, ar.time, ar.iphase, ar.amp, ar.azimuth, ar.slow from leb_arrival ar, leb_assoc ac, leb_origin og, static_siteid ss  where ss.sta=ar.sta and ar.arid=ac.arid and ac.orid=og.orid and og.evid=%d and ss.statype='ar'" % (evid))

res = cursor.fetchall()
for r in res:
    (lon, lat, depth, time) = earthmodel.InvertDetection(r[4], r[10], r[11], r[7])
    print r
    print time, lon, lat, depth

print (res[0][0] - 1237680000) / 3600.0
