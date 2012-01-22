import database.db
from database.dataset import *
import utils.geog
import sys
import sigvisa_util

if len(sys.argv) < 2:
    print "not enough arguments (need leb event ID): ", sys.argv
    sys.exit(1)

evid = int(sys.argv[1])

cursor = database.db.connect().cursor()
sites = read_sites(cursor)

sql_query = "select site.id-1, iarr.arid, iarr.time, iarr.deltim, iarr.azimuth, iarr.delaz, iarr.slow, iarr.delslo, iarr.snr, ph.id-1, iarr.amp, iarr.per from leb_arrival iarr, leb_assoc iass, leb_origin ior, static_siteid site, static_phaseid ph where iarr.delaz > 0 and iarr.delslo > 0 and iarr.snr > 0 and iarr.sta=site.sta and iarr.iphase=ph.phase and ascii(iarr.iphase) = ascii(ph.phase) and iarr.arid=iass.arid and iass.orid=ior.orid and ior.evid=%d and site.statype='ss' order by iarr.snr desc" %  (evid)

cursor.execute(sql_query)
dets = cursor.fetchall()

print "WARNING: THESE ARE SITEID-1s, NOT SITEIDS"
for det in dets:
    print sigvisa_util.real_to_fake_det(det)
