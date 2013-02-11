import sigvisa.database.db
from sigvisa.database.dataset import *
import sigvisa.utils.geog
import sys

cursor = database.db.connect().cursor()
sites = read_sites(cursor)

for siteid in range(1, 117):

    sql_query="SELECT l.time, lebo.mb, lebo.lon, lebo.lat, l.azimuth, lebo.evid FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where l.time between 1238889600 and 1245456000 and lebo.mb>4 and leba.arid=l.arid and l.snr > 1 and lebo.orid=leba.orid and sid.sta=l.sta and sid.statype='ss' and (l.iphase='S' or l.iphase='Sn') and sid.id=%d order by l.sta" % (siteid)
    cursor.execute(sql_query)
    arrivals = np.array(cursor.fetchall())
    if arrivals.shape[0] == 0:
        continue

    regional = 0
    for arrival in arrivals:
        distance = utils.geog.dist_km((arrival[2], arrival[3]), (sites[siteid-1][0], sites[siteid-1][1]))
        if distance <= 600:
            regional = regional+1
    print "siteid %d: %d regional events" % (siteid, regional)
