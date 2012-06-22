import sigvisa_util
from database import db
import numpy as np
import utils.geog

cursor = db.connect().cursor()

sql_query = "select distinct fit.arid, lebo.lon, lebo.lat, sid.lon, sid.lat, leba.seaz from leb_origin lebo, leb_assoc leba, leb_arrival l, sigvisa_coda_fits fit, static_siteid sid where fit.arid=l.arid and l.arid=leba.arid and leba.orid=lebo.orid and sid.sta=l.sta"
cursor.execute(sql_query)
rows = np.array(cursor.fetchall())
for r in rows:
    azi1 = utils.geog.azimuth((r[3], r[4]), (r[1], r[2]))
    sql_query = "update sigvisa_coda_fits set azi=%f where arid=%d" % (azi1, r[0])
    print sql_query
    cursor.execute(sql_query)



