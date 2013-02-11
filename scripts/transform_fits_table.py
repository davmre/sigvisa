
from sigvisa import Sigvisa
import numpy as np
import sigvisa.utils.geog

new = """
create table sigvisa_coda_fits_shadow (
 runid    int,
 evid	  int,
 sta	  varchar(10),
 chan	  varchar(10),
 lowband  float(24),
 highband  float(24),
 phase	  varchar(20),
 atime	  float(24),
 peak_delay float(24),
 coda_height float(24),
 coda_decay  float(24),
 optim_method  varchar(15),
 iid	    int,
 stime	    float,
 etime	    float,
 acost	    float,
 dist	    float,
 azi	    float,
 primary key(runid, evid, sta, chan, lowband, highband, phase)
);
"""

s = Sigvisa()
cursor = s.dbconn.cursor()
cursor.execute("drop table sigvisa_coda_fits_shadow")
cursor.execute(new)
s.dbconn.commit()

get_fit = "select runid, arid, chan, band, peak_delay, coda_height, coda_decay, optim_method, iid, stime, etime, acost, dist, azi from sigvisa_coda_fits"
getmisc = "select l.sta, leba.phase, round(l.time,4), lebo.evid from leb_arrival l, leb_assoc leba, leb_origin lebo where l.arid=%d and leba.arid=l.arid and leba.orid=lebo.orid"
cursor.execute(get_fit)

cursor2 = s.dbconn.cursor()

for row in cursor:
    sql_query = getmisc % row[1]
#    print sql_query
    cursor2.execute(sql_query)
    misc = cursor2.fetchone()
#    print row, misc
    print row[0], row[1]

    lowband, highband = tuple([float(st) for st in row[3].split('_')])

    insert_query = "insert into sigvisa_coda_fits_shadow (runid, evid, sta, chan, lowband, highband, phase, atime, peak_delay, coda_height, coda_decay, optim_method, iid, stime, etime, acost, dist, azi) values (%d, %d, '%s', '%s', %f, %f, '%s', %f, %f, %f, %f, '%s', %d, %f, %f, %f, %f, %f)" % (row[0], misc[3], misc[0], row[2], lowband, highband, misc[1], misc[2], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13])
#    print insert_query
    cursor2.execute(insert_query)

cursor.execute("drop table sigvisa_coda_fits")
cursor.execute("alter table sigvisa_coda_fits_shadow rename to sigvisa_coda_fits")

s.dbconn.commit()
