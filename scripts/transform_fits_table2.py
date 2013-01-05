
from sigvisa import Sigvisa
import numpy as np
import utils.geog

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
#s.cursor.execute("drop table sigvisa_coda_fit_shadow")
#s.cursor.execute(new)
#s.dbconn.commit()

get_fit = "select runid, evid, sta, chan, lowband, highband, phase, round(atime,4), peak_delay, coda_height, coda_decay, optim_method, iid, round(stime,4), round(etime,4), acost, dist, azi from sigvisa_coda_fits"
s.cursor.execute(get_fit)
print "executed"

cursor2 = s.dbconn.cursor()

fitids = dict()

i=1
for row in s.cursor:

    if i % 100 == 0:
        print i
    i += 1

    band = "freq_%.1f_%.1f" % (row[4], row[5])

    key = (row[0], row[1], row[2], row[3], band, row[11], row[12])
    if not key in fitids:
        insert_query = "insert into sigvisa_coda_fit (runid, evid, sta, chan, band, optim_method, iid, stime, etime, acost, dist, azi) values (%d, %d, '%s', '%s', '%s', '%s', %d, %f, %f, %f, %f, %f)" % (row[0], row[1], row[2], row[3], band, row[11], row[12], row[13], row[14], row[15], row[16], row[17])
        cursor2.execute(insert_query)
        cursor2.execute("select fitid from sigvisa_coda_fit where runid=%d and evid=%d and sta='%s' and chan='%s' and band='%s' and optim_method='%s' and iid=%d" % key)
        fitids[key] = cursor2.fetchone()[0]
    phase_insert_query = "insert into sigvisa_coda_fit_phase (fitid, phase, template_model, param1, param2, param3, param4) values (%d, '%s', 'paired_exp', %f, %f, %f, %f)" % (fitids[key], row[6], row[7], row[8], row[9], row[10])
    cursor2.execute(phase_insert_query)

    s.dbconn.commit()


#cursor.execute("drop table sigvisa_coda_fits")
#cursor.execute("alter table sigvisa_coda_fits_shadow rename to sigvisa_coda_fits")

s.dbconn.commit()
