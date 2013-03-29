from sigvisa import *
from sigvisa.source.event import get_event

s = Sigvisa()
cursor1 = s.dbconn.cursor()

sql_query = "select f.evid, f.band, fp.phase, fp.coda_height, fp.fpid from sigvisa_coda_fit f, sigvisa_coda_fit_phase fp where fp.fitid=f.fitid and fp.amp_transfer is null"
cursor1.execute(sql_query)

cursor2 = s.dbconn.cursor()

i = 0
for (evid, band, phase, amp, fpid) in cursor1:
    ev = get_event(evid=evid)
    try:
        transfer = amp - ev.source_logamp(band, phase)
    except Exception as e:
        print e
        continue

    sql_query = "update sigvisa_coda_fit_phase set amp_transfer=:t where fpid=:fpid"
    cursor2.execute(sql_query, t=transfer, fpid=fpid)
    i += 1
    if (i % 500) == 0:
        s.dbconn.commit()
        print "updated %d rows" % i
s.dbconn.commit()
