from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.models.ttime import tt_predict

import numpy as np

runid=14
s = Sigvisa()
query = "select f.evid, f.sta, f.fitid, fp.fpid, fp.phase, fp.arrival_time from sigvisa_coda_fit f, sigvisa_coda_fit_phase fp where f.fitid=fp.fitid and f.runid=%d" % runid
fits = s.sql(query)

bad_fits = []
for (evid, sta, fitid, fpid, phase, arrival_time) in fits:
    ev = get_event(evid)
    pred_at = ev.time + tt_predict(ev, sta, phase)
    ttr = np.abs(pred_at - arrival_time)
    if ttr > 25.0:
        bad_fits.append(fpid)
        print "fit", fitid, "has phase", phase, "with tt residual", ttr

cursor = s.dbconn.cursor()
for fpid in bad_fits:
    cursor.execute("delete from sigvisa_coda_fit_phase where fpid=%d" % fpid)
cursor.close()
s.dbconn.commit()
