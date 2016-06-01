import numpy as np
from sigvisa.source.event import get_event
from sigvisa.models.ttime import tt_predict
from sigvisa import Sigvisa

s = Sigvisa()
missing_ttrs = s.sql("select fp.fpid, fp.arrival_time, fp.phase, f.sta, f.evid from sigvisa_coda_fit_phase fp, sigvisa_coda_fit f where fp.tt_residual is NULL and f.fitid=fp.fitid")
for (fpid, atime, phase, sta, evid) in missing_ttrs:
    ev = get_event(evid=evid)
    try:
        pred_atime = tt_predict(ev, sta, phase) + ev.time
    except:
        continue

    tt_residual = atime - pred_atime
    query = "update sigvisa_coda_fit_phase set tt_residual=%f where fpid=%d" % (tt_residual, fpid)
    s.sql(query)
    print query

s.dbconn.commit()
