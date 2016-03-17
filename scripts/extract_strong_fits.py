import numpy as np

from sigvisa import Sigvisa

s = Sigvisa()

runid=12
target_evs=400
target_distant_evs=100

outfile="weaker_evids"

for site in "ANMO,ELK,ILAR,KDAK,NEW,NVAR,PDAR,PFO,TXAR,ULM,YBH,YKA".split(","):
    n_evids = 0
    min_height=14.0

    sta = s.get_default_sta(site)
    while n_evids < target_evs and min_height > 1.0:
        min_height -= 0.5
        query = "select distinct f.evid from sigvisa_coda_fit f, sigvisa_coda_fit_phase fp where f.fitid=fp.fitid and f.runid=%d and f.sta='%s' and fp.coda_height > %f and f.dist < 3000" % (runid, sta, min_height)
        evids = s.sql(query)
        n_evids = len(evids)

    print site, n_evids, min_height
    n_evids = 0
    min_height = 14
    while n_evids < target_distant_evs and min_height > 1.0:
        min_height -= 0.5
        query = "select distinct f.evid from sigvisa_coda_fit f, sigvisa_coda_fit_phase fp where f.fitid=fp.fitid and f.runid=%d and f.sta='%s' and fp.coda_height > %f and f.dist between 600 and 3000" % (runid, sta, min_height)
        distant_evids = s.sql(query)
        n_evids = len(distant_evids)

    big_evids = set([int(evid[0]) for evid in evids])
    distant_evids = set([int(evid[0]) for evid in distant_evids])
    evids = big_evids.union( distant_evids)
    

    print site, len(big_evids), len(distant_evids), len(evids), min_height
    with open(outfile, 'a') as f:
        for evid in evids:
            f.write("%s %d\n" % (site, int(evid)))
                    
