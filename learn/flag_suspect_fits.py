import numpy as np

from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.utils.geog import dist_km
from sigvisa.models.ttime import tt_predict

def build_neighbor_cache(runid, threshold_km=25.0):

    s = Sigvisa()

    r = s.sql("select fitid, evid, sta from sigvisa_coda_fit where runid=%d" % runid)
    reverse_cache = {(evid, sta): fitid for (fitid, evid, sta) in r}
    evids = set([evid for (fitid, evid, sta) in r])
    ev_cache = {evid: get_event(evid=evid) for evid in evids}
    
    print "got %d unique events" % (len(ev_cache))

    ev_neighbors = {}
    for evid, ev in ev_cache.items():
        if evid not in ev_neighbors:
            ev_neighbors[evid] = set()

        for evid2, ev2 in ev_cache.items():
            if evid2 == evid: continue

            d = dist_km((ev.lon, ev.lat), (ev2.lon, ev2.lat)) + np.abs(ev.depth - ev2.depth)
            if d < threshold_km:
                ev_neighbors[evid].add(evid2)
        print "evid %d has %d neighbors" % (evid, len(ev_neighbors[evid]))

    
    neighbor_cache = {}
    for fitid, evid, sta in r:
        neighbor_evids = ev_neighbors[evid]
        neighbor_fitids = set()
        for neighbor_evid in neighbor_evids:
            try:
                neighbor_fitid = reverse_cache[(neighbor_evid, sta)]
            except:
                continue
            neighbor_fitids.add(neighbor_fitid)
        neighbor_cache[fitid] = neighbor_fitids


    return neighbor_cache


def flag_fit(fitid):
    s = Sigvisa()
    s.sql("update sigvisa_coda_fit set human_approved='3' where fitid=%d" % fitid)
    s.dbconn.commit()
    
def get_neighbor_fits(fitid):
    return neighbor_cache[fitid]

def get_fit_params_by_phase(fitid):
    s = Sigvisa()
    query = s.sql("select f.evid, f.sta, fp.fpid, fp.phase, fp.tt_residual from sigvisa_coda_fit f, sigvisa_coda_fit_phase fp where f.fitid=%d and fp.fitid=f.fitid" % fitid)

    ttr = query[0, 4]
    #ev = get_event(evid=query[0][0])
    #sta = query[0][1]
    phase_params = {phase: {"tt_residual": ttr), "fpid": fpid} for  (_, _, fpid, phase, ttr) in query}

    return phase_params

def process_fit(fitid, neighbor_cache, param_cache):
    # need to compare fit params to params of neighbors
    # meaning I need to have the params
    # and also I need to know the neighbors and have their params
    
    s = Sigvisa()
    params = param_cache[fitid]

    # list of fitids
    neighbors = neighbor_cache[fitid]
    tt_residuals_by_phase = {phase: [] for phase in params.keys()}

    for neighbor_fitid in neighbors:
        neighbor_params = param_cache[neighbor_fitid]
        for phase in tt_residuals_by_phase.keys():
            try:
                neighbor_residual = neighbor_params[phase]["tt_residual"]
            except:
                continue
            tt_residuals_by_phase[phase].append(neighbor_residual)

    for phase, neighbor_residuals in tt_residuals_by_phase.items():
        if len(neighbor_residuals) < 2:
            continue
        
        fit_residual = params[phase]["tt_residual"]
        neighbor_median = np.median(neighbor_residuals)
        neighbor_stddev = np.std(neighbor_residuals)
        fpid = params[phase]["fpid"]

        s.sql("insert into sigvisa_ttr_consistency (fpid, tt_residual, ttr_neighbor_median, ttr_neighbor_stddev, neighbor_fitids) values (%d, %f, %f, %f, '%s')" % (fpid, fit_residual, neighbor_median, neighbor_stddev, repr(neighbors)))

        print "inserted ttrs for fitid", fitid

    s.dbconn.commit()

def flag_if_suspicious(fitid, min_neighbors=5.0, zscore_cutoff=2.5):
    r = s.sql("select abs(tc.tt_residual - tc.ttr_neighbor_median)/tc.ttr_neighbor_stddev, neighbor_fitids from sigvisa_coda_fit_phase fp, sigvisa_ttr_consistency tc where fp.fitid=%d and fp.fpid=tc.fpid" % (fitid))
    if len(r) == 0:
        return

    zscores, neighborstrs = zip(*r)
    neighbors = eval(neighborstrs[0])
    zscores = np.asarray(zscores)

    if len(neighbors) > min_neighbors and (zscores > zscore_cutoff).any():
        print "flagging", fitid, "for zscores", zscores
        flag_fit(fitid)
    
    

#neighbor_cache = build_neighbor_cache(runid=18)
#param_cache = {fitid: get_fit_params_by_phase(fitid) for fitid in neighbor_cache.keys()}
#for fitid in neighbor_cache.keys():
#    process_fit(fitid, neighbor_cache, param_cache)

runid=18
s = Sigvisa()
fitids = s.sql("select distinct fitid from sigvisa_coda_fit where runid=%d" % runid)


for (fitid,) in fitids:
    flag_if_suspicious(fitid)
