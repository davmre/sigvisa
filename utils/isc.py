import os
from sigvisa.source.event import get_event, Event



def load_isc(evid):

    idc = get_event(evid=evid)

    TIMESTAMP_COL, TERR_COL, TRMS_COL, LON_COL, LAT_COL, SMAJ_COL, SMIN_COL, STRIKE_COL, DEPTH_COL, DERR_COL, METHOD_COL, SOURCE_COL, ISCID_COL, N_ISC_COLS = range(14)

    try:
        with open("/home/dmoore/python/sigvisa/experiments/scraped_events/full/%d.txt" % evid, 'r') as f:
            d = eval(f.read())
    except IOError:
        return None

    try:
        isc = d["ISC"]
    except KeyError:
        isc = d["IDC"]
    ev = Event(lon = isc[LON_COL], lat=isc[LAT_COL], depth=isc[DEPTH_COL], mb=idc.mb, time=isc[TIMESTAMP_COL])
    #print "idv ev", idc
    #print "isc ev", ev
    #print 
    return ev
