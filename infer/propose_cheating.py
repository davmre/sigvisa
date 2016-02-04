import numpy as np
from sigvisa import Sigvisa

def load_true_events():
    # store events as an array with cols LON, LAT, DEPTH, TIME, MB
    s = Sigvisa()
    fname = os.path.join(s.homedir, "experiments", "cheating.npy")
    return np.load(fname)

def cheating_location_proposal(sg, fix_result=None):
    try:
        true_events = sg.cheating_events
    except:
        true_events = load_true_events()
        sg.cheating_events = true_events
    
    if fix_result is None:
        plon, plat, pdepth, ptime, pmb = np.random.choice(true_events)
    else:
        ev = fix_result
        plon, plat, pdepth, ptime, pmb = ev.lon, ev.lat, ev.depth, ev.time, ev.mb

    lp = -np.inf
    rv_loc = Gaussian(0.0, 0.01)
    rv_dt = Gaussian(0.0, 3.0)
    rv_mb = Gaussian(0.0, 0.2)
    for xx in true_events:
        evlp = 0
        lon, lat, depth, time, mb = xx
        evlp += rv_loc.log_p(lon-plon)
        evlp += rv_loc.log_p(lat-plat)
        evlp += rv_dt.log_p(depth-pdepth)
        evlp += rv_dt.log_p(time-ptime)
        evlp += rv_mb.log_p(mb-pmb)
        lp = np.logaddexp(lp, evlp)
    
    if fix_result is None:
        proposed_ev = Event(lon=plon, lat=plat, depth=pdepth, time=ptime, mb=pmb)
        return proposed_ev, lp, ()
    else:
        return lp
