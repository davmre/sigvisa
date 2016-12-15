import numpy as np

import sys

from datetime import datetime
import calendar 

from sigvisa import Sigvisa
from sigvisa.results.evaluate_serialized import get_bulletin
from sigvisa.results.compare import find_matching

def merge_bulletins(bulletin1, bulletin2, max_delta_deg=2.0, max_delta_time=50.0):

    """
    Merge events from two bulletins. If an event occurs in both bulletins, prefer bulletin 1. 
    """


    indices = find_matching(bulletin1, bulletin2, 
                            max_delta_deg=max_delta_deg, max_delta_time=max_delta_time)
    reverse_mapping = dict([(j, i) for (i,j) in indices])
    merged = list(bulletin1)
    for j, ev in enumerate(bulletin2):
        if j not in reverse_mapping.keys():
            merged.append(ev)
        #else:
        #    gold_ev = bulletin1[reverse_mapping[j]]
        #    print "merging event %s with %s" % (ev, gold_ev)
    merged = sorted(merged, key = lambda ev: ev[3]) # sort by event time
    return np.asarray(merged)

def adjust_magnitude_hack(bulletin1, bulletin2, max_delta_deg=2.0, max_delta_time=50.0):

    # given bulletins 1 and 2, find the events present in both, and learn a linear mapping from magnitudes in 1 to magnitudes in 2. 
    # then pass *all* magnitudes in bulletin1 through this mapping to generate a new bulletin1 with magnitudes on a consistent scale to bulletin2
    # this is a hack and really we should be doing smart analysis of body-wave, surface-wave, moment, etc magnitudes, but those are all hacks too so maybe it's okay. 

    bulletin1, bulletin2 = np.asarray(bulletin1), np.asarray(bulletin2)
    indices = find_matching(bulletin1, bulletin2, 
                            max_delta_deg=max_delta_deg, max_delta_time=max_delta_time)
    mbs1 = []
    mbs2 = []

    for (i,j) in indices:
        mb1, mb2 = bulletin1[i, 4], bulletin2[j,4]
        if mb1 <= -1 or mb2 <= -1: continue
        mbs1.append(mb1)
        mbs2.append(mb2)
    mbs1 = np.asarray(mbs1).reshape(-1, 1)
    mbs2 = np.asarray(mbs2)

    from sklearn.linear_model import LinearRegression
    regr = LinearRegression()

    # Train the model using the training sets
    regr.fit(mbs1, mbs2)

    print regr.coef_, regr.intercept_
    all_mbs1 = bulletin1[:, 4].reshape(-1, 1)
    y = regr.predict(all_mbs1)

    new_bulletin1 = bulletin1.copy()
    new_bulletin1[:, 4] = y
    return new_bulletin1

def sigvisa_knowngood_bulletin():
    s = Sigvisa()
    q = "select sb.lon, sb.lat, sb.depth, sb.time, sb.mb, sb.score, -1 from sigvisa_origin sb, sigvisa_origin_rating sbr where sb.orid=sbr.orid and sbr.rating='2'"
    return s.sql(q)

def read_ddphs_origins(fname):

    # Origin Time           LAT        LON        Z      Mag      EH     EZ      RMS    EVID
    # 2008 05  05 19 36 20.206   41.1534   -114.8430  10.76    1.28   2.32    1.68   0.084   244163  Phases:  12

    # ['2008', '05', '05', '19', '36', '20.206', '41.1534', '-114.8430', '10.76', '1.28', '2.32', 'read_ddphs_origins1.68', '0.084', '244163', 'Phases:', '12']

    def parse_origin_line(line):
        parts = line.split()
        
        _, yr, mo, day, hr, mn, sec, lat, lon, depth, mag, eh, ez, rms, evid, _, phases = parts
        yr, mo, day, hr, mn, sec = int(yr), int(mo), int(day), int(hr), int(mn), float(sec)
        lat, lon, depth, mag, eh, ez, rms, evid = float(lat), float(lon), float(depth), float(mag), float(eh), float(ez), float(rms), int(evid)

        secint = int(sec)
        ms = sec - secint

        dt = datetime(yr, mo, day, hr, mn, secint)
        ts = calendar.timegm(dt.timetuple()) + ms

        v = (lon, lat, depth, ts, mag, 0.0, evid)
        return v

    with open(fname, 'r') as f:
        lines = f.readlines()
    origin_lines = [line.strip() for line in lines if line.startswith("#")]
    origins = np.array([parse_origin_line(line) for line in origin_lines])

    #sizable_origins = np.array([origin for origin in origins if origin[4] > 2.0])

    start_time = 1203646562
    end_time = 1204856162
    origins = [ev for ev in origins if start_time < ev[3] < end_time]

    return origins

def main():

    ddphs_fname = sys.argv[1]
    isc_fname = sys.argv[2]

    origins = read_ddphs_origins(ddphs_fname)
    isc_bulletin = np.loadtxt(isc_fname)

    origins = adjust_magnitude_hack(origins, isc_bulletin)

    large_origins = [ev for ev in origins if ev[4] > 2.0]
    merged = merge_bulletins(origins, isc_bulletin)
    merged_large = merge_bulletins(large_origins, isc_bulletin)

    sigvisa_knowngood = sigvisa_knowngood_bulletin()
    merged_analyst = merge_bulletins(merged, sigvisa_knowngood)

    np.savetxt('wells_origins.txt', origins)
    np.savetxt('merged_origins.txt', merged)
    np.savetxt('merged_origins_large.txt', merged_large)
    np.savetxt('merged_analyst.txt', merged_analyst)

if __name__ == "__main__":
    main()
