import numpy as np
import sys
import os

from sigvisa.graph.serialization import load_serialized_from_file
from sigvisa import Sigvisa

from sigvisa.results.evaluate_serialized import get_bulletin
from sigvisa.results.compare import find_matching

def populate_db(serialized_file, run_name, dbtable="sigvisa_origin", isc_bulletin=None, max_delta_deg=2.0, max_delta_time=50.0):
    evdicts, uadicts_by_sta = load_serialized_from_file(serialized_file)
    bulletin = np.array(sorted([(d["lon"], d["lat"], d["depth"], d["time"], d["mb"], d['score'] if 'score' in d else 0.0) for d in evdicts], key = lambda x: x[3]))

    orid_mapping = {}
    if isc_bulletin is not None:
        indices = find_matching(isc_bulletin, bulletin, 
                                max_delta_deg=max_delta_deg, max_delta_time=max_delta_time)
        orid_mapping = {j: isc_bulletin[i, 6] for (i, j) in indices}

    s = Sigvisa()
    for j, row in enumerate(bulletin):
        lon, lat, depth, time, mb, score = row[:6]

        if j in orid_mapping:
            orid_str = "%d" % orid_mapping[j]
        else:
            orid_str = "NULL"


        q = "insert into %s (lon, lat, depth, time, mb, score, matched_isc_orid, run_name) values (%f, %f, %f, %f, %f, %f, %s, '%s')" % (dbtable, lon, lat, depth, time, mb, score, orid_str, run_name)
        s.sql(q)

 
    s.dbconn.commit()


def main():

    serialized = sys.argv[1]
    run_name = os.path.basename(os.path.dirname(serialized))
    print run_name

    timefile = os.path.join(os.path.dirname(serialized), 'times.txt')
    time_blocks = np.loadtxt(timefile).reshape((-1, 2))
    isc_bulletin = get_bulletin(time_blocks, origin_type="isc")

    populate_db(serialized, run_name, isc_bulletin=isc_bulletin)

if __name__ == "__main__":
    main()
