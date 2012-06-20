import os, errno, sys, time, traceback
import numpy as np, scipy

from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plot
import learn, sigvisa_util
import signals.SignalPrior
from utils.waveform import *
import utils.geog
import obspy.signal.util


import utils.nonparametric_regression as nr
from priors.coda_decay.coda_decay_common import *
from priors.coda_decay.plot_coda_decays import *


def main():
# boilerplate initialization of various things
    siteid = int(sys.argv[1])
    method="bfgs"
    if len(sys.argv) > 2:
        method = sys.argv[2]
    iid=True
    by_phase=False
    snr_threshold=2

    cursor, sigmodel, earthmodel, sites, dbconn = sigvisa_util.init_sigmodel()

# want to select all events, with certain properties, which have a P or S phase detected at this station
    phase_condition = "(" + " or ".join(["leba.phase='%s'" % (pn) for pn in S_PHASES + P_PHASES]) + ")"        
    sql_query="SELECT distinct lebo.mb, lebo.lon, lebo.lat, lebo.evid, lebo.time, lebo.depth FROM leb_arrival l , static_siteid sid, static_phaseid pid, leb_origin lebo, leb_assoc leba, dataset d where d.label='training' and l.time between d.start_time and d.end_time and lebo.mb>5 and leba.arid=l.arid and l.snr > 6 and lebo.orid=leba.orid and %s and sid.sta=l.sta and sid.statype='ss' and sid.id=%d and pid.phase=leba.phase" % (phase_condition, siteid)
#5308821
#5301405
# and lebo.evid=5301449

    print sql_query
    cursor.execute(sql_query)
    events = np.array(cursor.fetchall())

#    bands = ['narrow_envelope_4.00_6.00', 'narrow_envelope_2.00_3.00', 'narrow_envelope_1.00_1.50', 'narrow_envelope_0.70_1.00']
    short_bands = [b[16:] for b in bands]

    cursor.execute("select max(runid) from sigvisa_coda_fits")
    runid, = cursor.fetchone()
    if runid is None:
        runid=0
    else:
        runid = int(runid)+1

    base_coda_dir = get_base_dir(siteid, runid)

    for event in events:
        evid = int(event[EV_EVID_COL])
        cmd_str = "python2.6 -m priors.coda_decay.fit_shape_params  %d %s %d %d" % (siteid, method, evid, runid)
        print "running", cmd_str
        os.system(cmd_str)

    dbconn.close()

if __name__ == "__main__":
    main()






