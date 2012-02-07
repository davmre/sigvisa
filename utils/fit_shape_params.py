import os, sys
import numpy as np
import matplotlib.pyplot as plt

from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
from matplotlib.backends.backend_pdf import PdfPages


import learn, sigvisa_util
import priors.SignalPrior
from utils.waveform import *
import utils.geog


def arrival_peak_offset(trace):
    window_end = trace.stats['sampling_rate'] * 15
    return np.argmax(trace.data[0:window_end]) / trace.stats['sampling_rate']

def gen_logenvelope(length, sampling_rate, gamma, b):
    t = linspace(0, length, sampling_rate*length)
    f = t**(gamma * -1) * np.exp(b * t)
    return f

def logenv_l1_cost(true_env, logenv):
    return np.sum (np.abs(true_env - logenv))

def fit_logenvelope(trace, peak_time):
    srate = trace.stats['sampling_rate']
    true_env = trace.data[peak_time*srate :]
    length = len(trace.data) - srate*peak_time
    cost = lambda(gamma, b): logenv_l1_cost(true_env, gen_logenvelope(length, srate, gamma, b))

    bounds = ((0, 2), (-.02, 0))
    results = scipy.optimize.brute(cost, bounds, Ns=20, full_output=0, finish=None)
    return results

siteid = int(sys.argv[1])

cursor = db.connect().cursor()

sites = read_sites(cursor)
print sites[siteid-1]

sql_query="SELECT l.time, lebo.mb, lebo.lon, lebo.lat FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where l.time between 1238889600 and 1245456000 and lebo.mb>4 and leba.arid=l.arid and lebo.orid=leba.orid and (iphase='S' or iphase='Sn' or iphase='ScP') and sid.sta=l.sta and sid.statype='ss' and sid.id=%d order by l.sta" % (siteid)
cursor.execute(sql_query)
arrivals = np.array(cursor.fetchall())

segments = []
for arrival in arrivals:
    start_time = arrival[0] - 5
    end_time = arrival[0] + 200
    arrival_segment = sigvisa_util.load_and_process_traces(cursor, start_time, end_time, stalist=[siteid,])
    segments = segments + arrival_segment

    distance = utils.geog.dist_km((arrival[2], arrival[3]), (sites[siteid-1][0], sites[siteid-1][1]))

    trace = arrival_segment[0]['horiz_avg']['narrow_logenvelope_2.00_3.00']
    peak_time = arrival_peak_offset(trace)

    (gamma, b) = fit_logenvelope(trace, peak_time)
    print "fit params", gamma, b, "at station", siteid, ", event at distance", distance

#pp = PdfPages('logs/learned_env_%d_%d.pdf' % (evid, siteid))

# select detections of this event at this site


#    plot_segment(synthesized[0], title= "LEARNED ENVELOPE: siteid %d" % (siteid), all_det_times = arrtimes)
#    pp.savefig()


#pp.close()
