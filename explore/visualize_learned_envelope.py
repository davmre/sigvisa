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


evid = int(sys.argv[1])
siteid = int(sys.argv[2])

cursor = db.connect().cursor()
cursor.execute("select lon, lat, depth, time, mb, orid from leb_origin "
                   "where evid=%d"
                   % (evid))
events = np.array(cursor.fetchall())
event = events[0]
print "loaded event", event


pp = PdfPages('logs/learned_env_%d_%d.pdf' % (evid, siteid))

# select detections of this event at this site

sql_query = "select site.id-1, iarr.arid, iarr.time, iarr.deltim, iarr.azimuth, iarr.delaz, iarr.slow, iarr.delslo, iarr.snr, ph.id-1, iarr.amp, iarr.per from leb_arrival iarr, leb_assoc iass, leb_origin ior, static_siteid site, static_phaseid ph where iarr.delaz > 0 and iarr.delslo > 0 and iarr.snr > 0 and iarr.sta=site.sta and iarr.iphase=ph.phase and ascii(iarr.iphase) = ascii(ph.phase) and iarr.arid=iass.arid and iass.orid=ior.orid and ior.evid=%d and site.id=%d order by iarr.time, iarr.arid" %  (evid, siteid)
print sql_query

cursor.execute(sql_query)
detections = np.array(cursor.fetchall())
print "loaded detections", detections

start_time = detections[0][2] - 20
end_time = detections[-1][2] + 100

sites = read_sites(cursor)
site_up = read_uptime(cursor, start_time, end_time)
phasenames, phasetimedef = read_phases(cursor)

sigmodel = learn.load_sigvisa("parameters", start_time, end_time, "envelope", site_up,
                        sites, phasenames, phasetimedef)

fake_det = [sigvisa_util.real_to_fake_det(x) for x in detections]
sigmodel.set_fake_detections(fake_det)
print "set detections", fake_det

energies, traces = sigvisa_util.load_and_process_traces(cursor, start_time, end_time, stalist=[siteid,])
#energies = sigvisa_util.trim_first_n_seconds(energies, 5)
sigmodel.set_signals(energies)

params = {siteid: dict()}
params = priors.SignalPrior.learn_envelope_params(sigmodel, siteid, params)
sigmodel.set_all_signal_params(params)




for segment in energies:

    trc = segment[0]

    siteid = trc.stats["siteid"]
    trc_start_time = trc.stats["starttime_unix"]
    if "window_size" in trc.stats.keys():
        srate = 1/ ( trc.stats.window_size * (1- trc.stats.overlap) )
        npts = trc.stats.npts_processed
    else:
        srate = trc.stats.sampling_rate
        npts = trc.stats.npts
    trc_end_time = trc_start_time + npts/srate

    arrtimes = []
    for det in detections:
        det_time = det[2]
        if det_time >= trc_start_time and det_time <= trc_end_time:
            arrtimes.append(det_time)

    plot_segment(segment, title= "REAL ENVELOPE: siteid %d" % (siteid), all_det_times = arrtimes)
    pp.savefig()

    sigmodel.synthesize_signals_det((siteid,), trc_start_time, trc_end_time, 5, 0, 0)
    synthesized = sigmodel.get_signals()
    plot_segment(synthesized[0], title= "LEARNED ENVELOPE: siteid %d" % (siteid), all_det_times = arrtimes)
    pp.savefig()


pp.close()
