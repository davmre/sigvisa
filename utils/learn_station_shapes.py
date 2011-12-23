import os, sys
import numpy as np
import matplotlib.pyplot as plt

from database.dataset import *
from database import db

import learn, sigvisa_util
import priors.SignalPrior

hours = 100

start_time = 1237680000
end_time = start_time + hours*3600
earliest_event_time = start_time - MAX_TRAVEL_TIME



cursor = db.connect().cursor()

leb_detections, leb_arid2num = read_detections(cursor, start_time, end_time, "leb_arrival", 1)  
idcx_detections, arid2num = read_detections(cursor, start_time, end_time, "idcx_arrival", 1)

  
print start_time, earliest_event_time
events, orid2num = read_events(cursor, earliest_event_time, end_time, "leb", None)
print "loaded ", len(events), " events."
  
# read associations, as training data for learning
evlist = read_assoc(cursor, earliest_event_time, end_time, orid2num, arid2num, "leb", None)
print "loaded associations for ", len(events), " events."

sites = read_sites(cursor)
site_up = read_uptime(cursor, start_time, end_time)
phasenames, phasetimedef = read_phases(cursor)

sigmodel = learn.load_sigvisa("parameters", start_time, end_time, 1, site_up,
                        sites, phasenames, phasetimedef)

fake_det = [sigvisa_util.real_to_fake_det(x) for x in leb_detections]
sigmodel.set_fake_detections(fake_det)
print "set detections", fake_det



cursor.execute("select ss.id from static_siteid ss where ss.statype='ss'")
three_axis_stations = np.array(cursor.fetchall())
print three_axis_stations

sta_params=dict()
for sta in three_axis_stations:

    try:
        energies, traces = sigvisa_util.load_and_process_traces(cursor, start_time, end_time, window_size=1, overlap=0.5, stalist=[int(sta),])
        sigmodel.set_signals(energies)

        print "learning envelope params for station", sta
        env_params = priors.SignalPrior.learn_envelope_params(sigmodel)
        print env_params
        sta_params[sta[0]]=env_params
    except:
        print "error, continuing!"
        continue


print "done!"
for (k,v) in sta_params.items():
    print "station", k, "params", v
