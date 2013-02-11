import matplotlib
matplotlib.use('PDF')
import sigvisa.database.db
from sigvisa.database.dataset import *
import learn, netvisa, sigvisa
#from multiprocessing import Process
import sigvisa.utils.waveform
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sigvisa_util import *
import time
import sigvisa.infer

event_list = [5290918, 5288718, 5295646, 5288706, 5295770, 5288789, 5288706, 5295878, 5295625, 5288718, 5289106, 5291164, 5295646, 5290918, 5295559, 5295796, 5297206, 5295543, 5288506, 5289149, 5288896, 5289283, 5297348, 5287879, 5297354, 5290958, 5288911, 5288789, 5289307, 5288031, 5291235, 5289063, 5289234, 5289071, 5297650, 5288827]
cursor = database.db.connect().cursor()

start_time = 1237680000
end_time = start_time + 3600*36

loop=50

events, orid2num = read_events(cursor, start_time, end_time, "leb")

out = open("local_max.log", 'a')

for ev in events:
    if ev[EV_ORID_COL] not in event_list:
        continue

    try:
        print "trying inference on event", ev
        skip = (ev[EV_TIME_COL] - 1237680000) / 3600
        hours = 0.5
        start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
            sel3_evlist, site_up, sites, phasenames, phasetimedef, arid2num \
            = read_data("validation", hours=hours, \
                    skip=skip, noarrays=True)

        sites = read_sites(cursor)
        site_up = read_uptime(cursor, start_time, end_time)
        phasenames, phasetimedef = read_phases(cursor)

        earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
        netmodel = learn.load_netvisa("parameters", start_time, end_time, detections, site_up, sites, phasenames, phasetimedef)

#print "loading traces for SIGVISA..."
        cursor.execute("select sta, id from static_siteid where statype='ss'")
        stations = np.array(cursor.fetchall())
        siteids = dict(stations)
        stations = stations[:,0]

    # convert all values in the siteids dictionary from strings to ints
    # TODO: figure out the proper pythonic way to do this
        siteids_ints = dict()
        for sta in siteids.keys():
            siteids_ints[sta] = int(siteids[sta])
        siteids = siteids_ints

        sigmodel = learn.load_sigvisa("parameters",
                                      start_time, end_time,
                                      site_up, sites, phasenames,
                                      phasetimedef)


        ssites = utils.geog.stations_by_distance(ev[EV_LON_COL], ev[EV_LAT_COL],sites)[0:5]
        stalist = map(lambda site : site[0] + 1, ssites)
        print stalist, start_time, end_time

        stalist = tuple(stalist)
        energies, traces = load_and_process_traces(cursor, start_time, end_time, 1, .5, stalist)
        print "loaded, setting waves"
        sigmodel.set_waves(traces)
        print "loaded, setting energies"
        sigmodel.set_signals(energies)

        prop_events = read_events(cursor, start_time, end_time, "leb")[0]
        f = lambda e : e[EV_ORID_COL] == ev[EV_ORID_COL]
        propose_events = filter(f, prop_events)
        propose_events.sort(cmp=lambda x,y: cmp(x[EV_TIME_COL], y[EV_TIME_COL]))
        propose_events = np.array(propose_events)
        print propose_events

        pp = PdfPages('logs/compare_%d_%d.pdf' % (ev[EV_ORID_COL], loop))
        inf_events, assoc = sigmodel.infer(22, loop,
                                           2,
                                           1800, 1800,
                                           1,
                                           propose_events,
                                           True,
                                           lambda a,b,c,d,e,f: 0,
                                           lambda b,c,d,e,f: infer.log_envelope_plot(pp,b,c,d,e,f))
        pp.close()
        out.write("event " + str(ev[EV_ORID_COL]))
        out.write(", inferred: " + str(inf_events))
        out.write(" vs original: " + str(ev) + "\n")
    except:
        continue
