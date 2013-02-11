import matplotlib
matplotlib.use('PDF')
import sigvisa.database.db
from sigvisa.database.dataset import *
import learn, netvisa, sigvisa, sigvisa_util
#from multiprocessing import Process
import sigvisa.utils.waveform
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import time
import sys

def get_analagous_segment(t, segments):
    siteid = t[0].stats['siteid']
    start_time = t[0].stats['starttime_unix']
    if "window_size" in t[0].stats:
        srate = 1/ ( t[0].stats.window_size * (1- t[0].stats.overlap) )
        npts = t[0].stats.npts_processed
    else:
        srate = t[0].stats.sampling_rate
        npts = t[0].stats.npts
    end_time = start_time + npts/srate

    for s in segments:
        if s[0].stats['siteid'] != siteid:
            continue


        sstart_time = s[0].stats['starttime_unix']
        if sstart_time > start_time:
            continue

        if "window_size" in s[0].stats:
            srate = 1/ ( s[0].stats.window_size * (1- s[0].stats.overlap) )
            npts = s[0].stats.npts_processed
        else:
            srate = s[0].stats.sampling_rate
            npts = s[0].stats.npts
        s_end_time = sstart_time + npts/srate
        if s_end_time < end_time:
            continue

        start_idx = int((start_time-sstart_time)*srate)
        end_idx = int((end_time-start_time)*srate + start_idx)

        new_seg = []
        for c in s:
            new_data = c.data[start_idx:end_idx]

            new_header = c.stats.copy()
            new_header["npts_processed"] = len(new_data)
            new_header["starttime_unix"] = start_time
            new_c = Trace(new_data, new_header)
            new_seg.append(new_c)
        return new_seg
    return None


ar_perturb=1


evid = int(sys.argv[1])
stalist = map(lambda x : int(x), sys.argv[2].split(','))

print evid, stalist

cursor = database.db.connect().cursor()

cursor.execute("select lon, lat, depth, time, mb, orid from leb_origin "
                   "where evid=%d"
                   % (evid))
events = np.array(cursor.fetchall())
print events
event = events[0]
start_time = event[EV_TIME_COL]
end_time = event[EV_TIME_COL] + 2000

detections, arid2num = read_detections(cursor, start_time, end_time, arrival_table="leb_arrival", noarrays=False)
print detections

sites = read_sites(cursor)
print sites
site_up = read_uptime(cursor, start_time, end_time)
phasenames, phasetimedef = read_phases(cursor)

earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
netmodel = learn.load_netvisa("parameters", start_time, end_time, detections, site_up, sites, phasenames, phasetimedef)


sigmodel = learn.load_sigvisa("parameters", start_time, end_time, ar_perturb,
                              site_up, sites, phasenames,
                              phasetimedef)


sigvisa.srand(int((time.time()*100) % 1000 ))

print "synth", events, stalist, start_time, end_time, 2
sigmodel.synthesize_signals(events, tuple(stalist), start_time, end_time, 5, 0, 0)

sim_signals = sigmodel.get_signals()


energies, traces = sigvisa_util.load_and_process_event_traces(cursor, [evid,], stations=stalist)

pp = PdfPages('logs/event_log.pdf')
for real_segment in energies:
    print real_segment
    print real_segment[0], real_segment[0].stats
    sim_segment = get_analagous_segment(real_segment, sim_signals)
    print sim_segment
    print sim_segment[0], sim_segment[0].stats
    title = "evid " + str(evid) + " station " + str(real_segment[0].stats['siteid']) + " time " + str(real_segment[0].stats['starttime_unix'])



    start_time = real_segment[0].stats["starttime_unix"]
    srate = real_segment[0].stats.sampling_rate
    npts = real_segment[0].stats.npts
    end_time = start_time + npts/srate

    site_dets = filter(lambda x: x[DET_SITE_COL]+1 == int(real_segment[0].stats['siteid']) and x[DET_TIME_COL] < end_time and x[DET_TIME_COL] > start_time, detections)
    site_det_times = map(lambda x: x[DET_TIME_COL], site_dets)
    print "all det times", site_det_times
    utils.waveform.plot_segment(real_segment, title="real envelope: " + title, format="r-", all_det_times=site_det_times)
    pp.savefig()

    if sim_segment is not None:
        utils.waveform.plot_segment(sim_segment, title="sampled envelope: "+title, format="b-")
        pp.savefig()



pp.close()
