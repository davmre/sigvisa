import matplotlib
matplotlib.use('PDF')
import database.db
from database.dataset import *
import learn, netvisa, sigvisa, sigvisa_util
#from multiprocessing import Process
import utils.waveform
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import time
import sys

def get_analagous_segment(t, segments):
    siteid = t.stats['siteid']
    start_time = t.stats['starttime_unix']
    if t.stats["window_size"] is not None:
        srate = 1/ ( t.stats.window_size * (1- t.stats.overlap) )
        npts = t.stats.npts_processed
    else:
        srate = t.stats.sampling_rate
        npts = t.stats.npts
    end_time = start_time + npts/srate

    for s in segments:
        if s.stats['siteid'] != siteid:
            continue

        sstart_time = s.stats['starttime_unix']
        if sstart_time > start_time:
            continue
        
        if s.stats["window_size"] is not None:
            srate = 1/ ( s.stats.window_size * (1- s.stats.overlap) )
            npts = s.stats.npts_processed
        else:
            srate = s.stats.sampling_rate
            npts = s.stats.npts
        s_end_time = sstart_time + npts/srate
        if s_end_time < end_time:
            continue

        start_idx = int((start_time-sstart_time)*srate)
        end_idx = int((end_time-start_time)*srate + start_idx)

        new_data = s.data[start_idx:end_idx]

        new_header = s.stats.copy()
        new_header["npts_processed"] = len(new_data)
        new_header["starttime_unix"] = start_time
        new_seg = Trace(new_data, new_header)
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
sigmodel.synthesize_signals(events, tuple(stalist), start_time, end_time, 2, 1, 1)

sim_signals = sigmodel.get_signals()


energies, traces = sigvisa_util.load_and_process_traces(cursor, start_time, end_time, window_size=1, overlap=0.5, stalist=stalist)

pp = PdfPages('logs/sample_ar.pdf')
for real_segment in energies:
    print real_segment
    sim_segment = get_analagous_segment(real_segment, sim_signals)
    title = "evid " + str(evid) + " station " + str(real_segment.stats['siteid']) + " time " + real_segment.stats['start_time']
    utils.waveform.plot_segment(real_segment, title="real envelope: " + title, format="r-")
    pp.savefig()

    if sim_segment is not None:
        utils.waveform.plot_segment(sim_segment, title="sampled envelope: "+title, format="b-")
        pp.savefig()



pp.close()

