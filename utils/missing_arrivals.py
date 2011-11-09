import matplotlib
matplotlib.use('PDF')
import database.db
from database.dataset import *
import learn, netvisa, sigvisa
#from multiprocessing import Process
import utils.waveform
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import time


def find_detection(detections, siteid, expected_time):
    for det in detections:
        if det[0] == siteid and np.abs(det[2] - expected_time) < 10:
            return True
    return False

AMP_THRESHOLD = 5

start_time = 1237683600 + 3600*4
end_time = start_time + 3600*32

cursor = database.db.connect().cursor()
detections, arid2num = read_detections(cursor, start_time, end_time, arrival_table="leb_arrival", noarrays=False)

events, orid2num = read_events(cursor, start_time, end_time, "leb")

sites = read_sites(cursor)
site_up = read_uptime(cursor, start_time, end_time)
phasenames, phasetimedef = read_phases(cursor)

earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
netmodel = learn.load_netvisa("parameters", start_time, end_time, detections, site_up, sites, phasenames, phasetimedef)




missing = []

maxplots = 40

pp = PdfPages('logs/missing_detections.pdf')
for ev in events:
    
    if maxplots <= 0:
        break

    for (siteid, site) in enumerate(sites):

        if maxplots <= 0:
            break

        expected_time = earthmodel.ArrivalTime(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], ev[EV_TIME_COL], 0, siteid)
        if expected_time < 0:
            continue
        ttime = netmodel.mean_travel_time(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], siteid, 0)

        expected_amp = netmodel.mean_amplitude(ev[EV_MB_COL], ev[EV_DEPTH_COL], expected_time - ev[EV_TIME_COL], siteid, 0)
    
        if expected_amp > AMP_THRESHOLD:
            
            if find_detection(detections, siteid, expected_time):
                print "matching detection found for site %d at time %f" % (siteid, expected_time)
                
            else:
                missed = (ev[EV_ORID_COL], siteid, expected_time, expected_amp  )
                print " missed: ", missed
                missing.append( missed  )
                try:
                    utils.waveform.plot_ss_waveforms(siteid, expected_time - 30, expected_time + 30, detections, earthmodel, ev)
                    plt.title("event %d" % ev[EV_ORID_COL])
                    pp.savefig()
                    maxplots = maxplots - 1
                except:
                    continue

pp.close();


print "final account of %d missing detections: " % (len(missing))
print missing

