import matplotlib
matplotlib.use('PDF')
import database.db
from database.dataset import *
import learn, netvisa, sigvisa
import sigvisa_util
import utils.waveform
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import time

AVG_EARTH_RADIUS_KM = 6371

def dist2str(dist):
    if dist < 100:
        return "local"
    elif dist < 1400:
        return "regional"
    else:
        return "tele"

start_time = 1237680000
end_time = start_time + 3600*20

cursor = database.db.connect().cursor()
#detections, arid2num = read_detections(cursor, start_time, end_time, arrival_table="leb_arrival", noarrays=False)

#events, orid2num = read_events(cursor, start_time, end_time, "leb")
#evlist = read_assoc(cursor, start_time, end_time, orid2num, arid2num, "leb")
sites = read_sites(cursor)
#site_up = read_uptime(cursor, start_time, end_time)
phasenames, phasetimedef = read_phases(cursor)

earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
#netmodel = learn.load_netvisa("parameters", start_time, end_time, detections, site_up, sites, phasenames, phasetimedef)

cursor.execute("select sta, id from static_siteid")
stations = np.array(cursor.fetchall())
stations = stations[:,0]
print stations

#adifflists = dict()
#adifflists["local"] = []
#adifflists["regional"] = []
#adifflists["tele"] = []

idifflists = dict()
idifflists["local"] = []
idifflists["regional"] = []
idifflists["tele"] = []

phaseid=8
v1list = []

#pp = PdfPages('logs/arrival_histograms_%d.pdf' %(target_phase))
#for (evnum, ev) in enumerate(events):
#    for (phaseid, detnum) in evlist[evnum]:
for siteid in range(0, 100, 1):
    for dist in range(0, 181, 20):
        depth=0
        #siteid=2
        sitelon=sites[siteid-1][SITE_LON_COL]
        sitelat=sites[siteid-1][SITE_LAT_COL]

        evlon = sitelon - dist
        if evlon < -180:
            evlon = evlon+360
        evlat = sitelat

        delta = earthmodel.Delta(evlon, evlat, siteid)
        pred_iangle = earthmodel.ArrivalIncidentAngle(evlon, evlat, depth, phaseid, siteid)
#        pred_azi = earthmodel.ArrivalAzimuth(ev[EV_LON_COL], ev[EV_LAT_COL], siteid)
        pred_slow = earthmodel.ArrivalSlowness(evlon, evlat, depth, phaseid, siteid)
        if pred_iangle < 0:
            continue
        
        v1 = np.sin(pred_iangle* np.pi/180)/pred_slow
        v2 = 1/(pred_slow*np.sin(pred_iangle* np.pi/180))

 #       traces = sigvisa_util.load_traces(cursor, [stations[siteid]], det[DET_TIME_COL]-1, det[DET_TIME_COL]+6)
 #       # print traces, siteid, stations[siteid], det[DET_TIME_COL]
 #       if traces is None or len(traces) == 0:
 #           continue
 #       segment = traces[0]
 #       try:
 #           azi, amp, iangle = sigvisa_util.estimate_azi_amp_slo(segment)
 #       except sigvisa_util.MissingChannel:
 #           continue
 #       idiff = iangle - pred_iangle
 #       adiff = azi - pred_azi
        print "siteid", siteid, "dist", dist, "depth", depth, "delta", delta, "iangle", pred_iangle, "sin(iangle)", np.sin(pred_iangle* np.pi/180), "slow", pred_slow, "v1", v1, "v2", v2
        v1list.append(v1)
   
#for (k, l) in idifflists.items():
#    if len(l) < 1:
#        continue
#    plt.figure()
#    plt.title("IANGLE: " + str(k) + " phase " + str(target_phase) + " (n = " + str(len(l)) + ")" )
#    plt.hist(l, 20)
#    pp.savefig()
#pp.close();

print np.mean(v1list)
print np.median(v1list)
