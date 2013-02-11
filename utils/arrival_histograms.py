import matplotlib
matplotlib.use('PDF')
import sigvisa.database.db
from sigvisa.database.dataset import *
import learn, netvisa, sigvisa
#from multiprocessing import Process
import sigvisa.utils.waveform
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
end_time = start_time + 3600*168

cursor = database.db.connect().cursor()
detections, arid2num = read_detections(cursor, start_time, end_time, arrival_table="leb_arrival", noarrays=False)

events, orid2num = read_events(cursor, start_time, end_time, "leb")
evlist = read_assoc(cursor, start_time, end_time, orid2num, arid2num, "leb")
sites = read_sites(cursor)
site_up = read_uptime(cursor, start_time, end_time)
phasenames, phasetimedef = read_phases(cursor)

earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
netmodel = learn.load_netvisa("parameters", start_time, end_time, detections, site_up, sites, phasenames, phasetimedef)




for target_phase in range(0,14):

    adifflists = dict()
    adifflists["local"] = []
    adifflists["regional"] = []
    adifflists["tele"] = []
    sdifflists = dict()
    sdifflists["local"] = []
    sdifflists["regional"] = []
    sdifflists["tele"] = []
    tdifflists = dict()
    tdifflists["local"] = []
    tdifflists["regional"] = []
    tdifflists["tele"] = []

    pp = PdfPages('logs/arrival_histograms_%d.pdf' %(target_phase))
    for (evnum, ev) in enumerate(events):
        for (phaseid, detnum) in evlist[evnum]:
            det = detections[detnum]
            siteid = int(det[DET_SITE_COL])
            if phaseid != target_phase:
                continue




            pred_time = earthmodel.ArrivalTime(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], ev[EV_TIME_COL], phaseid, siteid)
            if pred_time < 0:
                continue

            pred_azi = earthmodel.ArrivalAzimuth(ev[EV_LON_COL], ev[EV_LAT_COL], siteid)
            pred_slo = earthmodel.ArrivalSlowness(ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL], phaseid, siteid)
            tdiff = det[DET_TIME_COL] - pred_time
            adiff = det[DET_AZI_COL] - pred_azi
            adiff = adiff if adiff < 180 else adiff -180
            adiff = adiff if adiff > -180 else adiff +180
            sdiff = det[DET_SLO_COL] - pred_slo
            print " pred azi ", pred_azi, " det azi ", det[DET_AZI_COL], " diff ", adiff

            dist = AVG_EARTH_RADIUS_KM * (np.pi/180) * earthmodel.Delta(ev[EV_LON_COL], ev[EV_LAT_COL], siteid)
            tdifflists[dist2str(dist)].append(tdiff)
            adifflists[dist2str(dist)].append(adiff)
            sdifflists[dist2str(dist)].append(sdiff)

    for (k, l) in tdifflists.items():
        if len(l) < 1:
            continue
        plt.figure()
        plt.title("TIME: " + str(k) + " phase " + str(target_phase) + " (n = " + str(len(l)) + ")" )
        plt.hist(l, 20)
        pp.savefig()
    for (k, l) in adifflists.items():
        if len(l) < 1:
            continue
        plt.figure()
        plt.title("AZI: " + str(k) + " phase " + str(target_phase) + " (n = " + str(len(l)) + ")" )
        plt.hist(l, 20)
        pp.savefig()
    for (k, l) in sdifflists.items():
        if len(l) < 1:
            continue
        plt.figure()
        plt.title("SLO: " + str(k) + " phase " + str(target_phase) + " (n = " + str(len(l)) + ")" )
        plt.hist(l, 20)
        pp.savefig()
    pp.close();
