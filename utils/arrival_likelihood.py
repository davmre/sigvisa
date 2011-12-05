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
import sys
import sigvisa_util

ar_perturb=1


arid = int(sys.argv[1])

cursor = database.db.connect().cursor()

cursor.execute("select ss.id, ar.time, ar.amp, ar.azimuth, ar.slow from leb_arrival ar, static_siteid ss where arid=%d and ss.sta=ar.sta" % (arid))
arr = np.array(cursor.fetchall())[0]

start_time = arr[1] - 5
end_time = start_time + 35

sites = read_sites(cursor)
site_up = read_uptime(cursor, start_time, end_time)
phasenames, phasetimedef = read_phases(cursor)

detections, arid2num = read_detections(cursor, start_time, end_time, "idcx_arrival", 1)


earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
#netmodel = learn.load_netvisa("parameters", start_time, end_time, detections, site_up, sites, phasenames, phasetimedef)


sigmodel = learn.load_sigvisa("parameters", start_time, end_time, ar_perturb, 
                              site_up, sites, phasenames,
                              phasetimedef)

energies, traces = sigvisa_util.load_and_process_traces(cursor, start_time, end_time, window_size=1, overlap=0.5, stalist=(int(arr[0]),))
sigmodel.set_waves(traces)
sigmodel.set_signals(energies)

for t in range(0,10):
    atime = start_time + t
    arrll = sigmodel.arrival_likelihood(atime, arr[2], arr[3], arr[4], 0)
    print "ll of start_time", atime, "is", arrll

#pp = PdfPages('logs/sample_ar.pdf')
#for signal in signals:
#    print signal
#    title = "orid " + str(orid) + " station " + str(signal[0].stats['siteid']) + " times " + str(start_time) + " " #+ str(end_time)
#    utils.waveform.plot_segment(signal, title=title)
#    pp.savefig()
#pp.close()
