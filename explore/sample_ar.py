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
import sys

ar_perturb=1


#start_time = 1237680000 + 3600*float(sys.argv[3])
#end_time = start_time + 3600*float(sys.argv[4])
start_time = float(sys.argv[3])
end_time = float(sys.argv[4])
stalist = map(lambda x : int(x), sys.argv[2].split(','))
orid = int(sys.argv[1])
perturb = int(sys.argv[5])

print orid, stalist, start_time, end_time

cursor = database.db.connect().cursor()
detections, arid2num = read_detections(cursor, start_time, end_time, arrival_table="leb_arrival", noarrays=False)
print detections

cursor.execute("select lon, lat, depth, time, mb, orid from leb_origin "
                   "where orid=%d"
                   % (orid))
events = np.array(cursor.fetchall())
print events

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
sigmodel.synthesize_signals(events, tuple(stalist), start_time, end_time, 2, perturb, perturb)

signals = sigmodel.get_signals()

pp = PdfPages('logs/sample_ar.pdf')
for signal in signals:
    print signal
    title = "orid " + str(orid) + " station " + str(signal[0].stats['siteid']) + " times " + str(start_time) + " " + str(end_time)
    utils.waveform.plot_trace(signal[2], title="sampled envelope", format="b-")
    pp.savefig()
pp.close()
