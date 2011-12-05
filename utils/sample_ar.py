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

ar_perturb=1


start_time = 1237680000 + 3600*float(sys.argv[3])
end_time = start_time + 3600*float(sys.argv[4])
stalist = map(lambda x : int(x), sys.argv[2].split(','))
orid = int(sys.argv[1])

cursor = database.db.connect().cursor()
detections, arid2num = read_detections(cursor, start_time, end_time, arrival_table="leb_arrival", noarrays=False)


cursor.execute("select lon, lat, depth, time, mb, orid from leb_origin "
                   "where orid=%d"
                   % (orid))
events = np.array(cursor.fetchall())

sites = read_sites(cursor)
site_up = read_uptime(cursor, start_time, end_time)
phasenames, phasetimedef = read_phases(cursor)

earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
netmodel = learn.load_netvisa("parameters", start_time, end_time, detections, site_up, sites, phasenames, phasetimedef)


sigmodel = learn.load_sigvisa("parameters", start_time, end_time, ar_perturb, 
                              site_up, sites, phasenames,
                              phasetimedef)


print "synth", events, stalist, start_time, end_time, 2
sigmodel.synthesize_signals(events, tuple(stalist), start_time, end_time, 2, 1, 1)

signals = sigmodel.get_signals()

pp = PdfPages('logs/sample_ar.pdf')
for signal in signals:
    print signal
    title = "orid " + str(orid) + " station " + str(signal[0].stats['siteid']) + " times " + str(start_time) + " " + str(end_time)
    utils.waveform.plot_segment(signal, title=title)
    pp.savefig()
pp.close()
