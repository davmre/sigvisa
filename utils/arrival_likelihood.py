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
import sigvisa_util

ar_perturb=1


arid = int(sys.argv[1])

cursor = database.db.connect().cursor()

cursor.execute("select ss.id, ar.time, ar.amp, ar.azimuth, ar.slow from leb_arrival ar, static_siteid ss where arid=%d and ss.sta=ar.sta" % (arid))
arr = np.array(cursor.fetchall())[0]

start_time = arr[1] - 10
end_time = start_time + 40

sites = read_sites(cursor)
site_up = read_uptime(cursor, start_time, end_time)
phasenames, phasetimedef = read_phases(cursor)

detections, arid2num = read_detections(cursor, start_time, end_time, "idcx_arrival", 1)


earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
#netmodel = learn.load_netvisa("parameters", start_time, end_time, detections, site_up, sites, phasenames, phasetimedef)


sigmodel = learn.load_sigvisa("parameters", start_time, end_time, ar_perturb,
                              site_up, sites, phasenames,
                              phasetimedef)
stalist=(int(arr[0]),)
print stalist
energies, traces = sigvisa_util.load_and_process_traces(cursor, start_time, end_time, window_size=1, overlap=0.5, stalist=stalist)
sigmodel.set_waves(traces)
sigmodel.set_signals(energies)

arrll = [0 for i in range(0,40)]
for i in range(0,40):
    t = float(i)/2
    atime = start_time + t
    arrll[i] = sigmodel.arrival_likelihood(atime, arr[2], arr[3], arr[4], 0)

bestt = 0
besttll = float("-inf")
for i in range(0,40):
    t = float(i)/2
    atime = start_time + t
    print "ll of start_time", atime, "is", arrll[i]
    if arrll[i] > besttll:
        bestt = t-10
        besttll = arrll[i]

print "logging ", bestt, besttll, sigmodel.arrival_likelihood(start_time+bestt, arr[2], arr[3], arr[4], 1)

pp = PdfPages('logs/ar_likelihood.pdf')
x =  np.arange(-10, 10,0.5)
plt.figure()
plt.xlabel("Time (s)")
plt.ylabel("Log-likelihood")
plt.plot(x, arrll[0:40], 'b-')
pp.savefig()
pp.close()
