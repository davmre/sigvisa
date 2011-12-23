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
import hashlib
import sigvisa_util
from utils.draw_earth import draw_events, draw_earth, draw_density


stalist = map(lambda x : int(x), sys.argv[2].split(','))
evid = int(sys.argv[1])
print "params ", stalist, evid

hsh = hashlib.sha1(str(evid) + str(stalist)).hexdigest()
print "param hash", hsh

cursor = database.db.connect().cursor()
cursor.execute("select lon, lat, depth, time, mb, orid from leb_origin "
                   "where evid=%d"
                   % (evid))
events = np.array(cursor.fetchall())
print events
event = events[0]

start_time = event[3] - 100
end_time = event[3] + 2000


format_strings = ','.join(['%s'] * len(stalist))
print format_strings
print tuple(stalist)
cursor.execute("select lon, lat from static_siteid where id IN (%s)" % format_strings, tuple(stalist))
sta_locations = np.array(cursor.fetchall())
print sta_locations

detections, arid2num = read_detections(cursor, start_time, end_time, arrival_table="leb_arrival", noarrays=False)
print detections
sites = read_sites(cursor)
print sites
site_up = read_uptime(cursor, start_time, end_time)
phasenames, phasetimedef = read_phases(cursor)

earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
netmodel = learn.load_netvisa("parameters", start_time, end_time, detections, site_up, sites, phasenames, phasetimedef)


sigmodel = learn.load_sigvisa("parameters", start_time, end_time, 1, 
                              site_up, sites, phasenames,
                              phasetimedef)


sigvisa.srand(int((time.time()*100) % 1000 ))


energies, traces = sigvisa_util.load_and_process_traces(cursor, start_time, end_time, window_size=1, overlap=0.5, stalist=stalist)
sigmodel.set_waves(traces)
sigmodel.set_signals(energies)

print event
print event[3], event[0], event[1], event[2], event[4]

f = open("logs/heatmap_%s.log" % (hsh), "w")

window = 10
resolution = 1.5

lon1 = event[0] - window
lon2 = event[0] + window
lat1 = event[1] - window
lat2 = event[1] + window

bmap = draw_earth("",
                  #"NET-VISA posterior density, NEIC(white), LEB(yellow), "
                  #"SEL3(red), NET-VISA(blue)",
                  projection="mill",
                  resolution="l",
                  llcrnrlon = lon1, urcrnrlon = lon2,
                  llcrnrlat = lat1, urcrnrlat = lat2,
                  nofillcontinents=True, figsize=(4.5,4))

draw_events(bmap, np.array(((event[0], event[1]),)),
            marker="o", ms=10, mfc="none", mec="yellow", mew=2)

draw_events(bmap, sta_locations,
            marker="x", ms=10, mfc="none", mec="red", mew=2)


# draw a density
LON_BUCKET_SIZE = resolution
# Z axis is along the earth's axis
# Z goes from -1 to 1 and will have the same number of buckets as longitude
Z_BUCKET_SIZE = (4.0 / 360.0) * LON_BUCKET_SIZE

# skip one degree at the bottom of the map to display the map scale
# otherwise, the density totally covers it up
lon_arr = np.arange(event[0] - window,
                    event[0] + window,
                    LON_BUCKET_SIZE)
z_arr = np.arange(np.sin(np.radians(event[1] - window *.88)),
                  np.sin(np.radians(event[1] + window)),
                  Z_BUCKET_SIZE)
lat_arr = np.degrees(np.arcsin(z_arr))

score = np.zeros((len(lon_arr), len(lat_arr)))
best, worst = -np.inf, np.inf
for loni, lon in enumerate(lon_arr):
    for lati, lat in enumerate(lat_arr):
        if lon<-180: lon+=360
        if lon>180: lon-=360

        sc = -np.inf
        for evtime in np.arange(event[3]-8, event[3] + 8.1, 2):
            sc = max(sc, sigmodel.event_likelihood(evtime, lon, lat, event[2], event[4]))
      
        score[loni, lati] = sc
        f.write(str(loni) + " " + str(lati) + " " + str(sc) + " " + str(lon) + " " + str(lat) + "\n")
        f.flush()
        print "set score for lon %f lat %f to %f" % (lon, lat, sc)

        if sc > best: best = sc
        if sc < worst: worst = sc

if worst < best * .9:
    levels = np.arange(worst, best*.9, (best*.9 - worst)/5).tolist() +\
        np.linspace(best*.9, best, 5).tolist()
else:
    levels = np.linspace(worst, best, 10).tolist()

  # round the levels so they are easier to display in the legend
levels = np.round(levels, 1).tolist()

print lon_arr, lat_arr, score
draw_density(bmap, lon_arr, lat_arr, score, levels = levels, colorbar=True)
  
  
  # add a map scale
scale_lon, scale_lat = event[0], \
    event[1]-window * .98
try:
    bmap.drawmapscale(scale_lon, scale_lat, scale_lon, scale_lat,
                      window*100,
                      fontsize=8, barstyle='fancy',
                      labelstyle='simple', units='km')
except:
    pass

f.close()
plt.savefig("logs/heat_%s.pdf" % (hsh))

