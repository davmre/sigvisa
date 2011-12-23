import matplotlib
matplotlib.use('PDF')
import database.db
from database.dataset import *
#from multiprocessing import Process
import utils.waveform
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import time
import sys
from utils.draw_earth import draw_events, draw_earth, draw_density


stalist = map(lambda x : int(x), sys.argv[2].split(','))
orid = int(sys.argv[1])

cursor = database.db.connect().cursor()
cursor.execute("select lon, lat, depth, time, mb, orid from leb_origin "
                   "where orid=%d"
                   % (orid))
events = np.array(cursor.fetchall())
print events
event = events[0]


f = open(sys.argv[3], "r")

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
                  nofillcontinents=True, figsize=(5.5,4))

draw_events(bmap, np.array(((event[0], event[1]),)),
            marker="*", ms=10, mfc="none", mec="yellow", mew=2)
draw_events(bmap, np.array(((-172.857833, -15.978006),)),
            marker="s", ms=10, mfc="none", mec="blue", mew=2)

# draw a density
LON_BUCKET_SIZE = resolution
# Z axis is along the earth's axis
# Z goes from -1 to 1 and will have the same number of buckets as longitude
Z_BUCKET_SIZE = (2.0 / 360.0) * LON_BUCKET_SIZE

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
for line in f:
    tokens = line.strip().split(" ")
    loni = int(tokens[0])
    lati = int(tokens[1])
    sc = float(tokens[2])
    score[loni, lati] = sc
    print loni, lati, "=", sc
    if sc > best: 
        print "new best", sc
        best = sc
    if sc < worst: 
        print "new worst", sc
        worst = sc

print worst, best
if worst < best * .9:
    levels = np.arange(worst, best*.9, (best*.9 - worst)/5).tolist() +\
        np.linspace(best*.9, best, 5).tolist()
else:
    levels = np.linspace(worst, best, 10).tolist()

  # round the levels so they are easier to display in the legend
levels = np.round(levels, 1).tolist()

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
plt.savefig("logs/heat_fromfile.pdf")

