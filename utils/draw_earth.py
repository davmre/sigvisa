from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import geog

def draw_earth(title, **args):
  """
  default projection="moll" and resolution="l"
  """
  if "nofillcontinents" in args:
    del args["nofillcontinents"]
    nofillcontinents = True
  else:
    nofillcontinents = False

  if "figsize" in args:
    figsize = args["figsize"]
    del args["figsize"]
  else:
    figsize = (8, 4.8)
  
  if "projection" not in args:
    args["projection"] = "moll"
  if "resolution" not in args:
    args["resolution"] = "l"
  
  # if no args other than projection and resolution then center the map
  if len(args) == 2:
    args["lat_0"] = 0
    args["lon_0"] = 0

  plt.figure(figsize=figsize)
  plt.title(title)
  bmap = Basemap(**args)
  try:
    bmap.drawmapboundary(fill_color=(.7,.7,1,1))
  except:
    try:
      bmap.drawmapboundary()
    except:
      pass

  try:
    bmap.drawcoastlines(zorder=10)
  except:
    bmap.drawcoastlines()
    
  plt.subplots_adjust(left=0.02, right=0.98)

  if not nofillcontinents:
    # fill the continents with a greenish color
    try:
      bmap.fillcontinents(color=(0.5,.7,0.5,1), lake_color=(.7,.7,1,1),
                          zorder=1)
    except:
      bmap.fillcontinents(color=(0.5,.7,0.5,1))
    # fill the oceans with a bluish color
    try:
      bmap.drawmapboundary(fill_color=(.7,.7,1))
    except:
      try:
        bmap.drawmapboundary()
      except:
        pass
  
  #bmap.drawmeridians(np.arange(-180,210,30))
  #bmap.drawparallels(np.arange(-90,120,30))
  return bmap

def draw_events(bmap, events, **args):
  """
  The args can be any collection of marker args like ms, mfc mec
  """
  if "zorder" not in args:
    args["zorder"] = 10
  
  for ev in events:
    x,y = bmap(ev[0], ev[1])
    bmap.plot([x], [y], **args)

def draw_events_mag(bmap, events, **args):
  """
  The args can be any collection of marker args like ms, mfc mec
  """
  if "zorder" not in args:
    args["zorder"] = 10

  base_ms = args["ms"]
  del args["ms"]
  
  for ev in events:
    x,y = bmap(ev[0], ev[1])
    args["ms"] = int(base_ms * pow(ev[4],2.01))
    bmap.plot([x], [y], **args)

  args["ms"] = base_ms                  # restore marker size

def draw_vectors(bmap, vectors, scale, **args):
  for (lon1, lat1, lon2, lat2) in vectors:
    x1, y1 = bmap(lon1, lat1)
    x2, y2 = bmap(lon2, lat2)
    plt.arrow(x1, y1, scale * (x2-x1), scale * (y2-y1), **args)
    
def draw_density(bmap, lons, lats, vals, levels=10, colorbar=True):
  loni, lati = np.mgrid[0:len(lons), 0:len(lats)]
  lon_arr, lat_arr = lons[loni], lats[lati]
  
  # convert to map coordinates
  x, y = bmap(list(lon_arr.flat), list(lat_arr.flat))
  x_arr = np.array(x).reshape(lon_arr.shape)
  y_arr = np.array(y).reshape(lat_arr.shape)

  cs1 = bmap.contour(x_arr, y_arr, vals, levels, linewidths=.5, colors="k",
                     zorder=6)
  cs2 = bmap.contourf(x_arr, y_arr, vals, levels, cmap=plt.cm.jet, zorder=5,
                      extend="both",
                      norm=matplotlib.colors.BoundaryNorm(cs1.levels,
                                                          plt.cm.jet.N))

  if colorbar:
    plt.colorbar(cs2, orientation="vertical", drawedges=True)

def draw_events_arrivals(bmap, events, arrivals, sites, ttime, quant=2):
  """
  threshold - time in seconds between expected arrival time and the arrival
  """
  evcolors = ["red", "yellow", "blue", "orange", "purple", "cyan", "magenta"]
  evcolnum = -1
  for (ev_lon, ev_lat, ev_depth, ev_time, ev_mb) in events:
    evcolnum = (evcolnum + 1) % len(evcolors)
    ev_x, ev_y = bmap(ev_lon, ev_lat)
    bmap.plot([ev_x], [ev_y], marker="s", ms=10, mfc="none",
              mec=evcolors[evcolnum], mew=2, zorder=4)
    
    for arr in arrivals.itervalues():
      sta = sites[arr.sta]
      dist = geog.dist_deg((sta.lon, sta.lat), (ev_lon, ev_lat))
      if dist <= 100:
        if int(arr.time / quant) \
               == int((ev_time + ttime[ev_depth, dist]) / quant):
          bmap.drawgreatcircle(ev_lon, ev_lat, sta.lon, sta.lat,
                               color = evcolors[evcolnum],
                               linewidth=2, zorder=5)

  draw_events(bmap, sites, marker="^", ms=10, mfc="none", mec="green",
              mew=2, zorder=3)

def show():
  plt.show()
  
