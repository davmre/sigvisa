# Copyright (c) 2012, Bayesian Logic, Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Bayesian Logic, Inc. nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# Bayesian Logic, Inc. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# 
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

  if "draw_grid" in args:
    del args["draw_grid"]
    draw_grid = True
  else:
    draw_grid = False
  
  # if no args other than those above then center the map
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

  if draw_grid:
    bmap.drawmeridians(np.arange(-180, 180, 30), zorder=10)
    bmap.drawparallels(np.arange(-90, 90, 30), zorder=10)
  
  return bmap

def draw_events(bmap, events, **args):
  """
  The args can be any collection of marker args like ms, mfc mec
  """
  if "zorder" not in args:
    args["zorder"] = 10
  
  # if there are any array args then we need to apply them to each event
  # separately
  array_args = []
  for argname, argval in args.iteritems():
    if not np.issctype(type(argval)):
      assert(len(events) == len(argval))
      array_args.append((argname, argval))
      
  for enum, ev in enumerate(events):
    x,y = bmap(ev[0], ev[1])
    # set each of the array argument for this event
    for (argname, argval) in array_args:
      args[argname] = argval[enum]
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
    
def draw_density(bmap, lons, lats, vals, levels=10, colorbar=True,
                 nolines = False, colorbar_orientation="vertical",
                 colorbar_shrink = 0.9):
  loni, lati = np.mgrid[0:len(lons), 0:len(lats)]
  lon_arr, lat_arr = lons[loni], lats[lati]
  
  # convert to map coordinates
  x, y = bmap(list(lon_arr.flat), list(lat_arr.flat))
  x_arr = np.array(x).reshape(lon_arr.shape)
  y_arr = np.array(y).reshape(lat_arr.shape)

  cs1 = bmap.contour(x_arr, y_arr, vals, levels, linewidths=.5, colors="k",
                     zorder=6 - int(nolines))
  cs2 = bmap.contourf(x_arr, y_arr, vals, levels, cmap=plt.cm.jet, zorder=5,
                      extend="both",
                      norm=matplotlib.colors.BoundaryNorm(cs1.levels,
                                                          plt.cm.jet.N))

  if colorbar:
    plt.colorbar(cs2, orientation=colorbar_orientation, drawedges=True,
                 shrink = colorbar_shrink)

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
  
