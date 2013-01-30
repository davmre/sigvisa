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

import matplotlib
from mpl_toolkits.basemap import Basemap
from matplotlib.figure import Figure
import matplotlib.cm

import os
from optparse import OptionParser

import numpy as np
import geog

from database.dataset import *
from database import db

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

def draw_events(bmap, events, labels = None, **args):
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


    if labels is not None and labels[enum] is not None and 'ax' in args:
      axes = args['ax']
      axes.annotate(
        labels[enum][0],
        xy = (x, y),
        size=4,
        arrowprops = None)

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
    bmap.ax.arrow(x1, y1, scale * (x2-x1), scale * (y2-y1), **args)


def draw_density(bmap, lons, lats, vals, levels=10, colorbar=True,
                 nolines = False, colorbar_orientation="vertical",
                 colorbar_shrink = 0.9):
  loni, lati = np.mgrid[0:len(lons), 0:len(lats)]
  lon_arr, lat_arr = lons[loni], lats[lati]

  # convert to map coordinates
  x, y = bmap(list(lon_arr.flat), list(lat_arr.flat))
  x_arr = np.array(x).reshape(lon_arr.shape)
  y_arr = np.array(y).reshape(lat_arr.shape)

  cm = matplotlib.cm.get_cmap('jet')
  cs1 = bmap.contour(x_arr, y_arr, vals, levels, linewidths=.5, colors="k",
                     zorder=6 - int(nolines))
  cs2 = bmap.contourf(x_arr, y_arr, vals, levels, cmap=cm, zorder=5,
                      extend="both",
                      norm=matplotlib.colors.BoundaryNorm(cs1.levels,
                                                          cm.N))

  if colorbar:
    bmap.ax.colorbar(cs2, orientation=colorbar_orientation, drawedges=True,
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


def main():

    parser = OptionParser()

    parser.add_option("-s", "--siteid", dest="siteid", default=None, type="int", help="siteid of station for which to generate plots, or None to just plot all stations (None)")

    parser.add_option("--min_azi", dest="min_azi", default=0, type="float", help="exclude all events with azimuth less than this value (0)")
    parser.add_option("--max_azi", dest="max_azi", default=360, type="float", help="exclude all events with azimuth greater than this value (360)")
    parser.add_option("--min_dist", dest="min_dist", default=0, type="float", help="exclude all events with distance less than this value (0)")
    parser.add_option("--max_dist", dest="max_dist", default=24000, type="float", help="exclude all events with distance greater than this value (24000)")
    parser.add_option("--min_depth", dest="min_depth", default=0, type="float", help="exclude all events with depth less than this value (0)")
    parser.add_option("--max_depth", dest="max_depth", default=8000, type="float", help="exclude all events with depth greater than this value (8000)")
    parser.add_option("--min_mb", dest="min_mb", default=0, type="float", help="exclude all events with mb less than this value (0)")
    parser.add_option("--max_mb", dest="max_mb", default=10, type="float", help="exclude all events with mb greater than this value (10)")
    parser.add_option("--start_time", dest="start_time", default=0, type="float", help="exclude all events with time less than this value (0)")
    parser.add_option("--end_time", dest="end_time", default=1237680000, type="float", help="exclude all events with time greater than this value (1237680000)")
    parser.add_option("-o", "--outfile", dest="outfile", default="geog.pdf", type="str", help="name of output file (geog.pdf)")

    (options, args) = parser.parse_args()

    from matplotlib.backends.backend_pdf import PdfPages

#    pp = PdfPages(os.path.join(pdf_dir, "geog.pdf"))
    pp = PdfPages(options.outfile)

    cursor = db.connect().cursor()
    sites = read_sites(cursor)

    max_lon=180
    min_lon=-180
    max_lat=90
    min_lat=-90
    bmap = draw_earth("",
                      projection="cyl",
                      resolution="l",
                      llcrnrlon = min_lon, urcrnrlon = max_lon,
                      llcrnrlat = min_lat, urcrnrlat = max_lat,
                      nofillcontinents=True,
                      figsize=(8,8))

    if options.siteid is not None:
      siteid = options.siteid
      site_lonlat = sites[siteid-1, 0:2]


      f = open("pn_evids", 'r')
      events = []
      for l in f:

        try:
          orid = int(l)
        except:
          continue
        sql_query="SELECT distinct mb, lon, lat, evid, time, depth FROM leb_origin where evid=%d" % (orid)
        cursor.execute(sql_query)
        ev = cursor.fetchone()
        events.append(ev)
      events = np.array(events)

#      sql_query="SELECT distinct lebo.mb, lebo.lon, lebo.lat, lebo.evid, lebo.time, lebo.depth FROM leb_arrival l , static_siteid sid, static_phaseid pid, leb_origin lebo, leb_assoc leba where l.time between %f and %f and lebo.mb between %f and %f and lebo.depth between %f and %f and leba.arid=l.arid and lebo.orid=leba.orid and sid.sta=l.sta and sid.id=%d and pid.phase=leba.phase" % (options.start_time, options.end_time, options.min_mb, options.max_mb, options.min_depth, options.max_depth, options.siteid)
#      cursor.execute(sql_query)
#      events = np.array(cursor.fetchall())

      d = lambda ev : geog.dist_km((sites[siteid-1][0], sites[siteid-1][1]), (ev[1], ev[2]))
      a = lambda ev : geog.azimuth((sites[siteid-1][0], sites[siteid-1][1]), (ev[1], ev[2]))
      distances = np.array([d(ev) for ev in events])
      azimuths = np.array([a(ev) for ev in events])
      dist_i = np.logical_and((distances >= options.min_dist), (distances <= options.max_dist))
      azi_i = np.logical_and((azimuths >= options.min_azi), (azimuths <= options.max_azi))
      i = np.logical_and(dist_i, azi_i)

      print events.shape
      print i.shape

      ev_lonlat = events[i, 1:3]

      for i,ev in enumerate(ev_lonlat):
        print "drawing event at (%.2f, %.2f)" % (ev[0], ev[1])
        draw_events(bmap, ((ev[0], ev[1]),), marker=".", ms=1, mfc="none", alpha=0.4, mec="red", mew=1)
      draw_events(bmap, (site_lonlat,),  marker="x", ms=4, mfc="none", mec="purple", mew=2)

      bmap.ax.suptitle("siteid %d azi (%d, %d) dist (%d, %d)\n mb (%.2f, %.2f) depth (%d, %d)" % (siteid, options.min_azi, options.max_azi, options.min_dist, options.max_dist, options.min_mb, options.max_mb, options.min_depth, options.max_depth))

    else:
      sql_query = "select sta from static_siteid"
      cursor.execute(sql_query)
      stas = cursor.fetchall()

      draw_events(bmap, sites[:, 0:2],  labels=stas, marker="x", ms=3, mfc="none", mec="purple", mew=1)

    pp.savefig()
    pp.close()


if __name__ == "__main__":
  main()
