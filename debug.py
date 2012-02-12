# draw the density around a point

import sys
from optparse import OptionParser

from database.dataset import *
import netvisa, learn
import database.db
from utils.geog import degdiff, dist_deg
from analyze import suppress_duplicates

# ignore warnings from matplotlib in python 2.4
import warnings
warnings.simplefilter("ignore",DeprecationWarning)
import matplotlib.pyplot as plt
# for type 1 fonts
#plt.rcParams['ps.useafm'] = True
#plt.rcParams['pdf.use14corefonts'] = True
from utils.draw_earth import draw_events, draw_earth, draw_density

def best_score(netmodel, event, event_detlist):
  
  curr_detlist = []
  curr_score = netmodel.score_event(event, curr_detlist)
  
  for item in event_detlist:
    curr_detlist.append(item)
    score = netmodel.score_event(event, curr_detlist)
    if score > curr_score:
      curr_score = score
    else:
      curr_detlist.pop(-1)
  return curr_score
    
def print_events(sitenames, netmodel, earthmodel, detections, leb_events, leb_evlist,
                 label):
  print "=" * 60
  if leb_evlist is None:
    for evnum in range(len(leb_events)):
      print_event(sitenames, netmodel, earthmodel, detections, leb_events[evnum],
                  None, label)
  else:
    score = 0
    for evnum in range(len(leb_events)):
      score += print_event(sitenames, netmodel, earthmodel, detections, leb_events[evnum],
                           leb_evlist[evnum], label)
      print "-" * 60
    print "Total: %.1f" % score
    print "=" * 60

def print_event(sitenames, netmodel, earthmodel, detections, event, event_detlist, label):
  print ("%s: lon %4.2f lat %4.2f depth %3.1f mb %1.1f time %.1f orid %d"
         % (label, event[ EV_LON_COL], event[ EV_LAT_COL],
            event[ EV_DEPTH_COL], event[ EV_MB_COL],
            event[ EV_TIME_COL], event[ EV_ORID_COL]))
  if event_detlist is None:
    return
  
  print "Detections:",
  detlist = [x for x in event_detlist]
  detlist.sort()
  for phaseid, detid in detlist:
    detsc = netmodel.score_event_det(event, phaseid, detid)
    if detsc is None:
      detsc = -1
    inv = netmodel.invert_det(detid, 0)
    if inv is None:
      asterisk = ""
    else:
      (ilon, ilat, idepth, itime) = inv
      idist = dist_deg((ilon, ilat), event[[EV_LON_COL, EV_LAT_COL]])
      itimediff = itime - event[EV_TIME_COL]
      ievent = event.copy()
      (ievent[EV_LON_COL], ievent[EV_LAT_COL], ievent[EV_DEPTH_COL],
       ievent[EV_TIME_COL]) = inv
      ievent [EV_MB_COL] = 3.0
      iscore = "%.1f" % best_score(netmodel, ievent, detlist)
      if abs(itimediff) < 50 and idist < 5:
        asterisk = " *"
      else:
        asterisk = ""
      
    print "(%s %s %.1f%s)" % (earthmodel.PhaseName(phaseid),
                               sitenames[int(detections[detid, DET_SITE_COL])],
                               detsc, asterisk),
  print
  score = netmodel.score_event(event, event_detlist)
  print "Ev Score: %.1f    (prior location logprob %.1f)" \
        % (score, netmodel.location_logprob(event[ EV_LON_COL],
                                            event[ EV_LAT_COL],
                                            event[ EV_DEPTH_COL]))
  return score

def main(param_dirname):
  parser = OptionParser()
  parser.set_usage("Usage: python debug.py [options] <runid> <leb|visa> <orid>")
  parser.add_option("-w", "--window", dest="window", default=10.0,
                    type="float",
                    help = "window size around event location (10.0)",
                    metavar="degrees")
  parser.add_option("-r", "--resolution", dest="resolution", default=.5,
                    type="float",
                    help = "resolution in degrees (.5)",
                    metavar="degrees")
  parser.add_option("-s", "--suppress", dest="suppress", default=False,
                    action = "store_true", help = "suppress duplicates")
  parser.add_option("-x", "--text_only", dest="text_only", default=False,
                    action = "store_true", help = "text only")
  parser.add_option("-a", "--arrival_table", dest="arrival_table",
                    default="idcx_arrival",
                    help = "arrival table to use for inferring on")
  (options, args) = parser.parse_args()
  
  if len(args) != 3:
    parser.print_help()
    sys.exit(1)
  
  runid, orid_type, orid = int(args[0]), args[1], int(args[2])
  if orid_type not in ("visa", "leb"):
    print "invalid orid_type %s" % orid_type
  print "Debugging run %d %s origin %d" % (runid, orid_type, orid)
  
  # find the event time and use that to configure start and end time
  cursor = database.db.connect().cursor()
  if orid_type == "visa":
    cursor.execute("select time from visa_origin where runid=%d and orid=%d"%
                   (runid, orid))
  elif orid_type == "leb":
    cursor.execute("select time from leb_origin where orid=%d"% (orid,))

  fetch = cursor.fetchone()
  if fetch is None:
    print "Event not found"
    sys.exit(2)
  evtime, = fetch
  print "Event Time %.1f" % evtime

  start_time, end_time, end_det_time = evtime-100, evtime+100,\
                                       evtime+100+MAX_TRAVEL_TIME

  # read all detections which could have been caused by events in the
  # time range
  detections, arid2num = read_detections(cursor, start_time, end_det_time,
                                         options.arrival_table)

  # read LEB events
  leb_events, leb_orid2num = read_events(cursor, start_time, end_time, "leb")
  leb_evlist = read_assoc(cursor, start_time, end_time, leb_orid2num,
                          arid2num, "leb")
  
  
  sel3_events, sel3_orid2num = read_events(cursor, start_time, end_time,"sel3")
  sel3_evlist = read_assoc(cursor, start_time, end_time, sel3_orid2num,
                           arid2num, "sel3")

  visa_events, visa_orid2num = read_events(cursor, start_time, end_time,"visa",
                                           runid=runid)

  cursor.execute("select orid, score from visa_origin where runid=%d" %
                 (runid,))
  visa_evscores = dict(cursor.fetchall())
  if options.suppress:
    visa_events, visa_orid2num = suppress_duplicates(visa_events, visa_evscores)
  
  visa_evlist = read_assoc(cursor, start_time, end_time, visa_orid2num,
                           arid2num, "visa", runid=runid)

  neic_events = read_isc_events(cursor, start_time, end_time, "NEIC")

  all_isc_events = read_isc_events(cursor, start_time, end_time, None)
  
  site_up = read_uptime(cursor, start_time, end_det_time,
                        options.arrival_table)
  
  sites = read_sites(cursor)
  
  cursor.execute("select sta from static_siteid site order by id")
  sitenames = np.array(cursor.fetchall())[:,0]
  
  phasenames, phasetimedef = read_phases(cursor)
  
  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_det_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)
  netmodel.disable_sec_arr()
  
  # print all the events
  print_events(sitenames, netmodel, earthmodel, detections, leb_events, leb_evlist, "LEB")
  print_events(sitenames, netmodel, earthmodel, detections, sel3_events, sel3_evlist,
               "SEL3")
  print_events(sitenames, netmodel, earthmodel, detections, visa_events, visa_evlist,
               "VISA")
  print_events(sitenames, netmodel, earthmodel, detections, neic_events, None, "NEIC")
  print_events(sitenames, netmodel, earthmodel, detections, all_isc_events, None, "ISC")

  if options.text_only:
    return
  
  # now draw a window around the event location
  if orid_type == "visa":
    evnum = visa_orid2num[orid]
    event = visa_events[evnum].copy()
    event_detlist = visa_evlist[evnum]
  elif orid_type == "leb":
    evnum = leb_orid2num[orid]
    event = leb_events[evnum].copy()
    event_detlist = leb_evlist[evnum]
  
  lon1 = event[EV_LON_COL] - options.window
  lon2 = event[EV_LON_COL] + options.window
  lat1 = event[EV_LAT_COL] - options.window
  lat2 = event[EV_LAT_COL] + options.window
  
  bmap = draw_earth("",
                    #"NET-VISA posterior density, NEIC(white), LEB(yellow), "
                    #"SEL3(red), NET-VISA(blue)",
                    projection="mill",
                    resolution="l",
                    llcrnrlon = lon1, urcrnrlon = lon2,
                    llcrnrlat = lat1, urcrnrlat = lat2,
                    nofillcontinents=True, figsize=(4.5,4))
  if len(leb_events):
    draw_events(bmap, leb_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="o", ms=10, mfc="none", mec="yellow", mew=2)
  if len(sel3_events):
    draw_events(bmap, sel3_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="o", ms=10, mfc="none", mec="red", mew=2)

  if len(visa_events):
    draw_events(bmap, visa_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="s", ms=10, mfc="none", mec="blue", mew=2)

  if len(neic_events):
    draw_events(bmap, neic_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="*", ms=10, mfc="white", mew=1)
  
  elif len(all_isc_events):
    draw_events(bmap, all_isc_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="*", ms=10, mfc="none", mec="yellow", mew=1)

  # draw a density
  LON_BUCKET_SIZE = options.resolution
  # Z axis is along the earth's axis
  # Z goes from -1 to 1 and will have the same number of buckets as longitude
  Z_BUCKET_SIZE = (2.0 / 360.0) * LON_BUCKET_SIZE

  # skip one degree at the bottom of the map to display the map scale
  # otherwise, the density totally covers it up
  lon_arr = np.arange(event[EV_LON_COL] - options.window,
                      event[EV_LON_COL] + options.window,
                      LON_BUCKET_SIZE)
  z_arr = np.arange(np.sin(np.radians(event[EV_LAT_COL] - options.window *.88)),
                    np.sin(np.radians(event[EV_LAT_COL] + options.window)),
                    Z_BUCKET_SIZE)
  lat_arr = np.degrees(np.arcsin(z_arr))
  
  score = np.zeros((len(lon_arr), len(lat_arr)))
  best, worst = -np.inf, np.inf
  for loni, lon in enumerate(lon_arr):
    for lati, lat in enumerate(lat_arr):
      if lon<-180: lon+=360
      if lon>180: lon-=360

      sc = -np.inf
      for evnum in range(len(visa_events)):
        tmp = visa_events[evnum].copy()
        tmp[EV_LON_COL] = lon
        tmp[EV_LAT_COL] = lat
        sc = max(sc, netmodel.score_event(tmp, visa_evlist[evnum]))
      for evnum in range(len(leb_events)):
        tmp = leb_events[evnum].copy()
        tmp[EV_LON_COL] = lon
        tmp[EV_LAT_COL] = lat
        sc = max(sc, netmodel.score_event(tmp, leb_evlist[evnum]))
      
      score[loni, lati] = sc

      if sc > best: best = sc
      if sc < worst: worst = sc

  if worst < best * .9:
    levels = np.arange(worst, best*.75, (best*.75 - worst)/5).tolist() +\
             np.linspace(best*.75, best, 5).tolist()
  else:
    levels = np.linspace(worst, best, 10).tolist()

  # round the levels so they are easier to display in the legend
  levels = np.round(levels, 1).tolist()

  draw_density(bmap, lon_arr, lat_arr, score, levels = levels, colorbar=True)
  
  
  # add a map scale
  scale_lon, scale_lat = event[EV_LON_COL], \
                         event[EV_LAT_COL]-options.window * .98
  try:
    bmap.drawmapscale(scale_lon, scale_lat, scale_lon, scale_lat,
                      options.window*100,
                      fontsize=8, barstyle='fancy',
                      labelstyle='simple', units='km')
  except:
    pass

  
  plt.savefig("output/debug_run_%d_%s_orid_%d.png" % (runid, orid_type, orid))

  bmap = draw_earth("",
                    #"NET-VISA posterior density, NEIC(white), LEB(yellow), "
                    #"SEL3(red), NET-VISA(blue)",
                    projection="mill",
                    resolution="l",
                    llcrnrlon = event[EV_LON_COL] - 30,
                    urcrnrlon = event[EV_LON_COL] + 30,
                    llcrnrlat = event[EV_LAT_COL] - 20,
                    urcrnrlat = event[EV_LAT_COL] + 20,
                    figsize=(4.5,4))
  if len(sel3_events):
    draw_events(bmap, sel3_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="o", ms=10, mfc="none", mec="red", mew=2)

  if len(visa_events):
    draw_events(bmap, visa_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="s", ms=10, mfc="none", mec="blue", mew=2)

  if len(leb_events):
    draw_events(bmap, leb_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="o", ms=10, mfc="none", mec="yellow", mew=2)
  
  if len(neic_events):
    draw_events(bmap, neic_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="*", ms=10, mfc="white", mew=1)
  elif len(all_isc_events):
    draw_events(bmap, all_isc_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="*", ms=10, mfc="none", mec="yellow", mew=1)

  scale_lon, scale_lat = event[EV_LON_COL], \
                         event[EV_LAT_COL]-19
  try:
    bmap.drawmapscale(scale_lon, scale_lat, scale_lon, scale_lat,5000,
                      fontsize=8, barstyle='fancy',
                      labelstyle='simple', units='km')
  except:
    pass
  
  plt.savefig("output/debug_area_run_%d_%s_orid_%d.png"
              % (runid, orid_type, orid))

  bmap = draw_earth("",
                    #"NET-VISA posterior density, NEIC(white), LEB(yellow), "
                    #"SEL3(red), NET-VISA(blue)",
                    projection="mill",
                    resolution="l",
                    llcrnrlon = event[EV_LON_COL] - options.window,
                    urcrnrlon = event[EV_LON_COL] + options.window,
                    llcrnrlat = event[EV_LAT_COL] - options.window,
                    urcrnrlat = event[EV_LAT_COL] + options.window,
                    figsize=(4.5,4))
  if len(all_isc_events):
    draw_events(bmap, all_isc_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="*", ms=10, mfc="none", mec="yellow", mew=1)

  if len(sel3_events):
    draw_events(bmap, sel3_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="o", ms=10, mfc="none", mec="red", mew=2)

  if len(visa_events):
    draw_events(bmap, visa_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="s", ms=10, mfc="none", mec="blue", mew=2)

  draw_density(bmap, lon_arr, lat_arr, score, levels = levels, colorbar=True)
  
  scale_lon, scale_lat = event[EV_LON_COL], \
                         event[EV_LAT_COL]-options.window * .98
##  try:
##    bmap.drawmapscale(scale_lon, scale_lat, scale_lon, scale_lat,500,
##                      fontsize=8, barstyle='fancy',
##                      labelstyle='simple', units='km')
##  except:
##    pass
  
  plt.savefig("output/debug_dens_isc_run_%d_%s_orid_%d.png"
              % (runid, orid_type, orid))

  bmap = draw_earth("",
                    #"NET-VISA posterior density, NEIC(white), LEB(yellow), "
                    #"SEL3(red), NET-VISA(blue)",
                    projection="ortho",
                    resolution="l",
                    lon_0 = event[EV_LON_COL],
                    lat_0 = event[EV_LAT_COL],
                    figsize=(4.5,4))
  if len(all_isc_events):
    draw_events(bmap, all_isc_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="*", ms=10, mfc="none", mec="yellow", mew=1)

  if len(sel3_events):
    draw_events(bmap, sel3_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="o", ms=10, mfc="none", mec="red", mew=2)

  if len(visa_events):
    draw_events(bmap, visa_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="s", ms=10, mfc="none", mec="blue", mew=2)

  scale_lon, scale_lat = event[EV_LON_COL], \
                         event[EV_LAT_COL]-options.window * .8
##  try:
##    bmap.drawmapscale(scale_lon, scale_lat, scale_lon, scale_lat,500,
##                      fontsize=8, barstyle='fancy',
##                      labelstyle='simple', units='km')
##  except:
##    pass
  
  
  plt.savefig("output/debug_isc_run_%d_%s_orid_%d.png"
              % (runid, orid_type, orid))

  bmap = draw_earth("",
                    #"NET-VISA posterior density, NEIC(white), LEB(yellow), "
                    #"SEL3(red), NET-VISA(blue)",
                    projection="ortho",
                    resolution="l",
                    lon_0 = event[EV_LON_COL], lat_0 = event[EV_LAT_COL],
                    figsize=(4.5,4))
  if len(leb_events):
    draw_events(bmap, leb_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="o", ms=10, mfc="none", mec="yellow", mew=2)
  if len(sel3_events):
    draw_events(bmap, sel3_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="o", ms=10, mfc="none", mec="red", mew=2)

  if len(visa_events):
    draw_events(bmap, visa_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="s", ms=10, mfc="none", mec="blue", mew=2)

  if len(neic_events):
    draw_events(bmap, neic_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="*", ms=10, mfc="white", mew=1)
  elif len(all_isc_events):
    draw_events(bmap, all_isc_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="*", ms=10, mfc="none", mec="yellow", mew=1)

  plt.savefig("output/debug_globe_run_%d_%s_orid_%d.png"
              % (runid, orid_type, orid))

  ########
  # next display all the inverted events in this window
  invert_evs = []
  for detnum in range(len(detections)):
    inv_ev = netmodel.invert_det(detnum,0)
    if inv_ev is not None and inv_ev[3] > start_time and inv_ev[3] < end_time:
      invert_evs.append(inv_ev)
  invert_evs = np.array(invert_evs)
  #bmap2 = draw_earth("")
  bmap = draw_earth("")
  if len(invert_evs):
    draw_events(bmap, invert_evs[:,[0, 1]],
                marker="s", ms=10, mfc="none", mec="blue", mew=2)
  if len(sel3_events):
    draw_events(bmap, sel3_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="o", ms=10, mfc="none", mec="red", mew=2)
  if len(leb_events):
    draw_events(bmap, leb_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="o", ms=10, mfc="none", mec="yellow", mew=2)

  if len(neic_events):
    draw_events(bmap, neic_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="*", ms=12, mfc="white", mew=1)
  elif len(all_isc_events):
    draw_events(bmap, all_isc_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="*", ms=10, mfc="none", mec="yellow", mew=1)

  plt.show()

if __name__ == "__main__":
  main("parameters")
