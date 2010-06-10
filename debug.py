# draw the density around a point

import sys

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
plt.rcParams['ps.useafm'] = True
#plt.rcParams['pdf.use14corefonts'] = True
from utils.draw_earth import draw_events, draw_earth, draw_density

def print_events(netmodel, earthmodel, detections, leb_events, leb_evlist,
                 label):
  print "=" * 60
  if leb_evlist is None:
    for evnum in range(len(leb_events)):
      print_event(netmodel, earthmodel, detections, leb_events[evnum],
                  None, label)
  else:
    score = 0
    for evnum in range(len(leb_events)):
      score += print_event(netmodel, earthmodel, detections, leb_events[evnum],
                           leb_evlist[evnum], label)
      print "-" * 60
    print "Total: %.1f" % score
    print "=" * 60

def print_event(netmodel, earthmodel, detections, event, event_detlist, label):
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
    print "(%s, %d, %d)" % (earthmodel.PhaseName(phaseid),
                            int(detections[detid, DET_SITE_COL]), detid),
  print
  score = netmodel.score_event(event, event_detlist)
  print "Ev Score: %.1f    (prior location logprob %.1f)" \
        % (score, netmodel.location_logprob(event[ EV_LON_COL],
                                            event[ EV_LAT_COL],
                                            event[ EV_DEPTH_COL]))
  return score

def main(param_dirname):
  if len(sys.argv) != 3:
    print "Usage: python density.py <runid> <orid>"
  
  runid, orid = int(sys.argv[1]), int(sys.argv[2])
  print "Debugging run %d origin %d" % (runid, orid)
  
  # find the event time and use that to configure start and end time
  cursor = database.db.connect().cursor()
  cursor.execute("select time from visa_origin where runid=%s and orid=%s",
                 (runid, orid))
  evtime, = cursor.fetchone()
  print "Event Time %.1f" % evtime

  start_time, end_time, end_det_time = evtime-100, evtime+100,\
                                       evtime+100+MAX_TRAVEL_TIME

  # read all detections which could have been caused by events in the
  # time range
  detections, arid2num = read_detections(cursor, start_time, end_det_time)

  # read LEB events
  leb_events, leb_orid2num = read_events(cursor, start_time, end_time, "leb")
  leb_evlist = read_assoc(cursor, start_time, end_time, leb_orid2num,
                          arid2num, "leb")
  
  
  sel3_events, sel3_orid2num = read_events(cursor, start_time, end_time,"sel3")
  sel3_evlist = read_assoc(cursor, start_time, end_time, sel3_orid2num,
                           arid2num, "sel3")

  visa_events, visa_orid2num = read_events(cursor, start_time, end_time,"visa",
                                           runid=runid)

  cursor.execute("select orid, score from visa_origin where runid=%s",
                 (runid,))
  visa_evscores = dict(cursor.fetchall())
  visa_events, visa_orid2num = suppress_duplicates(visa_events, visa_evscores)
  
  visa_evlist = read_assoc(cursor, start_time, end_time, visa_orid2num,
                           arid2num, "visa", runid=runid)

  neic_events = read_isc_events(cursor, start_time, end_time, "NEIC")
  
  site_up = read_uptime(cursor, start_time, end_det_time)
  
  sites = read_sites(cursor)
  
  phasenames, phasetimedef = read_phases(cursor)
  
  earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
  netmodel = learn.load_netvisa(param_dirname,
                                start_time, end_det_time,
                                detections, site_up, sites, phasenames,
                                phasetimedef)

  # print all the events
  print_events(netmodel, earthmodel, detections, leb_events, leb_evlist, "LEB")
  print_events(netmodel, earthmodel, detections, sel3_events, sel3_evlist,
               "SEL3")
  print_events(netmodel, earthmodel, detections, visa_events, visa_evlist,
               "VISA")
  print_events(netmodel, earthmodel, detections, neic_events, None, "NEIC")

  # now draw a window around the event location
  evnum = visa_orid2num[orid]
  event = visa_events[evnum].copy()

  lon1 = event[EV_LON_COL] - 10
  lon2 = event[EV_LON_COL] + 10
  lat1 = event[EV_LAT_COL] - 10
  lat2 = event[EV_LAT_COL] + 10
  
  bmap = draw_earth("NET-VISA posterior density, NEIC(white), LEB(yellow), "
                    "SEL3(red), NET-VISA(blue)",
                    projection="mill",
                    resolution="l",
                    llcrnrlon = lon1, urcrnrlon = lon2,
                    llcrnrlat = lat1, urcrnrlat = lat2,
                    nofillcontinents=True)
  if len(leb_events):
    draw_events(bmap, leb_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="o", ms=10, mfc="none", mec="yellow", mew=1)
  if len(sel3_events):
    draw_events(bmap, sel3_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="o", ms=10, mfc="none", mec="red", mew=1)
    
  draw_events(bmap, visa_events[:,[EV_LON_COL, EV_LAT_COL]],
              marker="s", ms=10, mfc="none", mec="blue", mew=1)

  if len(neic_events):
    draw_events(bmap, neic_events[:,[EV_LON_COL, EV_LAT_COL]],
                marker="*", ms=10, mfc="white", mew=1)

  # draw a density
  LON_BUCKET_SIZE = .5
  # Z axis is along the earth's axis
  # Z goes from -1 to 1 and will have the same number of buckets as longitude
  Z_BUCKET_SIZE = (2.0 / 360.0) * LON_BUCKET_SIZE
  
  lon_arr = np.arange(-180., 180., LON_BUCKET_SIZE)
  z_arr = np.arange(-1.0, 1.0, Z_BUCKET_SIZE)
  lat_arr = np.arcsin(z_arr) * 180. / np.pi
  
  score = np.zeros((len(lon_arr), len(lat_arr)))

  for loni, lon in enumerate(lon_arr):
    for lati, lat in enumerate(lat_arr):
      event = visa_events[evnum].copy()
      if dist_deg((event[EV_LON_COL], event[EV_LAT_COL]), (lon, lat)) > 15:
        continue
      event[EV_LON_COL] = lon
      event[EV_LAT_COL] = lat
      score[loni, lati] = netmodel.score_event(event, visa_evlist[evnum])

  draw_density(bmap, lon_arr, lat_arr, score, colorbar=False)

  plt.show()

if __name__ == "__main__":
  main("parameters")
