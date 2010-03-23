# python imports
import numpy as np

# local imports
import db

EV_LON_COL, EV_LAT_COL, EV_DEPTH_COL, EV_TIME_COL, EV_MB_COL, EV_NUM_COLS \
            = range(5+1)

MIN_MAGNITUDE = 2.0

def read_events(label, evtype = "leb", time_range = None):
  # connect to the database
  dbconn = db.connect()
  cursor = dbconn.cursor()
  
  # get the start and end time
  if time_range is None:
    cursor.execute("select start_time, end_time from dataset where label=%s",
                   (label,))
    row = cursor.fetchone()
    if row is None:
      raise ValueError("Unknown label %s" % label)
    start_time, end_time = row
  else:
    start_time, end_time = time_range
  
  cursor.execute("select lon, lat, depth, time, mb from %s_origin "
                 "where time between "
                 "(select start_time from dataset where label='%s') and "
                 "(select end_time from dataset where label='%s')"
                 % (evtype, label, label))
  events = np.array(cursor.fetchall())

  # change -999 mb to MIN MAG
  events[:, EV_MB_COL][events[:, EV_MB_COL] == -999] = MIN_MAGNITUDE

  return start_time, end_time, events
