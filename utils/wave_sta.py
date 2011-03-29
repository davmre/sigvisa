# plot the waveform at a station in a time range

import sys, MySQLdb,struct
import matplotlib.pyplot as plt
import numpy as np

from database import db, wave


def main(param_dirname):
  if len(sys.argv) != 4:
    print "Usage: python wave_sta.py <sta> <start time> <end time>"
    sys.exit(1)
  
  sta, start_time, end_time = sys.argv[1], float(sys.argv[2]),float(sys.argv[3])
  cursor = db.connect().cursor()
  cursor.execute("select distinct chan from idcx_wfdisc where sta='%s'" % sta)
  chans = [x for x, in cursor.fetchall()]
  print "Channels:", chans
  for chan in chans:
    data, samprate = wave.fetch_waveform(sta, chan, start_time, end_time)
    timerange = np.arange(start_time, end_time, 1.0/samprate)
    plt.figure()
    plt.suptitle("%s -- %s" % (sta, chan))
    plt.plot(timerange, data)
    plt.xlabel("Time")
    plt.ylabel("nm")
  plt.show()

if __name__ == "__main__":
  try:
    main("parameters")
  except SystemExit:
    raise
  except:
    import pdb, traceback, sys
    traceback.print_exc(file=sys.stdout)
    pdb.post_mortem(sys.exc_traceback)
    raise
