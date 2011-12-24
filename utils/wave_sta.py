# plot the waveform at a station in a time range
# example usage which shows a clear P (at 69s) and an S phase (324s)
# python -m utils.wave_sta ASAR 1237683000 1237683400

import sys, MySQLdb,struct
import matplotlib.pyplot as plt
import numpy as np

import obspy.signal

from database import db, wave


def main(param_dirname):
  if len(sys.argv) != 6:
    print "Usage: python wave_sta.py <sta> <start time> <end time>"\
          " <low frequency> <high frequency>"
    sys.exit(1)
  
  sta, start_time, end_time, lowf, highf = sys.argv[1],float(sys.argv[2]),float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
  cursor = db.connect().cursor()
  cursor.execute("select distinct chan from idcx_wfdisc where sta='%s'" % sta)
  chans = [x for x, in cursor.fetchall()]
  if not len(chans):
    # for array station use the reference station instead
    cursor.execute("select refsta from static_site where sta='%s' and "
                   "offdate=-1" % sta)
    sta, = cursor.fetchone()
    cursor.execute("select distinct chan from idcx_wfdisc where sta='%s'"% sta)
    chans = [x for x, in cursor.fetchall()]
    
  print "Channels:", chans
  for chan in chans:
    # read 100 extra second to get the STA/LTA working
    data, samprate = wave.fetch_waveform(sta, chan, start_time-100, end_time)
    filtered_data = obspy.signal.filter.bandpass(data, lowf, highf, samprate)
    cft_data = obspy.signal.recStalta(filtered_data, int(1.5 * samprate),
                                      int(60 * samprate))
    timerange = np.arange(start_time, end_time, 1.0/samprate)
    plt.figure()
    plt.suptitle("%s -- %s" % (sta, chan))
    plt.xlabel("Time (s)")
    
    #plt.plot(timerange, wave.lowpass_filter(data, samprate, 0.5))
    # remove the low-frequency noise
    #plt.plot(timerange, wave.highpass_filter(data, samprate, 0.5))

    axes = plt.subplot(3, 1, 1)
    plt.plot(timerange, filtered_data[100*samprate:])
    plt.ylabel("Raw %.1f - %.1f Hz" % (lowf, highf))
    plt.grid(True)

    plt.subplot(3, 1, 2, sharex=axes)
    plt.plot(timerange,
             obspy.signal.filter.envelope(filtered_data)[100*samprate:])
    plt.ylabel("Envelope")
    plt.grid(True)
    
    plt.subplot(3, 1, 3, sharex=axes)
    plt.plot(timerange, cft_data[100*samprate:])
    plt.ylabel("STA/LTA")

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
