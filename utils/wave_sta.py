# plot the waveform at a station in a time range
# example usage which shows a clear P (at 69s) and an S phase (324s)
# python -m utils.wave_sta ASAR 1237683000 1237683400
import warnings
warnings.filterwarnings("ignore")

import sys, MySQLdb,struct
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser

import obspy.signal

from database import db, wave

MIN_SAMPRATE = 5

DEFAULT_SPECTRUM = [None, (0.1, .5), (.5, .7), (1.0, 2.0), (2.0, 4.0)]

# we need to skip a certain early portion to allow for propery filtering
SKIP = 100

def main(param_dirname):
  parser = OptionParser()
  parser.set_usage("Usage: python wave_sta.py [options] <sta> <start time>"
                   " <duration>")
  parser.add_option("-e", "--envelope", dest="envelope", default=False,
                    action="store_true", help="display envelope")
  parser.add_option("-a", "--arrival", dest="arrivals", default=False,
                    action="store_true", help="display arrivals")
  parser.add_option("-f", "--frequencies", dest="frequencies", default=None,
                    help="frequency ranges "
                    "('[None,(0.1,.5),(.5,.7),(1,2),(2,4)]")
  (options, args) = parser.parse_args()
  
  if len(args) != 3:
    parser.print_help()
    sys.exit(1)

  if options.frequencies is None:
    options.frequencies = DEFAULT_SPECTRUM
  else:
    options.frequencies = eval(options.frequencies)
  
  sta, start_time, duration = args[0], float(args[1]), float(args[2])
  end_time = start_time + duration
  cursor = db.connect().cursor()

  # read static_site
  siterec = wave.read_site(cursor, sta, start_time)
  
  if siterec['statype'] == 'ss':
    # read all the channels
    sitechanrecs = wave.read_sitechan(cursor, sta, start_time)
    
    # fetch the raw data for each channel
    for sitechanrec in sitechanrecs:
      rawdata, samprate = wave.fetch_waveform(sta, sitechanrec['chan'],
                                              start_time-SKIP, end_time)

      if samprate < MIN_SAMPRATE:
        continue
      
      plot_spectrum(options, cursor, sta, sitechanrec['chan'], rawdata,
                    start_time, end_time, samprate)
      
  else:                                 # array
    # get all the array elements
    arrelems = wave.read_site_ss(cursor, siterec['refsta'], start_time)

    # for each array element, we will read all available channels
    channel_data = {}
    channel_cnt = {}
    channel_samprate = {}
    
    for arrelem in arrelems:
      elemchanrecs = wave.read_sitechan(cursor, arrelem['sta'], start_time)
      for elemchanrec in elemchanrecs:
        chan = elemchanrec['chan']
        try:
          data, samprate = wave.fetch_waveform(arrelem['sta'], chan,
                                               start_time-SKIP, end_time)
          if samprate < MIN_SAMPRATE:
            continue
          #print arrelem['sta'], chan, len(data), samprate
        except wave.MissingWaveform:
          print "Warning: missing sta %s chan %s" % (arrelem['sta'], chan)
          continue
        
        # is this the first time we are seeing this channel?
        if chan not in channel_samprate:
          channel_samprate[chan] = samprate
          channel_cnt[chan] = 1
          channel_data[chan] = np.array(data)
          
        else:
          # the sampling rate should not be different
          if channel_samprate[chan] != samprate:
            print "Sampling rate of channel %s on element %s is %d instead of"\
                  " %d" % (chan, arrelem['sta'], samprate,
                           channel_samprate[chan])
            # TODO: downsample if we need to
            continue

          channel_cnt[chan] += 1
          channel_data[chan] += np.array(data)
    
    # now we will filter and display each channel
    for chan, sumdata in channel_data.iteritems():
      if channel_cnt[chan] > 1:
        print "channel %s cnt %d" % (chan, channel_cnt[chan])
      rawdata = sumdata / float(channel_cnt[chan])
      samprate = channel_samprate[chan]
      
      plot_spectrum(options, cursor, sta, chan, rawdata, start_time, end_time,
                    samprate)

  plt.show()
  return
  
def plot_spectrum(options, cursor, sta, chan, rawdata, start_time, end_time,
                  samprate):
  timerange = np.arange(start_time, end_time, 1.0/samprate)

  assert(len(timerange) == (len(rawdata) - SKIP*samprate))
  plt.figure()
  plt.suptitle("%s -- %s" % (sta, chan))
  plt.xlabel("Time (s)")
  
  # for each frequency band
  for bandnum, freqrange in enumerate(options.frequencies):
    if freqrange is None:
      filt_data = rawdata
      filt_name = "raw"
    else:
      lowf, highf = freqrange
      try:
        filt_data = obspy.signal.filter.bandpass(rawdata, lowf, highf, samprate)
        filt_name = "%.1f - %.1f Hz" % (lowf, highf)
      except ValueError:
        # if the sampling rate is too low just show the raw data
        filt_data = rawdata
        filt_name = "raw"
    assert(len(filt_data) == len(rawdata))
    if bandnum == 0:
      axes = plt.subplot(len(options.frequencies)+1, 1, bandnum+1)
    else:
      plt.subplot(len(options.frequencies)+1, 1, bandnum+1, sharex=axes)

    if options.envelope:
      filt_data = obspy.signal.filter.envelope(np.array(filt_data))
    plt.plot(timerange, filt_data[int(SKIP*samprate):])
    plt.ylabel(filt_name)
    plt.grid(True)

  plt.subplot(len(options.frequencies)+1, 1, len(options.frequencies)+1,
              sharex=axes)
  plot_stalta(options, cursor, sta, start_time, end_time, filt_data, samprate)
  plt.xlabel("Time (s)")
  
def plot_stalta(options, cursor, sta, start_time, end_time, filt_data,
                  samprate):
  plt.ylabel("STA (1.5s) / LTA (60s)")

  if options.arrivals:
    cursor.execute("select time from idcx_arrival where sta='%s' and "
                   "time between %d and %d" % (sta, start_time, end_time))
    
    for arrtime, in cursor.fetchall():
      plt.plot([arrtime, arrtime], [.25, 1], linewidth=3, color="blue")
  
    cursor.execute("select time from leb_arrival where sta='%s' and "
                   "time between %d and %d" % (sta, start_time, end_time))
    
    for arrtime, in cursor.fetchall():
      plt.plot([arrtime, arrtime], [0, .75], linewidth=3, color="red")
  
    cursor.execute("select orid, phase, time from leb_assoc join leb_arrival "
                   "using (arid,sta) where sta='%s' and "
                   "time between %d and %d order by time"
                   % (sta, start_time, end_time))
    
    for orid, phase, arrtime, in cursor.fetchall():
      print "orid %d phase %s time %.1f" % (orid, phase, arrtime)
      plt.plot([arrtime, arrtime], [-1, 0], linewidth=3, color="red",
               linestyle="-.")

  stalta = obspy.signal.trigger.recStalta(filt_data, int(1.5 * samprate),
                                          int(10 * samprate))

  timerange = np.arange(start_time, end_time, 1.0/samprate)
  plt.plot(timerange, stalta[int(SKIP*samprate):], color="black")

  plt.grid(True)
  
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
