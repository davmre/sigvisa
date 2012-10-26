import os, cPickle, sys, tarfile, csv
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

from database.dataset import *

from datetime import datetime
from database.az_slow_corr import load_az_slow_corr

import priors.SecDetPrior
import priors.NumEventPrior
import priors.EventLocationPrior
import priors.EventMagPrior
import priors.EventDetectionPrior
import priors.NumFalseDetPrior
import priors.ArrivalTimePrior
import priors.ArrivalAzimuthPrior
import priors.ArrivalSlownessPrior
import priors.ArrivalPhasePrior
import priors.ArrivalSNR
import priors.ArrivalAmplitudePrior

import sigvisa_c, sigvisa

def log_trace(trc, filename, format):

  real_fn = 'logs/%s.pdf' % (filename)

#  while (os.path.exists(real_fn)):
#    real_fn = real_fn + "_"

  print "logging to file", real_fn

  pp = PdfPages(real_fn)

  siteid = trc.stats["siteid"]
  start_time = trc.stats["starttime_unix"]
  if trc.stats["window_size"] is not None:
    srate = 1/ ( trc.stats.window_size * (1- trc.stats.overlap) )
    npts = trc.stats.npts_processed
  else:
    srate = trc.stats.sampling_rate
    npts = trc.stats.npts
  end_time = start_time + npts/srate

  text = "%s: siteid %d" % (filename, siteid)
  print text
  utils.waveform.plot_trace(trc, title = text, format=format)
  pp.savefig()
  pp.close()

  return True


def load_earth(param_dirname, sites, phasenames, phasetimedef):
  model = sigvisa_c.EarthModel(sites, phasenames, phasetimedef,
                             os.path.join(param_dirname, "ttime", "iasp91."),
                             "",
                             os.path.join(param_dirname,"GA_dist_depth_ranges"),
                             os.path.join(param_dirname,"qfvc.mb"))

  return model


def load_sigvisa(param_dirname, start_time, end_time, signal_model_name, site_up,
                 sites, phasenames, phasetimedef, load_signal_params = True):
  earthmodel = load_earth(param_dirname, sites, phasenames, phasetimedef)

  sigmodel = sigvisa_c.SigModel(earthmodel, start_time, end_time, signal_model_name,
                              os.path.join(param_dirname, "NumEventPrior.txt"),
                              os.path.join(param_dirname, "EventLocationPrior.txt"),
                              os.path.join(param_dirname, "EventMagPrior.txt"),
                              os.path.join(param_dirname, "ArrivalTimePrior.txt"),
                              os.path.join(param_dirname, "ArrivalAzimuthPrior.txt"),
                              os.path.join(param_dirname, "ArrivalSlownessPrior.txt"),
                              os.path.join(param_dirname, "ArrivalAmplitudePrior.txt"),
                              log_trace)

#  if (load_signal_params):
#    sigmodel.set_all_signal_params(priors.SignalPrior.read_params(os.path.join(param_dirname, "EnvelopeSignalModel.txt")))

  return sigmodel


def learn_signal(param_dirname, earthmodel, hours, site_up, sites, phasenames, phasetimedef):
  print "learning signal parameters from %f hour of data" % (hours)

  MAX_TRAVEL_TIME = 2000
  start_time = 1237680000
  end_time = start_time + hours*3600
  earliest_event_time = start_time - MAX_TRAVEL_TIME
  cursor = db.connect().cursor()

  leb_detections, leb_arid2num = read_detections(cursor, start_time, end_time, "leb_arrival", 1)
  idcx_detections, arid2num = read_detections(cursor, start_time, end_time, "idcx_arrival", 1)


  print start_time, earliest_event_time
  events, orid2num = read_events(cursor, earliest_event_time, end_time, "leb", None)
  print "loaded ", len(events), " events."

  # read associations, as training data for learning
  evlist = read_assoc(cursor, earliest_event_time, end_time, orid2num, arid2num, "leb", None)
  print "loaded associations for ", len(events), " events."


  fake_det = [sigvisa_util.real_to_fake_det(x) for x in leb_detections]

  #cursor.execute("select lon, lat, depth, time, mb, orid from leb_origin where orid=5297348")
  #events = np.array(cursor.fetchall())
  # stalist = (48,)
  #sigmodel.synthesize_signals_det(stalist, start_time, end_time, 2, 0, 0)
  #energies = sigmodel.get_signals()


  sigmodel = load_sigvisa(param_dirname, start_time, end_time, "envelope", site_up, sites, phasenames, phasetimedef, load_signal_params = False)


  return priors.SignalPrior.learn(cursor, earthmodel, sigmodel, events,
                                  evlist,
                                  idcx_detections, arid2num, param_dirname,
                                  start_time, end_time, fake_det, sites)


def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-q", "--quick", dest="quick", default=False,
                    action = "store_true",
                    help = "quick training on a subset of data (False)")
  parser.add_option("-r", "--hours", dest="hours", default=None,
                    type="float",
                    help = "train on HOURS worth of data (all)")
  parser.add_option("-g", "--gui", dest="gui", default=False,
                    action = "store_true",
                    help = "display the model  (False)")
  parser.add_option("-s", "--silent", dest="verbose", default=True,
                    action = "store_false",
                    help = "silent, i.e. no output (False)")
  parser.add_option("-1", "--type1", dest="type1", default=False,
                    action = "store_true",
                    help = "Type 1 fonts (False)")
  parser.add_option("-i", "--visa_leb_runid", dest="visa_leb_runid",
                    default=None, help = "Visa runid to be treated as leb",
                    metavar="RUNID", type="int")
  parser.add_option("-m", "--model", dest="model",
                    default=None, help = "Which model(s) to learn (all)")
  parser.add_option("-w", "--writefig", dest="writefig",
                    default=None, help = "Directory to save figures (None)",
                    metavar="DIR")
  parser.add_option("-p", "--pdf", dest="pdf",
                    default=None, help = "pdf file to save figures (None)",
                    metavar="FILE")
  parser.add_option("-d", "--datadir", dest="datadir",
                    default=None, help = "Directory to save data (None)",
                    metavar="DIR")
  parser.add_option("--datafile", dest="datafile", default=None,
                    help = "tar file with data (None)", metavar="FILE")
  parser.add_option("-c", "--cache", dest="cache", default=False,
                    action = "store_true",
                    help = "write data to cache and read from cache (False)")
  parser.add_option("--sigvisa", dest="sigvisa", default=False,
                    action = "store_true",
                    help = "load waveforms and train signal prior (False)")


  (options, args) = parser.parse_args()

  if options.sigvisa:
    hours = options.hours

  # use Type 1 fonts by invoking latex
  if options.type1:
    plt.rcParams['text.usetex'] = True

  if options.pdf:
    options.pdf = PdfPages(options.pdf)

  if options.gui:
    options.plt = plt
    showgui = True
  # if the user has not request a GUI but he does want the pictures then we
  # will set the gui flag for the subcomponents, however, we won't display
  # the generated images
  elif options.pdf or options.writefig:
    options.plt = plt
    options.gui = True
    showgui = False
  else:
    showgui = False

  if options.datadir:
    param_dirname = options.datadir

  if options.quick:
    hours = 100
  else:
    hours = options.hours

  if options.model is not None:
    options.model = set(m for m in options.model.split(","))

  if options.datafile:
    start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
                sel3_evlist, site_up, sites, phasenames, phasetimedef, \
                sitenames = read_datafile(options.datafile, param_dirname,
                                          hours = hours)
  else:

    cache_fname = "cache-training-hours-%s-visa-%s.pic1" \
                  % (str(hours), str(options.visa_leb_runid))

    if options.cache and os.path.exists(cache_fname):
      start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
                  sel3_evlist, site_up, sites, phasenames, phasetimedef \
                  = cPickle.load(file(cache_fname, "rb"))
    else:
      start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
                  sel3_evlist, site_up, sites, phasenames, phasetimedef, arid2num \
                  = read_data(hours=hours,
                              visa_leb_runid=options.visa_leb_runid)

    if options.cache and not os.path.exists(cache_fname):
      cPickle.dump((start_time, end_time, detections, leb_events, leb_evlist,
                    sel3_events, sel3_evlist, site_up, sites, phasenames,
                    phasetimedef), file(cache_fname, "wb"), protocol=1)

    sitenames = read_sitenames()

  earthmodel = load_earth(param_dirname, sites, phasenames, phasetimedef)

  false_dets = priors.SecDetPrior.compute_false_detections(detections,
                                                           leb_evlist)

  if options.sigvisa:
    learn_signal(param_dirname, earthmodel, hours, site_up, sites, phasenames, phasetimedef)
    return

  if options.model is None or "SecDet" in options.model:
    priors.SecDetPrior.learn(os.path.join(param_dirname, "SecDetPrior.txt"),
                             options, earthmodel, detections, leb_events,
                             leb_evlist)

  if options.model is None or "NumEvent" in options.model:
    priors.NumEventPrior.learn(os.path.join(param_dirname, "NumEventPrior.txt"),
                               options, start_time, end_time, leb_events)

  if options.model is None or "EventLocation" in options.model:
    priors.EventLocationPrior.learn(os.path.join(param_dirname,
                                                 "EventLocationPrior.txt"),
                                    options, leb_events)

  if options.model is None or "EventMag" in options.model:
    priors.EventMagPrior.learn(os.path.join(param_dirname,
                                            "EventMagPrior.txt"),
                               options, leb_events)

  if options.model is None or "EventDetection" in options.model:
    priors.EventDetectionPrior.learn(os.path.join(param_dirname,
                                                  "EventDetectionPrior.txt"),
                                     earthmodel, start_time, end_time,
                                     detections, leb_events, leb_evlist,
                                     site_up)

  if options.model is None or "NumFalseDet" in options.model:
    priors.NumFalseDetPrior.learn(os.path.join(param_dirname,
                                               "NumFalseDetPrior.txt"),
                                  detections, false_dets, site_up)

  if options.model is None or "ArrivalTime" in options.model:
    priors.ArrivalTimePrior.learn(os.path.join(param_dirname,
                                               "ArrivalTimePrior.txt"),
                                  earthmodel, detections, leb_events,
                                  leb_evlist)

  if options.model is None or "ArrivalAzimuth" in options.model:
    priors.ArrivalAzimuthPrior.learn(os.path.join(param_dirname,
                                                  "ArrivalAzimuthPrior.txt"),
                                     earthmodel, detections, leb_events,
                                     leb_evlist)

  if options.model is None or "ArrivalSlowness" in options.model:
    priors.ArrivalSlownessPrior.learn(os.path.join(param_dirname,
                                                   "ArrivalSlownessPrior.txt"),
                                      earthmodel, detections, leb_events,
                                      leb_evlist)

  if options.model is None or "ArrivalPhase" in options.model:
    priors.ArrivalPhasePrior.learn(os.path.join(param_dirname,
                                                "ArrivalPhasePrior.txt"),
                                   options, earthmodel, detections, leb_events,
                                   leb_evlist, false_dets)

  if options.model is None or "ArrivalSNR" in options.model:
    priors.ArrivalSNR.learn(os.path.join(param_dirname,
                                         "ArrivalSNRPrior.txt"),
                            options, earthmodel, detections, leb_events,
                            leb_evlist, false_dets)

  if options.model is None or "ArrivalAmplitude" in options.model:
    priors.ArrivalAmplitudePrior.learn(
      os.path.join(param_dirname, "ArrivalAmplitudePrior.txt"),
      options, earthmodel, detections, leb_events, leb_evlist, false_dets)

  if options.model is None or "Site" in options.model:
    learn_site(os.path.join(param_dirname, "SitePrior.txt"), sitenames, sites)

  if options.model is None or "Phase" in options.model:
    learn_phase(os.path.join(param_dirname, "PhasePrior.txt"), phasenames,
                phasetimedef)

  if options.pdf:
    options.pdf.close()

  if showgui:
    plt.show()

def read_datafile_and_sitephase(data_fname, param_dirname, hours=None, skip=0,
                                verbose=True):
  return read_datafile(data_fname, param_dirname, hours=hours, skip=skip,
                   site_fname = os.path.join(param_dirname, "SitePrior.txt"),
                   phase_fname = os.path.join(param_dirname, "PhasePrior.txt"),
                       verbose=verbose)

def read_datafile(data_fname, param_dirname, hours=None, skip=0,
                  site_fname=None, phase_fname=None, verbose=True):
  tar = tarfile.open(name=data_fname, mode="r:*")
  fobj = tar.extractfile("dataset.csv")
  inp = csv.reader(fobj, quoting=csv.QUOTE_NONNUMERIC)
  inp.next()                            # skip header
  (label, file_start_time, file_end_time) = inp.next()

  # compute the start_time and end_time for the dataset
  if skip >= file_start_time and skip <= file_end_time:
    start_time = skip
  else:
    start_time = file_start_time + skip * 60 * 60

  if hours is None:
    end_time = file_end_time
  elif hours >= file_start_time and hours <= file_end_time:
    end_time = hours
  else:
    end_time = start_time + hours * 60 * 60

  if start_time > file_end_time or start_time < file_start_time \
     or end_time < file_start_time or end_time > file_end_time \
     or end_time < start_time:
    raise ValueError("invalid skip %d and hours %s parameter ; data range"
                     " %.1f to %.1f" % (skip, str(hours), file_start_time,
                                        file_end_time))

  if verbose:
    print "Importing data from", datetime.utcfromtimestamp(start_time),\
          "to", datetime.utcfromtimestamp(end_time)

  # sites

  if site_fname is not None:
    fobj = open(site_fname)
  else:
    fobj = tar.extractfile("site.csv")
  inp = csv.reader(fobj, quoting=csv.QUOTE_NONNUMERIC)
  inp.next()                            # skip header

  sitenames = []
  sites = []
  siteid = {}
  for sta, lon, lat, elev, isarray in inp:
    sitenames.append(sta)
    sites.append((lon, lat, elev, isarray))
    siteid[sta] = len(siteid)
  sites = np.array(sites)
  sitenames = np.array(sitenames)

  if verbose:
    print "%d sites, %d array" % (len(sites), sites[:,3].sum())

  # phases

  if phase_fname is not None:
    fobj = open(phase_fname)
  else:
    fobj = tar.extractfile("phase.csv")
  inp = csv.reader(fobj, quoting=csv.QUOTE_NONNUMERIC)
  inp.next()                            # skip header

  phasenames = []
  phasetimedef = []
  phaseid = {}

  for phase, istimedef in inp:
    phasenames.append(phase)
    phasetimedef.append(istimedef)
    phaseid[phase] = len(phaseid)

  phasenames = np.array(phasenames)
  phasetimedef = np.array(phasetimedef, bool)

  # arrivals and site_up
  # corrections need to be applied to the arrivals
  corr_dict = load_az_slow_corr(os.path.join(param_dirname, 'sasc'))

  fobj = tar.extractfile("idcx_arrival.csv")
  inp = csv.reader(fobj, quoting=csv.QUOTE_NONNUMERIC)
  inp.next()                            # skip header

  site_up = np.zeros((len(sites),
           int(ceil((end_time + MAX_TRAVEL_TIME - start_time)/ UPTIME_QUANT))),
                     bool)

  detections = []
  arid2detnum = {}
  for (sta, arid, time, deltim, azimuth, delaz, slow, delslo, snr, iphase, amp,
       per) in inp:
    if time < start_time:
      continue
    if time > (end_time + MAX_TRAVEL_TIME):
      break

    if sta not in siteid or delaz <= 0 or delslo <= 0 or snr <= 0:
      continue

    # apply the SASC correction
    if sta in corr_dict:
      azimuth, slow, delaz, delslo = corr_dict[sta].correct(azimuth, slow,
                                                            delaz, delslo)

    detections.append((siteid[sta], arid, time, deltim, azimuth, delaz, slow,
                       delslo, snr, phaseid[iphase], amp, per))
    arid2detnum[arid] = len(arid2detnum)
    site_up[siteid[sta], int((time - start_time) / 3600)] = True
  detections = np.array(detections)

  if verbose:
    print "%d detections" % len(detections)

  # LEB events

  fobj = tar.extractfile("leb_origin.csv")
  inp = csv.reader(fobj, quoting=csv.QUOTE_NONNUMERIC)
  inp.next()                            # skip header

  leb_events = []
  leb_orid2num = {}
  for lon, lat, depth, time, mb, orid in inp:
    if time < start_time or time > end_time:
      continue
    if depth < 0:
      depth = 0
    if mb < 0:
      mb = MIN_MAGNITUDE
    leb_events.append((lon, lat, depth, time, mb, orid))
    leb_orid2num[orid] = len(leb_orid2num)
  leb_events = np.array(leb_events)

  # LEB evlist

  fobj = tar.extractfile("leb_assoc.csv")
  inp = csv.reader(fobj, quoting=csv.QUOTE_NONNUMERIC)
  inp.next()                            # skip header

  leb_evlist = [[] for _ in leb_events]

  for orid, arid, phase, sta in inp:
    if orid in leb_orid2num and arid in arid2detnum:
      leb_evlist[leb_orid2num[orid]].append((phaseid[phase],arid2detnum[arid]))

  if verbose:
    print "LEB: %d events, avg %.1f site-phases" \
          % (len(leb_events), float(sum(len(evlist) for evlist in leb_evlist))
             /(.00001 + len(leb_evlist)))

  # SEL3 events

  fobj = tar.extractfile("sel3_origin.csv")
  inp = csv.reader(fobj, quoting=csv.QUOTE_NONNUMERIC)
  inp.next()                            # skip header

  sel3_events = []
  sel3_orid2num = {}
  for lon, lat, depth, time, mb, orid in inp:
    if time < start_time or time > end_time:
      continue
    if depth < 0:
      depth = 0
    if mb < 0:
      mb = MIN_MAGNITUDE
    sel3_events.append((lon, lat, depth, time, mb, orid))
    sel3_orid2num[orid] = len(sel3_orid2num)
  sel3_events = np.array(sel3_events)

  # SEL3 evlist

  fobj = tar.extractfile("sel3_assoc.csv")
  inp = csv.reader(fobj, quoting=csv.QUOTE_NONNUMERIC)
  inp.next()                            # skip header

  sel3_evlist = [[] for _ in sel3_events]

  for orid, arid, phase, sta in inp:
    if orid in sel3_orid2num and arid in arid2detnum:
      sel3_evlist[sel3_orid2num[orid]].append((phaseid[phase],
                                               arid2detnum[arid]))

  if verbose:
    print "SEL3: %d events, avg %.1f site-phases" \
          % (len(sel3_events),float(sum(len(evlist) for evlist in sel3_evlist))
             /(.00001 + len(sel3_evlist)))

  return start_time, end_time, detections, leb_events, leb_evlist,\
         sel3_events, sel3_evlist, site_up, sites, phasenames, phasetimedef,\
         sitenames

def learn_site(param_file, sitenames, sites):
  out = csv.writer(open(param_file, "wb"), quoting=csv.QUOTE_NONNUMERIC)
  out.writerow(["STA", "LON", "LAT", "ELEV", "ISARRAY"])
  for sta, (lon, lat, elev, isarray) in zip(sitenames, sites):
    out.writerow((sta, lon, lat, elev, int(isarray)))
  del out

def learn_phase(param_file, phasenames, phasetimedef):
  out = csv.writer(open(param_file, "wb"), quoting=csv.QUOTE_NONNUMERIC)
  out.writerow(["PHASE", "ISTIMEDEF"])
  for phase, istimedef in zip(phasenames, phasetimedef):
    out.writerow((phase, int(istimedef)))
  del out

if __name__ == "__main__":
  np.seterr(divide = 'raise', invalid='raise', over='ignore')
  try:
    main("parameters")
  except SystemExit:
    raise
  except:
    import pdb, traceback, sys
    traceback.print_exc(file=sys.stdout)
    pdb.post_mortem(sys.exc_traceback)
    raise


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

