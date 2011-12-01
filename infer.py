import os, sys, time
import numpy as np
from optparse import OptionParser
import matplotlib
matplotlib.use('PDF')
from matplotlib.backends.backend_pdf import PdfPages

from database.dataset import *
import database.db
import netvisa, sigvisa, learn
from results.compare import *
from utils.geog import dist_km, degdiff
from utils.waveform import *
from sigvisa_util import *
import ast

FDET_ARID_COL, FDET_SITEID_COL, FDET_TIME_COL, FDET_AMP_COL, FDET_AZI_COL, FDET_SLO_COL, FDET_PHASE_COL, FDET_NUM_COLS = range(7+1)

def analyze_leb(netmodel, earthmodel, leb_events, leb_evlist, detections,
                sel3_events, sel3_evlist):
  inv_evs = []
  inv_detnums = []
  inv_arids = []
  for detnum in range(len(detections)):
    ev = netmodel.invert_det(detnum,0)
    if ev is not None:
      inv_evs.append(ev)
      inv_detnums.append(detnum)
      inv_arids.append(detections[detnum, DET_ARID_COL])

  leb_to_sel3 = dict(find_matching(leb_events, sel3_events))
  
  for evnum in range(len(leb_events)):
    leb_event = leb_events[evnum]
    leb_detlist = leb_evlist[evnum]
    
    print_event(netmodel, earthmodel, leb_event, leb_detlist, "LEB")
    
    print "INV:",
    for invnum, (evlon, evlat, evdepth, evtime) in enumerate(inv_evs):
      detnum = inv_detnums[invnum]
      if abs(leb_event[EV_TIME_COL] - evtime) < 50 \
         and dist_deg((evlon, evlat),
                      (leb_event[EV_LON_COL], leb_event[EV_LAT_COL])) < 5:
        print "(%d, %s," % (detnum,
                            earthmodel.PhaseName(int(detections[detnum,
                                                         DET_PHASE_COL]))),
        # compute the score of the inverted event
        event = leb_event.copy()
        event[EV_LON_COL] = evlon
        event[EV_LAT_COL] = evlat
        event[EV_DEPTH_COL] = evdepth
        event[EV_TIME_COL] = evtime
        event[EV_MB_COL] = MIN_MAGNITUDE
        print "%.1f)" % netmodel.score_event(event, leb_detlist),
        #print_event(netmodel, earthmodel, event, leb_detlist, "INV")
        #netmodel.score_world(np.array([event]), [leb_detlist], 1)
        #netmodel.score_world(np.array([event]), [[(0,detnum)]], 1)
        
    print
    
    if evnum in leb_to_sel3:
      sel3_evnum = leb_to_sel3[evnum]
      print_event(netmodel, earthmodel, sel3_events[sel3_evnum],
                  sel3_evlist[sel3_evnum], "SEL3")
    print
  
def print_events(netmodel, earthmodel, leb_events, leb_evlist, label):
  print "=" * 60
  score = 0
  for evnum in range(len(leb_events)):
    score += print_event(netmodel, earthmodel, leb_events[evnum],
                         leb_evlist[evnum], label)
    print "-" * 60
  print "Total: %.1f" % score
  print "=" * 60
  
def print_event(netmodel, earthmodel, event, event_detlist, label):
  print ("%s: lon %4.2f lat %4.2f depth %3.1f mb %1.1f time %.1f orid %d"
         % (label, event[ EV_LON_COL], event[ EV_LAT_COL],
            event[ EV_DEPTH_COL], event[ EV_MB_COL],
            event[ EV_TIME_COL], event[ EV_ORID_COL]))
  print "Detections:",
  detlist = [x for x in event_detlist]
  detlist.sort()
  for phaseid, detid in detlist:
    print "(%s, %d)" % (earthmodel.PhaseName(phaseid), detid),
  print
  score = netmodel.score_event(event, event_detlist)
  print "Ev Score: %.1f    (prior location logprob %.1f)" \
        % (score, netmodel.location_logprob(event[ EV_LON_COL],
                                            event[ EV_LAT_COL],
                                            event[ EV_DEPTH_COL]))
  return score

def write_events(netmodel, earthmodel, events, ev_detlist, runid, maxtime,
                 detections):
  # store the events and associations

  print "called write_events on events", events, ", detlist ", ev_detlist

  conn = database.db.connect()
  cursor = conn.cursor()
  world_score = 0.0
  for evnum in range(len(events)):
    event = events[evnum]
    detlist = ev_detlist[evnum]
    evscore = netmodel.score_event(event, detlist)
    world_score += evscore
    
    cursor.execute("insert into visa_origin (runid, orid, lon, lat, depth, "
                   "time, mb, score) values (%d, %d, %f, %f, %f, %f, %f, %f)"%
                   (runid, event[EV_ORID_COL], event[EV_LON_COL],
                    event[EV_LAT_COL], event[EV_DEPTH_COL], event[EV_TIME_COL],
                    event[EV_MB_COL], evscore))
    
    for phaseid, detnum in detlist:
      arrtime = earthmodel.ArrivalTime(event[EV_LON_COL], event[EV_LAT_COL],
                                       event[EV_DEPTH_COL], event[EV_TIME_COL],
                                       phaseid,
                                       int(detections[detnum, DET_SITE_COL]))
      if arrtime < 0:
        print "Warning: visa orid %d impossible at site %d with phase %d"\
              % (event[EV_ORID_COL], int(detections[detnum, DET_SITE_COL]),
                 phaseid)
        continue

      timeres = detections[detnum, DET_TIME_COL] - arrtime
      
      seaz = earthmodel.ArrivalAzimuth(event[EV_LON_COL], event[EV_LAT_COL],
                                       int(detections[detnum, DET_SITE_COL]))
      azres = degdiff(seaz, detections[detnum, DET_AZI_COL])

      arrslo = earthmodel.ArrivalSlowness(event[EV_LON_COL], event[EV_LAT_COL],
                                          event[EV_DEPTH_COL], phaseid,
                                       int(detections[detnum, DET_SITE_COL]))
      
      slores = detections[detnum, DET_SLO_COL] - arrslo
      
      cursor.execute("insert into visa_assoc(runid, orid, phase, arid, score, "
                     "timeres, azres, slores) "
                     "values (%d, %d, '%s', %d, %f, %f, %f, %f)" %
                     (runid, event[EV_ORID_COL],
                      earthmodel.PhaseName(phaseid),
                      detections[detnum, DET_ARID_COL],
                      netmodel.score_event_det(event, phaseid, detnum),
                      timeres, azres, slores))
  
  cursor.execute("update visa_run set data_end=%f, run_end=now(), "
                 "score = score + %f where runid=%d" %
                 (maxtime, world_score, runid))
  conn.commit()

def write_events_sig(sigmodel, earthmodel, events, ev_arrlist, runid, maxtime):
  # store the events and associations

  print "called write_events on events", events, ", arrlist ", ev_arrlist

  conn = database.db.connect()
  cursor = conn.cursor()
  world_score = 0.0
  for evnum in range(len(events)):
    event = events[evnum]
    arrlist = ev_arrlist[evnum]

    # TODO: replace this with a true world score; this is just an
    # approximation.
    world_score += event[EV_SC_COL]
    
    cursor.execute("insert into visa_origin (runid, orid, lon, lat, depth, "
                   "time, mb, score) values (%d, %d, %f, %f, %f, %f, %f, %f)"%
                   (runid, event[EV_ORID_COL], event[EV_LON_COL],
                    event[EV_LAT_COL], event[EV_DEPTH_COL], event[EV_TIME_COL],
                    event[EV_MB_COL], event[EV_SC_COL]))
    
    for evid, siteid, phaseid, arrtime, arramp, arrazi, arrslo in arrlist:
   
      cursor.execute("insert into visa_arriv(runid, orid, siteid, phase, time, amp, azi, slo) "
                     "values (%d, %d, '%d', '%s', %f, %f, %f, %f)" %
                     (runid, event[EV_ORID_COL],
                      siteid,
                      earthmodel.PhaseName(phaseid),
                      arrtime, arramp, arrazi, arrslo))
  
  cursor.execute("update visa_run set data_end=%f, run_end=now(), "
                 "score = score + %f where runid=%d" %
                 (maxtime, world_score, runid))
  conn.commit()

def log_envelope_plot(pp, events, evarrlist, real_env, pred_env, text=""):
  
  trc = real_env[0]

  siteid = trc.stats["siteid"]
  start_time = trc.stats["starttime_unix"]
  if trc.stats["window_size"] is not None:
    srate = 1/ ( trc.stats.window_size * (1- trc.stats.overlap) )
    npts = trc.stats.npts_processed
  else:
    srate = trc.stats.sampling_rate
    npts = trc.stats.npts
  end_time = start_time + npts/srate

  arrtimes = []
  for ev_l in evarrlist:
    for sta_l in ev_l:
      if int(sta_l[1]) == siteid:
        if sta_l[3] >= start_time and sta_l[3] <= end_time:
          arrtimes.append(sta_l[3])

  text = "REAL ENVELOPE: siteid %d %s" % (siteid, text)
  plot_segment(real_env, pp, "pdf", title = text, all_det_times = arrtimes)

  text = "PRED ENVELOPE: siteid %d %s" % (siteid, text)
  plot_segment(pred_env, pp, "pdf", title= text, all_det_times = arrtimes)

  return True

def tryeval(val):
  try:
    val = ast.literal_eval(val)
  except ValueError:
    pass
  return val

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-n", "--numsamples", dest="numsamples", default=10,
                    type="int",
                    help = "number of samples per window step (10)")
  parser.add_option("-c", "--birthsteps", dest="birthsteps", default=2,
                    type="int",
                    help = "number of steps per dimension in birth move (2)")
  parser.add_option("-w", "--window", dest="window", default=3600,
                    type="int",
                    help = "window size in seconds (3600)")
  parser.add_option("-t", "--step", dest="step", default=3600,
                    type="int",
                    help = "window step-size in seconds (3600)")
  parser.add_option("-r", "--hours", dest="hours", default=None,
                    type="float",
                    help = "inference on HOURS worth of data (all)")
  parser.add_option("-k", "--skip", dest="skip", default=0,
                    type="float",
                    help = "skip the first HOURS of data (0)")
  parser.add_option("-s", "--seed", dest="seed", default=123456789,
                    type="int",
                    help = "random number generator seed (123456789)")
  parser.add_option("-g", "--sigvisa", dest="sigvisa", default=False,
                    action = "store_true",
                    help = "do inference using SIGVISA (False)")
  parser.add_option("-x", "--text", dest="gui", default=True,
                    action = "store_false",
                    help = "text only output (False)")
  parser.add_option("-v", "--verbose", dest="verbose", default=False,
                    action = "store_true",
                    help = "verbose output (False)")
  parser.add_option("-d", "--descrip", dest="descrip", default="",
                    help = "description of the run ('')")
  parser.add_option("-l", "--label", dest="label", default="validation",
                    help = "training, validation (default), or test")
  parser.add_option("-p", "--propose", dest="propose_run", default=None,
                    type = str,
                    help = "use run RUNID's events as proposal",
                    metavar="RUNID")
  parser.add_option("-a", "--arrival_table", dest="arrival_table",
                    default=None,
                    help = "arrival table to use for inferring on")
  parser.add_option("-y", "--noarrays", dest="noarrays", default=False,
                    action="store_true",
                    help = "omit data from array stations, used for sigvisa testing (False)")
  parser.add_option("--ar-perturb", dest="ar_perturb", default=False,
                    action="store_true", help = "use autoregressive perturbation model for signals (False)")
  parser.add_option("--synthetic", dest="synthetic", default=False,
                    action="store_true", help = "generate synthetic signals based on hard-coded events, used for sigvisa testing (False)")
  parser.add_option("--det-propose", dest="det_propose", default=False,
                    action="store_true", help = "use idcx_arrivals detections for event proposals used for sigvisa testing (False)")
  parser.add_option("--clean-det-propose", dest="clean_det_propose", default=None,
                    type=str, help = "propose using only detections associated with the specified event ()")
  parser.add_option("--dummy-propose", dest="dummy_propose", default=None,
                    type=str, help = "use the specified list of LEB orids as event proposals; used for sigvisa testing ()")
  parser.add_option("--siteids", dest="siteids", default=None,
                    type = str, help = "limit SIGVISA inference to a specific set of stations, used for sigvisa testing. ('')")
  parser.add_option("-z", "--threads", dest="threads", default=1,
                    type="int",
                    help = "number of threads (1)")
  (options, args) = parser.parse_args()

  if options.seed == 0:
    options.seed = int(time.time())

  if options.sigvisa:
    sigvisa.srand(options.seed)  
  else:
    netvisa.srand(options.seed)


  # connect to database
  conn = database.db.connect()
  cursor = conn.cursor()

  if options.arrival_table is None:

    start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
                sel3_evlist, site_up, sites, phasenames, phasetimedef, arid2num \
                = read_data(options.label, hours=options.hours,
                            skip=options.skip, noarrays=options.noarrays)
  else:
    # read the time range from the table
    cursor.execute("select min(time), max(time) from %s"
                   % options.arrival_table)
    start_time, end_time = cursor.fetchone()
    if options.skip >= start_time and options.skip <= end_time:
      start_time = options.skip
    else:
      start_time += options.skip * 60. * 60.
    
    if options.hours is None:
      pass
    elif options.hours >= start_time and options.hours <= end_time:
      end_time = options.hours
    else:
      end_time = start_time + options.hours * 60. * 60.

    
    print "Dataset: %.1f hrs from %d to %d" % ((end_time-start_time)/3600,
                                               start_time, end_time),
    print "i.e. %s to %s" % (strftime("%x %X", gmtime(start_time)),
                             strftime("%x %X", gmtime(end_time)))
    
    # read the detections and uptime
    detections, arid2num = read_detections(cursor, start_time, end_time,
                                 options.arrival_table, options.noarrays)
    site_up = read_uptime(cursor, start_time, end_time,
                          options.arrival_table)
    # read the rest of the static data
    sites = read_sites(cursor)
    phasenames, phasetimedef = read_phases(cursor)
    assert(len(phasenames) == len(phasetimedef))

    earthmodel = learn.load_earth(param_dirname, sites, phasenames, phasetimedef)
    netmodel = learn.load_netvisa(param_dirname,
                                  start_time, end_time,
                                  detections, site_up, sites, phasenames,
                                  phasetimedef)
    model = netmodel

    # load sigvisa stuff
  if options.sigvisa:

    #print "loading traces for SIGVISA..."
    cursor.execute("select sta, id from static_siteid where statype='ss'")
    stations = np.array(cursor.fetchall())
    siteids = dict(stations)
    stations = stations[:,0]

    # convert all values in the siteids dictionary from strings to ints
    # TODO: figure out the proper pythonic way to do this
    siteids_ints = dict()
    for sta in siteids.keys():
      siteids_ints[sta] = int(siteids[sta])
    siteids = siteids_ints

    print "creating sigvisa model..."
    sigmodel = learn.load_sigvisa("parameters",
                                  start_time, end_time, 1 if options.ar_perturb else 0,
                                  site_up, sites, phasenames,
                                  phasetimedef)

    if options.siteids is None:
      stalist = siteids
    else:
      stalist = [tryeval(x) for x in options.siteids.split(',')]
    stalist = tuple(stalist)

    if options.dummy_propose is not None:
      prop_ids = map(lambda x : int(x), options.dummy_propose.split(','))
      prop_events = read_events(cursor, start_time, end_time, "leb")[0]
      f = lambda e : e[EV_ORID_COL] in prop_ids
      propose_events = filter(f, prop_events)
      propose_events.sort(cmp=lambda x,y: cmp(x[EV_TIME_COL], y[EV_TIME_COL]))
      propose_events = np.array(propose_events)
    else:
      if options.det_propose:
        fake_det = [real_to_fake_det(x) for x in detections]
        fake_det = filter(lambda x : x[FDET_SITEID_COL]+1 in stalist, fake_det) 
        print "using %d detections" % len(fake_det)
      elif options.clean_det_propose:
        cursor.execute("select site.id-1, iarr.arid, iarr.time, iarr.deltim, iarr.azimuth, iarr.delaz, iarr.slow, iarr.delslo, iarr.snr, ph.id-1, iarr.amp, iarr.per FROM leb_arrival iarr, leb_assoc la, static_siteid site, static_phaseid ph where iarr.arid=la.arid and la.orid=5397597 and la.sta=site.sta and site.statype='ss' and iarr.iphase=ph.phase and ascii(iarr.iphase) = ascii(ph.phase)")
        clean_dets = np.array(cursor.fetchall())
        fake_det = [real_to_fake_det(x) for x in clean_dets]
        #fake_det = filter(lambda x : x[FDET_SITEID_COL]+1 in stalist, fake_det) 
        print "using %d detections" % len(fake_det)
        stalist2 = [tryeval(d[0])+1 for d in clean_dets]
        stalist = tuple(stalist2)
        print "stalist", stalist
        print "fake_det", fake_det
      else: 
        print "getting waves"
        signals = sigmodel.get_waves()
        print "got waves"
        sta_high_thresholds = dict()
        sta_low_thresholds = dict()
        for i in range(200):
          sta_high_thresholds[i] = 2
          sta_low_thresholds[i] = 0.8
        print "signals set, computing fake detections"
        fake_det = fake_detections(signals, sta_high_thresholds, sta_low_thresholds)

        print "real dets"
        real_det = [real_to_fake_det(x) for x in detections]
        real_det = filter(lambda x : x[FDET_SITEID_COL]+1 in stalist, real_det) 
        for det in real_det:
          print det
      
      print "fake_det", fake_det
      print "setting fake detections"
      sigmodel.set_fake_detections(fake_det)


    if options.synthetic:
      evlist = np.matrix( ( (0, 0, 0, 1237680500, 3.0, 1)))
#                           (-10, 10, 0, 123761000, 3.0, 2),
#                           (-10, -10, 0, 1237681500, 3.0, 3))  )
      sigmodel.synthesize_signals(evlist, stalist, start_time, end_time, 40)
    else:
      energies, traces = load_and_process_traces(cursor, start_time, end_time, 1, .5, stalist)
      print "loaded, setting waves"
      sigmodel.set_waves(traces)
      print "loaded, setting energies"
      sigmodel.set_signals(energies)

      

    model = sigmodel

##   if options.verbose:
##     print "===="
##     print "LEB:"
##     print "===="
##     analyze_leb(netmodel, earthmodel, leb_events, leb_evlist, detections,
##                 sel3_events, sel3_evlist)
  
  #print_events(netmodel, earthmodel, leb_events, leb_evlist, "LEB")
  #netmodel.score_world(leb_events, leb_evlist, 1)

  #print "===="
  #print "SEL3:"
  #print "===="
  #print_events(netmodel, earthmodel, sel3_events, sel3_evlist, "SEL3")
  #netmodel.score_world(sel3_events, sel3_evlist, 1)

  # create a runid

  cursor.execute ("insert into visa_run(run_start, numsamples, window, step, "
                  "seed, data_start, noarrays, score, descrip) values "
                  "(now(), %d, %d, %d, %d, %f, %s, 0, '%s')" %
                  (options.numsamples, options.window, options.step,
                   options.seed, start_time, "TRUE" if options.noarrays else "FALSE", options.descrip))
  conn.commit()
  cursor.execute("select max(runid) from visa_run")
  runid, = cursor.fetchone()

  print "===="
  print "NET runid %d" % runid
  print "===="
  if options.propose_run is not None:
    propose_events = []
    for prop_type in options.propose_run.split(","):
      # is this a NET-VISA runid?
      if prop_type.isdigit():
        prop_run = int(prop_type)
        prop_events = read_events(cursor, start_time, end_time, "visa",
                                  prop_run)[0]
      else:
        prop_events = read_events(cursor, start_time, end_time,
                                  prop_type)[0]
      propose_events.extend(prop_events.tolist())
    propose_events.sort(cmp=lambda x,y: cmp(x[EV_TIME_COL], y[EV_TIME_COL]))
    propose_events = np.array(propose_events)
    print "Using %d events as birth proposer" % len(propose_events)
  elif options.dummy_propose is None:
    propose_events = None

  if options.sigvisa:

    pp = PdfPages('logs/signals_%d.pdf' % (runid))
    events = sigmodel.infer(runid, options.numsamples,
                                      options.birthsteps,
                                      options.window, options.step,
                                      options.threads,
                                      propose_events,
                                      options.verbose,
                            lambda a,b,c,d,e,f:
                                      write_events_sig(a,b,c,d,e,f),
                            lambda b,c,d,e,f:
                                      log_envelope_plot(pp,b,c,d,e,f))
    pp.close()
    print "inferred events", events
    return
  else:
    events, ev_detlist = netmodel.infer(runid, options.numsamples,
                                      options.birthsteps,
                                      options.window, options.step,
                                      options.threads,
                                      propose_events,
                                      options.verbose,
                                      lambda a,b,c,d,e,f:
                                      write_events(a,b,c,d,e,f,detections))
  #print_events(netmodel, earthmodel, events, ev_detlist, "VISA")


  if options.arrival_table is None:
    # store the results
    f, p, r, err = f1_and_error(leb_events, events)
    
    conn = database.db.connect()
    cursor = conn.cursor()
    cursor.execute("update visa_run set f1=%f, prec=%f, "
                   "recall=%f, error_avg=%f, error_sd=%f "
                   "where runid=%d" %
                   (f, p, r, err[0], err[1], runid))
    conn.commit()
    msg = ("Runid=%d F1=%.2f, Prec=%.2f, Recall=%.2f, Avg Error = %.2f+-%.2f"
           % (runid, f, p, r, err[0], err[1]))
  
  else:
    msg = "Runid=%d #events=%d" % (runid, len(events))
    
  if options.verbose:      
    print "===="
    print "EVAL"
    print "===="  
    true_idx, false_idx, mat_idx = find_true_false_guess(leb_events, events)
    unmat_idx = [x for x in range(len(leb_events))]
    print "Matched:"
    for lebevnum, evnum in mat_idx:
      print "/--"
      print_event(netmodel, earthmodel, leb_events[lebevnum],
                  leb_evlist[lebevnum], "LEB")
      print "~~ %.1f km ~~>" % dist_km(leb_events[lebevnum,
                                                  [EV_LON_COL, EV_LAT_COL]],
                                       events[evnum, [EV_LON_COL, EV_LAT_COL]])
      print_event(netmodel, earthmodel, events[evnum], ev_detlist[evnum],
                  "VISA")
      print "\--"
      unmat_idx.remove(lebevnum)
    if len(unmat_idx):
      print "Unmatched:"
    for lebevnum in unmat_idx:
      print "--"
      print_event(netmodel, earthmodel, leb_events[lebevnum],
                  leb_evlist[lebevnum], "LEB")
      print "--"
    if len(false_idx):
      print "Spurious:"
    for evnum in false_idx:
      print "~~"
      print_event(netmodel, earthmodel, events[evnum], ev_detlist[evnum],
                  "VISA")
      print "~~"
      
  print(msg)

if __name__ == "__main__":
  main("parameters")
