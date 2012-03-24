# visualize the model
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os, cPickle, sys

from database.dataset import *
from learn import load_earth, load_netvisa, read_datafile_and_sitephase
from utils.draw_earth import draw_events, draw_earth, draw_density
import priors.SecDetPrior

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-q", "--quick", dest="quick", default=False,
                    action = "store_true",
                    help = "quick training on a subset of data (False)")
  parser.add_option("-r", "--hours", dest="hours", default=None,
                    type="float",
                    help = "train on HOURS worth of data (all)")
  parser.add_option("-1", "--type1", dest="type1", default=False,
                    action = "store_true",
                    help = "Type 1 fonts (False)")
  parser.add_option("-w", "--writefig", dest="writefig",
                    default=None, help = "Directory to save figures (None)",
                    metavar="DIR")
  parser.add_option("--pdf", dest="pdf",
                    default=None, help = "pdf file to save figures (None)",
                    metavar="FILE")
  parser.add_option("-m", "--model", dest="model",
                    default=None, help = "Which model(s) to visualize (all)")
  parser.add_option("-x", "--textonly", dest="textonly", default=False,
                    action = "store_true",
                    help = "Text only output (False)")
  parser.add_option("-c", "--cache", dest="cache", default=False,
                    action = "store_true",
                    help = "write data to cache and read from cache (False)")
  parser.add_option("-i", "--visa_leb_runid", dest="visa_leb_runid",
                    default=None, help = "Visa runid to be treated as leb",
                    metavar="RUNID", type="int")
  parser.add_option("-p", "--phaseid", dest="phaseid", type="int",
                    default=0, help = "phase to display")
  parser.add_option("-s", "--sta", dest="sta",
                    default="ASAR", help = "site to display (ASAR)",
                    metavar="STATION")
  parser.add_option("--datafile", dest="datafile", default=None,
                    help = "tar file with data (None)", metavar="FILE")
  
  (options, args) = parser.parse_args()
  
  # use Type 1 fonts by invoking latex
  if options.type1:
    plt.rcParams['text.usetex'] = True
    
  if options.pdf:
    options.pdf = PdfPages(options.pdf)
  
  if options.model is not None:
    options.model = set(m for m in options.model.split(","))
  
  if options.quick:
    hours = 100
  else:
    hours = options.hours

  if options.datafile:
    start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
                sel3_evlist, site_up, sites, phasenames, phasetimedef, \
                sitenames = read_datafile_and_sitephase(options.datafile,
                                                        param_dirname,
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
                  sel3_evlist, site_up, sites, phasenames, phasetimedef \
                  = read_data(hours=hours, visa_leb_runid=options.visa_leb_runid)
    sitenames = read_sitenames()

  earthmodel = load_earth(param_dirname, sites, phasenames, phasetimedef)
  netmodel = load_netvisa(param_dirname, start_time, end_time,
                          detections, site_up, sites, phasenames, phasetimedef)
  false_dets = priors.SecDetPrior.compute_false_detections(detections,
                                                           leb_evlist)
  # find the requested station
  if not len(np.where(sitenames == options.sta)[0]):
    print "Unknown station", options.sta
    sys.exit(1)
  options.siteid = int(np.where(sitenames == options.sta)[0])
  
  if options.model is None or "EventLocation" in options.model:
    EventLocation(options, earthmodel, netmodel, leb_events)
    
  if options.model is None or "EventDetection" in options.model:
    EventDetection(options, earthmodel, netmodel, start_time, end_time,
                   detections, leb_events, leb_evlist, site_up)

  if options.model is None or "ArrivalTime" in options.model:
    ArrivalTime(options, earthmodel, sitenames, netmodel,
                detections, leb_events, leb_evlist)
  
  if options.model is None or "ArrivalSlowness" in options.model:
    ArrivalSlowness(options, earthmodel, sitenames, netmodel,
                    detections, leb_events, leb_evlist)
  
  if options.model is None or "ArrivalAzimuth" in options.model:
    ArrivalAzimuth(options, earthmodel, sitenames, netmodel,
                   detections, leb_events, leb_evlist)

  if options.model is None or "ArrivalAmplitude" in options.model:
    ArrivalAmplitude(options, earthmodel, sitenames, netmodel,
                     detections, leb_events, leb_evlist, false_dets)

  if options.model is None or "ArrivalPhase" in options.model:
    ArrivalPhase(options, earthmodel, netmodel)
  
  if options.pdf:
    options.pdf.close()
  
  if not options.textonly:
    plt.show()

def EventLocation(options, earthmodel, netmodel, leb_events):
  DEPTH_STEP = 100
  for depth in range(0, int(MAX_DEPTH), DEPTH_STEP):
    bmap = draw_earth("Depth %.1f - %.1f" % (depth, depth+DEPTH_STEP))
    leb_idx = (leb_events[:,EV_DEPTH_COL]>=depth)\
              &(leb_events[:,EV_DEPTH_COL]<(depth+MAX_DEPTH))
    draw_events(bmap, leb_events[leb_idx][:,[EV_LON_COL, EV_LAT_COL]],
                marker="*", ms=8, mfc="yellow")

def ArrivalTime(options, earthmodel, sitenames, netmodel,
                detections, leb_events, leb_evlist):
  residuals = []
  for evnum, event in enumerate(leb_events):
    for phaseid, detnum in leb_evlist[evnum]:
      if phaseid == options.phaseid and detections[detnum, DET_SITE_COL] == options.siteid:
        evlon, evlat, evdepth, evtime = event[EV_LON_COL], event[EV_LAT_COL],\
                                        event[EV_DEPTH_COL], event[EV_TIME_COL]
        pred_arrtime =  earthmodel.ArrivalTime(evlon, evlat, evdepth, evtime,
                                               options.phaseid, options.siteid)
        if pred_arrtime > 0:
          res = detections[detnum, DET_TIME_COL] - pred_arrtime
          residuals.append(res)

  minr, maxr = min(residuals), max(residuals)
  phasename = earthmodel.PhaseName(options.phaseid)
  sitename = sitenames[options.siteid]
  
  plt.figure(figsize=(8,4.8))
  plt.title("Arrival Time - %s (%s)" % (phasename, sitename))
  
  xpts = np.linspace(minr, maxr, 100)
  plt.hist(residuals, bins=xpts, normed=True, label="data", facecolor="blue")
  ypts = [np.exp(netmodel.arrtime_logprob(x, 0, 0, options.siteid, options.phaseid))
          for x in xpts]
  plt.plot(xpts, ypts, label="model", color="black", linewidth=3)
  plt.xlabel("Time (s)")
  plt.ylabel("Probability density")
  plt.legend(loc="upper left")
  if options.writefig is not None:
    basename = os.path.join(options.writefig,
                            "ArrivalTime-%s-%s" % (sitename, phasename))
    if options.type1:
      plt.savefig(basename+".pdf")
    else:
      plt.savefig(basename+".png")
        
  if options.pdf:
    options.pdf.savefig()
  
  
def ArrivalSlowness(options, earthmodel, sitenames, netmodel,
                    detections, leb_events, leb_evlist):
  residuals = []
  for evnum, event in enumerate(leb_events):
    for phaseid, detnum in leb_evlist[evnum]:
      if phaseid == options.phaseid and detections[detnum, DET_SITE_COL] == options.siteid:
        evlon, evlat, evdepth, evtime = event[EV_LON_COL], event[EV_LAT_COL],\
                                        event[EV_DEPTH_COL], event[EV_TIME_COL]
        pred_arrslo =  earthmodel.ArrivalSlowness(evlon, evlat, evdepth,
                                                  options.phaseid, options.siteid)
        res = pred_arrslo - detections[detnum, DET_SLO_COL]
        residuals.append(res)

  minr, maxr = min(residuals), max(residuals)
  phasename = earthmodel.PhaseName(options.phaseid)
  sitename = sitenames[options.siteid]
  
  plt.figure(figsize=(8,4.8))
  plt.title("Arrival Slowness - %s (%s)" % (phasename, sitename))
  
  xpts = np.linspace(minr, maxr, 100)
  plt.hist(residuals, bins=xpts, normed=True, label="data", facecolor="blue")
  ypts = [np.exp(netmodel.arrslo_logprob(x, 0, 0, options.siteid, options.phaseid))
          for x in xpts]
  plt.plot(xpts, ypts, label="model", color="black", linewidth=3)
  plt.xlabel("Slowness (s/deg)")
  plt.ylabel("Probability density")
  plt.legend(loc="upper left")
  if options.writefig is not None:
    basename = os.path.join(options.writefig,
                            "ArrivalSlowness-%s-%s" % (sitename, phasename))
    if options.type1:
      plt.savefig(basename+".pdf")
    else:
      plt.savefig(basename+".png")
        
  if options.pdf:
    options.pdf.savefig()

def ArrivalAzimuth(options, earthmodel, sitenames, netmodel,
                   detections, leb_events, leb_evlist):
  residuals = []
  for evnum, event in enumerate(leb_events):
    for phaseid, detnum in leb_evlist[evnum]:
      if phaseid == options.phaseid and detections[detnum, DET_SITE_COL] == options.siteid:
        evlon, evlat, evdepth, evtime = event[EV_LON_COL], event[EV_LAT_COL],\
                                        event[EV_DEPTH_COL], event[EV_TIME_COL]
        pred_arraz =  earthmodel.ArrivalAzimuth(evlon, evlat, options.siteid)
        res = earthmodel.DiffAzimuth(pred_arraz,
                                     detections[detnum, DET_AZI_COL])
        residuals.append(res)

  minr, maxr = min(residuals), max(residuals)
  phasename = earthmodel.PhaseName(options.phaseid)
  sitename = sitenames[options.siteid]
  
  plt.figure(figsize=(8,4.8))
  plt.title("Arrival Azimuth - %s (%s)" % (phasename, sitename))
  
  xpts = np.linspace(minr, maxr, 100)
  plt.hist(residuals, bins=xpts, normed=True, label="data", facecolor="blue")
  ypts = [np.exp(netmodel.arraz_logprob(x, 0, 0, options.siteid, options.phaseid))
          for x in xpts]
  plt.plot(xpts, ypts, label="model", color="black", linewidth=3)
  plt.xlabel("Azimuth (deg)")
  plt.ylabel("Probability density")
  plt.legend(loc="upper left")
  if options.writefig is not None:
    basename = os.path.join(options.writefig,
                            "ArrivalAzimuth-%s-%s" % (sitename, phasename))
    if options.type1:
      plt.savefig(basename+".pdf")
    else:
      plt.savefig(basename+".png")
      
  if options.pdf:
    options.pdf.savefig()

def ArrivalAmplitude(options, earthmodel, sitenames, netmodel,
                     detections, leb_events, leb_evlist, false_dets):
  data = []
  for evnum, event in enumerate(leb_events):
    for phaseid, detnum in leb_evlist[evnum]:
      if phaseid == options.phaseid \
             and detections[detnum, DET_SITE_COL] == options.siteid:
        if detections[detnum, DET_AMP_COL] < 0:
          continue
        evmb, evdepth, evtime = event[EV_MB_COL], event[EV_DEPTH_COL],\
                                event[EV_TIME_COL]

        if evmb >= 3 and evmb <= 4 and evdepth <= 50:
          datum = (detections[detnum, DET_TIME_COL] - evtime,
                   np.log(detections[detnum, DET_AMP_COL]))

          data.append(datum)
                 

  mint, maxt = min(data)[0], max(data)[0]
  
  phasename = earthmodel.PhaseName(options.phaseid)
  sitename = sitenames[options.siteid]
  
  plt.figure(figsize=(8,4.8))
  plt.title("Arrival Amplitude - %s (%s)" % (phasename, sitename))

  stept = (maxt - mint) // 100.
  xpts = np.arange(mint, maxt+stept, stept)
  buckets = [[] for _ in xpts]

  for ttime, amp in data:
    buckets[int((ttime-mint)//stept)].append(amp)

  means, stds = [], []
  for b in buckets:
    if len(b):
      m = np.mean(b)
      s = np.std(b)
      means.append(m)
      stds.append(s)
    else:
      means.append(np.inf)
      stds.append(np.inf)

  plt.errorbar(xpts, means, yerr=stds, ecolor="blue", elinewidth=1, ms=5,
               capsize=3, label="data 3--4 mb") 
  ypts = [netmodel.logamp(3.5, 0, x, options.siteid, options.phaseid)
          for x in xpts]
  plt.plot(xpts, ypts, label="model 3.5 mb", color="black", linewidth=3)
  plt.xlabel("Travel Time (s)")
  plt.ylabel("Log Amplitude")
  plt.legend(loc="upper left")
  plt.ylim(-3,3)
  if options.writefig is not None:
    basename = os.path.join(options.writefig,
                            "ArrivalAmp-%s-%s" % (sitename, phasename))
    if options.type1:
      plt.savefig(basename+".pdf")
    else:
      plt.savefig(basename+".png")
      
  if options.pdf:
    options.pdf.savefig()

    
  plt.figure(figsize=(8,4.8))
  plt.title("False Arrival Amplitude (%s)" % (sitename,))
  bins = plt.hist([np.log(detections[detnum, DET_AMP_COL])
                   for detnum in false_dets
                   if int(detections[detnum, DET_SITE_COL]) == options.siteid
                   and detections[detnum, DET_AMP_COL] > 0], label="data",
                  normed=True, facecolor="blue", bins=100)[1]
  plt.plot(bins, [np.exp(netmodel.falseamp_logprob_cdf(options.siteid,
                                                       np.exp(x))[0])
                  for x in bins], label="model", color="black", linewidth=3)
  plt.ylabel("Probability density")
  plt.xlabel("log(amp)")
  plt.legend(loc="upper left")
  if options.writefig is not None:
    basename = os.path.join(options.writefig,
                            "FalseArrivalAmp-%s" % (sitename,))
    if options.type1:
      plt.savefig(basename+".pdf")
    else:
      plt.savefig(basename+".png")
      
  if options.pdf:
    options.pdf.savefig()
  
def EventDetection(options, earthmodel, netmodel, start_time, end_time,
                   detections, leb_events, leb_evlist, site_up):
  numtimedefphases = earthmodel.NumTimeDefPhases()
  numsites = earthmodel.NumSites()

  # construct a dataset for the requested phase and site
  phase_data = dict((i,[]) for i in range(numtimedefphases))  
  
  for evnum, event in enumerate(leb_events):
    det_phase_site = set((phaseid, detections[detnum, DET_SITE_COL])\
                         for phaseid, detnum in leb_evlist[evnum])
    for siteid in [options.siteid]:
      dist = earthmodel.Delta(event[EV_LON_COL], event[EV_LAT_COL], siteid)
      for phaseid in [options.phaseid]:
        arrtime = earthmodel.ArrivalTime(event[EV_LON_COL], event[EV_LAT_COL],
                                         event[EV_DEPTH_COL],
                                         event[EV_TIME_COL], phaseid, siteid)
        
        # check if the site is in the shadow zone of this phase
        if arrtime < 0:
          continue
        
        # check if the site was up at the expected arrival time
        if arrtime < start_time or arrtime >= end_time \
            or not site_up[siteid, int((arrtime - start_time) / UPTIME_QUANT)]:
          continue

        isdet = int((phaseid, siteid) in det_phase_site)
        
        phase_data[phaseid].append((siteid, isdet, event[EV_MB_COL],
                                    event[EV_DEPTH_COL], dist))

  DEGBUCKET = 5
  for plot_phaseid in [options.phaseid]:
    phasename = earthmodel.PhaseName(plot_phaseid)
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Detection probability at %s for %s phase, surface event"
                % (options.sta, phasename))
    else:
      plt.title("Detection Probability - %s (ASAR)" % phasename)
    
    # now, bucket the phase data for the P phase at ASAR
    occ = np.zeros(180 // DEGBUCKET, int) + .0001
    det = np.zeros(180 // DEGBUCKET, int)
    for siteid, isdet, evmag, evdepth, dist in phase_data[plot_phaseid]:
      if siteid == options.siteid and evmag >=3 and evmag <= 4 and evdepth<=50:
        distidx = int(dist// DEGBUCKET)
        occ[distidx] += 1
        if isdet:
          det[distidx] += 1
    prob = det.astype(float) / occ.astype(float)
    x_bucket_pts = range(0, 180, DEGBUCKET)
    # plot the data
    plt.bar(left=x_bucket_pts, height=prob, width=DEGBUCKET, alpha=1,
            label="data 3--4 mb", color="blue")
    
    # plot the model
    mindist, maxdist = earthmodel.PhaseRange(plot_phaseid)
    x_pts = range(int(mindist), int(maxdist)+1)
    y_pts = [np.exp(netmodel.detection_logprob(1, 0, 3.5, x,
                                               options.siteid, plot_phaseid))
             for x in x_pts]
    plt.plot(x_pts, y_pts, label="model 3.5 mb", color="black", linewidth=3)
    
    plt.xlim(0,180)
    plt.ylim(0, 1)
    plt.xlabel("Distance (deg)")
    plt.ylabel("Probability")
    plt.legend(loc="upper right")
    if options.writefig is not None:
      basename = os.path.join(options.writefig,
                              "EventDetection-%s" % (phasename,))
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
        
    if options.pdf:
      options.pdf.savefig()

  # Repeat for mb 5 - 6
  DEGBUCKET = 5
  for plot_phaseid in [options.phaseid]:
    phasename = earthmodel.PhaseName(plot_phaseid)
    plt.figure(figsize=(8,4.8))
    if not options.type1:
      plt.title("Detection probability at %s for %s phase, surface event"
                % (options.sta, phasename))
    else:
      plt.title("Detection Probability - %s (ASAR)" % phasename)
    
    # now, bucket the phase data for the P phase at ASAR
    occ = np.zeros(180 // DEGBUCKET)
    det = np.zeros(180 // DEGBUCKET)
    for siteid, isdet, evmag, evdepth, dist in phase_data[plot_phaseid]:
      if siteid == options.siteid and evmag >=5 and evmag <= 6 and evdepth<=50:
        distidx = int(dist// DEGBUCKET)
        occ[distidx] += 1
        if isdet:
          det[distidx] += 1
    print occ
    prob = det / (occ + .001)
    x_bucket_pts = range(0, 180, DEGBUCKET)
    # plot the data
    plt.bar(left=x_bucket_pts, height=prob, width=DEGBUCKET, alpha=1,
            label="data 5--6 mb", color="blue")
    
    # plot the model
    mindist, maxdist = earthmodel.PhaseRange(plot_phaseid)
    x_pts = range(int(mindist), int(maxdist)+1)
    y_pts = [np.exp(netmodel.detection_logprob(1, 0, 5.5, x,
                                               options.siteid, plot_phaseid))
             for x in x_pts]
    plt.plot(x_pts, y_pts, label="model 5.5 mb", color="black", linewidth=3)
    
    plt.xlim(0,180)
    plt.ylim(0, 1)
    plt.xlabel("Distance (deg)")
    plt.ylabel("Probability")
    plt.legend(loc="upper right")
    if options.writefig is not None:
      basename = os.path.join(options.writefig,
                              "EventDetection-%s" % (phasename,))
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
        
    if options.pdf:
      options.pdf.savefig()
      

def ArrivalPhase(options, earthmodel, netmodel):
  x = np.arange(0, earthmodel.NumPhases(), .1)
  y = np.arange(0, earthmodel.NumTimeDefPhases(), .1)
  X,Y = np.meshgrid(x,y)
  
  Z = np.zeros((len(y), len(x)))
  
  for xi, x_val in enumerate(x):
    for yi, y_val in enumerate(y):
      Z[yi, xi] = np.exp(netmodel.arrphase_logprob(int(x_val), int(y_val)))
  
  x_ticks = np.arange(earthmodel.NumPhases())
  y_ticks = np.arange(earthmodel.NumTimeDefPhases())
  
  fig = plt.figure(figsize=(8,4.8))
  plt.title("Arrival Phase Distribution")
  cs = plt.contourf(X, Y, Z, np.linspace(0,1.,11))
  plt.plot([0,13.9], [0,13.9], color="black", linewidth=3, linestyle=".")
  plt.xticks(x_ticks + .5, [earthmodel.PhaseName(i) for i in x_ticks],
             rotation="vertical")
  plt.yticks(y_ticks + .5, [earthmodel.PhaseName(i) for i in y_ticks])
  plt.xlabel("Arrival Phase")
  plt.ylabel("True Phase")
  cax = fig.add_axes((.91,.19,.025,.7))
  plt.colorbar(cs, cax)
  plt.subplots_adjust(bottom=.19, right=.9)
  if options.writefig is not None:
    basename = os.path.join(options.writefig, "ArrivalPhase")
    if options.type1:
      plt.savefig(basename+".pdf")
    else:
      plt.savefig(basename+".png")
  
  if options.pdf:
    options.pdf.savefig()


  plt.figure(figsize=(8,4.8))
  plt.title("False Arrival Phase Distribution")
  plt.bar(left=x_ticks,
          height=[np.exp(netmodel.falsephase_logprob(x)) for x in x_ticks])
  plt.xticks(x_ticks + .5, [earthmodel.PhaseName(i) for i in x_ticks],
             rotation="vertical")
  plt.ylabel("Probability")
  plt.subplots_adjust(bottom=.17)
  if options.writefig is not None:
    basename = os.path.join(options.writefig, "FalseArrivalPhase")
    if options.type1:
      plt.savefig(basename+".pdf")
    else:
      plt.savefig(basename+".png")
  
  if options.pdf:
    options.pdf.savefig()

if __name__ == "__main__":
  np.seterr(divide = 'raise', invalid='raise', over='raise')
  try:
    main("parameters")
  except SystemExit:
    raise
  except:
    import pdb, traceback, sys
    traceback.print_exc(file=sys.stdout)
    pdb.post_mortem(sys.exc_traceback)
    raise
