# checks the properties of the secondary detections
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from database.dataset import *
import database.db
from utils.geog import degdiff

MAX_SECDET_GAP = 30

def main():
  # read all the detections and identify secondaries using a simple 10 second
  # rule
  cursor = database.db.connect().cursor()
  print "Reading detections...",
  stime, etime = read_timerange(cursor, "training", None, 0)
  detections = read_detections(cursor, stime, etime)[0]
  print "done (%d)" % len(detections)

  HIGH_AMP, STEP_AMP = 300, 5
  HIGH_SNR, STEP_SNR = 100, 1
  
  all_amp = np.zeros(HIGH_AMP / STEP_AMP)
  all_snr = np.zeros(HIGH_SNR / STEP_SNR) + .0001

  primary_amp = np.zeros(HIGH_AMP / STEP_AMP)
  primary_snr = np.zeros(HIGH_SNR / STEP_SNR)

  az_res = []
  slo_res = []

  logamp_res = []
  logsnr_res = []

  time_res = []

  sec_phase = np.zeros(18)

  amp_time_corr = ([], [])

  discarded_amp, discarded_snr = 0, 0
  
  for detnum, det in enumerate(detections):
    if det[DET_AMP_COL] < HIGH_AMP:
      all_amp[ det[DET_AMP_COL] // STEP_AMP ] += 1
    else:
      discarded_amp += 1
      
    if det[DET_SNR_COL] < HIGH_SNR:
      all_snr[ det[DET_SNR_COL] // STEP_SNR ] += 1
    else:
      discarded_snr += 1
    
    for secdetnum in xrange(detnum+1, len(detections)):
      secdet = detections[secdetnum]
      if secdet[DET_TIME_COL] > (det[DET_TIME_COL] + MAX_SECDET_GAP):
        break
      if det[DET_SITE_COL] != secdet[DET_SITE_COL]:
        continue
      # we have identified a secondary now measure its attributes
      # w.r.t. the primary
      if det[DET_AMP_COL] < HIGH_AMP:
        primary_amp[ det[DET_AMP_COL] // STEP_AMP ] += 1
        amp_time_corr[0].append(det[DET_AMP_COL])
        amp_time_corr[1].append(secdet[DET_TIME_COL] - det[DET_TIME_COL])
        
      if det[DET_SNR_COL] < HIGH_SNR:
        primary_snr[ det[DET_SNR_COL] // STEP_SNR ] += 1
      
      time_res.append(secdet[DET_TIME_COL] - det[DET_TIME_COL])
      az_res.append(degdiff(secdet[DET_AZI_COL], det[DET_AZI_COL]))
      slo_res.append(secdet[DET_SLO_COL] - det[DET_SLO_COL])
      
      logamp_res.append(np.log(secdet[DET_AMP_COL]) - np.log(det[DET_AMP_COL]))
      logsnr_res.append(np.log(secdet[DET_SNR_COL]) - np.log(det[DET_SNR_COL]))

      sec_phase[int(secdet[DET_PHASE_COL])] += 1

  print "%d discarded due to amp, %d discarded due to SNR" % (discarded_amp,
                                                              discarded_snr)
  pp = PdfPages(os.path.join("output", "check_sec.pdf"))
  
  plt.figure()
  plt.title("Secondary detection probability vs primary amp")
  plt.bar(np.arange(0, HIGH_AMP, STEP_AMP), primary_amp / all_amp, STEP_AMP)
  plt.ylabel("Probability")
  plt.xlabel("Amplitude")
  pp.savefig()
  
  plt.figure()
  plt.title("Secondary detection probability vs primary SNR")
  plt.bar(np.arange(0, HIGH_SNR, STEP_SNR), primary_snr / all_snr, STEP_SNR)
  plt.ylabel("Probability")
  plt.xlabel("SNR")
  pp.savefig()
  
  plt.figure()
  plt.title("Azimuth diff")
  plt.hist(az_res, 100)
  pp.savefig()

  plt.figure()
  plt.title("Slowness diff")
  plt.hist(slo_res, 100)
  pp.savefig()
  
  plt.figure()
  plt.title("Logamp diff")
  plt.hist(logamp_res, 100)
  pp.savefig()
  
  plt.figure()
  plt.title("Log SNR diff")
  plt.hist(logsnr_res, 100)
  pp.savefig()

  plt.figure()
  plt.title("Time diff")
  plt.hist(time_res, 100)
  plt.xlabel("Time (s)")
  pp.savefig()

  plt.figure()
  plt.title("Time diff vs Primary Amp (log color scale)")
  plt.hexbin(amp_time_corr[0], amp_time_corr[1], bins="log")
  plt.xlabel("Amplitude")
  plt.ylabel("Time (s)")
  cb = plt.colorbar()
  cb.set_label('log10(N)')
  
  pp.savefig()
  
  plt.figure()
  plt.title("Secondary phase")
  plt.bar(np.arange(0, 18), sec_phase / sec_phase.sum(), 1)
  plt.ylabel("Probability")
  plt.xlabel("Phase index")
  pp.close()
  plt.show()

if __name__ == "__main__":
  main()
  
