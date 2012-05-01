# Copyright (c) 2012, Bayesian Logic, Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Bayesian Logic, Inc. nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# Bayesian Logic, Inc. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# 
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
  stime, etime = read_timerange(cursor, "training", None, 0)
  print "Reading detections...",
  detections = read_detections(cursor, stime, etime)[0]
  print "done (%d)" % len(detections)
  print "Reading associated arrival ids...",
  assoc_arids = set()
  cursor.execute("select lass.arid from leb_assoc lass, "
                 "static_phaseid ph, idcx_arrival iarr where "
                 "ph.timedef='d' and ascii(lass.phase) = ascii(ph.phase) and "
                 "iarr.arid=lass.arid and iarr.time between %f and %f"
                 % (stime, etime))
  for arid, in cursor.fetchall():
    assoc_arids.add(int(arid))
  print "done (%d)" % len(assoc_arids)
  
  LOW_LOGAMP, HIGH_LOGAMP, STEP_LOGAMP = -4, 10, .25
  logamp_det, logamp_tot = np.zeros((HIGH_LOGAMP - LOW_LOGAMP)/STEP_LOGAMP),\
                           np.zeros((HIGH_LOGAMP - LOW_LOGAMP)/STEP_LOGAMP)

  HIGH_SNR, STEP_SNR = 200, 1
  snr_det, snr_tot = np.zeros(HIGH_SNR/STEP_SNR), np.zeros(HIGH_SNR/STEP_SNR)

  az_res = []
  slo_res = []

  logamp_res = []
  snr_res = []

  time_res = []

  sec_phase = np.zeros(18)

  logamp_time_corr = ([], [])

  discarded_amp, discarded_snr = 0, 0

  for detnum, det in enumerate(detections):
    if int(det[DET_ARID_COL]) in assoc_arids:
      continue
    
    if (np.log(det[DET_AMP_COL]) < HIGH_LOGAMP and
        np.log(det[DET_AMP_COL]) >= LOW_LOGAMP):
      logamp_tot[(np.log(det[DET_AMP_COL])-LOW_LOGAMP) // STEP_LOGAMP]+= 1
    else:
      discarded_amp += 1
      
    if det[DET_SNR_COL] < HIGH_SNR:
      snr_tot[ det[DET_SNR_COL] // STEP_SNR ] += 1
    else:
      discarded_snr += 1
    
    for secdetnum in xrange(detnum+1, len(detections)):
      secdet = detections[secdetnum]
      if secdet[DET_TIME_COL] > (det[DET_TIME_COL] + MAX_SECDET_GAP):
        break
      if det[DET_SITE_COL] != secdet[DET_SITE_COL]:
        continue
      # an associated arrival can't be a secondary
      if int(secdet[DET_ARID_COL]) in assoc_arids:
        continue
      
      # we have identified a secondary now measure its attributes
      # w.r.t. the primary
      if (np.log(det[DET_AMP_COL]) < HIGH_LOGAMP and
          np.log(det[DET_AMP_COL]) >= LOW_LOGAMP):
        logamp_det[(np.log(det[DET_AMP_COL])-LOW_LOGAMP) // STEP_LOGAMP]+= 1
        logamp_time_corr[0].append(np.log(det[DET_AMP_COL]))
        logamp_time_corr[1].append(secdet[DET_TIME_COL] - det[DET_TIME_COL])
          
      if det[DET_SNR_COL] < HIGH_SNR:
        snr_det[ det[DET_SNR_COL] // STEP_SNR ] += 1

      time_res.append(secdet[DET_TIME_COL] - det[DET_TIME_COL])
      az_res.append(degdiff(secdet[DET_AZI_COL], det[DET_AZI_COL]))
      slo_res.append(secdet[DET_SLO_COL] - det[DET_SLO_COL])
      
      logamp_res.append(np.log(secdet[DET_AMP_COL]) - np.log(det[DET_AMP_COL]))
      snr_res.append(secdet[DET_SNR_COL] - det[DET_SNR_COL])

      sec_phase[int(secdet[DET_PHASE_COL])] += 1

      break

  print "%d discarded due to amp, %d discarded due to SNR" % (discarded_amp,
                                                              discarded_snr)
  logamp_det += .000001
  logamp_tot += .001

  snr_det += .000001
  snr_tot += .001
  
  pp = PdfPages(os.path.join("output", "check_sec.pdf"))
  
  plt.figure()
  plt.title("Secondary detection probability vs primary detection amp")
  plt.bar(np.arange(LOW_LOGAMP, HIGH_LOGAMP, STEP_LOGAMP),
          logamp_det / logamp_tot, STEP_LOGAMP)
  plt.xlabel("log(amp)")
  plt.ylabel("probability")
  pp.savefig()
  
  plt.figure()
  plt.title("Secondary detection probability vs primary detection SNR")
  plt.bar(np.arange(0, HIGH_SNR, STEP_SNR), snr_det / snr_tot, STEP_SNR)
  plt.xlabel("SNR")
  plt.ylabel("probability")
  pp.savefig()
  
  plt.figure()
  plt.title("Azimuth diff")
  plt.hist(az_res, 100, normed=True)
  pp.savefig()

  plt.figure()
  plt.title("Slowness diff")
  plt.hist(slo_res, 100, normed=True)
  pp.savefig()
  
  plt.figure()
  plt.title("Logamp diff")
  plt.hist(logamp_res, 100, normed=True)
  pp.savefig()
  
  plt.figure()
  plt.title("SNR diff")
  plt.hist(snr_res, np.arange(-20, 20, .1), normed=True)
  pp.savefig()

  plt.figure()
  plt.title("Time diff")
  plt.hist(time_res, 100, normed=True)
  plt.xlabel("Time (s)")
  pp.savefig()

  plt.figure()
  plt.title("Time diff vs Primary LogAmp (log color scale)")
  plt.hexbin(logamp_time_corr[0], logamp_time_corr[1], bins="log")
  plt.xlabel("Log Amplitude")
  plt.ylabel("Time (s)")
  cb = plt.colorbar()
  cb.set_label('log10(N)')
  
  pp.savefig()
  
  plt.figure()
  plt.title("Secondary phase")
  plt.bar(np.arange(0, 18), sec_phase / sec_phase.sum(), 1)
  plt.ylabel("Probability")
  plt.xlabel("Phase index")
  pp.savefig()
  
  pp.close()
  plt.show()

if __name__ == "__main__":
  main()
  
