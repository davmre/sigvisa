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
from database.dataset import *
import numpy as np
import os

def learn(param_fname, options, start_time, end_time, leb_events):
  rate = float(len(leb_events)) / (end_time - start_time)
  print ("event rate is %f per second or %.1f per hour"
         % (rate, rate * 60 * 60))

  if options.gui:
    hrly_rate = rate * 60 * 60
    num_hrs = int((end_time - start_time) / (60 * 60))
    bins = np.arange(0, num_hrs+1)
    options.plt.figure(figsize=(8,4.8))
    if not options.type1:
      options.plt.title("Event Rate per hour")
    options.plt.hist((leb_events[:, EV_TIME_COL] - start_time) / (60*60),
                     bins, label="data", alpha=1.0, edgecolor="none",
                     facecolor="blue")
    options.plt.plot([0, num_hrs],
                     [hrly_rate + np.sqrt(hrly_rate),
                      hrly_rate + np.sqrt(hrly_rate)],
                     label = "model +std", linewidth=3, linestyle=":",
                     color="black")
    options.plt.plot([0, num_hrs], [hrly_rate, hrly_rate],
                     label = "model", linewidth=3, linestyle="-", color="black")
    options.plt.plot([0, num_hrs],
                     [hrly_rate - np.sqrt(hrly_rate),
                      hrly_rate - np.sqrt(hrly_rate)],
                     label = "model -std", linewidth=3,
                     linestyle="-.", color="black")
    options.plt.xlabel("Hour index")
    options.plt.ylabel("Frequency")
    options.plt.legend(loc="upper left")
    options.plt.xlim(0, num_hrs)
    
    if options.writefig is not None:
      basename = os.path.join(options.writefig, "NumEventPrior")
      if options.type1:
        options.plt.savefig(basename+".pdf")
      else:
        options.plt.savefig(basename+".png")
  
  fp = open(param_fname, "w")
  print >>fp, rate
  fp.close()

