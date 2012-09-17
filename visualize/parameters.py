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
# visualize the model parameters

from optparse import OptionParser
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os, csv

from database import db
from database.dataset import *
from learn import load_earth
from priors import ArrivalAmplitudePrior

def main(param_dirname):
  parser = OptionParser()
  parser.add_option("-1", "--type1", dest="type1", default=False,
                    action = "store_true",
                    help = "Type 1 fonts (False)")
  parser.add_option("-w", "--writefig", dest="writefig",
                    default=None, help = "Directory to save figures (None)",
                    metavar="DIR")
  parser.add_option("-p", "--pdf", dest="pdf",
                    default=None, help = "pdf file to save figures (None)",
                    metavar="FILE")
  parser.add_option("-m", "--model", dest="model",
                    default=None, help = "Which model(s) to visualize (all)")
  parser.add_option("-x", "--textonly", dest="textonly", default=False,
                    action = "store_true",
                    help = "Text only output (False)")
  (options, args) = parser.parse_args()
  
  # use Type 1 fonts by invoking latex
  if options.type1:
    plt.rcParams['text.usetex'] = True
    
  if options.pdf:
    options.pdf = PdfPages(options.pdf)

  if options.model is not None:
    options.model = set(m for m in options.model.split(","))  

  cursor = db.connect().cursor()
  sites = read_sites(cursor)
  phasenames, phasetimedef = read_phases(cursor)
  earthmodel = load_earth(param_dirname, sites, phasenames, phasetimedef)

  if options.model is None or "EventDetection" in options.model:
    EventDetection(param_dirname, options)

  if options.model is None or "ArrivalTime" in options.model:
    Arrival(param_dirname, options, earthmodel, "ArrivalTime", "seconds")
    
  if options.model is None or "ArrivalSlowness" in options.model:
    Arrival(param_dirname, options, earthmodel, "ArrivalSlowness", "s/deg")
    
  if options.model is None or "ArrivalAzimuth" in options.model:
    Arrival(param_dirname, options, earthmodel, "ArrivalAzimuth", "degrees")

  if options.model is None or "ArrivalAmplitude" in options.model:
    ArrivalAmplitude(param_dirname, options, phasenames)

  if options.model is None or "NumFalseDet" in options.model:
    NumFalseDet(param_dirname, options)
  
  if options.pdf:
    options.pdf.close()
  
  if not options.textonly:
    plt.show()

def Arrival(param_dirname, options, earthmodel, name, unitname):
  fp = open(os.path.join(param_dirname, "%sPrior.txt" % name))
  numsites, numphases  = read_ints(fp)

  phase_loc = [[] for _ in xrange(numphases)]
  phase_scale = [[] for _ in xrange(numphases)]
  for siteid in xrange(numsites):
    for phaseid in xrange(numphases):
      loc, scale = read_floats(fp)
      phase_loc[phaseid].append(loc)
      phase_scale[phaseid].append(scale)
      
  fp.close()

  for phaseid in xrange(earthmodel.NumTimeDefPhases()):
    plt.figure(figsize=(8,4.8))
    phasename = earthmodel.PhaseName(phaseid)
    if options.type1:
      plt.title(phasename + " - mean")
    else:
      plt.title(name + " " + phasename + " - mean")
    
    plt.hist(phase_loc[phaseid])
    plt.xlabel(unitname)
    
    if options.writefig is not None:
      basename = os.path.join(options.writefig,
                              "%s-loc-%s" % (name, phasename))
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
        
    if options.pdf:
      options.pdf.savefig()
    
    plt.figure(figsize=(8,4.8))
    if options.type1:
      plt.title(phasename + " - scale")
    else:
      plt.title(name + " " + phasename + " - scale")
    
    plt.hist(phase_scale[phaseid])
    plt.xlabel(unitname)
    
    if options.writefig is not None:
      basename = os.path.join(options.writefig,
                              "%s-scale-%s" % (name, phasename))
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
        
    if options.pdf:
      options.pdf.savefig()
    

def EventDetection(param_dirname, options):
  NUMPARAM = 15
  # 3 x 5 subplots for all 15 parameters per phase
  fp = open(os.path.join(param_dirname, "EventDetectionPrior.txt"))
  numphases, numsites = fp.readline().split()
  numphases, numsites = int(numphases), int(numsites)
  
  param_names = fp.readline().rstrip().split(",")[2:]
  assert(len(param_names) == NUMPARAM)
  
  for phase in xrange(numphases):
    phasename = None
    param_vals = [[] for _ in xrange(NUMPARAM)]
    for site in xrange(numsites):
      line = fp.readline().rstrip().split(",")
      if phasename is None:
        phasename = line[0]
      assert(site == int(line[1]))

      for idx, val in enumerate(line[2:]):
        param_vals[idx].append(float(val))
    
    plt.figure(figsize=(16,9.6))
    plt.subplots_adjust(hspace=.7)
    plt.suptitle(phasename)
    for param in xrange(NUMPARAM):
      plt.subplot(5, 3, param+1)
      plt.hist(param_vals[param])
      plt.xlabel(param_names[param])

    if options.writefig is not None:
      basename = os.path.join(options.writefig,
                              "EventDetectionParam-%s" % phasename)
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    
    if options.pdf:
      options.pdf.savefig()
    
  fp.close()

def ArrivalAmplitude(param_dirname, options, phasenames):
  # read all the parameters
  cfp = csv.reader(open(os.path.join(param_dirname,
                                     "ArrivalAmplitudePrior.txt")),
                   delimiter=" ", quoting=csv.QUOTE_NONNUMERIC)
  numsites, numphases = [int(x) for x in cfp.next()]

  false_par = np.ndarray((numsites, 6))
  true_par = np.ndarray((numphases, numsites,
                         len(ArrivalAmplitudePrior.FEATURE_NAMES)+1))

  for siteid in xrange(numsites):
    false_par[siteid] = cfp.next()
    for phaseid in xrange(numphases):
      true_par[phaseid, siteid] = cfp.next()
  del cfp

  # first show all the false arrival parameters in a single 2 x 3 plot
  plt.figure(figsize=(8,4.8))
  plt.subplots_adjust(hspace=.4)
  plt.suptitle("False Arrival Amplitude")
  for idx, name in enumerate(["wt0", "wt1", "mean0", "mean1", "std0", "std1"]):
    plt.subplot(3, 2, idx+1)
    plt.hist(false_par[:,idx])
    plt.xlabel(name)

  if options.writefig is not None:
    basename = os.path.join(options.writefig, "FalseArrivalAmplitudeParam")
    if options.type1:
      plt.savefig(basename+".pdf")
    else:
      plt.savefig(basename+".png")
  
  if options.pdf:
    options.pdf.savefig()
  
  
  for phaseid in xrange(numphases):
    plt.figure(figsize=(16,9.6))
    #plt.subplots_adjust(hspace=.35)
    plt.suptitle("Arrival Amplitude - %s" % phasenames[phaseid])
    numfeats = len(ArrivalAmplitudePrior.FEATURE_NAMES)
    for numsubplots in xrange(100):
      if numsubplots ** 2 >= numfeats:
        break
    else:
      assert False                      # too many features
    
    for idx, name in enumerate(ArrivalAmplitudePrior.FEATURE_NAMES):
      plt.subplot(numsubplots, numsubplots, idx+1)
      plt.hist(true_par[phaseid,:,idx])
      plt.xlabel(name)

    if options.writefig is not None:
      basename = os.path.join(options.writefig,
                              "ArrivalAmplitudeParam-%s" % phasenames[phaseid])
      if options.type1:
        plt.savefig(basename+".pdf")
      else:
        plt.savefig(basename+".png")
    
    if options.pdf:
      options.pdf.savefig()
      
def NumFalseDet(param_dirname, options):
  # read all the parameters
  cfp = csv.reader(open(os.path.join(param_dirname, "NumFalseDetPrior.txt")),
                   delimiter=" ", quoting=csv.QUOTE_NONNUMERIC)
  numsites = int(cfp.next()[0])

  falserates = np.ndarray(numsites)

  for siteid in xrange(numsites):
    falserates[siteid] = cfp.next()[0]
  del cfp

  plt.figure(figsize=(8,4.8))
  plt.suptitle("Number of False Arrivals")
  plt.bar(range(numsites), falserates * 3600, facecolor="blue")
  plt.xlabel("station index")
  plt.ylabel("rate (per hour)")
  
  if options.writefig is not None:
    basename = os.path.join(options.writefig, "NumFalseDetParam")
    if options.type1:
      plt.savefig(basename+".pdf")
    else:
      plt.savefig(basename+".png")
  
  if options.pdf:
    options.pdf.savefig()

def read_floats(fp):
  return [float(x) for x in fp.readline().rstrip().split()]

def read_ints(fp):
  return [int(x) for x in fp.readline().rstrip().split()]

if __name__ == "__main__":
  main("parameters")
