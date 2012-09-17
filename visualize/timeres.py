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
# visualize the time residuals
import csv, gzip, sys
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def main():
  if len(sys.argv) != 5:
    print "Usage: python timeres.py <origin.csv.gz> <assoc.csv.gz> <sta>"\
          " <phase>"
    sys.exit(1)

  _, orig_fname, assoc_fname, filt_sta, filt_phase = sys.argv
  
  # draw a map
  plt.figure(figsize = (16, 9.6), facecolor='w')
  plt.title("%s phase at %s" % (filt_phase, filt_sta))
  bmap = Basemap(projection="moll", resolution="l", lat_0=0, lon_0=0)
  bmap.drawmapboundary(fill_color=(.7,.7,1,1), zorder=1)
  bmap.fillcontinents(color=(0.5,.7,0.5,1), lake_color=(.6,.6,1,1),
                      zorder=2)
  bmap.drawcoastlines(zorder=3)
  plt.subplots_adjust(left=0.02, right=0.98)
  
  # read the origins
  origins = {}
  data = csv.reader(gzip.open(orig_fname))
  # the first row is the column names
  colnames = data.next()
  
  for row in data:
    event = dict(zip(colnames, row))
    
    lon, lat = float(event['LON']), float(event['LAT'])
    orid = int(event['ORID'])
    
    origins[orid] = (lon, lat)

  # read the associations
  data = csv.reader(gzip.open(assoc_fname))
  # the first row is the column names
  colnames = data.next()
  
  x_arr, y_arr, s_arr, c_arr = [], [], [], []
  for row in data:
    assoc = dict(zip(colnames, row))
    if assoc['PHASE'] != filt_phase or assoc['STA'] != filt_sta:
      continue
    timeres = float(assoc['TIMERES'])
    orid = int(assoc['ORID'])
    
    lon, lat = origins[orid]
    
    x, y = bmap(lon, lat)
    x_arr.append(x)
    y_arr.append(y)
    
    c = timeres
    
    s = 5
    
    s_arr.append(s)
    c_arr.append(c)
    
    #if len(x_arr) > 40000:
    #  break
  
  print "Processed %d phase associations" % len(x_arr)
  
  bmap.scatter(x_arr, y_arr, s_arr, c_arr, cmap=cm.jet,
               edgecolors='none', zorder=5)
  
  plt.show()
  return
  
if __name__ == "__main__":
  main()
  
