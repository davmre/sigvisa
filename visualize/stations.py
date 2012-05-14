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

# visualize the location of all the stations

import numpy as np
import matplotlib.pyplot as plt

import database.db
from database.dataset import *
from utils.draw_earth import draw_earth, draw_events

def main():
  # read the timerange of the dataset
  cursor = database.db.connect().cursor()
  start_time, end_time = read_timerange(cursor, "training", None, 0)
  # read the locations of the stations
  cursor.execute("select lon, lat from static_siteid order by id")
  sitelocs = np.array(cursor.fetchall())
  # read the uptimes of the stations
  siteup = read_uptime(cursor, start_time, end_time)
  siteup_scale = 8 * siteup.sum(axis=1) / siteup.shape[1]
  # draw the stations
  bmap = draw_earth("Seismic Stations (size = % uptime)")
  draw_events(bmap, sitelocs, marker="o", mfc="red", mew=0,
              ms = np.where(siteup_scale < 1, 1, siteup_scale))
  plt.show()
  
  
if __name__ == "__main__":
  main()
  
