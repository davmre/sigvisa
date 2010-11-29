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
  
