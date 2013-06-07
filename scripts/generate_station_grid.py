import sys
import numpy as np
import itertools

def wraparound_lat(l):
    if l > 90:
        return 180-l
    if l < -90:
        return -180-l
    else:
        return l

def wraparound_lon(l):
    if l < -180:
        return 180 - (-180-l)
    elif l > 180:
        return -180 + (l - 180)
    else:
        return l

basegrid_lons = np.linspace(-180, 180, 361)
basegrid_lats = np.linspace(-90, 90, 181)
depths = [0,35,50,100,200,410,500,660]

outfile = open('tt_grid', 'w')
with open('sites.csv', 'r') as f:
    for line in f:
        sta = [float(s) for s in line.split(',')]
        rounded_sta_lat, rounded_sta_lon = np.round(sta[:2])


        within20_lons = [wraparound_lon(l) for l in np.linspace(rounded_sta_lon-19.5, rounded_sta_lon+19.5, 40)]
        #within5_lons = [wraparound_lon(l) for l in np.linspace(rounded_sta_lon-4.75, rounded_sta_lon+4.75, 20)]
        within20_lats = [wraparound_lat(l) for l in np.linspace(rounded_sta_lat-19.5, rounded_sta_lat+19.5, 40)]
        #within5_lats = [wraparound_lat(l) for l in np.linspace(rounded_sta_lat-4.75, rounded_sta_lat+4.75, 20)]

        lons = np.sort(np.concatenate([basegrid_lons, within20_lons]))
        lats = np.sort(np.concatenate([basegrid_lats, within20_lats]))
        pts = list(itertools.product(lats, lons, depths))

        for pt in pts:
            outfile.write('%f %f %f %f %f %f\n' % (pt[0], pt[1], pt[2], sta[0], sta[1], sta[2]))

outfile.close()
