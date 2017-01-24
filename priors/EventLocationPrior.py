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
import os
import random
import numpy as np
from numpy import exp, log, pi, degrees, radians, arange, arcsin, zeros, sin
import time

from sigvisa.database.dataset import *
from sigvisa.utils.geog import dist_deg


from optparse import OptionParser



def kernel(bandwidth, distance):
    # convert bandwidth and distance to radians
    b, d = radians(bandwidth), radians(distance)

    return (1.0 + 1.0 / b ** 2)  / (2 * pi * AVG_EARTH_RADIUS_KM ** 2) \
        * exp(-d / b) / (1.0 + exp(-pi / b))


def compute_density(bandwidth, events, lon_arr, lat_arr, lon_grid, lat_grid):
    density = zeros((len(lon_arr), len(lat_arr)), float)
    for ev in events:
        density += kernel(bandwidth,
                          dist_deg((lon_grid, lat_grid),
                                   ev[[EV_LON_COL, EV_LAT_COL]]))
    density /= len(events)

    return density


def leave_oneout_avg_loglike(bandwidth, events, 
                             lon_arr, lat_arr,
                             lon_grid, lat_grid,
                             region_left,
                             bottom_range,
                             uniform_prob, earth_area,
                             lon_interval,
                             z_interval):
    print "computing leave one out bw %f" % bandwidth,
    t1 = time.time()
    n = float(len(events))
    density = compute_density(bandwidth, events, lon_arr, lat_arr, lon_grid, lat_grid)
    tot = 0.
    for ev in events:
        evdens = kernel(bandwidth, dist_deg((lon_grid, lat_grid),
                                            ev[[EV_LON_COL, EV_LAT_COL]]))
        netdensity = (n / (n - 1)) * density - (1 / (n - 1)) * evdens
        netdensity = (1 - uniform_prob) * netdensity + uniform_prob / earth_area

        loni = int((ev[EV_LON_COL] -region_left) // lon_interval)
        lati = int((sin(radians(ev[EV_LAT_COL])) -bottom_range) // z_interval)
        tot += log(netdensity[loni, lati])

    ans = tot / len(events)

    print "done (in %.1f secs) -> %f" % (time.time() - t1, ans)
    return ans


def learn(param_fname, options, leb_events, 
          sample_events=1000,
          uniform_prob=0.001,
          lon_interval=0.1,
          region_left=-180.,
          region_right=180.,
          region_top=90.,
          region_bottom=-90.):
    ########
    # first we will learn the optimal bandwidth (in degrees)
    ########

    # sample_events specifies how many events to use to compute the
    # optimal bandwidth

    # use half as many latitude buckets as longitude buckets


    lon_arr = arange(region_left, region_right, lon_interval)

    bottom_range = sin(radians(region_bottom))
    top_range = sin(radians(region_top))

    lon_coverage = (region_right - region_left)/360.
    lat_coverage = float(top_range - bottom_range) / 2.0

    n_lon_buckets = len(lon_arr)
    n_lat_buckets = int(n_lon_buckets * lat_coverage / (2* lon_coverage ))
    z_interval =  2.0 * lat_coverage / n_lat_buckets
    lat_arr = degrees(arcsin(arange(bottom_range, top_range, z_interval)))
    lat_grid, lon_grid = np.meshgrid(lat_arr, lon_arr)

    EARTH_AREA = 4 * pi * AVG_EARTH_RADIUS_KM ** 2
    solid_angle_sr = (top_range - bottom_range) * (radians(region_right) - radians(region_left))
    region_area = solid_angle_sr * EARTH_AREA / (4.*pi)
    patch_area = region_area / (len(lon_arr) * len(lat_arr))

    # to do so we will pick a random sample of 500 events
    indices = [i for i in range(len(leb_events))
               if random.random() < (float(sample_events) / len(leb_events))]

    # then we'll use leave-one-out cross-validation to select the best
    # bandwidth
    all_bandw, all_loglike = np.array([]), np.array([])
    for bandw in np.arange(0, 2, .2) + .2:
        all_bandw = np.append(all_bandw, bandw)
        all_loglike = np.append(all_loglike,
                                leave_oneout_avg_loglike(bandw,
                                                         leb_events[indices],
                                                         lon_arr=lon_arr,
                                                         lat_arr=lat_arr,
                                                         lon_grid=lon_grid, 
                                                         lat_grid=lat_grid,
                                                         region_left=region_left,
                                                         bottom_range=bottom_range,
                                                         uniform_prob=uniform_prob,
                                                         earth_area=region_area,
                                                         lon_interval=lon_interval,
                                                         z_interval=z_interval))

    best_bandw = all_bandw[all_loglike.argmax()]

    for bandw in np.arange(best_bandw - .2, best_bandw + .2, .05):
        if bandw == 0:
            continue
        if bandw in all_bandw:
            continue
        all_bandw = np.append(all_bandw, bandw)
        all_loglike = np.append(all_loglike,
                                leave_oneout_avg_loglike(bandw,
                                                         leb_events[indices],
                                                         lon_arr=lon_arr,
                                                         lat_arr=lat_arr,
                                                         lon_grid=lon_grid, 
                                                         lat_grid=lat_grid,
                                                         region_left=region_left,
                                                         bottom_range=bottom_range,
                                                         uniform_prob=uniform_prob,
                                                         earth_area=region_area,
                                                         lon_interval=lon_interval,
                                                         z_interval=z_interval))


    best_bandw = all_bandw[all_loglike.argmax()]

    print "best bandwidth", best_bandw

    """
    if options.gui:
        plt.figure(figsize=(8, 4.8))
        if not options.type1:
            plt.title("Leave-One-Out Avg. Log Likelihood")
        plt.scatter(all_bandw, all_loglike)
        plt.xlabel("bandwidth (degrees)")
        plt.ylabel("avg. log likelihood")
        plt.xlim(0, 2.)

        if options.writefig is not None:
            basename = os.path.join(options.writefig, "EventLocBandwidth")
            if options.type1:
                plt.savefig(basename + ".pdf")
            else:
                plt.savefig(basename + ".png")
    """

    density = compute_density(best_bandw, leb_events, 
                              lon_arr=lon_arr, lat_arr=lat_arr,
                              lon_grid=lon_grid, 
                              lat_grid=lat_grid)
    # fold-in a uniform prior
    density = (1 - uniform_prob) * density + uniform_prob / region_area


    if options.gui:
        title = "Event Location Log Density (b=%.2f)" % best_bandw

        f = Figure(figsize=(15, 15))
        hm = EventHeatmap(f=None, left_lon=region_left, right_lon=region_right, top_lat=region_top, bottom_lat=region_bottom, calc=False)
        ax = f.add_subplot(111)
        hm.init_bmap(axes=ax, nofillcontinents=True, projection="cyl")
        hm.plot_earth()
        
        lons = lon_arr
        lats = lat_arr

        #loni, lati = np.mgrid[0:len(lons), 0:len(lats)]
        #lon_arr, lat_arr = lons[loni], lats[lati]
        # convert to map coordinates
        x, y = hm.bmap(list(lon_arr.flat), list(lat_arr.flat))
        x_arr = np.array(x).reshape(lon_arr.shape)
        y_arr = np.array(y).reshape(lat_arr.shape)

        cs1 = hm.bmap.contour(x_arr, y_arr, vals, levels, linewidths=.5, colors="k",
                           zorder=6 - int(nolines))
        cs2 = hm.bmap.contourf(x_arr, y_arr, vals, levels, cmap=plt.cm.jet, zorder=5,
                               extend="both",
                               norm=matplotlib.colors.BoundaryNorm(cs1.levels,
                                                                   plt.cm.jet.N))


        plt.colorbar(cs2, orientation="vertical", drawedges=True)


        draw_density(bmap, lon_arr, lat_arr, log(density),
                     colorbar_orientation="horizontal", colorbar_shrink=0.75)
        if options.writefig is not None:
            basename = os.path.join(options.writefig, "EventLocationPrior")
            # a .pdf file would be too huge
            plt.savefig(basename + ".png")


    # convert the density at the grid points into a probability for each
    # bucket (the grid point is in the lower-left corner of the bucket)
    prob = density * patch_area

    print "Total prob:", prob.sum()

    fp = open(param_fname, "w")

    print >>fp, uniform_prob
    print >>fp, region_left, region_right, bottom_range, top_range
    print >>fp, lon_interval, z_interval

    np.savetxt(fp, prob)                 # writes out row by row
    fp.close()


class Dummy:
    pass


def main():


    parser = OptionParser()
    parser.add_option("--outfile", dest="outfile", default=os.path.join("parameters", "EventLocationPrior.txt"), type="str", help="")
    parser.add_option("--llnl_hack", dest="llnl_hack", default=False, action="store_true", help="")

    (options, args) = parser.parse_args()



    cursor = db.connect().cursor()
    param_fname = options.outfile
    if options.llnl_hack:
        print "reading events...",
        st1 = 1167609600
        et1 = 1199145600
        events1 = read_events(cursor, st1, et1, "isc")[0]

        st2 = 1203603300
        et2 = 1203624962
        events2 = read_events(cursor, st2, et2, "isc")[0]
        
        leb_events = np.vstack((events1, events2))


        region_left=-126.
        region_right=-100.
        region_bottom=32.
        region_top=49.
        uniform_prob=0.01
    else:
        start_time, end_time = read_timerange(cursor, "training", None, 0)
        print "reading events...",
        leb_events = read_events(cursor, start_time, end_time, "leb")[0]

        region_left=-180.
        region_right=180
        region_bottom=-90.
        region_top=90.
        uniform_prob=0.001

    print "done (%d events)" % len(leb_events)
    options = Dummy()

    options.gui = False
    options.type1 = False
    options.writefig = None

    learn(param_fname, options, leb_events, 
          uniform_prob=uniform_prob,
          region_left=region_left,
          region_right=region_right,
          region_bottom=region_bottom,
          region_top=region_top)


if __name__ == "__main__":
    main()
