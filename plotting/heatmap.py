import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

from utils.draw_earth import draw_events, draw_earth, draw_density

import multiprocessing


def multi_f(f_args):
    """Takes a tuple (f, loni, lati, lon, lat), evaluates and returns f(lon, lat)"""
    return f_args[0](*f_args[3:5])

class Heatmap:

    def __init__(self, fn, n=20, center=None, width=None, lonbounds=None, latbounds=None, fname=None, calc=True):
        """ Arguments:

        fn: the function to plot, with signature f(lon, lat)
        n: number of points to plot in each dimension

        we require EITHER:
        lonbounds: (minlon, maxlon)
        latbounds: (minlat, maxlat)
        OR:
        center: (lon, lat)
        width: width of heat map in degrees (i.e. the
               heat map includes points of distance
               width/2 from the center in all four directions)
        """

        self.fn = fn

        self.n = n

        if fname is not None:
            self.__load(fname)
        else:

            if lonbounds is not None and latbounds is not None:
                self.min_lon = lonbounds[0]
                self.max_lon = lonbounds[1]
                self.min_lat = latbounds[0]
                self.max_lat = latbounds[1]
            elif center is not None and width is not None:
                self.min_lon = center[0]-width/2.0
                self.max_lon = center[0]+width/2.0
                self.min_lat = center[1]-width/2.0
                self.max_lat = center[1]+width/2.0
            else:
                raise RuntimeError("Heat map requires either a bounding box, or a center and a width")

            self.fnvals = np.empty((n, n))
            self.fnvals.fill(np.nan)


        self.lon_arr = np.linspace(self.min_lon, self.max_lon, self.n)
        self.lat_arr = np.linspace(self.min_lat, self.max_lat, self.n)

        self.bmap = draw_earth("",
                               projection="cyl",
                               resolution="l",
                               llcrnrlon = self.min_lon, urcrnrlon = self.max_lon,
                               llcrnrlat = self.min_lat, urcrnrlat = self.max_lat,
                               nofillcontinents=True,
                               figsize=(8,8))

    def save(self, fname):
        data_file = open(fname, 'w')
        data_file.write('%f %f %f %f %d\n' % (self.min_lon, self.max_lon,
                                              self.min_lat, self.max_lat, self.n))

        for loni in range(self.n):
            for lati in range(self.n):
                v = self.fnvals[loni, lati]
                if np.isnan(v):
                    continue
                else:
                    data_file.write('%d %d %f\n' % (loni, lati, v))
        data_file.close()

    def __load(self, fname):
        print "loading heat map values from %s" % fname

        data_file = open(fname, 'r')
        meta_info = data_file.readline()
        self.min_lon, self.max_lon, self.min_lat, self.max_lat, self.n = [float(x) for x in meta_info.split()]
        self.n = int(self.n)

        self.fnvals = np.empty((self.n, self.n))
        self.fnvals.fill(np.nan)

        for l in data_file:
            v = l.split()
            loni = int(v[0])
            lati = int(v[1])
            fval = float(v[2])
            self.fnvals[loni, lati] = fval
        data_file.close()

    def calc_parallel(self, processes=4, checkpoint=None):
        inputs = []
        for loni, lon in enumerate(self.lon_arr):
            for lati, lat in enumerate(self.lat_arr):
                inputs.append((self.fn, loni, lati, lon, lat))

        pool = multiprocessing.Pool(processes=processes)
        outputs = pool.map(multi_f, inputs)
        for io in zip(inputs, outputs):
            self.fnvals[io[0][1], io[0][2]] = io[1]

        # note at the moment this is kind of useless since it only
        # saves once finished. still including for compatibility with
        # the serial method.
        if checkpoint is not None:
            self.save(checkpoint)

    def calc(self, checkpoint=None):
        for loni, lon in enumerate(self.lon_arr):
            for lati, lat in enumerate(self.lat_arr):
                if np.isnan(self.fnvals[loni, lati]):
                    self.fnvals[loni, lati] = self.fn(lon, lat)

                    if checkpoint is not None:
                        self.save(checkpoint)

    def plot_locations(self, locations, labels=None, **plotargs):
        normed_locations = [self.normalize_lonlat(*location) for location in locations]
        draw_events(self.bmap, normed_locations, labels=labels, **plotargs)

    def plot_density(self, colorbar = True):

        if colorbar:
            minlevel = scipy.stats.scoreatpercentile(self.fnvals.flatten(), 20)
            levels = np.linspace(minlevel, np.max(self.fnvals), 10)
        else:
            levels = None

        draw_density(self.bmap, self.lon_arr, self.lat_arr, self.fnvals,
                     levels = levels, colorbar=colorbar)


    def normalize_lonlat(self, lon, lat):
        """
        Return the given location represented within the coordinate
        scheme of the current heatmap (e.g. [-180, 180] vs [0, 360]).
        If the location isn't part of the current heatmap,
        its representation is undefined.
        """

        while lon < self.min_lon:
            lon += 360
        while lon > self.max_lon:
            lon -= 360
        while lat < self.min_lat:
            lon += 180
        while lat > self.max_lat:
            lon -= 180

        return (lon, lat)


    def max(self):
        maxlon = 0
        maxlat = 0
        maxval = np.float("-inf")
        for loni, lon in enumerate(self.lon_arr):
            for lati, lat in enumerate(self.lat_arr):
                if self.fnvals[loni, lati] > maxval:
                    maxval = self.fnvals[loni, lati]
                    maxlon = lon
                    maxlat = lat
        return (maxlon, maxlat, maxval)


    def min(self):
        minlon = 0
        minlat = 0
        minval = np.float("-inf")
        for loni, lon in enumerate(self.lon_arr):
            for lati, lat in enumerate(self.lat_arr):
                if self.fnvals[loni, lati] < minval:
                    minval = self.fnvals[loni, lati]
                    minlon = lon
                    minlat = lat

        return (minlon, minlat, minval)


    def __mul__(self, other):

        if self.min_lon != other.min_lon or self.min_lat != other.min_lat or self.max_lon != other.max_lon or self.max_lat != other.max_lat or self.n != other.n:
            raise Exception("cannot multiply heatmaps with different gridpoints!")

        newf = lambda (lon, lat): self.fn(lon, lat) * other.fn(lon, lat)
        new_vals = self.fnvals * other.fnvals

        hm = Heatmap(fn=newf, n=self.n, lonbounds=[self.min_lon, self.max_lon],
                     latbounds=[self.min_lat, self.max_lat])
        hm.fnvals = new_vals
        return hm

    def __add__(self, other):

        if self.min_lon != other.min_lon or self.min_lat != other.min_lat or self.max_lon != other.max_lon or self.max_lat != other.max_lat or self.n != other.n:
            raise Exception("cannot add heatmaps with different gridpoints!")

        newf = lambda (lon, lat): self.fn(lon, lat) + other.fn(lon, lat)
        new_vals = self.fnvals + other.fnvals

        hm = Heatmap(fn=newf, n=self.n, lonbounds=[self.min_lon, self.max_lon],
                     latbounds=[self.min_lat, self.max_lat])
        hm.fnvals = new_vals
        return hm

def testfn(lon, lat):
    return lon*lat

def main():

    h = Heatmap(fn = testfn, n=20, lonbounds=[-180, 180], latbounds=[-90, 90])
    h.calc(checkpoint = "test.heatmap")

    h2 = Heatmap(fn = testfn, n=20, lonbounds=[-180, 180], latbounds=[-90, 90])
    h2.calc_parallel(checkpoint = "test2.heatmap")

    hnew = Heatmap(fn =f, fname = "test.heatmap")
    hnew.plot_density()

    plt.savefig("heatmap.png")

if __name__ == "__main__":
    main()
