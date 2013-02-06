import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

from utils.draw_earth import draw_events, draw_earth, draw_density
import utils.geog

import multiprocessing
import copy


def multi_f(f_args):
    """Takes a tuple (f, loni, lati, lon, lat), evaluates and returns f(lon, lat)"""
    return f_args[0](*f_args[3:5])

class Heatmap(object):

    def __init__(self, f, n=20, center=None, width=None, lonbounds=None, latbounds=None, autobounds=None, fname=None, calc=True):
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

        self.f = f

        self.n = n

        self.fname = fname
        try:
            self.load()
        except:
            if autobounds is not None:
                lonbounds, latbounds, _ = self.event_bounds(autobounds)
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

            if self.min_lat < -90:
                self.min_lat = -90

            if self.max_lat > 90:
                self.max_lat = 90

            self.fvals = np.empty((n, n))
            self.fvals.fill(np.nan)

        self.lon_arr = np.linspace(self.min_lon, self.max_lon, self.n)
        self.lat_arr = np.linspace(self.min_lat, self.max_lat, self.n)

        if calc:
            self.calc(checkpoint = self.fname)

    def save(self, fname=None):
        if fname is None:
            fname = self.fname

        data_file = open(fname, 'w')
        data_file.write('%f %f %f %f %d\n' % (self.min_lon, self.max_lon,
                                              self.min_lat, self.max_lat, self.n))

        for loni in range(self.n):
            for lati in range(self.n):
                v = self.fvals[loni, lati]
                if np.isnan(v):
                    continue
                else:
                    data_file.write('%d %d %f\n' % (loni, lati, v))
        data_file.close()

    def load(self):
        fname = self.fname
        print "loading heat map values from %s" % fname

        data_file = open(fname, 'r')
        meta_info = data_file.readline()
        self.min_lon, self.max_lon, self.min_lat, self.max_lat, self.n = [float(x) for x in meta_info.split()]
        self.n = int(self.n)

        self.fvals = np.empty((self.n, self.n))
        self.fvals.fill(np.nan)

        for l in data_file:
            v = l.split()
            loni = int(v[0])
            lati = int(v[1])
            fval = float(v[2])
            self.fvals[loni, lati] = fval
        data_file.close()

    def calc_parallel(self, processes=4, checkpoint=None):
        inputs = []
        for loni, lon in enumerate(self.lon_arr):
            for lati, lat in enumerate(self.lat_arr):
                inputs.append((self.f, loni, lati, lon, lat))

        pool = multiprocessing.Pool(processes=processes)
        outputs = pool.map(multi_f, inputs)
        for io in zip(inputs, outputs):
            self.fvals[io[0][1], io[0][2]] = io[1]

        # note at the moment this is kind of useless since it only
        # saves once finished. still including for compatibility with
        # the serial method.
        if checkpoint is not None:
            self.save(checkpoint)

    def calc(self, checkpoint=None):
        for loni, lon in enumerate(self.lon_arr):
            for lati, lat in enumerate(self.lat_arr):
                if np.isnan(self.fvals[loni, lati]):
                    self.fvals[loni, lati] = self.f(lon, lat)

                    if checkpoint is not None:
                        self.save(checkpoint)
                    print "computed (%.2f, %.2f) = %.6f" % (lon, lat, self.fvals[loni, lati])

    def init_bmap(self, axes=None):
        self.bmap = draw_earth("",
                               projection="cyl",
                               resolution="l",
                               llcrnrlon = self.min_lon, urcrnrlon = self.max_lon,
                               llcrnrlat = self.min_lat, urcrnrlat = self.max_lat,
                               nofillcontinents=True,
                               ax=axes,
                           )


    def plot_earth(self):
        try:
            bmap = self.bmap
        except:
            self.init_bmap()

        parallels = np.arange(int(self.min_lat)-1,int(self.max_lat+1))
        if len(parallels) > 10:
            parallels = [int(k) for k in np.linspace(int(self.min_lat)-1,int(self.max_lat+1), 10)]

        self.bmap.drawparallels(parallels,labels=[False,True,True,False])
        meridians = np.arange(int(self.min_lon)-1, int(self.max_lon)+1)
        if len(meridians) > 10:
            meridians = [int(k) for k in np.linspace(int(self.min_lon)-1,int(self.max_lon+1), 10)]

        self.bmap.drawmeridians(meridians,labels=[True,False,False,True])

    def plot_locations(self, locations, labels=None, **plotargs):
        try:
            bmap = self.bmap
        except:
            self.init_bmap()

        normed_locations = [self.normalize_lonlat(*location) for location in locations]
        draw_events(self.bmap, normed_locations, labels=labels, **plotargs)

    def plot_density(self, f_preprocess=None, **density_args):
        try:
            bmap = self.bmap
        except:
            self.init_bmap()


        fv = copy.copy(self.fvals)
        if f_preprocess:
            for i in range(fv.shape[0]):
                for j in range(fv.shape[1]):
                    v = self.fvals[i,j]
                    for fp in f_preprocess:
                        v = fp(v, self.fvals.flatten())
                    fv[i,j] = v


        minlevel = scipy.stats.scoreatpercentile([v for v in fv.flatten() if not np.isnan(v)], 20)

        levels = np.linspace(minlevel, np.max(fv), 10)


        draw_density(self.bmap, self.lon_arr, self.lat_arr, fv,
                     levels = levels, **density_args)

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
            lat += 180
        while lat > self.max_lat:
            lat -= 180

        return (lon, lat)

    def event_bounds(self, X, quantile=0.9):

        X = X[:, 0:2]

        def find_center(X):

            rX = np.radians(X) + np.array((0, np.pi/2.0))

            pts_cartesian = [(np.sin(lat)*np.cos(lon), np.sin(lat)*np.sin(lon), np.cos(lat)) for (lon, lat) in rX]
            center_cartesian = np.mean(pts_cartesian, axis=0)
            (x,y,z) = center_cartesian
            lat_center = np.degrees(np.arccos(z) - np.pi/2.0)
            lon_center = np.degrees(np.arctan2(y,x))

            return (lon_center, lat_center)


        center = find_center(X)
        lon_distances = sorted(np.abs([utils.geog.degdiff(pt[0], center[0]) for pt in X]))
        lat_distances = sorted(np.abs([utils.geog.degdiff(pt[1], center[1]) for pt in X]))
        lon_width = lon_distances[int(np.ceil(len(lon_distances) * float(quantile)))]
        lat_width = lat_distances[int(np.ceil(len(lat_distances) * float(quantile)))]


        min_lon = center[0]-lon_width
        max_lon = center[0]+lon_width
        min_lat = center[1]-lat_width
        max_lat = center[1]+lat_width

        return [min_lon, max_lon], [min_lat, max_lat], center


    def max(self):
        maxlon = 0
        maxlat = 0
        maxval = np.float("-inf")
        for loni, lon in enumerate(self.lon_arr):
            for lati, lat in enumerate(self.lat_arr):
                if self.fvals[loni, lati] > maxval:
                    maxval = self.fvals[loni, lati]
                    maxlon = lon
                    maxlat = lat
        return (maxlon, maxlat, maxval)


    def min(self):
        minlon = 0
        minlat = 0
        minval = np.float("-inf")
        for loni, lon in enumerate(self.lon_arr):
            for lati, lat in enumerate(self.lat_arr):
                if self.fvals[loni, lati] < minval:
                    minval = self.fvals[loni, lati]
                    minlon = lon
                    minlat = lat

        return (minlon, minlat, minval)


    def __mul__(self, other):
        if other is None:
            # Treat "None" as the identity heatmap, so that we can use
            # it as the initialization for iterative updating.
            return self

        if self.min_lon != other.min_lon or self.min_lat != other.min_lat or self.max_lon != other.max_lon or self.max_lat != other.max_lat or self.n != other.n:
            raise Exception("cannot multiply heatmaps with different gridpoints!")

        newf = lambda lon, lat: self.f(lon, lat) * other.f(lon, lat)
        new_vals = self.fvals * other.fvals

        hm = type(self)(f=newf, n=self.n, lonbounds=[self.min_lon, self.max_lon],
                     latbounds=[self.min_lat, self.max_lat], calc=False)
        hm.fvals = new_vals
        return hm

    def __add__(self, other):
        if other is None:
            # Treat "None" as the identity heatmap, so that we can use
            # it as the initialization for iterative updating.
            return self

        if self.min_lon != other.min_lon or self.min_lat != other.min_lat or self.max_lon != other.max_lon or self.max_lat != other.max_lat or self.n != other.n:
            raise Exception("cannot add heatmaps with different gridpoints!")

        newf = lambda lon, lat: self.f(lon, lat) + other.f(lon, lat)
        new_vals = self.fvals + other.fvals

        hm = type(self)(f=newf, n=self.n, lonbounds=[self.min_lon, self.max_lon],
                     latbounds=[self.min_lat, self.max_lat], calc = False)
        hm.fvals = new_vals
        return hm


def testfn(lon, lat):
    return lon*lat

def main():

    h = Heatmap(f = testfn, n=20, lonbounds=[-180, 180], latbounds=[-90, 90])
    h.calc(checkpoint = "test.heatmap")

    h2 = Heatmap(f = testfn, n=20, lonbounds=[-180, 180], latbounds=[-90, 90])
    h2.calc_parallel(checkpoint = "test2.heatmap")

    hnew = Heatmap(f =f, fname = "test.heatmap")
    hnew.plot_density()

    plt.savefig("heatmap.png")

if __name__ == "__main__":
    main()
