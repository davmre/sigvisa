import numpy as np
import scipy
import scipy.stats

import sigvisa.utils.geog as geog
import multiprocessing
import copy
from mpl_toolkits.basemap import Basemap
import matplotlib

def multi_f(f_args):
    """Takes a tuple (f, loni, lati, lon, lat), evaluates and returns f(lon, lat)"""
    return f_args[0](*f_args[3:5])


class Heatmap(object):

    def _init_coord_bounds(self, center=None, width_deg=None, height_deg=None, width=None, left_lon=None, right_lon=None, top_lat=None, bottom_lat=None):

        # all latitudes will be in the range [center_lat - 180,
        # center_lat + 180], so we first need to compute the
        # center.
        if left_lon is not None and right_lon is not None and top_lat is not None and bottom_lat is not None:
            while right_lon < left_lon:
                right_lon += 360

            width_deg = (right_lon - left_lon)
            height_deg = (top_lat - bottom_lat)
            center = (left_lon + width_deg/2.0, bottom_lat + height_deg/2.0 )

        if center is None or width_deg is None or height_deg is None:
            raise RuntimeError("Heat map requires either a bounding box, or a center and a width/height")

        center = list(center)
        while center[0] > 180:
            center[0] -= 360
        while center[0] < -180:
            center[0] += 360
        center = tuple(center)

        self.left_lon = center[0] - width_deg/2.0
        self.right_lon = center[0] + width_deg/2.0
        self.bottom_lat = center[1] - height_deg/2.0
        self.top_lat = center[1] + height_deg/2.0

        assert ( -180 < self.left_lon < 180 )
        assert ( 0 < self.right_lon - self.left_lon < 360 )


        # poles are complicated; let's never try to plot them
        if self.top_lat > 90:
            self.top_lat = 90
        if self.bottom_lat < -90:
            self.bottom_lat = -90
        assert( 90 >= self.top_lat > self.bottom_lat >= -90)

        self.center = center
        self.lon_arr = np.linspace(self.left_lon, self.right_lon, self.n)
        self.lat_arr = np.linspace(self.bottom_lat, self.top_lat, self.n)


    def __init__(self, f, n=20, center=None, width=None, width_deg=None, height_deg=None, left_lon=None, right_lon=None, top_lat=None, bottom_lat=None, autobounds=None, fname=None, calc=True):
        """ Arguments:

        fn: the function to plot, with signature f(lon, lat)
        n: number of points to plot in each dimension

        we require EITHER:
        a bounding box: left/right lon and top/bottom lat
        OR:
        center: (lon, lat) and
        width: either width_deg and height_deg, or width (shortcut for both of those)
        """

        self.f = f
        self.n = n
        self.fname = fname

        # for backwards compatibility, accept "width" as a shortcut for both width_deg and height_deg
        if width is not None:
            if width_deg is None and height_deg is None:
                width_deg = width
                height_deg = width
            else:
                raise RuntimeError("can't specify both width and width_deg/height_deg")

        try:
            self.load()
        except:
            if autobounds is not None:
                left_lon, right_lon, bottom_lat, top_lat = self.event_bounds(autobounds)
            self._init_coord_bounds(center=center, width_deg=width_deg, height_deg=height_deg,
                                    left_lon=left_lon, right_lon=right_lon,
                                    bottom_lat=bottom_lat, top_lat=top_lat)
            self.fvals = np.empty((n, n))
            self.fvals.fill(np.nan)

        if calc:
            self.calc(checkpoint=self.fname)

    def save(self, fname=None):
        if fname is None:
            fname = self.fname

        data_file = open(fname, 'w')
        data_file.write('%f %f %f %f %d\n' % (self.left_lon, self.right_lon,
                                              self.bottom_lat, self.top_lat, self.n))
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
        left_lon, right_lon, bottom_lat, top_lat, n = [float(x) for x in meta_info.split()]
        self.n = int(n)

        self._init_coord_bounds(left_lon=left_lon, right_lon=right_lon,
                                bottom_lat=bottom_lat, top_lat=top_lat)

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
                    lon, lat = self.standardize_lonlat(lon, lat)
                    self.fvals[loni, lati] = self.f(lon, lat)

                    if checkpoint is not None:
                        self.save(checkpoint)
                    print "computed (%.2f, %.2f) = %.6f" % (lon, lat, self.fvals[loni, lati])

    def coord_list(self):
        cl = []
        for loni, lon in enumerate(self.lon_arr):
            for lati, lat in enumerate(self.lat_arr):
                lon, lat = self.standardize_lonlat(lon, lat)
                cl.append((lon, lat))
        return cl

    def set_coord_fvals(self, fvals):
        """

        Set fvals, given a list of values representing the function
        being evaluated at each point in self.coord_list().

        """
        i = 0
        for loni, lon in enumerate(self.lon_arr):
            for lati, lat in enumerate(self.lat_arr):
                self.fvals[loni, lati] = fvals[i]
                i += 1

    def init_bmap(self, nofillcontinents=True, resolution="l", projection="cyl", axes=None, **kwargs):
        bmap = Basemap(resolution = resolution, projection = projection, ax=axes,
                       llcrnrlon=self.left_lon, llcrnrlat=self.bottom_lat,
                       urcrnrlon=self.right_lon, urcrnrlat=self.top_lat,
                       **kwargs)
        bmap.drawmapboundary(fill_color=(.7, .7, 1, 1))
        bmap.drawcoastlines(zorder=10)

        if not nofillcontinents:
            # fill the continents with a greenish color
            bmap.fillcontinents(color=(0.5, .7, 0.5, 1), lake_color=(.7, .7, 1, 1),
                                zorder=1)
            # fill the oceans with a bluish color
            bmap.drawmapboundary(fill_color=(.7, .7, 1))

        self.bmap = bmap

    def plot_earth(self):
        try:
            bmap = self.bmap
        except:
            self.init_bmap()

        parallels = np.arange(int(self.bottom_lat) - 1, int(self.top_lat + 1))
        if len(parallels) > 10:
            parallels = [int(k) for k in np.linspace(int(self.bottom_lat) - 1, int(self.top_lat + 1), 10)]

        self.bmap.drawparallels(parallels, labels=[False, True, True, False])
        meridians = np.arange(int(self.left_lon) - 1, int(self.right_lon) + 1)
        if len(meridians) > 10:
            meridians = [int(k) for k in np.linspace(int(self.left_lon) - 1, int(self.right_lon + 1), 10)]

        self.bmap.drawmeridians(meridians, labels=[True, False, False, True])

    def plot_locations(self, locations, labels=None, zorder = 10, offmap_arrows=False, **plotargs):
        try:
            bmap = self.bmap
        except:
            self.init_bmap()

        normed_locations = [self.normalize_lonlat(*location) for location in locations]

        for enum, ev in enumerate(normed_locations):
            x, y = bmap(ev[0], ev[1])
            bmap.plot([x], [y], zorder=zorder, **plotargs)

            if offmap_arrows:
                edge_pt, edge_arrow = self.project_to_bounds(ev[0], ev[1])
                x, y = edge_pt
                if edge_arrow is not None:
                    base_scale = (self.right_lon - self.left_lon) / 200.0
                    edge_arrow *= base_scale * 10
                    bmap.ax.arrow( edge_pt[0] - edge_arrow[0], edge_pt[1] - edge_arrow[1],
                                   edge_arrow[0], edge_arrow[1], fc="white", ec="white",
                                   length_includes_head=True, overhang = .6,
                                   head_starts_at_zero=False, width=0.01,
                                   head_width= 3 * base_scale, head_length = 3 * base_scale, zorder=zorder)

            if labels is not None and labels[enum] is not None:
                axes = bmap.ax
                xbounds = bmap(self.right_lon, self.top_lat)
                x_off = 6 if x < self.right_lon else -30
                y_off = 6 if y < self.top_lat else -30
                axes.annotate(
                    labels[enum],
                    xy=(x, y),
                    xytext=(x_off, y_off),
                    textcoords='offset points',
                    size=8,
                    color = 'white',
                    zorder=zorder,
                    arrowprops = None)



    def plot_density(self, f_preprocess=None, colorbar=True, nolines=False,
                     colorbar_orientation="vertical", colorbar_shrink=0.9, smooth=False):
        try:
            bmap = self.bmap
        except:
            self.init_bmap()

        fv = copy.copy(self.fvals)
        if f_preprocess:
            for i in range(fv.shape[0]):
                for j in range(fv.shape[1]):
                    v = self.fvals[i, j]
                    for fp in f_preprocess:
                        v = fp(v, self.fvals.flatten())
                    fv[i, j] = v

        # fvals are indexed by [loni, lat], which means that longitude
        # is the row index (i.e. the vertical axis) and latitude is
        # the column index. which is backwards from how we want to
        # plot it, hence we need to transpose.
        fv = fv.T

        minlevel = scipy.stats.scoreatpercentile([v for v in fv.flatten() if not np.isnan(v)], 20)
        levels = np.linspace(minlevel, np.max(fv), 10)

        lon_arr, lat_arr, x_arr, y_arr = bmap.makegrid(nx = self.n, ny = self.n, returnxy=True)


        cm = matplotlib.cm.get_cmap('jet')
        if not smooth:
            cs1 = bmap.contour(x_arr, y_arr, fv, levels, linewidths=.5, colors="k",
                               zorder=6 - int(nolines))
            norm = matplotlib.colors.BoundaryNorm(cs1.levels, cm.N)
            cs2 = bmap.contourf(x_arr, y_arr, fv, levels, cmap=cm, zorder=5,
                                extend="both",
                                norm=norm)
        else:
            norm = matplotlib.colors.Normalize()
            cs2 = bmap.pcolormesh(x_arr, y_arr, fv, cmap=cm, zorder=5, norm=norm, shading='gouraud')

        if colorbar:
            from mpl_toolkits.axes_grid import make_axes_locatable
            import matplotlib.axes as maxes

            divider = make_axes_locatable(bmap.ax)
            cax = divider.new_horizontal("4%", pad=.5, axes_class=maxes.Axes)
            bmap.ax.figure.add_axes(cax)

            bmap.ax.figure.colorbar(cs2, orientation=colorbar_orientation, drawedges=not nolines,
                                    cax=cax, format='%.1f')


    def normalize_lonlat(self, lon, lat):
        """
        Return the given location represented within the coordinate
        scheme of the current heatmap (e.g. [-180, 180] vs [0, 360]).
        If the location isn't part of the current heatmap,
        its representation is undefined.
        """

        while lon < self.center[0] - 180:
            lon += 360
        while lon > self.center[0] + 180:
            lon -= 360
        assert ( -90 < lat < 90 )

        return (lon, lat)

    def standardize_lonlat(self, lon, lat):
        """

        Convert from map coordinates (longitudes ranging from
        self.center - width/2 to self.center + width/2) to standard
        coordinates (longitudes from -180 to 180).

        """

        while lon < -180:
            lon += 360
        while lon > 180:
            lon -= 360
        assert ( -90 < lat < 90 )
        return (lon, lat)

    def event_bounds(self, X, quantile=0.9):

        X = X[:, 0:2]

        def find_center(X):
            # Find the center of a set of points on the Earth's
            # surface, by first finding their center of mass, then
            # projecting onto the surface of the Earth.

            rX = np.radians(X) + np.array((0, np.pi / 2.0))

            pts_cartesian = [(np.sin(lat) * np.cos(lon), np.sin(lat) * np.sin(lon), np.cos(lat)) for (lon, lat) in rX]
            center_cartesian = np.mean(pts_cartesian, axis=0)
            (x, y, z) = center_cartesian
            lat_center = np.degrees(np.arccos(z) - np.pi / 2.0)
            lon_center = np.degrees(np.arctan2(y, x))

            return (lon_center, lat_center)

        center = find_center(X)
        lon_distances = sorted(np.abs([geog.degdiff(pt[0], center[0]) for pt in X]))
        lat_distances = sorted(np.abs([geog.degdiff(pt[1], center[1]) for pt in X]))
        lon_width = lon_distances[int(np.ceil(len(lon_distances) * float(quantile)))] * 2
        lat_width = lat_distances[int(np.ceil(len(lat_distances) * float(quantile)))] * 2

        left_lon = center[0] - lon_width/2.0
        right_lon = center[0] + lon_width/2.0
        bottom_lat = center[1] - lat_width/2.0
        top_lat = center[1] + lat_width/2.0

        return left_lon, right_lon, bottom_lat, top_lat

    def max(self):
        maxlon = 0
        maxlat = 0
        maxval = np.float("-inf")
        for loni, lon in enumerate(self.lon_arr):
            for lati, lat in enumerate(self.lat_arr):
                lon, lat = self.standardize_lonlat(lon, lat)
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
                lon, lat = self.standardize_lonlat(lon, lat)
                if self.fvals[loni, lati] < minval:
                    minval = self.fvals[loni, lati]
                    minlon = lon
                    minlat = lat

        return (minlon, minlat, minval)

    def round_point_to_grid(self, lon, lat):
        best_lon = 0
        best_lat = 0
        best_dist = np.float('inf')
        for loni, glon in enumerate(self.lon_arr):
            for lati, glat in enumerate(self.lat_arr):
                lon, lat = self.standardize_lonlat(lon, lat)
                dist = geog.dist_km((glon, glat), (lon, lat))
                if dist < best_dist:
                    best_dist = dist
                    best_lon = glon
                    best_lat = glat
        return best_lon, best_lat

    def project_to_bounds(self, lon, lat):

        # if the point is already within bounds, nothing to do
        if lon > self.left_lon and lon < self.right_lon and lat > self.bottom_lat and lat < self.top_lat:
            return np.array(self.bmap(lon, lat)), None

        # compute the great circle path to the point, and figure out where it leaves the bounds
        x_arr, y_arr = self.bmap.gcpoints(lon, lat, self.center[0], self.center[1], 100)
        left_border_x, top_border_y = self.bmap(self.left_lon, self.top_lat)
        right_border_x, bottom_border_y = self.bmap(self.right_lon, self.bottom_lat)
        pt = np.array(self.center)
        x_interp, y_interp = None, None
        for i, (x, y) in enumerate(zip(x_arr, y_arr)):
            last_pt = pt
            pt = np.array((x,y))
            v = pt - last_pt

            if x < left_border_x:
                x_interp = (left_border_x - last_pt[0]) / (x - last_pt[0] )
            elif x > right_border_x:
                x_interp = (right_border_x - last_pt[0]) / (x - last_pt[0] )
            if y < bottom_border_y:
                y_interp = (bottom_border_y - last_pt[1]) / (y - last_pt[1] )
            elif y < top_border_y:
                y_interp = (top_border_y - last_pt[1]) / ( y - last_pt[1] )

            if x_interp and y_interp:
                interp = min(x_interp, y_interp)
                break
            elif x_interp:
                interp = x_interp
                break
            elif y_interp:
                interp = y_interp
                break

        border_pt = last_pt + interp * v
        print "center", self.center, "border", border_pt, "v", v
        return border_pt, v / np.linalg.norm(v, 2)

    def __mul__(self, other):
        if other is None:
            # Treat "None" as the identity heatmap, so that we can use
            # it as the initialization for iterative updating.
            return self

        if self.left_lon != other.left_lon or self.bottom_lat != other.bottom_lat or self.right_lon != other.right_lon or self.top_lat != other.top_lat or self.n != other.n:
            raise Exception("cannot multiply heatmaps with different gridpoints!")

        newf = lambda lon, lat: self.f(lon, lat) * other.f(lon, lat)
        new_vals = self.fvals * other.fvals

        hm = type(self)(f=newf, n=self.n, left_lon=self.left_lon, right_lon=self.right_lon,
                        bottom_lat=self.bottom_lat, top_lat=self.top_lat, calc=False)
        hm.fvals = new_vals
        return hm

    def __add__(self, other):
        if other is None:
            # Treat "None" as the identity heatmap, so that we can use
            # it as the initialization for iterative updating.
            return self

        if self.left_lon != other.left_lon or self.bottom_lat != other.bottom_lat or self.right_lon != other.right_lon or self.top_lat != other.top_lat or self.n != other.n:
            raise Exception("cannot add heatmaps with different gridpoints!")

        newf = lambda lon, lat: self.f(lon, lat) + other.f(lon, lat)
        new_vals = self.fvals + other.fvals

        hm = type(self)(f=newf, n=self.n, left_lon=self.left_lon, right_lon=self.right_lon,
                        bottom_lat=self.bottom_lat, top_lat=self.top_lat, calc=False)
        hm.fvals = new_vals
        return hm

def main():
    pass
if __name__ == "__main__":
    main()
