import matplotlib.pyplot as plt

from sigvisa.plotting.heatmap import Heatmap
from sigvisa.utils.geog import dist_km, lonlatstr
from sigvisa.database.dataset import *
from sigvisa.database import db
from sigvisa import Sigvisa


class EventHeatmap(Heatmap):

    def __init__(self, f, **args):
        Heatmap.__init__(self, f=f, **args)

        self.event_locations = []
        self.event_labels = []
        self.event_yvals = []
        self.stations = []
        self.true_event = None

        self.ref_events = []
        self.ref_covs = []

    def add_events(self, locations=None, labels=None, yvals=None, evs=None):

        if locations is None:
            locations = np.array([(ev.lon, ev.lat) for ev in evs])

        if labels is not None:
            if len(labels) != len(locations):
                raise Exception("wrong number of labels given for these locations!")
        else:
            labels = [None for loc in locations]

        self.event_locations.extend([(l[0], l[1]) for l in locations])
        if yvals is not None:
            self.event_yvals.extend(yvals)
        if labels is not None:
            self.event_labels.extend(labels)

    def add_reference_events(self, locations=None, evs=None, covs=None):
        if locations is None:
            locations = np.array([(ev.lon, ev.lat) for ev in evs])

        self.ref_events.extend(locations)
        self.ref_covs.extend(covs)


    def add_stations(self, names):
        if isinstance(names, str):
            names = [names, ]
        self.stations.extend(names)

    def set_true_event(self, lon, lat):
        self.true_event = (lon, lat)

    def savefig(self, fname, title=None, **args):
        plt.figure()
        self.plot(**args)
        if title is None:
            title = self.title()
        plt.title(title)
        plt.savefig(fname)
        plt.close()

        self.save(fname + ".log")

    def plot(self, event_alpha=0.6, cov_alpha=0.1, axes=None, offmap_station_arrows=True, label_stations=True, nofillcontinents=True, meridians=True, projection="cyl", drawlines=True, colorseed=0, **density_args):

        self.init_bmap(axes=axes, nofillcontinents=nofillcontinents, projection=projection)

        if meridians:
            self.plot_earth()

        s = Sigvisa()

        if self.f is not None or not np.isnan(self.fvals).all():
            self.plot_density(**density_args)

        if self.ref_events:
            np.random.seed(colorseed)
            colors = [np.random.rand(3) for i in range(len(self.ref_events))]
            self.plot_locations(self.ref_events,
                                marker="*", ms=6, mec="none", mew=0,
                                alpha=1, colors=colors)
            if self.ref_covs:
                self.plot_covs(self.ref_events, self.ref_covs, alpha=cov_alpha, colors=colors)

            if drawlines:
                for color, l1, l2 in zip(colors, self.ref_events, self.event_locations):
                    self.drawline(l1, l2, color=color)

        if self.event_yvals:
            self.plot_locations(self.event_locations,
                                marker=".", s=6, facecolors="none",
                                yvals=self.event_yvals, alpha=event_alpha)
        else:
            evcolors=None
            if self.ref_events and len(self.ref_events)==len(self.event_locations):
                evcolors = colors

            self.plot_locations(self.event_locations, labels=self.event_labels,
                                marker=".", ms=6, mec="none", mew=0,
                                alpha=event_alpha, colors=evcolors)




        if self.true_event is not None:
            (lon, lat) = self.true_event
            self.plot_locations(((lon, lat),), labels=None,
                                marker="*", ms=16, mfc="none", mec="#44FF44", mew=2, alpha=1)

        sta_locations = [s.earthmodel.site_info(n, 0)[0:2] for n in self.stations]
        self.plot_locations(sta_locations, labels=self.stations if label_stations else None,
                            marker="^", ms=4, mfc="none", mec="blue", mew=1, alpha=1,
                            offmap_arrows=offmap_station_arrows)



    def title(self):
        peak = self.max()[0:2]
        title = "Peak: " + lonlatstr(*peak)
        if self.true_event is not None:
            title += "\nTrue: " + lonlatstr(*self.true_event)
            title += "\nDistance: %.2f km" % dist_km(self.true_event, peak)

        return title

    def __aggregate(self, other, hm):
        #hm.sitenames = self.sitenames

        try:
            my_ev = zip(self.event_locations, self.event_labels)
            other_ev = zip(other.event_locations, other.event_labels)
            all_ev = list(set(my_ev + other_ev))
            hm.event_locations, hm.event_labels = zip(*all_ev)
        except Exception as e:
            print "could not combine event locations:", e

        hm.stations = list(set(self.stations + other.stations))

        if self.true_event is None:
            hm.true_event = other.true_event
        elif other.true_event is not None and self.true_event != other.true_event:
            raise Exception("tried to combine heatmaps with different true events")
        else:
            hm.true_event = self.true_event

        return hm

    def __mul__(self, other):
        hm = Heatmap.__mul__(self, other)
        return self.__aggregate(other, hm)

    def __add__(self, other):
        hm = Heatmap.__add__(self, other)
        return self.__aggregate(other, hm)
