import matplotlib.pyplot as plt

from plotting.heatmap import Heatmap
from utils.geog import dist_km, lonlatstr
from database.dataset import *
from database import db

class EventHeatmap(Heatmap):

    def __init__(self, f, sitenames=None, **args):
        Heatmap.__init__(self, f=f, **args)

        if sitenames is None:
            cursor = db.connect().cursor()
            stations, name_to_siteid_minus1, siteid_minus1_to_name = read_sites_by_name(cursor)
            self.sitenames = stations
        else:
            self.sitenames = self.sitenames

        self.event_locations = []
        self.event_labels = []
        self.stations = []
        self.true_event = None

    def add_events(self, locations, labels=None):

        if labels is not None:
            if len(labels) != len(locations):
                raise Exception("wrong number of labels given for these locations!")
        else:
            labels = [None for loc in locations]

        self.event_locations.extend([(l[0], l[1]) for l in  locations])
        self.event_labels.extend(labels)

    def add_stations(self, names):
        if isinstance(names, str):
            names = [names,]
        self.stations.extend(names)

    def set_true_event(self, lon, lat):
        self.true_event = (lon, lat)
        return dist_km((lon, lat), self.max()[0:2])

    def savefig(self, fname, title = None, **args):
        plt.figure()
        self.plot(**args)
        if title is None:
            title = self.title()
        plt.title(title)
        plt.savefig(fname)
        plt.close()

        self.save(fname + ".log")


    def plot(self, colorbar=True, event_alpha=0.6):

        self.plot_earth()
        self.plot_density(colorbar=colorbar)

        self.plot_locations(self.event_locations, labels=self.event_labels,
                            marker=".", ms=12, mfc="none", mec="red", mew=2, alpha=event_alpha)

        if self.true_event is not None:
            (lon, lat) = self.true_event
            self.plot_locations(((lon, lat),), labels=None,
                                marker="*", ms=26, mfc="none", mec="#44FF44", mew=2, alpha=1)


        sta_locations = [self.sitenames[n][0:2] for n in self.stations]
        self.plot_locations(sta_locations, labels=self.stations,
                            marker="x", ms=7, mfc="none", mec="white", mew=2, alpha=1)


    def title(self):
        peak = self.max()[0:2]
        title = "Peak: " + utils.geog.lonlatstr(*peak)
        if self.true_event is not None:
            title += "\nTrue: " + utils.geog.lonlatstr(*self.true_event)
            title += "\nDistance: %.2f km" % dist_km(self.true_event, peak)

        return title

    def __aggregate(self, other, hm):
        hm.sitenames = self.sitenames

        my_ev = zip(self.event_locations, self.event_labels)
        other_ev = zip(other.event_locations, other.event_labels)
        all_ev = list(set(my_ev + other_ev))
        hm.event_locations, hm.event_labels = zip(*all_ev)

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
