import numpy as np
import source.brune_source as brune
import source.mm_source as mm

from database.dataset import *
from sigvisa import Sigvisa

from utils.geog import lonlatstr

class Event(object):


    __slots__ = ['lon', 'lat', 'depth', 'time', 'mb', 'evid', 'natural_source']


    def __new__(cls, *args, **kwargs):

        e = None
        if "evid" in kwargs and "evtype" in kwargs:
            evid = kwargs["evid"]
            evtype = kwargs["evtype"]
            try:
                e = Sigvisa().events[(evid,evtype)]
            except KeyError:
                e = super(Event, cls).__new__(cls, *args, **kwargs)
                Sigvisa().events[(evid,evtype)] = e
        if e is None:
            e = super(Event, cls).__new__(cls, *args, **kwargs)
        return e

    def __init__(self, evid=None, evtype="leb", mb=None, depth=None, lon=None, lat=None, time=None, natural_source=True):

        if evid is not None and evtype is not None:

            self.lon, self.lat, self.depth, self.time, self.mb, _, self.evid = \
                read_event(Sigvisa().cursor, evid, evtype)

            self.natural_source = True

        else:
            self.lon=lon
            self.lat=lat
            self.depth=depth
            self.time=time
            self.mb=mb
            self.evid=evid
            self.natural_source=natural_source

    def source_logamp(self, band, phase):
        if natural_source:
            return brune.source_logamp(mb=self.mb, band=band, phase=phase)
        else:
            return mm.source_logamp(event=self, band=band, phase=phase)

    def __str__(self):
        s = "evid %d, loc %s, depth %.1fkm, time %.1f, mb %.1f, %s source" % (self.evid, lonlatstr(self.lon, self.lat), self.depth, self.time, self.mb, "natural" if self.natural_source else "explosion")
        return s

