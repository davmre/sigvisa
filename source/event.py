import numpy as np
import sigvisa.source.brune_source as brune
import sigvisa.source.mm_source as mm

from sigvisa.database.dataset import *
from sigvisa import Sigvisa

from sigvisa.utils.geog import lonlatstr
import functools32

KNOWN_EXPLOSIONS = (5393637,)  # 2009 DPRK event


class EventNotFound(Exception):
    pass


@functools32.lru_cache(1024)
def get_event(*args, **kwargs):
    return Event(*args, **kwargs)


class Event(object):

    __slots__ = ['lon', 'lat', 'depth', 'time', 'mb', 'orid', 'evid', 'natural_source']

    def __init__(self, evid=None, evtype="leb", mb=None, depth=None, lon=None, lat=None, time=None, natural_source=True, orid=None):

        if (evid is not None or orid is not None) and evtype is not None:

            try:
                ev = read_event(Sigvisa().dbconn.cursor(), evid=evid, evtype=evtype, orid=orid)
                self.lon, self.lat, self.depth, self.time, self.mb, self.orid, self.evid = ev
            except TypeError as e:
                raise EventNotFound("couldn't load evid %d" % evid)

            self.natural_source = False if evid in KNOWN_EXPLOSIONS else True

        else:
            self.lon = lon
            self.lat = lat
            self.depth = depth
            self.time = time
            self.mb = mb
            self.orid = None
            self.evid = evid
            self.natural_source = natural_source

    def as_tuple(self):
        return (self.lon, self.lat, self.depth, self.time, self.mb, self.orid, self.evid)

    def source_logamp(self, band, phase):
        if self.natural_source:
            return brune.source_logamp(event=self, band=band, phase=phase)
        else:
            return mm.source_logamp(event=self, band=band, phase=phase)

    def __str__(self):
        s = "evid %d, loc %s, depth %.1fkm, time %.1f, mb %.1f, %s source" % (self.evid, lonlatstr(
            self.lon, self.lat), self.depth, self.time, self.mb, "natural" if self.natural_source else "explosion")
        return s
