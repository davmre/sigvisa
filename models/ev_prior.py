import numpy as np

from sigvisa import Sigvisa
from sigvisa.graph.nodes import VectorNode
from sigvisa.models import Distribution
from sigvisa.source.event import Event


EV_LON, EV_LAT, EV_DEPTH, EV_TIME, EV_MB, EV_NATURAL_SOURCE, EV_FLAG, EV_VECTOR_LEN = range(6+1+1)
EV_FLAG_VAL = -999

class EventPriorModel(Distribution):

    def log_p(self, x, cond=None):
        s = Sigvisa()
        if x[EV_NATURAL_SOURCE]:
            source_lp = 0 # eventually I should have a real model for event source type
            loc_lp = s.sigmodel.event_location_prior_logprob(x[EV_LON], x[EV_LAT], x[EV_DEPTH])
        else:
            raise NotImplementedError("prior for artificial-source events is not currently implemented")

        mb_lp = s.sigmodel.event_mag_prior_logprob(x[EV_MB])

        return loc_lp + mb_lp + source_lp


class EventNode(VectorNode):

    def __init__(self, event, fixed_values = True, **kwargs):

        dimension = 6
        self._id = event.id
        self.evid = event.evid
        super(EventNode, self).__init__(model = EventPriorModel(), dimension = dimension, initial_value=self.ev_to_vector(event), fixed_values=fixed_values, **kwargs)

    def ev_to_vector(self, ev):
        v = np.array((ev.lon, ev.lat, ev.depth, ev.time, ev.mb, 1 if ev.natural_source else 0, EV_FLAG_VAL))
        return v

    def vector_to_ev(self, v):
        lon, lat, depth, time, mb, source, flag = v
        ev = Event(lon=lon, lat=lat, depth=depth, time=time, mb=mb, natural_source=bool(source), internal_id = self._id, evid=self.evid)
        return ev

    def get_event(self):
        return self.vector_to_ev(self.get_value())

    def prior_predict(self, parent_values=None):
        pass
