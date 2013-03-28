import numpy as np

from sigvisa import Sigvisa
from sigvisa.graph.nodes import VectorNode
from sigvisa.models import Distribution
from sigvisa.source.event import Event


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


class EventNode(DictNode):

    def __init__(self, event, fixed = True, **kwargs):

        self._event = event

        super(EventNode, self).__init__(model = EventPriorModel(), dimension = dimension, initial_value=event.__dict__, fixed=fixed, **kwargs)

    def get_event(self):
        return self._event

    def prior_predict(self, parent_values=None):
        pass

    def low_bounds(self):
        bounds = { k: -np.inf for k in self.keys() }
        bounds['lon'] = -185
        bounds['lat'] = -95
        #bounds['time'] = np.float('-inf')
        bounds['depth'] = 0
        bounds['mb'] = 0
        bounds['natural_source'] = None

        # only return bounds for the mutable params, since these are what we're optimizing over
        bounds = np.array([bounds[k] for k in self.keys() if self._mutable[k]])

        return bounds

    def high_bounds(self):
        bounds = { k: np.inf for k in self.keys() }
        bounds['lon'] = 185
        bounds['lat'] = 95
        #bounds['time'] = np.float('-inf')
        bounds['depth'] = 400
        bounds['mb'] = 10
        bounds['natural_source'] = None

        # only return bounds for the mutable params, since these are what we're optimizing over
        bounds = np.array([bounds[k] for k in self.keys() if self._mutable[k]])

        return bounds
