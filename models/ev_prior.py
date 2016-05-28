import numpy as np

from sigvisa import Sigvisa
from sigvisa.graph.nodes import Node
from sigvisa.models import Distribution, DummyModel, Constraint
from sigvisa.models.distributions import Bernoulli, Exponential
from sigvisa.source.event import Event


class EventLocationPrior(Distribution):

    def __init__(self, inference_region=None, **kwargs):
        super(EventLocationPrior, self).__init__(**kwargs)
        self.inference_region=inference_region


    def log_p(self, x, cond=None, key_prefix=""):
        s = Sigvisa()
        lon, lat, depth = x[key_prefix + 'lon'], x[key_prefix + 'lat'], x[key_prefix + 'depth']
        if self.inference_region is not None:
            
            if not self.inference_region.contains_event(lon=lon, lat=lat):
                return -np.inf

        loc_lp = s.sigmodel.event_location_prior_logprob(lon, lat, depth)

        if np.isnan(loc_lp):
            # we are outside the implicit inference region of the currently trained prior model
            loc_lp = -np.inf

        return loc_lp

    def sample(self):
        s = Sigvisa()
        s.sigmodel.srand(np.random.randint(sys.maxint))
        lon, lat, depth = s.sigmodel.event_location_prior_sample()
        return (lon, lat, depth)

def event_from_evnodes(evnodes):
    # TEMP: evnodes will return a single node, until I get around to splitting it up.
    #return evnodes.get_event()

    evdict = {}
    for k in ('lon', 'lat', 'depth', 'time', 'mb', 'natural_source'):
        evdict[k] = evnodes[k].get_local_value(key=k)
    n = evnodes['mb']

    return Event(autoload=False, eid=n.eid, evid=n.evid, orid=n.orid, **evdict)


def setup_event(event, mag_prior_model, fixed=True, inference_region=None, **kwargs):

    key_prefix = '%d;' % event.eid

    loc_node = EventLocationNode(event, fixed=fixed, label=key_prefix + 'loc', 
                                 inference_region=inference_region, **kwargs)
    mb_node = Node(model=mag_prior_model, 
                   initial_value=event.mb, keys=(key_prefix + "mb",), 
                   low_bound=0.0, high_bound=10.0, fixed=fixed, 
                   label=key_prefix + 'mb', **kwargs)
    source_node = Node(model=Bernoulli(.999), initial_value=event.natural_source, keys=(key_prefix + "natural_source",), fixed=fixed, label=key_prefix + 'source', **kwargs)

    if inference_region is not None:
        tmodel = Constraint(default_value=event.time, 
                            a=inference_region.stime, 
                            b=inference_region.etime)
    else:
        tmodel = DummyModel(default_value=event.time)
    time_node = Node(model=tmodel, initial_value=event.time, keys=(key_prefix + "time",), fixed=fixed, label=key_prefix + 'time', **kwargs)

    evnodes = {"loc": loc_node,
               "lon": loc_node,
               "lat": loc_node,
               "depth": loc_node,
               "mb": mb_node,
               "natural_source": source_node,
               "time": time_node,
           }
    for n in evnodes.values():
        n.eid = event.eid
        n.evid = event.evid
        n.orid = event.orid
        n.key_prefix = key_prefix

    return evnodes

class EventLocationNode(Node):

    def __init__(self, event, fixed = True, 
                 inference_region=None, **kwargs):

        self.eid = event.eid
        self.evid = event.evid
        self.orid = event.orid
        key_prefix = '%d;' % self.eid
        initial_value=dict()
        d = event.to_dict()
        for k  in ("lon", "lat", "depth"):
            initial_value[key_prefix + k] = d[k]

        super(EventLocationNode, self).__init__(model = EventLocationPrior(inference_region=inference_region), 
                                                initial_value=initial_value,
                                                fixed=fixed, **kwargs)
        self.key_prefix = key_prefix

    def prior_predict(self, parent_values=None):
        pass

    def low_bounds(self):
        bounds = { k: -np.inf for k in self.keys() }
        bounds['lon'] = -185
        bounds['lat'] = -95
        bounds['depth'] = 0

        # only return bounds for the mutable params, since these are what we're optimizing over
        bounds = np.array([bounds[k] for k in self.keys() if self._mutable[k]])

        return bounds

    def high_bounds(self):
        bounds = { k: np.inf for k in self.keys() }
        bounds['lon'] = 185
        bounds['lat'] = 95
        bounds['depth'] = 700

        # only return bounds for the mutable params, since these are what we're optimizing over
        bounds = np.array([bounds[k] for k in self.keys() if self._mutable[k]])

        return bounds
        # only return bounds for the mutable params, since these are what we're optimizing over
        bounds = np.array([bounds[k] for k in self.keys() if self._mutable[k]])

        return bounds

class NoisyEventLocModel(Distribution):
    """
    when adding an event, we can have an "observe_event(eid, ev, stds=None)" method that takes the event to observe, the event to observe it as, and the noise levelk of the observation (obviously with some default).
   this will just create a bunch of child nodes, each for a particular aspect of the event. each with a parent-conditional Gaussian model.
    """
