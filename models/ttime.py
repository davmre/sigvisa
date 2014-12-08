import numpy as np

from sigvisa import Sigvisa

from sigvisa.graph.nodes import DeterministicNode
from sigvisa.graph.graph_utils import get_parent_value

class ArrivalTimeNode(DeterministicNode):

    def __init__(self, eid, sta, phase, **kwargs):
        s = Sigvisa()
        self.sta = sta
        self.phase = phase

        self.sta = sta
        self.ref_siteid = s.ref_siteid[sta]

        try:
            self.phaseid = s.phaseids[phase]
        except KeyError:
            print "WARNING: unrecognized phase %s, treating as P phase" % phase
            self.phaseid = 1



        super(ArrivalTimeNode, self).__init__(**kwargs)

        pv = self._parent_values()
        self.tt_residual_key, _ = get_parent_value(eid, phase, sta, param_name="tt_residual", parent_values = pv, return_key=True)
        self.ev_time_key = '%d;time' % eid
        self.ev_lon_key = '%d;lon' % eid
        self.ev_lat_key = '%d;lat' % eid
        self.ev_depth_key = '%d;depth' % eid

    def compute_value(self, parent_values=None):
        if self._fixed: return
        if parent_values is None:
            parent_values = self._parent_values()

        lon = parent_values[self.ev_lon_key]
        lat = parent_values[self.ev_lat_key]
        depth = parent_values[self.ev_depth_key]
        evtime = parent_values[self.ev_time_key]

        residual = parent_values[self.tt_residual_key]

        s = Sigvisa()
        meantt = s.sigmodel.mean_travel_time(lon, lat, depth, evtime, self.sta, self.phaseid - 1)
        self._dict[self.single_key] = evtime + meantt + residual

        for child in self.children:
            child.parent_keys_changed.add((self.single_key, self))

    def default_parent_key(self):
        return self.tt_residual_key

    def invert(self, value, parent_key, parent_values=None):
        if parent_values is None:
            parent_values = self._parent_values()

        lon = parent_values[self.ev_lon_key]
        lat = parent_values[self.ev_lat_key]
        depth = parent_values[self.ev_depth_key]
        evtime = parent_values[self.ev_time_key]

        s = Sigvisa()
        meantt = s.sigmodel.mean_travel_time(lon, lat, depth, evtime, self.sta, self.phaseid - 1)

        if parent_key == self.tt_residual_key:
            return value - evtime - meantt
        else:
            raise KeyError("can't invert arrival time with respect to %s" % parent_key)

    def deriv_value_wrt_parent(self, value=None, key=None, parent_values=None, parent_key=None):
        if parent_values is None:
            parent_values = self._parent_values()
        if value is None:
            value = self.get_value()

        key = key if key else self.single_key
        parent_key = parent_key if parent_key else self.parent_amp_transfer_key

        if key != self.single_key:
            raise AttributeError("don't know how to compute derivative of %s at coda height node" % key)
        if parent_key == self.ev_time_key:
            return 1.0
        elif parent_key == self.tt_residual_key:
            return 1.0
        elif parent_key.endswith("tt_residual"):
            return 0.0
        else:
            raise AttributeError("don't know how to compute arrival time derivative with respect to parent %s" % parent_key)

def tt_predict(event, sta, phase=None, phaseid=None):
    s = Sigvisa()

    if phaseid is None:
        try:
            phaseid = s.phaseids[phase]
        except KeyError:
            print "WARNING: unrecognized phase %s, treating as P phase" % phase
            phaseid = 1

    if event.time < 1:
        import pdb; pdb.set_trace()


    meantt = s.sigmodel.mean_travel_time(event.lon, event.lat, event.depth, event.time, sta, phaseid - 1)
    return meantt

def tt_predict_grad(lon, lat, depth, origin_time, sta, phase=None, phaseid=None):
    s = Sigvisa()

    if phaseid is None:
        try:
            phaseid = s.phaseids[phase]
        except KeyError:
            print "WARNING: unrecognized phase %s, treating as P phase" % phase
            phaseid = 1

    assert(origin_time >= 1)

    meantt, dlon, dlat, ddepth = s.sigmodel.mean_travel_time_grad(lon, lat, depth, origin_time, sta, phaseid - 1)
    return meantt, np.array((dlon, dlat, ddepth, 1.0))


def tt_log_p(x, event, sta, phase):
    s = Sigvisa()
    phaseid = s.phaseids[phase]
    ref_siteid = s.ref_siteid[sta]

    if event.time < 1:
        import pdb; pdb.set_trace()


    meantt = s.sigmodel.mean_travel_time(event.lon, event.lat, event.depth, event.time, sta, phaseid - 1)
    ll = s.sigmodel.arrtime_logprob(x, meantt, 0, ref_siteid-1, phaseid - 1)
    return ll

def tt_residual(event, sta, atime, phase=None, phaseid=None):
    pred_atime = event.time + tt_predict(event, sta, phase=phase, phaseid=phaseid)
    return atime - pred_atime
