import re
import numpy as np
import copy
from sigvisa import Sigvisa
from sigvisa.graph.graph_utils import parse_key
from sigvisa.graph.nodes import Node
from sigvisa.learn.train_param_common import load_modelid
import sigvisa.utils.geog as geog

from bisect import bisect_left

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

def lldlld_X(ev, sta):
    X = np.zeros((1,8))
    slon, slat, sdepth = self.s.earthmodel.site_info(sta, evtime)[0:3]
    X[0, 0:3] = (slon, slat, sdepth)
    X[0, 3:6] = (ev.lon, ev.lat, ev.depth)
    X[0, 7] = geog.dist_km((ev.lon, ev.lat), (slon, slat))
    X[0, 8] = geog.azimuth((slon, slat), (ev.lon, ev.lat))

class ArrayNode(Node):

    def __init__(self, sorted_keys, **kwargs):

        super(ArrayNode, self).__init__(keys=sorted_keys, **kwargs)

        self.sorted_keys = sorted_keys
        self.s = Sigvisa()
        self.r = re.compile("([-\d]+);(.+);(.+);(.+);(.+);(.+)")
        self.X = np.zeros((len(self.sorted_keys),8))
        pv = super(ArrayNode, self)._parent_values()
        self._update_X(keys = sorted_keys, parent_values=pv)

    def _update_X(self, keys, parent_values):
        for k in keys:
            i = index(self.sorted_keys, k)
            eid, phase, sta, chan, band, param = parse_key(k, self.r)
            evlon = parent_values["%d;lon" % eid]
            evlat = parent_values["%d;lat" % eid]
            evdepth = parent_values["%d;depth" % eid]
            evtime = parent_values["%d;time" % eid]
            self.X[i, 3:6] = (evlon, evlat, evdepth)
            self.X[i, 0:3] = self.s.earthmodel.site_info(sta, evtime)[0:3]
            stalon, stalat = self.X[i, 0:2]
            self.X[i, 6] = geog.dist_km((evlon, evlat), (stalon, stalat))
            self.X[i, 7] = geog.azimuth((stalon, stalat), (evlon, evlat))


    def _parent_values(self):
        parent_keys_changed = [k for (k,n) in self.parent_keys_changed]
        parent_values = super(ArrayNode, self)._parent_values()
        self._update_X(keys = parent_keys_changed, parent_values=parent_values)
        return parent_values

    def log_p(self, parent_values=None):
        #  log probability of the values at this node, conditioned on all parent values
        if parent_values is None:
            parent_values = self._parent_values()
        y = self._transform_values_for_model(values = self.get_dict(), parent_values=parent_values)
        return self.model.log_p(x = y, cond=self.X)


    def deriv_log_p(self, key=None, parent_values=None, parent_key=None, parent_idx=None, **kwargs):
        if parent_values is None:
            parent_values = self._parent_values()
        y = self._transform_values_for_model(values = self.get_dict(), parent_values=parent_values)
        import pdb; pdb.set_trace()
        return self.model.deriv_log_p(x = y, cond=parent_values, idx=key, cond_key=parent_key, cond_idx=parent_idx, key_prefix=self.key_prefix, **kwargs)



    def deriv_log_p(self, key=None, parent_values=None, parent_key=None, lp0=None,  eps=1e-4, **kwargs):
        # derivative of the log probability at this node, with respect
        # to a key at this node, or with respect to a key provided by
        # a parent.

        if parent_values is None:
            parent_values = self._parent_values()
        y = self._transform_values_for_model(values = self.get_dict(), parent_values=parent_values)

        if key is not None:
            idx = index(self.sorted_keys, key)
            return self.model.deriv_log_p(x = y[idx], cond=self.X[idx:idx+1,:],  **kwargs)
        elif parent_key is not None:
            print "warning: inefficient numerical derivative of arraynode wrt parents"
            lp0 = lp0 if lp0 is not None else self.model.log_p(x = y, cond=self.X)
            old_pv = parent_values[parent_key]
            parent_values[parent_key] += eps
            self._update_X(keys = [parent_key,], parent_values=parent_values)
            lp1 = self.model.log_p(x = y, cond=self.X)
            parent_values[parent_key] -= eps
            self._update_X(keys = [parent_key,], parent_values=parent_values)
            return (lp1 - lp0) / eps
        else:
            return self.model.deriv_log_p(x = y, cond=self.X, **kwargs)

    #turns dictionary into array and sort it
    def _transform_values_for_model(self, values, parent_values):
        y = np.zeros((len(self.sorted_keys),))
        for (i, k) in enumerate(self.sorted_keys):
            y[i] = values[k]
        return y

    def parent_predict(self, parent_values=None):
        if self._fixed: return
        for (i, k) in enumerate(self.sorted_keys):
            v = self.model.predict(cond = self.X[i:i+1, :])
            self.set_value(value=v, key=k)

    def parent_sample(self, parent_values=None):
        if self._fixed: return
        for (i, k) in enumerate(self.sorted_keys):
            v = self.model.predict(cond = self.X[i:i+1, :])
            self.set_value(value=v, key=k)



    # use custom getstate() and setstate() methods to avoid pickling
    # param models when we pickle a graph object (since these models
    # can be large, and GP models can't be directly pickled anyway).

    # use custom getstate() and setstate() methods to avoid pickling
    # param models when we pickle a graph object (since these models
    # can be large, and GP models can't be directly pickled anyway).
    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d['model']
        del d['s']
        return d

    def __setstate__(self, state):
        if "model" not in state:
            if state['modelid'] is None:
                state['model'] = DummyModel()
            else:
                state["model"] = load_modelid(modelid=state['modelid'], gpmodel_build_trees=False)
        if "s" not in state:
            state["s"] = Sigvisa()
        self.__dict__.update(state)

