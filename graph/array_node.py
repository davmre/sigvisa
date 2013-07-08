import re
import numpy as np

from sigvisa import Sigvisa
from sigvisa.graph.graph_utils import parse_key
from sigvisa.graph.nodes import Node

from bisect import bisect_left

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

class ArrayNode(Node):

    def __init__(self, sorted_keys, **kwargs):

        super(ArrayNode, self).__init__(keys=sorted_keys, **kwargs)

        self.sorted_keys = sorted_keys
        self.s = Sigvisa()
        self.r = re.compile("([-\d]+);(.+);(.+);(.+);(.+);(.+)")
        self.X = np.zeros((len(self.sorted_keys),6))
        pv = super(ArrayNode, self)._parent_values()
        self._update_X(keys = sorted_keys, parent_values=pv)

    def _update_X(self, keys, parent_values):
        for k in keys:
            i = index(self.sorted_keys, k)
            eid, phase, sta, chan, band, param = parse_key(k, self.r)
            evlon = parent_values["%d;lon" % eid]
            evlat = parent_values["%d;lat" % eid]
            evtime = parent_values["%d;time" % eid]
            self.X[i, 3:6] = (evlon, evlat, evtime)
            self.X[i, 0:3] = self.s.earthmodel.site_info(sta, evtime)[0:3]

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
