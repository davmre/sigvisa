import numpy as np

import copy
from sigvisa.learn.train_param_common import load_modelid
from sigvisa.models import DummyModel

class Node(object):

    def __init__(self, model=None, label="", initial_value = None, fixed=False, keys=None, children=(), parents = ()):

        self.model = model
        self._fixed = fixed
        self.label = label
        self.mark = 0
        self.key_prefix = ""

        if not keys:
            if isinstance(initial_value, dict):
                keys = initial_value.keys()
            elif isinstance(fixed, dict):
                keys = fixed.keys()
            else:
                keys = (label,)

        if isinstance(fixed, bool):
            self._mutable = { k : not fixed for k in keys }
        elif isinstance(fixed, dict):
            self._mutable = { k : not fixed[k] for k in keys }
        elif fixed is None:
            self._mutable = { k : True for k in keys }
        else:
            raise ValueError("passed invalid fixed-values setting %s" % fixed)
        self._fixed = not any(self._mutable.itervalues())

        if len(keys) > 1:
            if initial_value:
                assert (set(keys) == set(initial_value.iterkeys()) )
            if fixed and isinstance(fixed, dict):
                assert (set(keys) == set(fixed.iterkeys()) )
            self.single_key = None
            self._dict = initial_value
        else:
            self.single_key = keys[0]
            self._dict = dict()
            self._dict[self.single_key] = initial_value

        self._low_bounds = dict()
        self._high_bounds = dict()
        for key in keys:
            self._low_bounds[key] = np.float("-inf")
            self._high_bounds[key] = np.float("inf")

        self.children = set()
        self.parents = dict()
        for child in children:
            self.addChild(child)
        for parent in parents:
            self.addParent(parent)


    def addChild(self, child):
        self.children.add(child)
        for key in self.keys():
            child.parents[key] = self

    def addParent(self, parent):
        parent.children.add(self)
        for key in parent.keys():
            self.parents[key] = parent

    def deterministic(self):
        # deterministic nodes are treated specially (see
        # DeterministicNode below).
        return False

    def keys(self):
        # return the list of keys provided by this node
        return sorted(self._dict.keys())

    def get_value(self, key=None):
        key = key if key else self.single_key
        return self._dict[key]

    def set_value(self, value, key=None):
        key = key if key else self.single_key
        if self._mutable[key]:
            self._dict[key] = value

    def get_dict(self):
        return self._dict

    def set_dict(self, value, override_fixed=False):
        assert(set(value.iterkeys()) == set(self._mutable.iterkeys()))
        if override_fixed:
            self._dict = value
        else:
            self._dict = {k : value[k] if self._mutable[k] else self._dict[k] for k in value.iterkeys() }

    def _set_values_from_model(self, value):
        # here "value" is assumed to be whatever is returned when
        # sampling from the model at this node. it can be a single
        # value or a dict/vector of values. in the latter case, we'll
        # generally need to override this method to do the right thing
        # with respect to whatever the model actually returns.

        if self.single_key:
            self.set_value(value=value)
        else:
            self.set_all_values(value)

    def fix_value(self, key=None):
        # hold constant (i.e., do not allow to be set or resampled) the value
        # of a particular key at this node, or of all keys (if no key is specified)

        if key is None:
            self._mutable = {k : False for k in self._mutable.iterkeys() }
        else:
            self._mutable[key] = False
        self._fixed = not any(self._mutable.itervalues())

    def unfix_value(self, key=None):
        if key is None:
            self._mutable = {k : True for k in self._mutable.iterkeys() }
        else:
            self._mutable[key] = True
        self._fixed = False

    def _parent_values(self):
        # return a dict of all keys provided by parent nodes, and their values
        return dict([(k, v) for p in self.parents.values() for (k,v) in p.get_dict().items()])

    def log_p(self, parent_values=None):
        #  log probability of the values at this node, conditioned on all parent values

        if parent_values is None:
            parent_values = self._parent_values()
        v = self.get_dict()
        v = v[self.single_key] if self.single_key else v
        return self.model.log_p(x = v, cond=parent_values, key_prefix=self.key_prefix)

    def deriv_log_p(self, key=None, parent_values=None, parent_key=None, parent_idx=None,  **kwargs):
        # derivative of the log probability at this node, with respect
        # to a key at this node, or with respect to a key provided by
        # a parent.
        if parent_values is None:
            parent_values = self._parent_values()

        v = self.get_dict()
        if self.single_key:
            v = v[self.single_key]
            return self.model.deriv_log_p(x = v, cond=parent_values, idx=None, cond_key=parent_key, cond_idx=parent_idx, key_prefix=self.key_prefix,  **kwargs)
        else:
            return self.model.deriv_log_p(x = v, cond=parent_values, idx=key, cond_key=parent_key, cond_idx=parent_idx, key_prefix=self.key_prefix, **kwargs)

    def parent_sample(self, parent_values=None):
        # sample a new value at this node conditioned on its parents
        if self._fixed: return
        if parent_values is None:
            parent_values = self._parent_values()
        self._set_values_from_model(self.model.sample(cond=parent_values))

    def parent_predict(self, parent_values=None):
        # predict a new value at this node conditioned on its parents.
        # the meaning of "predict" varies with the model, but is
        # usually the mean or mode of the conditional distribution.
        if self._fixed: return
        if parent_values is None:
            parent_values = self._parent_values()
        self._set_values_from_model(self.model.predict(cond=parent_values))

    def get_children(self):
        return self.children

    def set_mark(self, v=1):
        self.mark = v

    def clear_mark(self):
        self.mark = 0

    def get_mark(self):
        return self.mark

    # use custom getstate() and setstate() methods to avoid pickling
    # param models when we pickle a graph object (since these models
    # can be large, and GP models can't be directly pickled anyway).
    def __getstate__(self):
        try:
            self.modelid
            d = copy.copy(self.__dict__)
            del d['model']
            state = d
        except AttributeError:
            state = self.__dict__
        return state

    def __setstate__(self, state):
        if "model" not in state:
            if state['modelid'] is None:
                state['model'] = DummyModel()
            else:
                state["model"] = load_modelid(modelid=state['modelid'])
        self.__dict__.update(state)

    def mutable_dimension(self):
        # return the number of values at this node that are not fixed
        # (and can thus be optimized over, etc).
        return sum(self._mutable.itervalues())

    def mutable_keys(self):
        # return the set of keys at this node whose values are not
        # fixed (and can thus be optimized over, etc).
        return [k for k in self.keys() if self._mutable[k]]

    def get_mutable_values(self):
        return [self._dict[k] for k in self.keys() if self._mutable[k]]

    def set_mutable_values(self, values):
        assert(len(values) == self.mutable_dimension())
        i = 0
        for k in self.keys():
            if self._mutable[k]:
                self._dict[k] = values[i]
                i += 1

    def low_bounds(self):
        return [self._low_bounds[k] for k in self.keys() if self._mutable[k]]

    def high_bounds(self):
        return [self._high_bounds[k] for k in self.keys() if self._mutable[k]]


class DeterministicNode(Node):

    def deterministic(self):
        return True

    def parent_sample(self, parent_values=None):
        self.compute_value(parent_values=parent_values)

    def parent_predict(self, parent_values=None):
        self.compute_value(parent_values=parent_values)

    def compute_value(self, parent_values=None):
        raise NotImplementedError("compute_value method not implemented at this node!")

    def invert(self, value, parent_key, parent_values=None):
        raise NotImplementedError("invert method not implemented at this node!")

    def default_parent_key():
        raise NotImplementedError("default_parent_key method not implemented at this node!")

    def log_p(self, value=None, parent_values=None):
        raise AttributeError("cannot compute log_p for a deterministic node!")

    def deriv_log_p(**kwargs):
        raise AttributeError("cannot compute deriv_log_p for a deterministic node!")

    def set_value(self, value, key=None, parent_key=None):
        if not parent_key:
            parent_key = self.default_parent_key()

        parent_val = self.invert(value=value, parent_key=parent_key)
        self.parents[parent_key].set_value(value=parent_val, key=parent_key)
        self.compute_value()

    def get_mutable_values(self, **kwargs):
        raise AttributeError("deterministic node has no mutable values!")

    def set_mutable_values(self, **kwargs):
        raise AttributeError("deterministic node has no mutable values!")

    def fix_value(self, **kwargs):
        raise AttributeError("cannot fix/unfix values for a deterministic node!")

    def unfix_value(self, **kwargs):
        raise AttributeError("cannot fix/unfix values for a deterministic node!")
