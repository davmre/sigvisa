import numpy as np

from collections import deque, Iterable
import sigvisa.infer.optimize.optim_utils as optim_utils

class Node(object):


    def __init__(self, model=None, label="", initial_value = None, fixed=False, children=None, parents = None):

        self.children = set(children) if children is not None else set()
        self.parents = parents if parents is not None else dict()
        self.model = model
        self._value = initial_value
        self._fixed = fixed
        self.label = label
        self.mark = 0

    def addChild(self, child):
        self.children.add(child)
        child.parents[self.label] = self

    def addParent(self, parent):
        parent.children.add(self)
        self.parents[parent.label] = parent

    def get_value(self):
        return self._value

    def set_value(self, value, override_fixed=False):
        if not self._fixed or override_fixed:
            self._value = value

    def fix_value(self):
        self._fixed = True

    def unfix_value(self):
        self._fixed = False

    def _parent_values(self):
        return dict([(k, p.get_value()) for (k, p) in self.parents.iteritems()])

    def log_p(self, value=None, parent_values=None):
        if parent_values is None:
            parent_values = self._parent_values()
        if value is None:
            value = self.get_value()

        return self.model.log_p(x = value, cond=parent_values)

    def deriv_log_p(self, value=None, key=None, parent_values=None, parent_name=None, parent_key=None, **kwargs):
        if parent_values is None:
            parent_values = self._parent_values()
        if value is None:
            value = self.get_value()
        return self.model.deriv_log_p(x = value, cond=parent_values, idx=key, cond_key=parent_name, cond_idx=parent_key, **kwargs)

    def prior_sample(self, parent_values=None):
        if self._fixed: return
        if parent_values is None:
            parent_values = self._parent_values()
        new_value = self.model.sample(cond=parent_values)
        self.set_value(new_value)

    def prior_predict(self, parent_values=None):
        if self._fixed: return
        if parent_values is None:
            parent_values = self._parent_values()
        new_value = self.model.predict(cond=parent_values)
        self.set_value(new_value)

    def get_children(self):
        return self.children

    def set_mark(self, v=1):
        self.mark = v

    def clear_mark(self):
        self.mark = 0

    def get_mark(self):
        return self.mark

class DictNode(Node):

    """

    A Node whose value is guaranteed to be a vector (technically, a 1D
    numpy array). Conceptually, models the joint distribution over some
    set of variables.

    """

    def __init__(self, initial_value=None, fixed=None, keys=None, **kwargs):
        super(DictNode, self).__init__(**kwargs)

        if keys:
            if initial_value:
                assert (set(keys) == set(initial_value.iterkeys()) )
            if fixed and isinstance(fixed, dict):
                assert (set(keys) == set(fixed.iterkeys()) )
        else:
            keys = initial_value.keys() if initial_value else fixed.keys()

        if isinstance(fixed, bool):
            self._mutable = { k : not fixed for k in keys }
        elif isinstance(fixed, dict):
            self._mutable = { k : not fixed[k] for k in keys }
        elif fixed is None:
            self._mutable = { k : True for k in keys() }
        else:
            raise ValueError("passed invalid fixed-values setting %s" % fixed)
        self._fixed = not any(self._mutable.itervalues())

        self._value = initial_value

    def keys(self):
        return sorted(self._mutable.keys())

    def fix_value(self, key=None):
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

    def mutable_dimension(self):
        return sum(self._mutable.itervalues())

    def mutable_keys(self):
        return [k for k in self.keys() if self._mutable[k]]

    def get_mutable_values(self):
        """
        Get the values of all of the sub-variables that are not fixed. (used for optimization, etc.)
        """
        return [self._value[k] for k in self.keys() if self._mutable[k]]

    def set_mutable_values(self, values):
        """
        Set the values of all of the sub-variables that are not fixed. (used for optimization, etc.)
        """
        assert(len(values) == self.mutable_dimension())
        i = 0
        for k in self.keys():
            if self._mutable[k]:
                self._value[k] = values[i]
                i += 1

    def low_bounds(self):
        return [np.float('-inf'),] * self.mutable_dimension()

    def high_bounds(self):
        return [np.float('inf'),] * self.mutable_dimension()

    def set_value(self, value, override_fixed=False):
        assert(set(value.iterkeys()) == set(self._mutable.iterkeys()))
        if override_fixed:
            self._value = value
        else:
            self._value = {k : value[k] if self._mutable[k] else self._value[k] for k in value.iterkeys() }

    def set_key(self, value, key):
        if self._mutable[key]:
            self._value[key] = value


class ClusterNode(DictNode):

    """

    A collection of Nodes, each having the same parents and children,
    which can therefore be collapsed into a single node conceptually, even
    though they are still treated as independent for computational
    purposes.

    """

    def __init__(self, nodes, label="", parents = None, children=None):

        # setup below here assumes that the set of nodes is finalized,
        # so subclasses should only call this constructor after
        # constructing all sub-nodes.
        assert( isinstance(nodes, dict) )
        self._nodes = nodes

        # invariant: self._mutable[i] == self._nodes[i]._fixed for all nodes i
        super(ClusterNode, self).__init__(model=None, label=label,
                                          parents=parents, children=children,
                                          fixed = { k : nodes[k]._fixed for k in nodes.keys() })

        for node in self._nodes.itervalues():
            node.parents = self.parents
            node.children = self.children

        self._value = None

    def fix_value(self, key=None):
        if key is None:
            for node in self._nodes.itervalues():
                node.fix_value()
            self._mutable = {k : False for k in self._mutable.iterkeys() }
        else:
            self._nodes[key].fix_value()
            self._mutable[key] = False
        self._fixed = not any(self._mutable.itervalues())

    def unfix_value(self, key=None):
        if key is None:
            for node in self._nodes.itervalues():
                node.unfix_value()
            self._mutable = {k : True for k in self._mutable.iterkeys() }
        else:
            self._nodes[key].unfix_value()
            self._mutable[key] = True
        self._fixed = False

    def get_mutable_values(self):
        values = []
        for (k,n) in sorted(self._nodes.iteritems()):
            if self._mutable[k]:
                values.append(n.get_value())
        return values

    def set_mutable_values(self, values):
        assert(len(values) == self.mutable_dimension())
        i = 0
        for k in self.keys():
            if self._mutable[k]:
                self._nodes[k].set_value(values[i])
                i += 1

    def get_value(self):
        return { k : self._nodes[k].get_value() for k in self._nodes.iterkeys() }

    def set_value(self, value, override_fixed=False):
        assert(set(value.iterkeys()) == set(self._mutable.iterkeys()))
        for (k, v) in value.iteritems():
            self._nodes[k].set_value(v, override_fixed=override_fixed)

    def set_key(self, value, key):
        if self._mutable[key]:
            self._nodes[key].set_value(value = value)

    def log_p(self, value = None):
        if value is None:
            value = self.get_value()

        lp = sum( [self._nodes[key].log_p(value = value[key]) for key in self.keys() ] )
        return lp

    def deriv_log_p(self, value = None, key=None, **kwargs):
        if value is None:
            value = self.get_value()

        # ignore initial log_prob optimization since it doesn't
        # (currently) decompose across clusternodes
        if 'lp0' in kwargs:
            del kwargs['lp0']

        return self._nodes[key].deriv_log_p(value = value[key], **kwargs)

    def prior_sample(self, parent_values=None):
        if parent_values is None:
            parent_values = self._parent_values()

        for node in self._nodes.itervalues():
            node.prior_sample(parent_values = parent_values)

    def prior_predict(self, parent_values=None):
        if parent_values is None:
            parent_values = self._parent_values()

        for node in self._nodes.itervalues():
            node.prior_predict(parent_values = parent_values)
