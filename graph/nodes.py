import numpy as np

from collections import deque, Iterable
import sigvisa.infer.optimize.optim_utils as optim_utils

class Node(object):


    def __init__(self, model=None, label="", initial_value = None, fixed_value=False, children=None, parents = None):

        self.children = set(children) if children is not None else set()
        self.parents = parents if parents is not None else dict()
        self.model = model
        self._value = initial_value
        self._fixed_value = fixed_value
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
        if not self._fixed_value or override_fixed:
            self._value = value

    def fix_value(self):
        self._fixed_value = True

    def unfix_value(self):
        self._fixed_value = False

    def _parent_values(self):
        return dict([(k, p.get_value()) for (k, p) in self.parents.items()])

    def log_p(self, value=None, parent_values=None):
        if parent_values is None:
            parent_values = self._parent_values()
        if value is None:
            value = self.get_value()

        return self.model.log_p(x = value, cond=parent_values)

    def deriv_log_p(self, value=None, parent=None, parent_i=None, eps=1e-4, lp0=None):
        if parent is not None and parent_i is not None:
            deriv = self.parent_deriv(parent=parent, parent_i=parent_i, eps=eps, lp0=lp0)
        else:
            raise NotImplementedError("derivative not implemented for non-vector node %s" % self.label)
        return deriv

    def parent_deriv(self, parent, parent_i, eps=1e-4, lp0=None):
        lp0 = lp0 if lp0 else self.log_p()
        pvals = self._parent_values()
        pv = np.array(pvals[parent]).copy()
        mi = self.parents[parent].mutable_i_to_i(mutable_i=parent_i)
        pv[mi] += eps
        pvals[parent] = pv
        deriv = ( self.log_p(parent_values=pvals) - lp0 ) / eps
        return deriv

    def prior_sample(self, parent_values=None):
        if self._fixed_value: return
        if parent_values is None:
            parent_values = self._parent_values()
        new_value = self.model.sample(cond=parent_values)
        self.set_value(new_value)

    def prior_predict(self, parent_values=None):
        if self._fixed_value: return
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

class VectorNode(Node):

    """

    A Node whose value is guaranteed to be a vector (technically, a 1D
    numpy array). Conceptually, models the joint distribution over some
    set of variables.

    """

    def __init__(self, dimension, fixed_values=None, initial_value=None, **kwargs):
        super(VectorNode, self).__init__(**kwargs)

        self._dimension = dimension

        if isinstance(fixed_values, bool):
            fixed_values = [fixed_values,] * dimension
        elif isinstance(fixed_values, Iterable):
            assert(len(fixed_values) == dimension)
        elif fixed_values is None:
            fixed_values = [False,] * dimension
        else:
            raise ValueError("passed invalid fixed-values setting %s" % fixed_values)
        fixed_values = np.array(fixed_values, dtype=bool)
        self._unfixed_values = ~fixed_values
        self._fixed_value = fixed_values.all()

        if initial_value is None:
            self._value = np.zeros((dimension,))
        else:
            self._value = initial_value

    def fix_value(self, i=None):
        if i is None:
            self._unfixed_values = np.array([False,] * self.dimension(), dtype=bool)
        else:
            self._unfixed_values[i] = False
        self._fixed_value = ~self._unfixed_values.any()

    def unfix_value(self, i=None):
        if i is None:
            self._unfixed_values = np.array([True,] * self.dimension(), dtype=bool)
        else:
            self._unfixed_values[i] = True
        self._fixed_value = False

    def dimension(self):
        return self._dimension

    def mutable_dimension(self):
        return np.sum(self._unfixed_values)

    def get_mutable_values(self):
        """

        Get the values of all of the sub-variables that are not fixed. (used for optimization, etc.)

        """

        return self._value[self._unfixed_values]

    def set_mutable_values(self, values):
        """

        Set the values of all of the sub-variables that are not fixed. (used for optimization, etc.)

        """

        self._value[self._unfixed_values] = values

    def low_bounds(self):
        return [np.float('-inf'),] * self.mutable_dimension()

    def high_bounds(self):
        return [np.float('inf'),] * self.mutable_dimension()

    def set_value(self, value, override_fixed=False):
        assert(len(value) == self.dimension())

        if override_fixed:
            self._value = value
        else:
            self._value[self._unfixed_values] = value[self._unfixed_values]

    def set_index(self, value, i):
        if self._unfixed_values[i]:
            self._value[i] = value

    def mutable_i_to_i(self, mutable_i):
        return np.arange(self.dimension())[self._unfixed_values][mutable_i]

    def deriv_log_p(self, i=None, parent=None, parent_i=None, eps=1e-4, lp0=None):
        lp0 = lp0 if lp0 else self.log_p()
        if i is not None:
            v_new = np.array(self.get_value()).copy()
            mi = self.mutable_i_to_i(mutable_i = i)
            v_new[i] += eps
            deriv = ( self.log_p(value=v_new) - lp0 ) / eps
        elif parent is not None and parent_i is not None:
            deriv = self.parent_deriv(parent=parent, parent_i=parent_i, lp0=lp0, eps=eps)
        return deriv





class ClusterNode(VectorNode):

    """

    A collection of Nodes, each having the same parents and children,
    which can therefore be collapsed into a single node conceptually, even
    though they are still treated as independent for computational
    purposes.

    """

    def __init__(self, label="", nodes = None, parents = None, children=None):

        # setup below here assumes that the set of nodes is finalized,
        # so subclasses should only call this constructor after
        # constructing all sub-nodes.
        self._nodes = np.array(nodes, dtype=object) if nodes is not None else np.array((), dtype=object)

        # invariant: self._unfixed_values[i] == self._nodes[i]._fixed_value for all nodes i
        super(ClusterNode, self).__init__(model=None, label=label,
                                          parents=parents, children=children,
                                          dimension=len(nodes),
                                          fixed_values = [n._fixed_value for n in self._nodes])

        for node in self._nodes:
            node.parents = self.parents
            node.children = self.children

        self._value = None

    def fix_value(self, i=None):
        if i is None:
            for node in self._nodes:
                node.fix_value()
            self._unfixed_values = np.array([False,] * self.dimension(), dtype=bool)
        else:
            self._nodes[i].fix_value()
            self._unfixed_values[i] = False
        self._fixed_value = ~self._unfixed_values.any()

    def unfix_value(self, i=None):
        if i is None:
            for node in self._nodes:
                node.unfix_value()
            self._unfixed_values = np.array([True,] * self.dimension(), dtype=bool)
        else:
            self._nodes[i].unfix_value()
            self._unfixed_values[i] = True
        self._fixed_value = False

    def get_mutable_values(self):
        return np.array([n.get_value() for n in self._nodes[self._unfixed_values]])

    def set_mutable_values(self, values):
        assert(len(values) == self.mutable_dimension())
        for (i, node) in enumerate([n for n in self._nodes[self._unfixed_values]]):
            node.set_value(values[i])

    def get_value(self):
        return [n.get_value() for n in self._nodes]

    def set_value(self, value, override_fixed=False):
        assert(len(value) == self.dimension())
        for (i, node) in enumerate(self._nodes):
            node.set_value(value[i], override_fixed=override_fixed)

    def set_index(self, value, i):
        if self._unfixed_value[i]:
            self.nodes[i].set_value(value = value)

    def log_p(self, value = None):
        if value is None:
            value = self.get_value()

        lp = 0
        for (i, node) in enumerate(self._nodes):
            lp += node.log_p(value = value[i])
        return lp

    def prior_sample(self, parent_values=None):
        if parent_values is None:
            parent_values = self._parent_values()

        for node in self._nodes:
            node.prior_sample(parent_values = parent_values)

    def prior_predict(self, parent_values=None):
        if parent_values is None:
            parent_values = self._parent_values()

        for node in self._nodes:
            node.prior_predict(parent_values = parent_values)


