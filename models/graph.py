import numpy as np

from collections import deque
import sigvisa.infer.optimize.optim_utils as optim_utils

class CyclicGraphError(Exception):
    pass

class Node(object):

    def __init__(self, model, label="", initial_value = None, fixed_value=False, children=None, parents = None):

        self.children = set(children) if children is not None else set()
        self.parents = parents if parents is not None else dict()
        self.model = model
        self._value = initial_value
        self.fixed_value = fixed_value
        self.label = label
        self.mark = 0

    def addChild(self, child):
        self.children.add(child)
        child.parents[self.label] = self

    def addParent(self, parent):
        parent.children.add(self)
        self.parents[parent.label] = parent

    def dimension(self):
        return len(self._value)

    def mutable_dimension(self):
        if self.fixed_value:
            return 0
        else:
            return len(self._value)

    def get_mutable_values(self):
        if self.fixed_value:
            return []
        else:
            return self.get_value()

    def set_mutable_values(self, values):
        if not self.fixed_value:
            self.set_value(values)

    def get_value(self):
        return self._value

    def set_value(self, value):
        if not self.fixed_value:
            self._value = value

    def low_bounds(self):
        return [np.float('-inf'),] * self.dimension()

    def high_bounds(self):
        return [np.float('inf'),] * self.dimension()

    def _parent_values(self):
        return dict([(k, p.get_value()) for (k, p) in self.parents.items()])

    def log_p(self, value=None, parent_values=None):
        if parent_values is None:
            parent_values = self._parent_values()
        if value is None:
            value = self.get_value()

        return self.model.log_p(x = value, cond=parent_values)

    def prior_sample(self, parent_values=None):
        if self.fixed_value: return
        if parent_values is None:
            parent_values = self._parent_values()
        new_value = self.model.sample(cond=parent_values)
        self.set_value(new_value)

    def prior_predict(self, parent_values=None):
        if self.fixed_value: return
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

class ClusterNode(Node):

    """

    A collection of Nodes, each having the same parents and children,
    which can therefore be collapsed into a single node conceptually, even
    though they are still treated as independent for computational
    purposes.

    """

    def __init__(self, label="", nodes = None, parents = None, children=None):
        super(ClusterNode, self).__init__(model=None, label=label, parents=parents, children=children)

        self.nodes = nodes if nodes is not None else []

        for node in self.nodes:
            node.parents = self.parents
            node.children = self.children

    def dimension(self):
        return len(self.nodes)

    def mutable_dimension(self):
        return len([n for n in self.nodes if not n.fixed_value])


    def get_mutable_values(self):
        """

        Get the values of all of the subnodes that are not fixed. (used for optimization, etc.)

        """

        values = []
        for (i, node) in enumerate([n for n in self.nodes if not n.fixed_value]):
            values.append(node.get_value())
        return np.array(values)

    def set_mutable_values(self, value = None):
        """

        Set the values of all of the subnodes that are not fixed. (used for optimization, etc.)

        """

        for (i, node) in enumerate([n for n in self.nodes if not n.fixed_value]):
            node.set_value(value[i])


    def get_value(self):
        values = np.zeros(len(self.nodes))
        for (i, node) in enumerate(self.nodes):
            values[i] = node.get_value()
        return values

    def set_value(self, value):
        for (i, node) in enumerate(self.nodes):
            node.set_value(value[i])

    def log_p(self, value = None):
        if value is None:
            value = self.get_value()

        lp = 0
        for (i, node) in enumerate(self.nodes):
            lp += node.log_p(value = value[i])
        return lp

    def prior_sample(self, parent_values=None):
        if parent_values is None:
            parent_values = self._parent_values()

        for node in self.nodes:
            node.prior_sample(parent_values = parent_values)

    def prior_predict(self, parent_values=None):
        if parent_values is None:
            parent_values = self._parent_values()

        for node in self.nodes:
            node.prior_predict(parent_values = parent_values)


class DAG(object):

    """
    Represents a directed acyclic graph.

    """

    def __init__(self, toplevel_nodes=None, leaf_nodes=None):
        self.toplevel_nodes = toplevel_nodes if toplevel_nodes is not None else []
        self.leaf_nodes = leaf_nodes if leaf_nodes is not None else []

        # invariant: self._topo_sorted_list should always be a topologically sorted list of nodes
        self._topo_sort()

    def __ts_visit(self, node):
        m = node.get_mark()
        if m == 2:
            import pdb; pdb.set_trace()
            raise CyclicGraphError("graph contains a cycle!")
        elif m == 0:
            node.set_mark(2) # visit node "temporarily"
            for pn in node.parents.values():
                self.__ts_visit(pn)
            node.set_mark(1)
            self._topo_sorted_list.append(node)
    def _topo_sort(self):

        # check graph invariants
        for tn in self.toplevel_nodes:
            assert(len(tn.parents) == 0)
        for ln in self.leaf_nodes:
            assert(len(ln.children) == 0)

        self._topo_sorted_list = []
        for leaf in self.leaf_nodes:
            self.__ts_visit(leaf)
        self.clear_visited()

    def topo_sorted_nodes(self):
        return self._topo_sorted_list


    def clear_visited(self):
        q = deque(self.toplevel_nodes)
        while len(q) > 0:
            node = q.pop()
            node.clear_mark()
            q.extendleft(node.children)


class DirectedGraphModel(DAG):

    """
    A directed graphical probability model.

    """

    def __init__(self, **kwargs):
        super(DirectedGraphModel, self).__init__(**kwargs)

    def current_log_p(self):
        logp = 0
        for node in self.topo_sorted_nodes():
            lp = node.log_p()
            logp += lp
        return logp

    def prior_predict_all(self):
        for node in self.topo_sorted_nodes():
            if not node.fixed_value:
                node.prior_predict()

    def prior_sample_all(self):
        for node in self.topo_sorted_nodes():
            node.prior_predict()

    def joint_optimize_nodes(self, node_list, optim_params):
        """
        Assume that the value at each node is a 1D array.
        """

        node_list = list(node_list) # it's important that the nodes have a consistent order
        all_children = [child for node in node_list for child in node.children]
        relevant_nodes = set(node_list + all_children)

        def set_all(values):
            i = 0
            for node in node_list:
                n = node.mutable_dimension()
                node.set_mutable_values(values[i:i+n])
                i += n

        def get_all():
            return np.concatenate([node.get_mutable_values() for node in node_list])

        def joint_prob(values, c=-1):
            set_all(values)
            ll = np.sum([node.log_p() for node in relevant_nodes])
            return c * ll

        start_values = get_all()
        low_bounds = np.concatenate([node.low_bounds() for node in node_list])
        high_bounds = np.concatenate([node.high_bounds() for node in node_list])
        bounds = zip(low_bounds, high_bounds)

        result_vector, cost = optim_utils.minimize(joint_prob, start_values, optim_params=optim_params, bounds=bounds)
        set_all(result_vector)

