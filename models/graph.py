import numpy as np

from collections import deque

import sigvisa.infer.optimize.optim_utils as optim_utils

class Node(object):

    def __init__(self, model, label="", initial_value = None, fixed_value=False, children=[], parents = {}):

        self.children=children
        self.parents = parents
        self.model = model
        self.value = initial_value
        self.fixed_value = fixed_value
        self.label = label
        self.mark = 0

    def addChild(self, child):
        self.children.append(child)

    def addParent(self, parent):
        self.parents[parent.label] = parent

    def dimension(self):
        return len(self.value)

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    @staticmethod
    def low_bounds(self):
        return [np.float('-inf'),] * self.dimension()

    @staticmethod
    def high_bounds(self):
        return [np.float('inf'),] * self.dimension()

    def log_p(self, value=None, parent_values=None):
        if parent_values is None:
            parent_values = dict([(k, p.get_value()) for (k, p) in self.parents])
        if value is None:
            value = self.value

        return self.model.log_p(x = val, cond=parent_values)

    def prior_sample(self, parent_values=None):
        if self.fixed_value: raise Exception("trying to change the value of a fixed-value node!")
        if parent_values is None:
            parent_values = dict([(k, p.get_value()) for (k, p) in self.parents])
        new_value = self.model.sample(cond=parent_values)
        self.value = new_val

    def prior_predict(self, parent_values=None):
        if self.fixed_value: raise Exception("trying to change the value of a fixed-value node!")
        if parent_values is None:
            parent_values = dict([(k, p.get_value()) for (k, p) in self.parents])
        new_value = self.model.predict(cond=parent_values)
        self.value = new_val

    def get_children(self):
        return self.children


    def set_mark(m=1):
        self.mark = v

    def clear_mark():
        self.mark = 0

    def get_mark():
        return self.mark

class ClusterNode(Node):

    """

    A collection of Nodes, each having the same parents and children,
    which can therefore be collapsed into a single node conceptually, even
    though they are still treated as independent for computational
    purposes.

    """

    def __init__(self, label="", nodes = [], parents = {}, children=[]):
        self.nodes = nodes
        self.label = label
        self.parents = parents
        self.children = children

        for node in self.nodes:
            node.parents = self.parents
            node.children = self.children

    def dimension(self):
        return len(self.nodes)

    def get_value(self):
        values = np.zeros(len(self.nodes))
        for (i, node) in enumerate(self.nodes):
            values[i] = node.get_value()
        return values

    def set_value(self, value):
        for (i, node) in enumerate(self.nodes):
            node.set_value(value[i])

    def log_p(self, value = None):
        lp = 0
        for (i, node) in enumerate(self.nodes):
            lp += node.log_p(value = value[i])
        return lp

    def

    def prior_sample(self, parent_values=None):
        if parent_values is None:
            parent_values = dict([k, p.get_value()) for (k, p) in self.parents])

        for node in self.nodes:
            node.prior_sample(parent_values = parent_values)

    def prior_predict(self, parent_values=None):
        if parent_values is None:
            parent_values = dict([k, p.get_value()) for (k, p) in self.parents])

        for node in self.nodes:
            node.prior_predict(parent_values = parent_values)


def DAG(object):

    """
    Represents a directed acyclic graph.

    """

    def __init__(self, toplevel_nodes[], leaf_nodes=[]):
        self.toplevel_nodes = []
        self.leaf_nodes = []

        # invariant: self.__topo_sorted_list should always be a topologically sorted list of nodes
        self.__topo_sort()

    def __ts_visit(self, node):
        m = node.get_mark()
        if m == 2:
            raise Exception("graph contains a cycle!")
        elif m == 0:
            node.set_mark(2) # visit node "temporarily"
            for pn in node.parents.values():
                self.__ts_visit(pn)
            node.set_mark(1)
            self.__topo_sorted_list.append(node)
    def __topo_sort(self):
        self.__topo_sorted_list = []
        for leaf in self.leaf_nodes:
            self.__ts_visit(self, leaf)
        self.clear_visited()

    def topo_sorted_nodes(self):
        return self.__topo_sorted_list

    def clear_visited(self):
        q = deque(self.toplevel_nodes)
        while len(q) > 0:
            node = q.pop()
            node.clear_visited()
            q.extendLeft(node.children)


class DirectedGraphModel(DAG):

    """
    A directed graphical probability model.

    """

    def __init__(self):
        super(DAG, self).__init__()

    def current_log_p(self):
        logp = 0
        for node in self.topo_sorted_nodes():
            logp += node.log_p()
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

        all_children = [child for node in node_list for child in node.children]
        relevant_nodes = set(node_list + all_children)

        def set_all(values):
            i = 0
            for node in node_list:
                n = node.dimension()
                node.set_value(values[i:i+n])
                i += n

        def get_all():
            return np.concatenate([node.get_value() for node in node_list])

        def joint_prob(values):
            set_all(values)
            ll = np.sum([node.log_p() for node in relevant_nodes])
            return ll

        start_values = get_all()
        low_bounds = np.concatenate([node.low_bounds() for node in node_list])
        high_bounds = np.concatenate([node.high_bounds() for node in node_list])
        bounds = zip(low_bounds, high_bounds)

        result_vector, cost = optim_utils.minimize(joint_prob, start_values, optim_params=optim_params, bounds=bounds)
        set_all(result_vector)

    def optimize_bottom_up(self):
        for node in self.topo_sorted_nodes()[-1]:
            if node.fixed_value: continue

            v = node.get_value()
            f = lambda x : -1 * node.markov_blanket_logp(value=x)

            if instanceof(n, np.ndarray):
                v = minimize_matrix(f=f, start=v, optim_params=optim_params, low_bounds=node.low_bound(), high_bounds=node.high_bound())
            else:
                raise NotImplementedError("optimizing non-array params is not tested / guaranteed to work")
                v = minimize(f=f, x0=v, optim_params=optim_params, bounds=(node.low_bound(), node.high_bound()))



