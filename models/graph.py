import numpy as np

from collections import deque

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

    def get_value(self):
        return self.value

    def log_p(self, val=None, parent_vals=None):
        if parent_vals is None:
            parent_vals = dict([(k, p.get_value()) for (k, p) in self.parents])
        if val is None:
            val = self.value

        return self.model.log_p(x = val, cond=parent_vals)

    def prior_sample(self, parent_vals=None):
        if self.fixed_value: raise Exception("trying to change the value of a fixed-value node!")
        if parent_vals is None:
            parent_vals = dict([(k, p.get_value()) for (k, p) in self.parents])
        new_val = self.model.sample(cond=parent_vals)
        self.value = new_val

    def prior_predict(self, parent_vals=None):
        if self.fixed_value: raise Exception("trying to change the value of a fixed-value node!")
        if parent_vals is None:
            parent_vals = dict([(k, p.get_value()) for (k, p) in self.parents])
        new_val = self.model.predict(cond=parent_vals)
        self.value = new_val

    def get_children(self):
        return self.children

    def markov_blanket_logp(self, val=None):
        logp = self.log_p(val=val)
        for child in self.children:
            logp += child.log_p(val=val)

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

    def get_value(self):
        raise NotImplementedError("abstract base class")

    def prior_sample(self, parent_vals=None):
        if parent_vals is None:
            parent_vals = dict([k, p.get_value()) for (k, p) in self.parents])

        for node in self.nodes:
            node.prior_sample(parent_vals = parent_vals)

    def prior_predict(self, parent_vals=None):
        if parent_vals is None:
            parent_vals = dict([k, p.get_value()) for (k, p) in self.parents])

        for node in self.nodes:
            node.prior_predict(parent_vals = parent_vals)

    def log_p(self, parent_vals = None):
        if parent_vals is None:
            parent_vals = dict([k, p.get_value()) for (k, p) in self.parents])

        lp = 0
        for node in self.nodes:
            lp += node.log_p(parent_vals = parent_vals)

        return lp

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

    def optimize_bottom_up(self):
        for node in self.topo_sorted_nodes()[-1]:
            if node.fixed_value: continue

            v = node.get_value()
            f = lambda x : -1 * node.markov_blanket_logp(val=x)

            if instanceof(n, np.ndarray):
                v = minimize_matrix(f=f, start=v, optim_params=optim_params, low_bounds=node.low_bound(), high_bounds=node.high_bound())
            else:
                raise NotImplementedError("optimizing non-array params is not tested / guaranteed to work")
                v = minimize(f=f, x0=v, optim_params=optim_params, bounds=(node.low_bound(), node.high_bound()))



