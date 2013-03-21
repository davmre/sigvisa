import numpy as np

from collections import deque, Iterable
import sigvisa.infer.optimize.optim_utils as optim_utils

class CyclicGraphError(Exception):
    pass


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
            if not node._fixed_value:
                node.prior_predict()

    def prior_sample_all(self):
        for node in self.topo_sorted_nodes():
            node.prior_predict()

    def get_all(self, node_list):
        return np.concatenate([node.get_mutable_values() for node in node_list])

    def set_all(self, values, node_list):
        i = 0
        for node in node_list:
            n = node.mutable_dimension()
            node.set_mutable_values(values[i:i+n])
            i += n

    def joint_prob(self, values, node_list, relevant_nodes, c=1):
        v = self.get_all(node_list = node_list)
        self.set_all(values=values, node_list=node_list)
        ll = np.sum([node.log_p() for node in relevant_nodes])
        self.set_all(values=v, node_list=node_list)
        return c * ll

    def log_p_grad(self, values, node_list, relevant_nodes, eps=1e-4, c=1):
        try:
            eps0 = eps[0]
        except:
            eps = (eps,) * len(values)

        v = self.get_all(node_list = node_list)
        self.set_all(values=values, node_list=node_list)
        initial_lp = dict([(node.label, node.log_p()) for node in relevant_nodes])
        grad = np.zeros((len(values),))
        i = 0
        for node in node_list:
            n = node.mutable_dimension()
            lp0 = node.log_p()
            for ni in range(n):
                deriv = node.deriv_log_p(i=ni, lp0 = initial_lp[node.label], eps=eps[i + ni])
                for child in node.children:
                    deriv += child.deriv_log_p(parent=node.label, parent_i = ni, lp0 = initial_lp[child.label], eps=eps[i + ni])
                grad[i + ni] = deriv
            i += n
        self.set_all(values=v, node_list=node_list)
        return grad * c

    def joint_optimize_nodes(self, node_list, optim_params, use_grad=False):
        """
        Assume that the value at each node is a 1D array.
        """

        node_list = list(node_list) # it's important that the nodes have a consistent order
        all_children = [child for node in node_list for child in node.children]
        relevant_nodes = set(node_list + all_children)




        start_values = self.get_all(node_list=node_list)
        low_bounds = np.concatenate([node.low_bounds() for node in node_list])
        high_bounds = np.concatenate([node.high_bounds() for node in node_list])
        bounds = zip(low_bounds, high_bounds)

        jp = lambda v: self.joint_prob(values=v, relevant_nodes=relevant_nodes, node_list=node_list, c=-1)

        # this is included for profiling / debugging -- not real code
        def time_joint_prob():
            import time
            st = time.time()
            for i in range(500):
                joint_prob(start_values, relevant_nodes=relevant_nodes)
            et = time.time()
            print "joint prob took %.3fs on average" % ((et-st)/500.0)

        if use_grad:
            g = lambda v, eps=1e-4: self.log_p_grad(values=v, node_list=node_list, relevant_nodes=relevant_nodes, c=-1, eps=eps)
        else:
            g = None

        result_vector, cost = optim_utils.minimize(f=jp, x0=start_values, fprime=g, optim_params=optim_params, bounds=bounds)
        self.set_all(values=result_vector, node_list=node_list)
        print "got optimized x", result_vector
