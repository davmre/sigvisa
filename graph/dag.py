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

        self.all_nodes = dict()
        self.nodes_by_key = dict()


        def add_children(n):
            self.add_node(n)
            for c in n.children:
                add_children(c)

        if self.toplevel_nodes is not None:
            for n in self.toplevel_nodes:
                add_children(n)

    def current_log_p(self):
        logp = 0
        for node in self.topo_sorted_nodes():
            if node.deterministic(): continue
            lp = node.log_p()
            logp += lp
        return logp

    def prior_predict_all(self):
        for node in self.topo_sorted_nodes():
            if not node._fixed:
                node.prior_predict()

    def prior_sample_all(self):
        for node in self.topo_sorted_nodes():
            node.prior_predict()

    def get_all(self, node_list):
        return np.concatenate([node.get_mutable_values() for node in node_list if not node.deterministic()])

    def set_all(self, values, node_list):
        i = 0
        for node in node_list:
            if node.deterministic(): continue
            n = node.mutable_dimension()
            node.set_mutable_values(values[i:i+n])
            i += n

            for dn in self.get_deterministic_children(node):
                node.prior_predict()


    def joint_prob(self, values, node_list, relevant_nodes, c=1):
        # node_list: list of nodes whose values we are interested in

        # relevant_nodes: all nodes whose log_p() depends on a value
        # from a node in node_list.

        v = self.get_all(node_list = node_list)
        self.set_all(values=values, node_list=node_list)
        ll = np.sum([node.log_p() for node in relevant_nodes])
        self.set_all(values=v, node_list=node_list)
        return c * ll


    def get_stochastic_children(self, node):
        # return all stochastic nodes that depend directly on this
        # node or a deterministic function of this node. For each
        # such stochastic child, also include the chain of
        # deterministic nodes connecting it to the given node.

        child_list = []
        def traverse_child(n, intermediates):
            if not n.deterministic():
                child_list.append((n, intermediates))
                return
            else:
                for c in n.children:
                    traverse_child((c, intermediates + (n,)))
        for c in node.children:
            traverse_child(c, ())
        return child_list

    def get_deterministic_children(self, node):
        # return all nodes that compute a deterministic function of
        # this node, in topologically sorted order.

        # TODO: currently assumes a tree structure to the
        # deterministic children, i.e. doesn't do a true topo
        # sort. Results will be INCORRECT if this is not the case.

        child_list = []
        def traverse_child(n):
            if n.deterministic():
                child_list.append(n)
                for c in n.children:
                    traverse_child(c)
            else:
                return

        traverse_child(node)
        return child_list


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
            keys = node.mutable_keys()
            for (ni, key) in enumerate(keys):
                deriv = node.deriv_log_p(key=key, eps=eps[i + ni], lp0=initial_lp[node.label])

                # sum the derivatives of all child nodes wrt to this value, including
                # any deterministic nodes along the way
                child_list = self.get_stochastic_children(node)
                for (child, intermediate_nodes) in child_list:
                    d = 1
                    for i in intermediate_nodes:
                        d *= i.deriv_value_wrt_parent(parent_key = key)
                        key = i.label
                    d *= child.deriv_log_p(parent_key = key,
                                           eps=eps[i + ni],
                                           lp0=initial_lp[child.label])
                    deriv += d

                grad[i + ni] = deriv
            i += len(keys)
        self.set_all(values=v, node_list=node_list)
        return grad * c

    def joint_optimize_nodes(self, node_list, optim_params, use_grad=True):
        """
        Assume that the value at each node is a 1D array.
        """

        # note, it's important that the nodes have a consistent order, since
        # we represent their joint values as a vector.
        node_list = [node for node in node_list if not node.deterministic()]
        all_stochastic_children = [child for node in node_list for (child, intermediates) in self.get_stochastic_children(node)]
        relevant_nodes = set(node_list + all_stochastic_children)

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

    def add_node(self, node):
        self.all_nodes[node.label] = node
        for key in node.keys():
            self.nodes_by_key[key] = node
        self._topo_sort()

    def topo_sorted_nodes(self):
        assert(len(self._topo_sorted_list) == len(self.all_nodes))
        return self._topo_sorted_list

    def get_node_from_key(self, key):
        return self.nodes_by_key[key]

    def set_value(self, key, value, **kwargs):
        n = self.nodes_by_key[key]
        n.set_value(value=value, key=key, **kwargs)

    def get_value(self, key):
        n = self.nodes_by_key[key]
        return n.get_value(key=key)
