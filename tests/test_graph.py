import numpy

from sigvisa.models.graph import Node, DAG, DirectedGraphModel, CyclicGraphError


import unittest


class TestGraph(unittest.TestCase):

    def testTopoSort(self):
        node1 = Node(model=None, label="node1", initial_value=1)
        node2 = Node(model=None, label="node2", initial_value=2)
        node3 = Node(model=None, label="node3", initial_value=3)
        node4 = Node(model=None, label="node4", initial_value=4)
        node5 = Node(model=None, label="node5", initial_value=5)

        node1.addChild(node3)
        node2.addChild(node3)
        node3.addChild(node5)
        node3.addChild(node4)

        dm = DirectedGraphModel(toplevel_nodes = [node1, node2], leaf_nodes = [node4, node5])

        ts = dm.topo_sorted_nodes()
        print [n.label for n in ts]
        self.assertLess(ts.index(node1), ts.index(node3))
        self.assertLess(ts.index(node2), ts.index(node3))
        self.assertLess(ts.index(node3), ts.index(node4))
        self.assertLess(ts.index(node3), ts.index(node5))

        with self.assertRaises(AssertionError):
            node3.addChild(node1)
            dm._topo_sort()


if __name__ == '__main__':
    unittest.main()

