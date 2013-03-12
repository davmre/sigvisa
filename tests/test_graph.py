import numpy as np

from sigvisa.graph.nodes import Node, VectorNode, ClusterNode
from sigvisa.graph.dag import DAG, DirectedGraphModel, CyclicGraphError

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
        self.assertLess(ts.index(node1), ts.index(node3))
        self.assertLess(ts.index(node2), ts.index(node3))
        self.assertLess(ts.index(node3), ts.index(node4))
        self.assertLess(ts.index(node3), ts.index(node5))

        with self.assertRaises(AssertionError):
            node3.addChild(node1)
            dm._topo_sort()

    def testVectorNode(self):
        n = VectorNode(dimension=5, fixed_values=False)
        n.set_value(value=np.ones(5))
        self.assertTrue(  (n.get_value() == np.ones(5)).all()   )

        n.fix_value(i=3)
        self.assertTrue(n.mutable_dimension() == 4)
        self.assertTrue(  (n.get_mutable_values() == np.ones(4)).all()   )
        n.set_mutable_values(values = np.ones(4) * 2)

        target_value = np.ones(5) * 2
        target_value[3] = 1
        self.assertTrue(  (n.get_value() == target_value).all()   )

        n.set_value(value=np.ones(5) * 9)
        target_value = np.ones(5) * 9
        target_value[3] = 1
        self.assertTrue(  (n.get_value() == target_value).all()   )

        n.set_value(value=np.ones(5) * 11, override_fixed=True)
        self.assertTrue(  (n.get_value() == np.ones(5) * 11).all()   )

        n.unfix_value()
        n.set_value(value=np.ones(5) * 13, override_fixed=False)
        self.assertTrue(  (n.get_value() == np.ones(5) * 13).all()   )

    def testClusterNode(self):
        nodes = [Node() for i in range(5)]
        n = ClusterNode(nodes=nodes)

        n.set_value(value=np.ones(5))
        self.assertTrue(  (n.get_value() == np.ones(5)).all()   )

        n.fix_value(i=3)
        self.assertTrue(n.mutable_dimension() == 4)
        self.assertTrue(  (n.get_mutable_values() == np.ones(4)).all()   )
        n.set_mutable_values(values = np.ones(4) * 2)

        target_value = np.ones(5) * 2
        target_value[3] = 1
        self.assertTrue(  (n.get_value() == target_value).all()   )

        n.set_value(value=np.ones(5) * 9)
        target_value = np.ones(5) * 9
        target_value[3] = 1
        self.assertTrue(  (n.get_value() == target_value).all()   )

        n.set_value(value=np.ones(5) * 11, override_fixed=True)
        self.assertTrue(  (n.get_value() == np.ones(5) * 11).all()   )

        n.unfix_value()
        n.set_value(value=np.ones(5) * 13, override_fixed=False)
        self.assertTrue(  (n.get_value() == np.ones(5) * 13).all()   )

        self.assertTrue( n._value is None )

if __name__ == '__main__':
    unittest.main()

