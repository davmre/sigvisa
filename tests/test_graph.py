import numpy as np

from sigvisa.graph.nodes import Node
from sigvisa.graph.dag import DAG, DirectedGraphModel, CyclicGraphError

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.source.event import get_event
from sigvisa.signals.io import load_event_station

import pickle
import os
import unittest

class TestSigvisaGraph(unittest.TestCase):

    def setUp(self):
        self.seg = load_event_station(evid=5301405, sta="URZ").with_filter('freq_2.0_3.0;env')
        self.wave = self.seg['BHZ']
        self.event = get_event(evid=5301405)

        self.sg = SigvisaGraph(phases = ['P', 'S'])
        self.sg.add_event(self.event)
        self.sg.add_wave(self.wave)

    def testPickle(self):
        f = open('pickle_test', 'wb')
        pickle.dump(self.sg, f)
        f.close()
        f = open('pickle_test', 'rb')
        sg = pickle.load(f)
        f.close()
        os.remove('pickle_test')




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

    def testDictNode(self):
        n = Node(keys = ['A', 'B', 'C'], fixed=False)
        v = {'A': 1, 'B': 1, 'C': 1 }
        n.set_dict(value=v)
        self.assertEqual(n.get_dict(), v)

        n.fix_value(key='A')
        self.assertTrue(n.mutable_dimension() == 2)
        self.assertTrue(  (n.get_mutable_values() == np.ones(2)).all()   )
        n.set_mutable_values(values = np.ones(2) * 2)

        target_value = {'A': 1, 'B': 2, 'C': 2 }
        self.assertTrue(  (n.get_dict() == target_value)  )

        v = {'A': 3, 'B': 3, 'C': 3 }
        n.set_dict(value=v)
        target_value = {'A': 1, 'B': 3, 'C': 3 }
        self.assertTrue(  (n.get_dict() == target_value)  )

        v = {'A': 4, 'B': 4, 'C': 4}
        n.set_dict(value=v, override_fixed=True)
        self.assertTrue(  (n.get_dict() == {'A': 4, 'B': 4, 'C': 4} )   )

        n.unfix_value()
        v = {'A': 5, 'B': 5, 'C': 5}
        target_value = {'A': 5, 'B': 5, 'C': 5}
        n.set_dict(value=v, override_fixed=False)
        self.assertTrue(  n.get_dict() == target_value   )

if __name__ == '__main__':
    unittest.main()
