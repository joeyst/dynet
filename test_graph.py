import unittest as ut
from graph import Graph
from copy import deepcopy

class TestGraph(ut.TestCase):
	def test_add_node(self):
		net = Graph(2, 1)
		net.add_node()
		self.assertEqual(net.points(), 4)
		self.assertEqual(net.get_dependency_dict(), {2: set(), 3: set()})
		self.assertEqual(net.ready, {0: False, 1: False, 2: False, 3: False})
		self.assertEqual(net.PRE, {2: 0, 3: 0})
		self.assertEqual(net.POST, {0: 0, 1: 0, 3: 0})
		self.assertEqual(net.ready_error, {2: False, 3: False})
		self.assertEqual(net.PRE_error, {2: 0, 3: 0})
		self.assertEqual(net.POST_error, {3: 0})
		# Only intermediate nodes should have values error bc output nodes don't have following nodes to 
		# connect to and input nodes don't get error signal propagated to them. 
		self.assertEqual(net.values_error, {3: {}})
		self.assertEqual(net.weights_error, {0: {}, 1: {}, 3: {}})

		net.add_node()
		self.assertEqual(net.points(), 5)
		self.assertEqual(net.get_dependency_dict(), {2: set(), 3: set(), 4: set()})
		self.assertEqual(net.ready, {0: False, 1: False, 2: False, 3: False, 4: False})
		self.assertEqual(net.PRE, {2: 0, 3: 0, 4: 0})
		self.assertEqual(net.POST, {0: 0, 1: 0, 3: 0, 4: 0})
		self.assertEqual(net.ready_error, {2: False, 3: False, 4: False})
		self.assertEqual(net.PRE_error, {2: 0, 3: 0, 4: 0})
		self.assertEqual(net.POST_error, {3: 0, 4: 0})
	
	def test_add_edge(self):
		net = Graph(2, 1)
		net.add_node()
		net.add_edge(3, 2, 0.5)
	
		self.assertTrue(2 in net.weights[3])
		self.assertEqual(net.get_dependency_dict(), {2: {3}, 3: set()})
		self.assertEqual(net.values_error, {3: {2: 0}})
		self.assertTrue(2 in net.weights_error[3] and 0 in net.weights_error)

	def test_fwd_with_edge(self):
		net = Graph(2, 1)
		net.add_node()
		net.add_edge(3, 2)

		prev_net = deepcopy(net)
		net.fwd([1, 1])
		net.print()
		self.assertEqual(net.weights, prev_net.weights)
		self.assertEqual(net.get_dependency_dict(), prev_net.get_dependency_dict())
		self.assertEqual(net.values_error, prev_net.values_error)
		self.assertEqual(net.weights_error, prev_net.weights_error)
		self.assertEqual(net.ready, {0: True, 1: True, 2: True, 3: True})
		self.assertEqual(net.PRE, {2: 0, 3: 0})
		self.assertEqual(net.POST, {0: 1, 1: 1, 3: 0})
		self.assertEqual(net.ready_error, {2: False, 3: False})
		self.assertEqual(net.PRE_error, {2: 0, 3: 0})
		self.assertEqual(net.POST_error, {3: 0})

	def test_bwd_with_edge(self):
		net = Graph(2, 1)
		net.add_node()
		net.add_edge(3, 2)

		net.fwd([1, 1])
		print("After fwd:")
		net.print()
		net.bwd([1])
		print("After bwd:")
		net.print()
		self.assertEqual(net.ready_error, {2: True, 3: True})
		self.assertEqual(net.PRE_error, {2: 1, 3: 0.5})
		self.assertEqual(net.POST_error, {3: 0.5})
		self.assertEqual(net.values_error, {3: {2: 0.5}})
		self.assertEqual(net.weights_error, {0: {}, 1: {}, 3: {2: 0}})
		self.assertEqual(net.weights, {0: {}, 1: {}, 3: {2: 0.5}})
