import unittest as ut
from graph import Graph

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
