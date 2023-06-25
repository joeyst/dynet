import unittest as ut
from graph import Graph
from copy import deepcopy
import random

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
		net.bwd([1])
		self.assertEqual(net.ready_error, {2: True, 3: True})
		self.assertEqual(net.PRE_error, {2: 1, 3: 0.5})
		self.assertEqual(net.POST_error, {3: 0.5})
		self.assertEqual(net.values_error, {3: {2: 0.5}})
		self.assertEqual(net.weights_error, {0: {}, 1: {}, 3: {2: 0}})
		self.assertEqual(net.weights, {0: {}, 1: {}, 3: {2: 0.5}})

	def test_bwd_with_connection_from_input_to_output(self):
		net = Graph(2, 1)
		net.add_edge(0, 2, 0.5)
		net.fwd([1, 1])
		net.bwd([1])
	
		prev_error = net.get_error()
		for i in range(5):
			net.fwd([1, 1])
			net.bwd([1])
			error = net.get_error()
			print(error)
			self.assertTrue(error < prev_error)
			prev_error = error
		print("test_bwd_with_connection_from_input_to_output passed")

	def test_bwd_with_intermediate_connection(self):
		net = Graph(2, 1)
		net.add_node()
		net.add_edge(3, 2, 0.5)
		net.add_edge(0, 3, 0.6)
		net.fwd([1, 1])
		net.bwd([1])

		prev_error = net.get_error()
		for i in range(5):
			net.fwd([1, 1])
			net.bwd([1])
			net.print()
			error = net.get_error()
			# print(error)
			self.assertTrue(error < prev_error)
			prev_error = error

	def test_bwd_with_multiple_intermediate_connections(self):
		net = Graph(2, 1)
		net.add_node()
		net.add_edge(3, 2, 0.5)
		net.add_edge(0, 3, 0.6)
		net.add_edge(1, 3, 0.7)
		net.add_edge(0, 2, 0.8)
		net.add_edge(1, 2, 0.9)
	
		net.fwd([1, 1])
		net.bwd([1])

		prev_error = net.get_error()
		for i in range(5):
			net.fwd([1, 1])
			net.bwd([1])
			error = net.get_error()
			self.assertTrue(error < prev_error)
			prev_error = error

	def test_bwd_with_multiple_intermediate_connects_and_multiple_values(self):
		net = Graph(2, 1)
		net.add_node()
		net.add_edge(3, 2, 0.5)
		net.add_edge(0, 3, 0.6)
		net.add_edge(1, 3, 0.7)
		net.add_edge(0, 2, 0.8)
		net.add_edge(1, 2, 0.9)

		net.fwd([1, 1])
		net.bwd([1])
		net.fwd([2, 2])
		net.bwd([0])
		prev_error = net.get_error()

		for i in range(5):
			net.fwd([1, 1])
			net.bwd([1])
			net.fwd([2, 2])
			net.bwd([0])
			net.fwd([3, 3])
			net.bwd([-1])
			net.fwd([4, 4])
			net.bwd([-2])

		error = net.get_error()
		self.assertTrue(error < prev_error)
		prev_error = error

	def test_bwd_with_multiple_intermediate_connects_and_multiple_values_with_random_weights(self):
		net = Graph(2, 1)
		net.add_node()
		# random weights from -.5 to .5
		net.add_edge(3, 2, random.random() - 0.5)
		net.add_edge(0, 3, random.random() - 0.5)
		net.add_edge(1, 3, random.random() - 0.5)
		net.add_edge(0, 2, random.random() - 0.5)
		net.add_edge(1, 2, random.random() - 0.5)

		net.fwd([1, 1])
		net.bwd([1])
		net.fwd([2, 2])
		net.bwd([0])
		prev_error = net.get_error()

		for i in range(5000):
			net.fwd([1, 1])
			net.bwd([1])
			net.fwd([2, 2])
			net.bwd([0])
			net.fwd([3, 3])
			net.bwd([-1])
			net.fwd([4, 4])
			net.bwd([-2])
	
		error = net.get_error()
		self.assertTrue(error < prev_error)
		prev_error = error

	def test_dense(self):
		net = Graph(2, 1)
		net.add_edge(0, 2)
		net.add_edge(1, 2)

		for i in range(20):
			net.add_node()

		for i in range(20):
			net.add_edge(0, i + 3)
			net.add_edge(1, i + 3)
	 
		for i in range(20):
			net.add_node()

		for i in range(20):
			for j in range(20):
				net.add_edge(i + 3, j + 23, random.random() - 0.5)
		
		for i in range(20):
			net.add_edge(i + 23, 2, random.random() - 0.5)
		
		net.fwd([1, 1])
		net.bwd([1])

		for i in range(5):
			net.fwd([1, 1])
			net.bwd([1])
			net.fwd([2, 2])
			net.bwd([0])
			net.fwd([3, 3])
			net.bwd([-1])
			net.fwd([4, 4])
			net.bwd([-2])
	 
		
	 
		print(net.fwd([1, 1]))
		print(net.fwd([2, 2]))
		print(net.fwd([3, 3]))
		print(net.fwd([4, 4]))
	 
		error = net.get_error()
		print(error)
		print(net.weights)
		self.assertTrue(error < 0.1)
	
	def test_dense_with_square_numbers(self):
		net = Graph(2, 1)
		net.add_edge(0, 2)
		net.add_edge(1, 2)

		for i in range(20):
			net.add_node()

		for i in range(20):
			net.add_edge(0, i + 3)
			net.add_edge(1, i + 3)
	 
		for i in range(20):
			net.add_node()

		for i in range(20):
			for j in range(20):
				net.add_edge(i + 3, j + 23, random.random() - 0.5)
		
		for i in range(20):
			net.add_edge(i + 23, 2, random.random() - 0.5)
		
		nums_a = [i for i in range(20)]
		nums_b = [i for i in range(3, 23)]
		sums = [i * j for i, j in list(zip(nums_a, nums_b))]
  
		for i in range(50):
			for j in range(20):
				if j % 2 == 0:
					net.fwd([nums_a[j], nums_b[j]])
					net.bwd([sums[j]])
		
		errors = []
		for i in range(20):
			if i % 2 == 1:
				net.fwd([nums_a[i], nums_b[i]])
				errors.append(net.bwd([sums[i]]))
		
		print(errors)
		print("SUM OF ERRORS:", sum(errors))
		
		error = net.get_error()
		print(error)
		print(net.weights)
		self.assertTrue(error < 0.1)


