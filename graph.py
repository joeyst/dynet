
def ReLU(x):
	return max(0, x)

def ReLUDerivative(x):
	return max(0, 1)

def SquaredDifference(preds, actus):
	return sum(list(map(lambda vals : (vals[0] - vals[1]) ** 2, list(zip(preds, actus)))))

print("SquaredDifference:", SquaredDifference([1, 2], [3, 4]))
print("SquaredDifference:", SquaredDifference([1, 2], [3, 5]))

"""
Each node has a POST and a PRE. 
"""

class Graph:
	def __init__(self, ilen, olen, lr=0.05, fn=ReLU, fnd=ReLUDerivative):
		"""
		Net parameters 
		"""
		self.ilen = ilen
		self.olen = olen
		# Increments when adding a node to the network. 
		self.nodes = 0
		self.fn = fn
		self.fnd = fnd
		self.lr = lr

		"""
		Output nodes don't have an activation function, so only need PRE error signal. 
		Input nodes don't have a PRE, so only need POST error signal. 
		"""
		
		"""
		Forward pass info 
		"""
		# Nested dictionary of nodes to their weights => {node1: {node2: .5, node3: .4}}
		self.weights = {}
		for i in range(self.points()):
			if not self.is_output_node(i):
				self.weights[i] = {}

		# (node1: {node2: .5, node3: .4, node7: .7})
		# Stores `True`/`False` values for if nodes have had all their dependencies computed. 
		self.ready = {}
		# Stores forward pass values. One trial only. 
		# Non-nested dictionary => {node1: .5, node2: .4, ...}
		self.POST = {}
		self.PRE = {}
	
		"""
		Backward pass info
		"""
		self.weights_error = {}
		# Stores whether PRE error signal has been computed for each node.
		self.ready_error = {}
		self.POST_error = {}
		self.PRE_error = {}
		# Stores the error signal for each value, which is summed to get dE/dPOST
		self.values_error = {}
	
		self.reset_for_trial()

	def trial(self, inps, outs, verbose=True):
		print("Pre-forward pass:", self.repr())
		self.fwd(inps)
		print("Post-forward pass:", self.repr())
		self.bwd(outs)
		print("Post-backward pass:", self.repr())
	
	def reset_for_trial(self):
		self.set_all_four_errors_to_zero()
		self.set_PRE_and_POST_to_zero()
		self.set_readiness_and_readiness_error_to_false()

	def set_all_four_errors_to_zero(self):
		for i in range(self.points()):
			if not self.is_input_node(i):
				self.PRE_error[i] = 0
			if not self.is_output_node(i):
				self.weights_error[i] = {}
				for next_node in self.get_following_nodes(i):
					self.weights_error[i][next_node] = 0
			if not self.is_input_node(i) and not self.is_output_node(i):
				self.POST_error[i] = 0
				self.values_error[i] = {}
				for next_node in self.get_following_nodes(i):
					self.values_error[i][next_node] = 0
 
	def set_PRE_and_POST_to_zero(self):
		# Input nodes don't have a PRE. 
		# Output nodes don't have a POST. 
		for i in range(self.points()):
			if not self.is_input_node(i):
				self.PRE[i] = 0
			if not self.is_output_node(i):
				self.POST[i] = 0
		
	def set_readiness_and_readiness_error_to_false(self):
		# All nodes start out not ready.
		# Input nodes don't need to compute any error. 
		for i in range(self.points()):
			self.ready[i] = False
			if not self.is_input_node(i):
				self.ready_error[i] = False
		
		for i in range(self.points()):
			if self.is_input_node(i) and i in self.ready_error:
				raise Exception("Input node should not have a ready error.")
 
	def reset(self):
		"""
		Resets: POST, PRE, POST_error, PRE_error, values_error, ready, ready_error, weights_error
		"""
		for i in range(self.points()):
			if not self.is_output_node(i):
				self.POST[i] = 0
				self.values_error[i] = {}
				self.weights_error[i] = {}

			if not self.is_input_node(i):
				self.PRE[i] = 0
				self.PRE_error[i] = 0

			if not self.is_output_node(i) and not self.is_input_node(i):
				self.POST_error[i] = 0

			self.ready[i] = False
			self.ready_error[i] = False
	
	def bwd(self, outs):
		self.update_output_PRE_errors(outs)
		self.update_node_errors()
		self.update_weight_errors()
		self.adjust_weights()
	
	def fwd(self, inps):
		self.reset_for_trial()
		self.copy_inputs(inps)

		i = self.ilen
		while not self.all_nodes_computed():
			if i >= self.points():
				i = self.ilen
			if self.is_node_ready_for_compute(i):
				self.update_PRE_and_POST_for_node(i)
			i += 1
			
		return self.outputs()
	
	def update_node_errors(self):
		i = self.ilen
		while not self.all_nodes_error_computed():
			if i >= self.points():
				i = self.ilen
			if not self.is_input_node(i) and not self.is_output_node(i) and self.is_node_error_ready_for_compute(i):
				self.update_PRE_and_POST_error_for_node(i)
			i += 1

	def update_weight_errors(self):
		for i in range(self.ilen + self.olen, self.points()):
			for j in self.get_following_nodes(i):
				# Weights error = next node's PRE error * current node's POST value 
				self.weights_error[i][j] = self.POST[i] * self.PRE_error[j]
		
	def adjust_weights(self):
		for i in range(self.points()):
			if not self.is_output_node(i):
				for j in self.get_following_nodes(i):
					self.weights[i][j] -= self.lr * self.weights_error[i][j]

	def is_node_error_ready_for_compute(self, i):
		for j in self.get_following_nodes(i):
			if not self.ready_error[j]:
				return False
		return True

	def update_PRE_and_POST_error_for_node(self, previous):
	 
		# We have a node that points to a bunch of other nodes, so 
		# its responsibility in the error will be the sum of how it 
		# affects those other nodes. 
	
		# So, get a list of nodes it points to, and sum the PRE values of those nodes 
		# multiplied with the weights 
	
		"""
		NEXT1_PRE = sum(w1v1 + w2v2 ...)
		NEXT2_PRE = sum(...)
		NEXT3_PRE = sum(...)
	
		# We need to figure out the error that CURR_NODE contributes. 
		CURR_NODE => VALUE 
		VALUE = Weight * PREVIOUS_POST 
		Take derivative w/ respect to previous post, and we get the WEIGHT. 

		Take each of the next PREs and multiply them with their corresponding weights 
		=> dot product 


	 
	 
	 
		PRE => apply nonlinear function to PRE to get POST 
		We know the POST error contribution 
		dError/dPOST => dError/dPRE * dPRE/dPOST 
		"""
	 
		print("PREVIOUS:", previous)
		for next_node in self.get_following_nodes(previous):
			# Might be wrong bc should be [i]
			if self.values_error[previous] != {}:
				print("Creating new dictionary for tabbing errors")
				self.values_error[previous] = {}
			# Error that j node is contributing to i node.
			self.values_error[previous][next_node] = self.PRE_error[next_node] * self.weights[previous][next_node]

		self.POST_error[previous] = sum(list(self.values_error[previous].values()))
		self.PRE_error[previous] = self.POST_error[previous] * self.fnd(self.PRE[previous])
		self.ready_error[previous] = True
	
	def get_following_nodes(self, i):
		return self.weights[i].keys()

	def get_node_dependencies(self, i):
		# Returns the nodes that must be evaluated before the node at index `i`.
		return self.get_dependency_dict()[i]

	def is_input_node(self, i):
		return i < self.ilen

	def is_output_node(self, i):
		return i >= self.ilen and i < self.ilen + self.olen

	def is_intermediate_node(self, i):
		return i >= self.ilen + self.olen

	def get_dependency_dict(self):
		# Map of node indices to the node indices that must be evaluated 
		# BEFORE them. 
		
		print("WEIGHTS:", self.weights)
		deps = {}
		for i in range(self.ilen, self.points()):
			deps[i] = set()

		for start_node, node_weight_dict in self.weights.items():
			for end_node, weight in node_weight_dict.items():
				if not end_node in deps.keys():
					print("Error: {} not in dictionary. graph.py's get_dependency_dict")
				deps[end_node].add(start_node)
		return deps 

	def points(self):
		return self.ilen + self.olen + self.nodes
	
	def outputs(self):
		return [self.PRE[i] for i in range(self.ilen, self.ilen + self.olen)]

	def copy_inputs(self, inps):
		for i in range(self.ilen):
			self.POST[i] = inps[i]
			self.ready[i] = True

	def error(self, outs):
		return SquaredDifference(outs, self.outputs())

	def update_output_PRE_errors(self, outs):
		error_list = list(map(lambda pair : (pair[0] - pair[1]) ** 2, list(zip(outs, self.outputs()))))
		for i in range(self.ilen, self.ilen + self.olen):
			print("Setting PRE_error[{}]".format(i))
			self.PRE_error[i] = error_list[i - self.ilen]
			self.ready_error[i] = True
		return error_list

	def add_edge(self, prior, successor, weight=0.5, verbose=True):
		# If the node is beyond the end of the network, return `False`.
		if prior >= self.points() or successor >= self.points():
			if verbose:
				print("Node index out of bounds.")
			return False
		
		# If the node is not in the dependencies map, then we must be trying to 
		# add a dependency to an input node. Return `False`.
		if self.is_input_node(successor):
			print("Cannot add dependency to input node.")
			return False
		
		# Otherwise, add the dependency and return `True`.
		self.weights[prior][successor] = weight
		self.values_error[prior][successor] = 0
		self.weights_error[prior][successor] = 0
		return True

	def add_node(self, verbose=True):
		node_id = self.points()
		if verbose:
			print("Adding node with id: {}".format(node_id))
		self.weights[node_id] = {}
		self.weights_error[node_id] = {}
		self.ready[node_id] = False
		self.ready_error[node_id] = False
		self.PRE[node_id] = 0
		self.POST[node_id] = 0
		self.PRE_error[node_id] = 0
		self.POST_error[node_id] = 0
		self.values_error[node_id] = {}
		self.nodes += 1
		return self.nodes - 1

	def is_node_ready_for_compute(self, i):
		for index in self.get_node_dependencies(i):
			if self.ready[index] is False:
				return False
		return True
	
	def all_nodes_computed(self):
		for i in range(self.ilen, self.points()):
			if self.ready[i] is False:
				return False
		return True

	def all_nodes_error_computed(self):
		# print("Ready error: {}".format(self.ready_error))
		for i in range(self.ilen, self.points()):
			if self.ready_error[i] is False:
				return False
		return True
		
	def update_PRE_and_POST_for_node(self, i):
		self.ready[i] = True
		if self.is_input_node(i):
			print("Uh oh. Updating PRE and POST for input node. In graph.py's update_PRE_and_POST_for_node.")
		self.PRE[i] = self.dot(i)
		if not self.is_output_node(i):
			self.POST[i] = self.fn(self.PRE[i])
			return self.POST[i]
		else:
			print("Not updating POST value for output node. graph.py's update_PRE_and_POST_for_node.")
	
	def dot(self, i):
		return sum([self.weights[j][i] * self.POST[j] for j in self.get_node_dependencies(i)])
	
	def repr(self):
		return "\n===\n# nodes       : {}\nweights       : {}\ndepends       : {}\nready         : {}\nPRE           : {}\nPOST          : {}\nready_error   : {}\nPRE_error     : {}\nPOST_error    : {}\nvalues_error  : {}\nweights_error : {}\n\n\n".format(self.points(), self.weights, self.get_dependency_dict(), 
			self.ready, self.PRE, self.POST, self.ready_error, self.PRE_error, self.POST_error, self.values_error, self.weights_error)
	
	def print(self):
		print(self.repr())
	
net = Graph(2, 1)
net.add_node()
net.print()
net.add_edge(3, 2)
net.print()
net.add_node()
net.add_edge(4, 2)
net.print()
print(net.fwd([0, 0]))
print(net.fwd([1, 1])) 
net.print()
net.trial([1, 0], [1])
print(net.fwd([1, 0]))
net.trial([1, 0], [1])
print(net.fwd([1, 0]))
