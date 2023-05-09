
def ReLU(x):
  return max(0, x)

def ReLUDerivative(x):
  return max(0, 1)

def SquaredDifference(preds, actus):
  return sum(list(map(lambda vals : (vals[0] - vals[1]) ** 2, list(zip(preds, actus)))))

print("SquaredDifference:", SquaredDifference([1, 2], [3, 4]))
print("SquaredDifference:", SquaredDifference([1, 2], [3, 5]))

class Graph:
  def __init__(self, ilen, olen, lr=0.05, fn=ReLU, fnd=ReLUDerivative):
    self.ilen = ilen
    self.olen = olen
    # Increments when adding a node to the network. 
    self.nodes = 0
    self.fn = fn
    self.fnd = fnd
    self.lr = lr
    
    # Map of node indices to the node indices that must be evaluated 
    # BEFORE them. 
    self.dependencies = {}
    self.weights = {}
    self.initialize_output_dependencies()
    # (node1: {node2: .5, node3: .4, node7: .7})
    # Stores `True`/`False` values for if nodes have had all their dependencies computed. 
    self.ready = {}
    # Stores forward pass values. One trial only. 
    self.vals = {}
    # Stores backward pass effects on error. 
		# (node1: .5, node2: .4, node7: .7)
    self.node_changes = {}
    # (node1: {node2: .5, node3: .4, node7: .7})
		self.weight_changes = {}
  
		# Stores the derivative of error with respect to the current node's PRE
		# Non-nested dictionary => {node1: .5, node2: .4, ...}
		self.dE_dPRE = {}
  
		# Stores contributions to dE/dPOST. 
		# Nested dictionary of (weight, dE/dPRE) pairs => {node1: {node2: .5, node3: .4}} 
		self.dE_dPOSTs = {}
  
		# Stores the derivative of error with respect to the current node's 
		# POST => dE/dPOST{curr} = sum(dE/dPRE{next} * weight{curr}{next})
		# Non-nested dictionary => {node1: .5, node2: .4, ...}
		self.dE_dPOST = {}

		# Stores the derivative of error with respect to the current weight 
		# Nested dictionary => {node1: {node2: .5, node3: .4}, ...}
		self.dE_dW = {}
  
		# Stores the 

  def points(self):
    return self.ilen + self.olen + self.nodes
    
  def fwd(self, inps):
    self.reset()
    self.copy_inputs(inps)

    i = self.ilen
    while not self.all_nodes_computed():
      if i >= self.points():
        i = self.ilen
      if self.is_node_ready_for_compute(i):
        self.compute_value_for_node(i)
      i += 1
      
    return self.outputs()
  
  def initialize_output_dependencies(self):
    for i in range(self.ilen, self.ilen + self.olen):
      self.dependencies[i] = []
      self.weights[i] = {}
  
  def reset(self):
    for i in range(self.ilen, self.points()):
      self.ready[i] = False
    for i in range(self.ilen, self.ilen + self.olen):
      self.vals[i] = 0
  
  def outputs(self):
    return [self.vals[i] for i in range(self.ilen, self.ilen + self.olen)]

  def copy_inputs(self, inps):
    for i in range(self.ilen):
      self.vals[i] = inps[i]
      self.ready[i] = True
  
  def bwd(self, outs, error):
		# Getting how much each output node needs to change to reduce error.
		for i in range(self.ilen, self.ilen + self.olen):
			# Change = -learning_rate * error * derivative 

			self.node_changes[i] = self.fnd(self.vals[i]) * (self.vals[i] - outs[i - self.ilen])

	def error(self, outs):
		return SquaredDifference(outs, self.outputs())

  def add_edge(self, idep, iindep, weight=0.5, verbose=True):
    # If the node is beyond the end of the network, return `False`.
    if idep >= self.points() or iindep >= self.points():
      if verbose:
        print("Node index out of bounds.")
      return False
    
    # If the node is not in the dependencies map, then we must be trying to 
    # add a dependency to an input node. Return `False`.
    if idep not in self.dependencies:
      print("Cannot add dependency to input node.")
      return False
    
    # Otherwise, add the dependency and return `True`.
    self.dependencies[iindep].append(idep)
    self.weights[iindep][idep] = weight
    return True

  def add_node(self, verbose=True):
    node_id = self.points()
    if verbose:
      print("Adding node with id: {}".format(node_id))
    self.dependencies[node_id] = []
    self.weights[node_id] = {}
    self.ready[node_id] = False
    self.vals[node_id] = 0
    self.nodes += 1
    return self.nodes - 1
    
  def is_node_ready_for_compute(self, i):
    for index in self.dependencies[i]:
      if self.ready[index] is False:
        return False
    return True
  
  def all_nodes_computed(self):
    for i in range(self.ilen, self.points()):
      if self.ready[i] is False:
        return False
    return True
    
  def compute_value_for_node(self, i):
    self.vals[i] = self.fn(self.dot(i))
    self.ready[i] = True
    return self.vals[i]
  
  def dot(self, i):
    return sum([self.weights[i][j] * self.vals[j] for j in self.dependencies[i]])
  
  def repr(self):
    return "# nodes: {}\nweights: {}\ndepends: {}\n".format(self.points(), self.weights, self.dependencies)
  
  def print(self):
    print(self.repr())
  
net = Graph(2, 1)
net.add_node()
net.print()
net.add_edge(3, 2)
net.print()
print(net.fwd([0, 0]))
print(net.fwd([1, 1])) 
net.add_node()
net.add_edge(4, 2)
net.print()
net = Graph(2, 2)
print(net.fwd([0, 0]))
