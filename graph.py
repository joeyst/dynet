
class Graph:
  def __init__(self, ilen, olen):
    self.ilen = ilen
    self.olen = olen
    
    self.num_points = 0
    self.dependencies = {}
    self.edges = {}
    self.vals = {}
    
  def fwd(self, inps):
    pass
  
  def bwd(self, inps, error):
    pass

  def adj(self, inps, outs):
    pass
  