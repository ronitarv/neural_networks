# hasattr(ln, "update")
import time
import numpy as np

class Network:
  def __init__(self, layers):
    self.layers = layers
    self.inputs = None
  def forward(self, inputs):
    next_inputs = inputs
    for layer in self.layers:
      #start = time.time()
      next_inputs = layer.forward(next_inputs)
      #print(f'forward - {layer}: {time.time()-start}')
      #print("forward:", layer, time.time()-start)
    #self.input = next_input
    return next_inputs
  
  def backpropagate(self, path_error_terms):
    breakpoint
    next_error_terms = path_error_terms
    for layer in self.layers[::-1]:
      #start = time.time()
      next_error_terms = layer.backpropagate(next_error_terms)
      #print(f'backpropagate - {layer}: {time.time()-start}')
      #print("backpropagate:", layer, np.max(next_error_terms), time.time()-start)
    return next_error_terms
  
  def update(self):
    breakpoint
    #start = time.time()
    [layer.update() for layer in self.layers if hasattr(layer, "update")]
    #print(f'update - all: {time.time()-start}')
    #print("update:", time.time()-start)