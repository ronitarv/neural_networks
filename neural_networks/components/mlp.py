import numpy as np
from numba import njit
from numpy import float32 as float32
import math

@njit()
def activation_func_derivative(a):
   return a*(1-a)
   #return 1 if a > 0 else 0

@njit()
def activation_func(x):
   return 1 / (1 + math.exp(-x))
   #return max(0.0, x)

@njit()
def numba_forward(input, layers, weights, biases):
   neurons = [np.array([0.0]*l, dtype=float32) for l in layers[1::]]
   neurons.insert(0, input)
   for i, l in enumerate(layers[0:-1]):
      for j in range(layers[i+1]):
         neurons[i+1][j] = activation_func(sum([x[0]*x[1] for x in zip(neurons[i], weights[i][j*l : (j+1)*l])])+biases[i][j])
   return neurons

@njit()
def numba_backpropagate(layers, weights, path_error_terms, neurons):
   weight_deltas = [np.array([0]*(layers[i]*layers[i+1]), dtype=float32) for i in range(len(layers)-1)]
   bias_deltas = [np.array([0]*l, dtype=float32) for l in layers[1::]]
   neurons = [[x[i] for x in neurons] for i in range(len(path_error_terms))]
   input_error_terms = []
   for train_index in range(len(path_error_terms)):
      error_terms = [np.array([0]*l, dtype=float32) for l in layers[0:-1]]
      error_terms.append(path_error_terms[train_index])
      for i in range(len(layers)-1, 0, -1):
         for j in range(layers[i-1]):
            error_terms[i-1][j] = np.sum(error_terms[i]*weights[i-1][j::layers[i-1]]*activation_func_derivative(neurons[train_index][i-1][j]))
      input_error_terms.append(error_terms[0])
      for i in range(len(layers)-1, 0, -1):
         for j in range(layers[i]):
            weight_deltas[i-1][layers[i-1]*j:(j+1)*layers[i-1]] += \
            np.asarray([error_terms[i][j]*x for x in neurons[train_index][i-1]])
            bias_deltas[i-1][j] += error_terms[i][j]
   return weight_deltas, bias_deltas, input_error_terms

class Mlp:
   def __init__(self, layers, lr, weight_decay=0, momentum=0, has_residual_connections=False, clipnorm=1.0):
      self.layers = layers
      self.lr = lr
      self.wd = weight_decay
      self.momentum = momentum
      self.clipnorm = clipnorm
      self.weights = [np.random.uniform(-1, 1, layers[i]*layers[i+1]).astype("float") for i in range(len(layers)-1)] #[np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]), np.array([0.7, -0.8, 0.9, 0.11, -0.12, 0.13])]
      self.biases = [np.array([0]*l, dtype=float32) for l in layers[1::]] # [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5])] 
      self.weight_deltas = [np.array([0]*(layers[i]*layers[i+1]), dtype=float32) for i in range(len(layers)-1)]
      self.bias_deltas = [np.array([0]*l, dtype=float32) for l in layers[1::]]
      self.batch_size = 0
      self.neurons = None
      self.previous_weight_deltas = [np.array([0]*(layers[i]*layers[i+1]), dtype=float32) for i in range(len(layers)-1)]
      self.has_residual_connections = has_residual_connections
   
   def forward(self, inputs):
      """return output"""
      if len(inputs[0]) != self.layers[0]:
         raise Exception("size of provided and defined input do not match")
      neurons = []
      for input in inputs:
         neurons.append(numba_forward(input, self.layers, self.weights, self.biases))
      self.neurons = neurons
      outputs = np.array(list(map(lambda x: x[-1], neurons)), dtype=float32)
      return outputs + inputs if self.has_residual_connections else outputs

   def backpropagate(self, path_error_terms):
      """return path_error_terms"""
      weight_deltas, bias_deltas, new_path_error_terms = numba_backpropagate(self.layers, self.weights, path_error_terms, [np.array([x[i] for x in self.neurons]) for i in range(len(self.layers))])
      self.previous_weight_deltas = weight_deltas
      for i in range(len(self.layers)-1):
         self.weight_deltas[i] += weight_deltas[i] + [2*self.wd*x for x in self.weights[i]] + [self.momentum*x for x in self.previous_weight_deltas[i]]
         self.bias_deltas[i] += bias_deltas[i]
      # self.weight_deltas += weight_deltas + [2*self.wd*x for x in self.weights] + [self.momentum*x for x in self.previous_weight_deltas]
      # self.bias_deltas += bias_deltas
      self.batch_size += len(path_error_terms)
      #print(f'mlp  old / new path error terms   -   {np.mean(np.absolute(np.absolute(path_error_terms)))} / {np.mean(np.absolute(np.absolute(new_path_error_terms)))}')
      return np.array(new_path_error_terms)
   
   def update(self) -> None:
      for i in range(len(self.layers)-1):
         self.weights[i] -= self.lr * ((self.clipnorm * (self.weight_deltas[i] / np.linalg.norm(self.weight_deltas[i] / self.batch_size))) 
                                       if (np.linalg.norm(self.weight_deltas[i] / self.batch_size) > self.clipnorm) 
                                       else (self.weight_deltas[i] / self.batch_size))
         self.biases[i] -= self.lr * ((self.clipnorm * (self.bias_deltas[i] / np.linalg.norm(self.bias_deltas[i] / self.batch_size))) 
                                       if (np.linalg.norm(self.bias_deltas[i] / self.batch_size) > self.clipnorm) 
                                       else (self.bias_deltas[i] / self.batch_size))
         # self.weights[i] -= self.lr * (self.weight_deltas[i] / self.batch_size)
         # self.biases[i] -= self.lr * (self.bias_deltas[i] / self.batch_size)
      self.weight_deltas = [np.array([0]*(self.layers[i]*self.layers[i+1]), dtype=float32) for i in range(len(self.layers)-1)]
      self.bias_deltas = [np.array([0]*l, dtype=float32) for l in self.layers[1::]]
      self.batch_size = 0
   
   def weight_sum(self):
      return np.mean(np.absolute(self.weights))

   def __str__(self):
      return "Mlp"