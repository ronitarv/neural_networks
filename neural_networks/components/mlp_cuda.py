import numpy as np
from numba import njit
from numpy import float32
import math
import ctypes

#@njit()
def activation_func_derivative(a):
   return a*(1-a)
   #return 1 if a > 0 else 0

#@njit()
def activation_func(x):
   return 1 / (1 + math.exp(-x))
   #return max(0.0, x)

#@njit()
#def numba_forward(input, shape, weights, biases):
   # neurons = [np.zeros((l), dtype=float32) for l in shape[1::]]
   # neurons.insert(0, input)
   # activations_array = (ctypes.c_float * len([float(x) for y in neurons for x in y]))(*[float(x) for y in neurons for x in y])
   # weights_array = (ctypes.c_float * len([float(x) for y in self.weights for x in y.ravel()]))(*[float(x) for y in self.weights for x in y.ravel()])
   # biases_array = (ctypes.c_float * len([int(x) for y in self.biases for x in y]))(*[float(x) for y in self.biases for x in y])
   # shape_array = (ctypes.c_int * len(shape))(*shape)
   # for i, l in enumerate(shape[0:-1]):
   #    for j in range(shape[i+1]):
   #       neurons[i+1][j] = activation_func(sum([x[0]*x[1] for x in zip(neurons[i], weights[i][j*l : (j+1)*l])])+biases[i][j])
   # return neurons

#@njit()
def numba_backpropagate(shape, weights, path_error_terms, neurons):
   weight_deltas = [np.array([0]*(shape[i]*shape[i+1]), dtype=float32) for i in range(len(shape)-1)]
   bias_deltas = [np.array([0]*l, dtype=float32) for l in shape[1::]]
   neurons = [[x[i] for x in neurons] for i in range(len(path_error_terms))]
   input_error_terms = []
   for train_index in range(len(path_error_terms)):
      error_terms = [np.array([0]*l, dtype=float32) for l in shape[0:-1]]
      error_terms.append(path_error_terms[train_index])
      for i in range(len(shape)-1, 0, -1):
         error_terms[i] *= activation_func_derivative(neurons[train_index][i])
         for j in range(shape[i-1]):
            error_terms[i-1][j] = np.sum(error_terms[i]*weights[i-1][j::shape[i-1]])
      input_error_terms.append(error_terms[0])
      for i in range(len(shape)-1, 0, -1):
         for j in range(shape[i]):
            weight_deltas[i-1][shape[i-1]*j:(j+1)*shape[i-1]] += \
            np.asarray([error_terms[i][j]*x for x in neurons[train_index][i-1]])
            bias_deltas[i-1][j] += error_terms[i][j]
   return weight_deltas, bias_deltas, input_error_terms

class Mlp:
   def __init__(self, shape, lr, weight_decay=0, momentum=0, has_residual_connections=False, clipnorm=1.0, weights=None, biases=None):
      self.shape = shape
      self.rshape = shape[::-1]
      self.wshape = np.array([np.prod(shape[i:i+2]) for i in range(0,len(shape)-1)])
      self.rwshape=self.wshape[::-1]
      self.shape_length = len(shape)
      self.nr_weights = int(self.wshape.sum())
      self.nr_biases = int(shape[1:].sum())
      self.nr_neurons = int(shape.sum())
      self.nr_threads = int(np.max(self.shape))
      self.lr = lr
      self.wd = weight_decay
      self.momentum = momentum
      self.clipnorm = clipnorm
      self.weights = np.random.uniform(-1, 1, self.nr_weights).astype(float32) if weights is None else weights
      self.biases = np.zeros(self.nr_biases, dtype=float32) if biases is None else biases
      # self.weights = [np.random.uniform(-1, 1, (shape[i+1],shape[i])).astype("float") for i in range(len(shape)-1)] if not weights else weights #[np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]), np.array([0.7, -0.8, 0.9, 0.11, -0.12, 0.13])]
      # self.biases = [np.array([0]*l, dtype=float32) for l in shape[1::]] if not biases else biases # [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5])] 
      self.weight_deltas = np.zeros(self.nr_weights, dtype=float32) #[np.zeros((shape[i+1],shape[i]), dtype=float32) for i in range(len(shape)-1)]
      self.bias_deltas = np.zeros(self.nr_biases, dtype=float32) #[np.array([0]*l, dtype=float32) for l in shape[1::]]
      self.batch_size = 0
      self.neurons = None
      self.previous_weight_deltas = np.zeros(self.nr_weights, dtype=float32) #[np.zeros((shape[i+1],shape[i]), dtype=float32) for i in range(len(shape)-1)]
      self.has_residual_connections = has_residual_connections
      self.cuda_lib = ctypes.CDLL('neural_networks/components/mlp_cuda.so')  # Update with the correct path
      self.cuda_lib.mlp_forward_cuda.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
                                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), 
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
      self.cuda_lib.mlp_forward_cuda.restype = None
      self.cuda_lib.mlp_backward_cuda.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
                                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), 
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
      self.cuda_lib.mlp_backward_cuda.restype = None

   
   def forward(self, inputs):
      """return output"""
      nr_inputs = len(inputs)
      activations_array = (ctypes.c_float * (self.nr_neurons*nr_inputs))(*np.append(inputs.ravel(), np.zeros(self.nr_biases*nr_inputs, dtype=np.float32)))
      weights_array = (ctypes.c_float * self.nr_weights)(*self.weights)
      biases_array = (ctypes.c_float * self.nr_biases)(*self.biases)
      shape_array = (ctypes.c_int * self.shape_length)(*self.shape)
      self.cuda_lib.mlp_forward_cuda(activations_array, weights_array, biases_array, shape_array, self.shape_length, self.nr_neurons, self.nr_weights, self.nr_biases, self.nr_threads, nr_inputs)
      activations = np.array(activations_array, dtype=float32)
      self.neurons = activations
      outputs = activations[-self.shape[-1]*nr_inputs:].reshape((nr_inputs, self.shape[-1]))
      return outputs + inputs if self.has_residual_connections else outputs

   def backpropagate(self, path_error_terms):
      """return path_error_terms"""
      nr_inputs = len(path_error_terms)
      reversed_activations = self.neurons[::-1] #[x for y in list(reversed(neurons)) for x in y]
      reversed_weights = self.weights[::-1] #[float(x) for y in list(reversed(self.weights)) for x in y.ravel()] #[float(x) for y in list(reversed([x.transpose() for x in self.weights])) for x in y.ravel()]
      reversed_shape = self.shape[::-1]
      weight_deltas = np.zeros(self.nr_weights*nr_inputs)
      reversed_error_terms = np.append(path_error_terms.ravel()[::-1],np.zeros(self.shape[:-1].sum()))
      activations_array = (ctypes.c_float * (self.nr_neurons*nr_inputs))(*reversed_activations)
      weights_array = (ctypes.c_float * (self.nr_weights))(*reversed_weights)
      weight_deltas_array = (ctypes.c_float * (self.nr_weights*nr_inputs))(*weight_deltas)
      #bias_deltas_array = (ctypes.c_float * self.nr_biases)(*np.zeros(self.nr_biases))
      error_terms_array = (ctypes.c_float * (self.nr_neurons*nr_inputs))(*reversed_error_terms)
      shape_array = (ctypes.c_int * self.shape_length)(*reversed_shape)
      self.cuda_lib.mlp_backward_cuda(activations_array, weights_array, weight_deltas_array, error_terms_array, shape_array, self.shape_length, self.nr_neurons, self.nr_weights, self.nr_biases, self.nr_threads, nr_inputs)
      wd = np.array(weight_deltas_array)[::-1] #list(reversed([np.array(weight_deltas_array[self.rwshape[:i].sum():self.rwshape[:i+1].sum()]).reshape((self.rshape[i],self.rshape[i+1])) for i, _ in enumerate(self.wshape)])) #list(reversed([np.array(list(weight_deltas_array)[self.rwshape[:i].sum():self.rwshape[:i+1].sum()]).reshape((self.rshape[i+1],self.rshape[i])).transpose() for i, _ in enumerate(self.wshape)])) 
      wd = wd.reshape((nr_inputs, self.nr_weights)).sum(axis=0)
      new_error_terms = np.array(error_terms_array)[::-1] #list(reversed([np.array(list(error_terms_array)[self.rshape[:i].sum():self.rshape[:i+1].sum()]) for i, _ in enumerate(self.rshape)]))
      new_error_terms = [new_error_terms[nr_inputs*self.shape[:i].sum():nr_inputs*self.shape[:i+1].sum()].reshape((nr_inputs, self.shape[i])) for i, _ in enumerate(self.shape)]
      bd = np.array([y for x in new_error_terms[1:] for y in x.sum(axis=0)]) #np.array(bias_deltas_array) #new_error_terms[self.shape[0]:] #new_error_terms[1:]
      new_path_error_terms = new_error_terms[0]
      self.weight_deltas += wd + [2*self.wd*x for x in self.weights] + [self.momentum*x for x in self.previous_weight_deltas]
      self.bias_deltas += bd
      self.previous_weight_deltas = wd
      self.batch_size += len(path_error_terms)
      #print(f'mlp  old / new path error terms   -   {np.mean(np.absolute(np.absolute(path_error_terms)))} / {np.mean(np.absolute(np.absolute(new_path_error_terms)))}')
      return np.array(new_path_error_terms)
   
   def update(self) -> None:
      self.weights -= self.lr * ((self.clipnorm * (self.weight_deltas / np.linalg.norm(self.weight_deltas / self.batch_size))) 
                                       if (np.linalg.norm(self.weight_deltas / self.batch_size) > self.clipnorm) 
                                       else (self.weight_deltas / self.batch_size))
      self.biases -= self.lr * ((self.clipnorm * (self.bias_deltas / np.linalg.norm(self.bias_deltas / self.batch_size))) 
                                       if (np.linalg.norm(self.bias_deltas / self.batch_size) > self.clipnorm) 
                                       else (self.bias_deltas / self.batch_size))
      # for i in range(len(self.shape)-1):
      #    self.weights[i] -= self.lr * ((self.clipnorm * (self.weight_deltas[i] / np.linalg.norm(self.weight_deltas[i] / self.batch_size))) 
      #                                  if (np.linalg.norm(self.weight_deltas[i] / self.batch_size) > self.clipnorm) 
      #                                  else (self.weight_deltas[i] / self.batch_size))
      #    self.biases[i] -= self.lr * ((self.clipnorm * (self.bias_deltas[i] / np.linalg.norm(self.bias_deltas[i] / self.batch_size))) 
      #                                  if (np.linalg.norm(self.bias_deltas[i] / self.batch_size) > self.clipnorm) 
      #                                  else (self.bias_deltas[i] / self.batch_size))
         # self.weights[i] -= self.lr * (self.weight_deltas[i] / self.batch_size)
         # self.biases[i] -= self.lr * (self.bias_deltas[i] / self.batch_size)
      # self.weight_deltas = [np.array([0]*(self.shape[i]*self.shape[i+1]), dtype=float32) for i in range(len(self.shape)-1)]
      # self.bias_deltas = [np.array([0]*l, dtype=float32) for l in self.shape[1::]]
      self.weight_deltas = np.zeros(self.nr_weights, dtype=float32)
      self.bias_deltas = np.zeros(self.nr_biases, dtype=float32)
      self.batch_size = 0
   
   def weight_sum(self):
      return np.mean(np.absolute(self.weights))

   def __str__(self):
      return "Mlp"