import numpy as np
from numba import njit, cuda
from numpy import float64 as float32
import math
from neural_networks.components.mlp import Mlp
from neural_networks.components.linear import Linear
from functools import reduce
import time

#@njit
def activation_func_derivative(a):
   return a*(1-a)
   #return 1 if a > 0 else 0

#@njit
def activation_func(x):
   return 1 / (1 + math.exp(-x))
   #return max(0.0, x)

#@cuda.jit
#@njit
def softmax(v):
   # shiftx = v - np.max(v)
   # exps = np.exp(shiftx)
   # if ((np.isnan(exps / np.sum(exps))).any()):
   #    raise Exception("softmax return nan")
   # return exps / np.sum(exps)
   if ((np.isnan(np.exp(v) / np.exp(v).sum())).any()):
      raise Exception("softmax return nan")
   #    time.sleep(1)
   return np.exp(v) / np.exp(v).sum()

def softmax_derivative(softmax_values, softmax_error_terms):
   score_error_terms = np.zeros((len(softmax_error_terms)), dtype=float32)
   for i, _ in enumerate(softmax_values):
      score_error_terms[i] = np.sum(np.append(softmax_values[i]*(1-softmax_values[i])*softmax_error_terms[i], 
                                                -np.delete(softmax_values,[i])*softmax_values[i]*np.delete(softmax_error_terms,[i])),axis=0)
   return score_error_terms

#@njit
def numba_init_sentence(dimensions, embeddings, query_weights, key_weights, value_weights):
   queries = []
   for emb in embeddings:
      queries.append(np.array([sum(emb*query_weights[i::dimensions]) for i in range(dimensions)]))
      #queries.append(np.array([sum(emb*query_weights[i*dimensions: (i+1)*dimensions]) for i in range(dimensions)]))
   
   
   keys = []
   for emb in embeddings:
      keys.append(np.array([sum(emb*key_weights[i::dimensions]) for i in range(dimensions)]))
      #keys.append(np.array([sum(emb*key_weights[i*dimensions: (i+1)*dimensions]) for i in range(dimensions)]))
   
   values = []
   for emb in embeddings:
      values.append(np.array([sum(emb*value_weights[i::dimensions]) for i in range(dimensions)]))
      #values.append(np.array([sum(emb*value_weights[i*dimensions: (i+1)*dimensions]) for i in range(dimensions)]))

   # if np.max(queries) > 1000 or np.max(keys) > 1000 or np.max(values) > 1000:
   #    print()

   #softmax_error_terms = np.array([sum(x) for x in values])
   #print(np.max(queries), np.max(keys), np.max(values))
   return queries, keys, values #, softmax_error_terms

#@cuda.jit
def numba_forward(queries, keys, values, num_embeddings, dimensions):
   softmax_values = []
   outputs = []
   for train_index in range(num_embeddings):
      softmax_values.append(softmax(np.array([sum(queries[train_index]*key) for key in keys[0:train_index+1]])/np.sqrt(dimensions)))
      outputs.append(np.sum([z[0]*z[1] for z in zip(softmax_values[-1], values[0:train_index+1])],axis=0))
   
   return np.array(outputs), softmax_values

def forward_softmax(query, keys):
   return softmax((query*keys).sum(axis=1))

def forward_output(softmax_values, values):
   return np.sum([z[0]*z[1] for z in zip(softmax_values, values)],axis=0)

def forward_qkv(qkv_weights, emb, dimensions=3):
   return np.array([sum(emb*qkv_weights[i::dimensions]) for i in range(dimensions)])

# def numba_backpropagate(dimensions, path_error_terms, softmax_values, queries, keys, values, query_weights, key_weights, value_weights, query_weight_deltas, key_weight_deltas, value_weight_deltas, embeddings):
#    # update softmax error terms of train index to include whole path
#    attention_error_terms = []
#    for train_index in range(len(path_error_terms)):
#       # softmax_error_terms[train_index] = sum(softmax_error_terms[train_index] * path_error_terms[train_index])
#       # softmax_error_term = sum(softmax_error_terms[train_index] * path_error_terms[train_index])
#       #values = np.array(values, dtype=float32)
#       softmax_error_terms = np.sum(values * path_error_terms[train_index], axis=1)
#       # value backpropagation
#       #np.array(keys)
#       start = time.time()
#       softmax_values[train_index] = softmax_values[train_index].reshape(train_index+1, 1)
#       value_error_terms = softmax_values[train_index]*path_error_terms[train_index]
#       value_weight_deltas += sum([(value_error_terms[i].reshape(dimensions,1)*embeddings[i]).ravel() for i in range(train_index+1)])

#       query_error_terms = softmax_error_terms[:train_index+1].reshape(train_index+1,1)*softmax_values[train_index]*(1-softmax_values[train_index])*keys[:train_index+1]
#       query_weight_deltas += sum([(query_error_terms[i].reshape(dimensions,1)*embeddings[i]).ravel() for i in range(train_index+1)])

#       key_error_terms = softmax_error_terms[:train_index+1].reshape(train_index+1, 1)*softmax_values[train_index]*(1-softmax_values[train_index])*queries[train_index]
#       key_weight_deltas += sum([(key_error_terms[i].reshape(dimensions,1)*embeddings[i]).ravel() for i in range(train_index+1)])

#       print("deltas", time.time()-start)
#       start=time.time()

#       attention_error_terms.append(np.array(dimensions*[0.0], dtype=float32))
#       for error_terms in query_error_terms:
#          for j in range(dimensions):
#             attention_error_terms[train_index][j] = sum([error_terms[k]*query_weights[j::dimensions][k]*activation_func_derivative(embeddings[train_index][j]) for k in range(dimensions)])
      
#       for error_terms in key_error_terms:
#          for j in range(dimensions):
#             attention_error_terms[train_index][j] = sum([error_terms[k]*key_weights[j::dimensions][k]*activation_func_derivative(embeddings[train_index][j]) for k in range(dimensions)])
      
#       for error_terms in value_error_terms:
#          for j in range(dimensions):
#             attention_error_terms[train_index][j] = sum([error_terms[k]*value_weights[j::dimensions][k]*activation_func_derivative(embeddings[train_index][j]) for k in range(dimensions)])
#       print("terms", time.time()-start)
   
#    return query_weight_deltas, key_weight_deltas, value_weight_deltas, attention_error_terms

##@njit
def numba_backpropagate_scores(dimensions, path_error_terms, softmax_values, queries, keys, values):
   # update softmax error terms of train index to include whole path
   sum_value_error_terms = np.zeros((len(path_error_terms), dimensions), dtype=float32)
   sum_query_error_terms = np.zeros((len(path_error_terms), dimensions), dtype=float32)
   sum_key_error_terms = np.zeros((len(path_error_terms), dimensions), dtype=float32)
   #attention_error_terms.append(np.array(dimensions*[0.0], dtype=float32))
   for train_index in range(len(path_error_terms)):
      # softmax_error_terms[train_index] = sum(softmax_error_terms[train_index] * path_error_terms[train_index])
      # softmax_error_term = sum(softmax_error_terms[train_index] * path_error_terms[train_index])
      #values = np.array(values, dtype=float32)
      softmax_error_terms = np.sum(values * path_error_terms[train_index], axis=1)[:train_index+1]
      # value backpropagation
      #np.array(keys)
      #start = time.time()
      softmax_values[train_index] = softmax_values[train_index].reshape(train_index+1, 1)
      value_error_terms = softmax_values[train_index]*path_error_terms[train_index]
      #value_weight_deltas += sum([(value_error_terms[i].reshape(dimensions,1)*embeddings[i]).ravel() for i in range(train_index+1)])
      # score_error_terms = np.zeros((train_index+1), dtype=float32)
      # for i in range(train_index+1):
      #    score_error_terms[i] = np.sum(np.append(softmax_values[train_index][i]*(1-softmax_values[train_index][i])*softmax_error_terms[i], 
      #                                            -np.delete(softmax_values[train_index],[i])*softmax_values[train_index][i]*np.delete(softmax_error_terms,[i])),axis=0)
         #for j in range(train_index+1):
            # if j == i:
            #    d_softmax[i] += softmax_values[train_index][i]*(1-softmax_values[train_index][i])*softmax_error_terms[j]
            # else:
            #    d_softmax[i] += -softmax_values[train_index][j]*softmax_values[train_index][i]*softmax_error_terms[j]
      score_error_terms = softmax_derivative(softmax_values[train_index], softmax_error_terms)
      score_error_terms *= (1/np.sqrt(dimensions))
      score_error_terms = np.expand_dims(score_error_terms, axis=1)  
      #score_error_terms = softmax_error_terms[:train_index+1].reshape(train_index+1,1)*softmax_values[train_index]*(1-softmax_values[train_index])
      query_error_terms = np.sum(score_error_terms*keys[:train_index+1], axis=0)
      #query_error_terms = softmax_error_terms[:train_index+1].reshape(train_index+1,1)*softmax_values[train_index]*(1-softmax_values[train_index])*keys[:train_index+1]
       #np.sum([(embeddings[i].reshape(dimensions,1)*query_error_terms).ravel() for i in range(train_index+1)],axis=0)
      #query_weight_deltas += sum([(query_error_terms[i].reshape(dimensions,1)*embeddings[i]).ravel() for i in range(train_index+1)])

      key_error_terms = (score_error_terms*queries[train_index])
      #key_error_terms = softmax_error_terms[:train_index+1].reshape(train_index+1, 1)*softmax_values[train_index]*(1-softmax_values[train_index])*queries[train_index]
      
      #key_weight_deltas += sum([(key_error_terms[i].reshape(dimensions,1)*embeddings[i]).ravel() for i in range(train_index+1)])

      #print("deltas", time.time()-start)
      #start=time.time()
      #print()
      sum_value_error_terms[:train_index+1] += value_error_terms
      sum_query_error_terms[train_index] += query_error_terms
      sum_key_error_terms[:train_index+1] += key_error_terms
      #attention_error_terms.append(np.array(dimensions*[0.0], dtype=float32))

   

      #attention_error_terms[train_index] /= 3

         # for j in range(dimensions):
         #    attention_error_terms[train_index][j] = sum([error_terms[k]*query_weights[j::dimensions][k]*activation_func_derivative(embeddings[train_index][j]) for k in range(dimensions)])
      
      # for error_terms in key_error_terms:
      #    for j in range(dimensions):
      #       attention_error_terms[train_index][j] = sum([error_terms[k]*key_weights[j::dimensions][k]*activation_func_derivative(embeddings[train_index][j]) for k in range(dimensions)])
      
      # for error_terms in value_error_terms:
      #    for j in range(dimensions):
      #       attention_error_terms[train_index][j] = sum([error_terms[k]*value_weights[j::dimensions][k]*activation_func_derivative(embeddings[train_index][j]) for k in range(dimensions)])
      #print("terms", time.time()-start)
   #print(f'qkv:  {np.mean(np.absolute(queries)):.7f} / {np.mean(np.absolute(keys)):.7f} / {np.mean(np.absolute(values)):.7f}          -             qkv w:  {np.mean(np.absolute(query_weights)):.7f} / {np.mean(np.absolute(key_weights)):.7f} / {np.mean(np.absolute(value_weights)):.7f}            -         qkv wd:  {np.mean(np.absolute(query_weight_deltas)):.7f} / {np.mean(np.absolute(key_weight_deltas)):.7f} / {np.mean(np.absolute(value_weight_deltas)):.7f}')
   #print(f'query weight deltas {query_weight_deltas[0:4]}')
   #return query_weight_deltas / len(path_error_terms), key_weight_deltas / len(path_error_terms), value_weight_deltas / len(path_error_terms), attention_error_terms
   return sum_query_error_terms, sum_key_error_terms, sum_value_error_terms

def numba_backpropagate_qkv(dimensions, embeddings, query_error_terms, key_error_terms, value_error_terms, query_weights, key_weights, value_weights, query_weight_deltas, key_weight_deltas, value_weight_deltas, sequence_length):
   query_weight_deltas += np.sum([(embeddings[train_index].reshape(dimensions,1)*query_error_terms[train_index]).ravel() 
                                  for train_index in range(sequence_length)],axis=0)
   key_weight_deltas += np.sum([(embeddings[i].reshape(dimensions,1)*key_error_terms[i]).ravel() 
                                for i in range(sequence_length)],axis=0)
   value_weight_deltas += np.sum([(embeddings[i].reshape(dimensions,1)*value_error_terms[i]).ravel() 
                                  for i in range(sequence_length)],axis=0)
   attention_error_terms = np.zeros((sequence_length, dimensions), dtype=float32)

   for train_index, error_terms in enumerate(query_error_terms):
      attention_error_terms[train_index] += np.sum(query_weights.reshape(dimensions,dimensions) * error_terms, axis=1)

   for train_index, error_terms in enumerate(key_error_terms):
      attention_error_terms[train_index] += np.sum(key_weights.reshape(dimensions,dimensions) * error_terms, axis=1)

   for train_index, error_terms in enumerate(value_error_terms):
      attention_error_terms[train_index] += np.sum(value_weights.reshape(dimensions,dimensions) * error_terms, axis=1)
   
   return query_weight_deltas, key_weight_deltas, value_weight_deltas, attention_error_terms

class AttentionHead:
   def __init__(self, dimensions, lr, clipnorm=1.0):
      self.dimensions = dimensions
      self.lr = lr
      self.clipnorm = clipnorm
      # self.query_weights = np.array([-0.8, 0.4, -1.7, 2.8], dtype=float32)
      # self.key_weights = np.array([-1.5,1.5,0.7,-2.1], dtype=float32)
      # self.value_weights = np.array([1.0,0.6,-0.5,0.1], dtype=float32)
      
      self.queries = None
      self.keys = None
      self.values = None
      self.softmax_values = None
      self.outputs = None
      self.embeddings = None
      self.softmax_error_terms = None
      self.batch_size = 1

   def forward(self, queries, keys, values):
      """return: output"""
      # initialize qkv and partly softmax error terms
      # self.queries, self.keys, self.values = \
      # numba_init_sentence(self.dimensions, embeddings, self.query_weights, self.key_weights, self.value_weights)
      #self.embeddings = embeddings
      self.queries, self.keys, self.values = queries, keys, values
      # feed forward
      # self.outputs = []
      # self.softmax_values = []
      # for train_index in range(len(embeddings)):
      self.outputs, self.softmax_values = numba_forward(self.queries, self.keys, self.values, len(queries), self.dimensions)
         # self.outputs.append(output)
         # self.softmax_values.append(softmax_values)
      
      return self.outputs
   
   def backpropagate(self, path_error_terms):
      """return: new path error terms"""
      #start = time.time()
      #print("------------------------------------------", len(self.softmax_values[10]))
      query_error_terms, key_error_terms, value_error_terms = numba_backpropagate_scores(
         self.dimensions, path_error_terms, 
         self.softmax_values, np.array(self.queries), np.array(self.keys), np.array(self.values))
      #print("attenion head backpropagation -------- ", time.time()-start)
      
      self.queries = None
      self.keys = None
      self.values = None
      self.softmax_values = None
      self.outputs = None
      self.embeddings = None
      self.softmax_error_terms = None
      #self.softmax_values = None
      #self.output = None
      #print(f'attention  old / new path error terms   -   {np.mean(np.absolute(np.absolute(path_error_terms)))} / {np.mean(np.absolute(np.absolute(new_path_error_terms)))}')
      return query_error_terms, key_error_terms, value_error_terms
   
   def update(self) -> None:
      pass
      # self.query_weights -= self.lr * ((self.clipnorm * (self.query_weight_deltas / np.linalg.norm(self.query_weight_deltas / self.batch_size))) 
      #                                  if (np.linalg.norm(self.query_weight_deltas / self.batch_size) > self.clipnorm) 
      #                                  else (self.query_weight_deltas / self.batch_size))
      # self.key_weights -= self.lr * ((self.clipnorm * (self.key_weight_deltas / np.linalg.norm(self.key_weight_deltas / self.batch_size))) 
      #                                  if (np.linalg.norm(self.key_weight_deltas / self.batch_size) > self.clipnorm) 
      #                                  else (self.key_weight_deltas / self.batch_size))
      # self.value_weights -= self.lr * ((self.clipnorm * (self.value_weight_deltas / np.linalg.norm(self.value_weight_deltas / self.batch_size))) 
      #                                  if (np.linalg.norm(self.value_weight_deltas / self.batch_size) > self.clipnorm) 
      #                                  else (self.value_weight_deltas / self.batch_size))
      # # self.query_weights -= self.lr * self.query_weight_deltas
      # # self.key_weights -= self.lr * self.key_weight_deltas
      # # self.value_weights -= self.lr * self.value_weight_deltas
      # self.query_weight_deltas = np.array([0]*self.dimensions**2, dtype=float32)
      # self.key_weight_deltas = np.array([0]*self.dimensions**2, dtype=float32)
      # self.value_weight_deltas = np.array([0]*self.dimensions**2, dtype=float32)
   
   def weight_sum(self):
      return np.array([np.mean(np.absolute(self.query_weights)), np.mean(np.absolute(self.key_weights)), np.mean(np.absolute(self.value_weights))])


class Attention:
   def __init__(self, num_heads, dimensions, lr, has_residual_connections=False, clipnorm=1.0):
      self.heads = np.array([AttentionHead(int(dimensions/num_heads), lr, clipnorm) for i in range(num_heads)])
      self.num_heads = num_heads
      self.sections = int(dimensions / num_heads)
      self.lr = lr
      self.batch_size = 1
      self.linear = Linear((dimensions, dimensions), lr)
      self.clipnorm = clipnorm
      self.query_weights = np.random.uniform(-0.01, 0.01, dimensions**2).astype("float")
      self.key_weights = np.random.uniform(-0.01, 0.01, dimensions**2).astype("float")
      self.value_weights = np.random.uniform(-0.01, 0.01, dimensions**2).astype("float")
      # self.query_weights = np.array([-0.8, 0.4, -1.7, 2.8], dtype=float32)
      # self.key_weights = np.array([-1.5,1.5,0.7,-2.1], dtype=float32)
      # self.value_weights = np.array([1.0,0.6,-0.5,0.1], dtype=float32)
      self.query_weight_deltas = np.array([0]*dimensions**2, dtype=float32)
      self.key_weight_deltas = np.array([0]*dimensions**2, dtype=float32)
      self.value_weight_deltas = np.array([0]*dimensions**2, dtype=float32)
      self.dimensions = dimensions
      self.embeddings = None
      self.has_residual_connections = has_residual_connections

   def forward(self, embeddings):
      """return output"""
      queries, keys, values = numba_init_sentence(self.dimensions, embeddings, self.query_weights, self.key_weights, self.value_weights)
      queries = np.array(queries)
      keys = np.array(keys)
      values = np.array(values)
      self.embeddings = embeddings
      head_outputs = np.array([head.forward(queries[:, i*self.sections:(i+1)*self.sections], 
                                            keys[:, i*self.sections:(i+1)*self.sections], 
                                            values[:, i*self.sections:(i+1)*self.sections]) for i, head in enumerate(self.heads)])
      #outputs = head_outputs[0]
      outputs = self.linear.forward(np.array([head_outputs[:,i].ravel() for i in range(len(embeddings))])) if self.num_heads > 1 else head_outputs[0]
      return outputs + embeddings if self.has_residual_connections else outputs
   
   def backpropagate(self, path_error_terms):
      """return path_error_terms"""
      #linear_error_terms = path_error_terms
      linear_error_terms = self.linear.backpropagate(path_error_terms) if self.num_heads > 1 else path_error_terms
      score_error_terms = np.array([head.backpropagate(linear_error_terms[:, i*self.sections:(i+1)*self.sections]) for i, head in enumerate(self.heads)]) #  ---------------------------------- test comment
      query_error_terms = np.concatenate(score_error_terms[:,0], axis=1)
      key_error_terms = np.concatenate(score_error_terms[:,1], axis=1)
      value_error_terms = np.concatenate(score_error_terms[:,2], axis=1)
      query_weight_deltas, key_weight_deltas, value_weight_deltas, attention_error_terms = numba_backpropagate_qkv(
         self.dimensions, self.embeddings, query_error_terms, key_error_terms, value_error_terms, 
         self.query_weights, self.key_weights, self.value_weights, 
         self.query_weight_deltas, self.key_weight_deltas, self.value_weight_deltas, len(path_error_terms))
      self.query_weight_deltas += query_weight_deltas
      self.key_weight_deltas += key_weight_deltas
      self.value_weight_deltas += value_weight_deltas
      #new_path_error_terms = self.heads[0].backpropagate(path_error_terms)
      return attention_error_terms
   
   def update(self):
      self.query_weights -= self.lr * ((self.clipnorm * (self.query_weight_deltas / np.linalg.norm(self.query_weight_deltas / self.batch_size))) 
                                       if (np.linalg.norm(self.query_weight_deltas / self.batch_size) > self.clipnorm) 
                                       else (self.query_weight_deltas / self.batch_size))
      self.key_weights -= self.lr * ((self.clipnorm * (self.key_weight_deltas / np.linalg.norm(self.key_weight_deltas / self.batch_size))) 
                                       if (np.linalg.norm(self.key_weight_deltas / self.batch_size) > self.clipnorm) 
                                       else (self.key_weight_deltas / self.batch_size))
      self.value_weights -= self.lr * ((self.clipnorm * (self.value_weight_deltas / np.linalg.norm(self.value_weight_deltas / self.batch_size))) 
                                       if (np.linalg.norm(self.value_weight_deltas / self.batch_size) > self.clipnorm) 
                                       else (self.value_weight_deltas / self.batch_size))
      # self.query_weights -= self.lr * self.query_weight_deltas
      # self.key_weights -= self.lr * self.key_weight_deltas
      # self.value_weights -= self.lr * self.value_weight_deltas
      self.query_weight_deltas = np.array([0]*self.dimensions**2, dtype=float32)
      self.key_weight_deltas = np.array([0]*self.dimensions**2, dtype=float32)
      self.value_weight_deltas = np.array([0]*self.dimensions**2, dtype=float32)
      #[head.update() for head in self.heads]
      if self.num_heads > 1: self.linear.update()
   
   def weight_sum(self):
      return [head.weight_sum() for head in self.heads]
   
   def __str__(self):
      return "Attention"
