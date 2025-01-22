#from tensorflow.keras.datasets import mnist
import numpy as np
from numba import njit
import math
from numpy import float32
from playsound import playsound
import math

# 28*28 = 784
# 784*16 = 12544
# 784*16+16*16 = 12800

num_hidden_neurons = 20
batch_size = 1
learning_rate = 0.1
wd = 0.001 # 0.0000001 # 0.000001
momentum = 0.9
epochs = 20000
dictionary = ["pertti", "eat", "pizza", "ing", "is", "warm", "good"]
dictionary_size = len(dictionary)
x_train = [
   np.array(["pertti", "is", "eat", "ing", "pizza"]),
   np.array(["pizza", "is", "warm"]),
   np.array(["pertti", "is", "eat", "ing", "warm", "pizza"]),
   np.array(["eat", "ing", "pizza", "is", "good"])
   ]
word_dimensions = 3

# 77.34
# 80 hn200
# 85.22 hn300
# 87,26 hn532 bs100

# best batch size for 50 hidden neurons is 80


# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train.shape = (60000, 28*28)
# x_train = np.array([x/255 for x in x_train])
# x_train.shape = (int(60000/batch_size),batch_size, 28*28)
# y_train.shape = (int(60000/batch_size),batch_size)
# x_test.shape = (10000, 28*28)
# x_test = np.array([x/255 for x in x_test])


##@njit()
def activation_func_derivative(a):
   return a*(1-a)
   #return 1 if a > 0 else 0

##@njit()
def activation_func(x):
   return 1 / (1 + math.exp(-x))
   #return max(0.0, x)

def softmax(v):
   return np.exp(v) / np.exp(v).sum()

def decode_mlp(input, weights, biases, num_neurons):
   neurons = [input, np.array(num_neurons[1]*[0.0], dtype=float32), np.array(num_neurons[2]*[0.0], dtype=float32)]
   for i in range(num_neurons[1]):
      neurons[1][i] = activation_func(sum([x[0]*x[1] for x in zip(neurons[0], weights[0][i*num_neurons[0] : (i+1)*num_neurons[0]])])+biases[0][i])
   for i in range(num_neurons[2]):
      neurons[2][i] = activation_func(sum([x[0]*x[1] for x in zip(neurons[1], weights[1][i*num_neurons[1] : (i+1)*num_neurons[1]])])+biases[1][i])
   return neurons

class Mlp:
   def __init__(self, layers):
      self.layers = layers
      self.weights = [np.random.uniform(-1, 1, layers[i]*layers[i+1]).astype("float") for i in range(len(layers)-1)]
      self.biases = [np.array([0]*l, dtype=float32) for l in layers[1::]]
      self.weight_deltas = [np.array([0]*(layers[i]*layers[i+1]), dtype=float32) for i in range(len(layers)-1)]
      self.bias_deltas = [np.array([0]*l, dtype=float32) for l in layers[1::]]
      self.neurons = None
   
   def forward(self, input):
      if len(input) != self.layers[0]:
         raise Exception("size of provided and defined input do not match")
      neurons = [input] + [np.array([0]*l, dtype=float32) for l in self.layers[1::]]
      for i, l in enumerate(self.layers[0:-1]):
         for j in range(self.layers[1]):
            neurons[i+1][j] = activation_func(sum([x[0]*x[1] for x in zip(neurons[i], self.weights[i][j*l : (j+1)*l])])+self.weights[i][j])
      return neurons[-1]
   
   def backpropagate(self, path_error_terms):
      error_terms = [np.array([0]*l, dtype=float32) for l in self.layers[0:-1]].append(path_error_terms)
      for i in range(len(self.layers)-1, -1, -1):
         for j in range(self.layers[i]):
            error_terms[i][j] = sum([error_terms[i+1][k]*self.weights[i][j::self.layers[i]][k]*activation_func_derivative(self.neurons[i][j]) for k in range(self.layers[i+1])])
   
      for i in range(len(self.layers)-1, -1, -1):
         for j in range(self.layers[i]):
            self.weight_deltas[i][self.layers[i]*j:(j+1)*self.layers[i]] += \
            np.asarray([error_terms[i+1][j]*x for x in self.neurons[i]])
            self.bias_deltas[i][j] = error_terms[i][j]
      return error_terms[0]
   
   def update(self):
      for i in range(len(self.layers)):
         self.weights[i] += learning_rate * self.weight_deltas[i]
         self.biases[i] += learning_rate * self.bias_deltas[i]


##@njit()
def train_network(x_train, embedding_weights, embedding_biases, query_weights, key_weights, value_weights):
   lap = 0
   for sentence in x_train:
      decode_weights =  [] #list(map(lambda arr: np.array([z for i in range(2) for z in arr[i::2]]), embedding_weights[::-1]))
      decode_weights.append(np.array([z for i in range(num_hidden_neurons) for z in embedding_weights[1][i::num_hidden_neurons]]))
      decode_weights.append(np.array([z for i in range(dictionary_size) for z in embedding_weights[0][i::dictionary_size]]))
      decode_biases = [np.array([0]*num_hidden_neurons, dtype=float32),np.array([0]*dictionary_size, dtype=float32)]
      embedding_weight_deltas = [np.array([0]*(dictionary_size*num_hidden_neurons), dtype=float32), np.array([0]*(num_hidden_neurons*word_dimensions), dtype=float32)]
      #embedding_bias_deltas = [np.array([0]*num_hidden_neurons, dtype=float32),np.array([0]*word_dimensions, dtype=float32)]
      query_weight_deltas = [np.array([0]*(word_dimensions**2), dtype=float32)]
      key_weight_deltas = [np.array([0]*(word_dimensions**2), dtype=float32)]
      value_weight_deltas = [np.array([0]*(word_dimensions**2), dtype=float32)]
      decode_weight_deltas = embedding_weight_deltas[::-1]
      mean_cost = []
      
      previous_weight_changes = [np.array(784*num_hidden_neurons*[0.0], dtype=float32), np.array(num_hidden_neurons**2*[0.0], dtype=float32), np.array(num_hidden_neurons*10*[0.0], dtype=float32)]
      
      # forward pass ------------------------------------------------------------------------------------------------

      # embedding --------------
      
      embeddings = []
      embedding_neurons = []
      for word in sentence:
         #embedding_weight_deltas = [np.random.uniform(-1, 1, dictionary_size*num_hidden_neurons).astype("float"), np.random.uniform(-1, 1, num_hidden_neurons*word_dimensions).astype("float")]
         #embedding_bias_deltas = [np.array([0]*num_hidden_neurons, dtype=float32),np.array([0]*word_dimensions, dtype=float32)]
         neurons = [np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(word_dimensions*[0.0], dtype=float32)]
         input = [1 if word == x else 0 for x in dictionary]
         # for i in range(num_hidden_neurons):
         #    neurons[0][i] = activation_func(sum([x[0]*x[1] for x in zip(input, embedding_weights[0][i*dictionary_size : dictionary_size*i+dictionary_size])])+embedding_biases[0][i])
         # for i in range(word_dimensions):
         #    neurons[1][i] = activation_func(sum([x[0]*x[1] for x in zip(neurons[0], embedding_weights[1][num_hidden_neurons*i : num_hidden_neurons*i+num_hidden_neurons])])+embedding_biases[1][i])
         neurons = decode_mlp(input, embedding_weights, embedding_biases, (dictionary_size, num_hidden_neurons, word_dimensions))
         embeddings.append(neurons[2])
         embedding_neurons.append(neurons)

         # test -----

         # decode_neurons = decode_mlp(neurons[2], decode_weights, decode_biases, (word_dimensions, num_hidden_neurons, dictionary_size))
         # predictions = softmax(decode_neurons[2])
         # next_word = dictionary[np.argmax(predictions)]
         # print(word, next_word)

      # positional encoding -----

      for i, emb in enumerate(embeddings):
         embeddings[i] += [math.sin(i/10000**(2*j/word_dimensions)) if j%2==0 else math.cos(i/10000**(2*j/word_dimensions)) for j, x in enumerate(emb)]
      
      # attention ---------------

      queries = []
      for emb in embeddings:
         queries.append(np.array([sum(emb*query_weights[0][i*word_dimensions: (i+1)*word_dimensions]) for i in range(word_dimensions)]))
      
      keys = []
      for emb in embeddings:
         keys.append(np.array([sum(emb*key_weights[0][i*word_dimensions: (i+1)*word_dimensions]) for i in range(word_dimensions)]))
      
      values = []
      for emb in embeddings:
         values.append(np.array([sum(emb*value_weights[0][i*word_dimensions: (i+1)*word_dimensions]) for i in range(word_dimensions)]))
      
      softmax_error_terms = np.sum(values, axis=1)

      num_correct = 0
      for train_index in range(len(sentence)-1):
         softmax_values = softmax([sum(queries[train_index]*key) for key in keys[0:train_index+1]])
         attention_values = np.add.reduce([x[0]*x[1] for x in zip(softmax_values, values[0:train_index+1])])
         print(attention_values)
         # residual connections ----
         
         attention_values += embeddings[train_index]

         # decoding ---------------
         decode_neurons = decode_mlp(attention_values, decode_weights, decode_biases, (word_dimensions, num_hidden_neurons, dictionary_size))
         predictions = softmax(decode_neurons[2])
         next_word = dictionary[np.argmax(predictions)]
         #print(next_word)
         if (next_word == sentence[train_index+1]):
            num_correct += 1
            #print(next_word)
         # backpropagation -----------------------------------------------------------------------------------------------

         # decode backpropagation -------------------

         decode_error_terms = [np.array(word_dimensions*[0.0], dtype=float32), np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(dictionary_size*[0.0], dtype=float32)]
         expected_predictions = [1 if x == sentence[train_index+1] else 0 for x in dictionary]
         decode_error_terms[2] = [2*(x[0]-x[1])*x[0]*(1-x[1])*x[2]*(1-x[2]) for x in zip(predictions, expected_predictions, decode_neurons[2])]
         for j in range(num_hidden_neurons):
           decode_error_terms[1][j] = sum([decode_error_terms[2][k]*decode_weights[1][j::num_hidden_neurons][k]*activation_func_derivative(decode_neurons[1][j]) for k in range(dictionary_size)])
         for j in range(word_dimensions):
           decode_error_terms[0][j] = sum([decode_error_terms[1][k]*decode_weights[0][j::word_dimensions][k] for k in range(num_hidden_neurons)])
         for i in range(dictionary_size):
            decode_weight_deltas[1][num_hidden_neurons*i:(i+1)*num_hidden_neurons] += \
            np.asarray([decode_error_terms[2][i]*x for x in decode_neurons[1]])
         for i in range(num_hidden_neurons):
            decode_weight_deltas[0][word_dimensions*i:(i+1)*word_dimensions] += np.asarray([decode_error_terms[1][i]*x for x in decode_neurons[0]])
         
         # attention backpropagation ----------------

         # update softmax error terms of train index to include whole path

         softmax_error_terms[train_index] = sum(softmax_error_terms[train_index] * decode_error_terms[0])

         # value backpropagation

         for i in range(train_index+1):
            for j in range(word_dimensions):
               value_weight_deltas[0][j*word_dimensions:(j+1)*word_dimensions] += decode_error_terms[0][j] * softmax_values[i] * embeddings[i]


         # key backpropagation

         for i in range(train_index+1):
            for j in range(word_dimensions):
               key_weight_deltas[0][j*word_dimensions:(j+1)*word_dimensions] += softmax_error_terms[train_index]*softmax_values[i]*(1-softmax_values[i])*queries[train_index][j]*embeddings[i]
         
         # query backpropagation

         for i in range(train_index+1):
            for j in range(word_dimensions):
               query_weight_deltas[0][j*word_dimensions:(j+1)*word_dimensions] += softmax_error_terms[train_index]*softmax_values[i]*(1-softmax_values[i])*keys[train_index][j]*embeddings[i]
         
         # embedding error terms

         embedding_error_terms = [np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(word_dimensions*[0.0], dtype=float32)]
         
         for i in range(train_index+1):
            for j in range(word_dimensions):
               embedding_error_terms[1] += softmax_error_terms[train_index]*softmax_values[i]*(1-softmax_values[i])*queries[train_index][j]*query_weights[0][j*word_dimensions:(j+1)*word_dimensions]

         for i in range(train_index+1):
            for j in range(word_dimensions):
               embedding_error_terms[1] += softmax_error_terms[train_index]*softmax_values[i]*(1-softmax_values[i])*keys[train_index][j]*key_weights[0][j*word_dimensions:(j+1)*word_dimensions]
         
         for i in range(train_index+1):
            for j in range(word_dimensions):
               embedding_error_terms[1] += decode_error_terms[0][j] * softmax_values[i] * value_weights[0][j*word_dimensions:(j+1)*word_dimensions]
         
         # embedding backpropagation ------------

         for j in range(num_hidden_neurons):
           embedding_error_terms[0][j] = sum([embedding_error_terms[1][k]*embedding_weights[1][j::num_hidden_neurons][k]*activation_func_derivative(embedding_neurons[train_index][1][j]) for k in range(word_dimensions)])
         for i in range(word_dimensions):
            embedding_weight_deltas[1][num_hidden_neurons*i:(i+1)*num_hidden_neurons] += \
            np.asarray([embedding_error_terms[1][i]*x for x in embedding_neurons[train_index][1]])
         for i in range(num_hidden_neurons):
            embedding_weight_deltas[0][dictionary_size*i:(i+1)*dictionary_size] += np.asarray([embedding_error_terms[0][i]*x for x in embedding_neurons[train_index][0]])


         # unite decode and embedding weight deltas ----

         embedding_weight_deltas[1] += np.array([z for i in range(word_dimensions) for z in decode_weight_deltas[0][i::word_dimensions]])
         embedding_weight_deltas[0] += np.array([z for i in range(num_hidden_neurons) for z in decode_weight_deltas[1][i::num_hidden_neurons]])

      # adding weight deltas ----------------
      for i, deltas in enumerate(embedding_weight_deltas):
         embedding_weights[i] -= learning_rate * deltas
      query_weights[0] -= learning_rate * query_weight_deltas[0]
      key_weights[0] -= learning_rate * key_weight_deltas[0]
      value_weights[0] -= learning_rate * value_weight_deltas[0]

      # accuracy --------

      #print(f'{num_correct}/{len(sentence)} correct')
   
   return embedding_weights, query_weights, key_weights, value_weights

  #       error_terms = [np.array(dictionary_size*[0.0], dtype=float32), np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(word_dimensions*[0.0], dtype=float32)]
  #       for i in range(10):
  #         error_terms[2][i] = (2*(neurons[2][i]-answer[i]))*activation_func_derivative(neurons[2][i]) # +2*wd*sum(np.array([y for x in weights for y in x]))       -   neurons[2][i]*(1-neurons[2][i])
  #       for j in range(num_hidden_neurons):
  #         error_terms[1][i] = sum([error_terms[2][k]*weights[2][j::num_hidden_neurons][k]*activation_func_derivative(neurons[1][j]) for k in range(10)])
  #       for j in range(num_hidden_neurons):
  #         error_terms[0][i] = sum([error_terms[1][k]*weights[1][j::num_hidden_neurons][k]*activation_func_derivative(neurons[0][j]) for k in range(num_hidden_neurons)])
  #       for i in range(10):
  #         weight_changes[2][num_hidden_neurons*i:num_hidden_neurons*i+num_hidden_neurons] = \
  #         np.asarray([error_terms[2][i]*x for x in neurons[1]])
  #         bias_changes[2][i] = error_terms[2][i]
  #       for i in range(num_hidden_neurons):
  #         weight_changes[1][num_hidden_neurons*i:num_hidden_neurons*i+num_hidden_neurons] = \
  #         np.asarray([error_terms[1][i]*x for x in neurons[0]])
  #         bias_changes[1][i] = error_terms[1][i]
  #       for i in range(num_hidden_neurons):
  #         weight_changes[0][784*i:784*i+784] = np.asarray([error_terms[0][i]*x for x in img[0]])
  #         bias_changes[0][i] = error_terms[0][i]
  #       mean_cost.append(sum([x**2 for x in answer-neurons[2]]))
  #       for i in range(3):
  #         imgs_weight_changes[i] += weight_changes[i]+2*wd*weights[i]+momentum*previous_weight_changes[i]
  #         imgs_bias_changes[i] += bias_changes[i]
  #       previous_weight_changes = weight_changes
  #     lap += 1
  #     if (lap%(60000/batch_size/10) == 0):
  #        print("\r", round(lap/(60000/batch_size)*100), "%   -   cost",  round(sum(mean_cost)/len(mean_cost),3))
  #     for i in range(3):
  #        weights[i] -= learning_rate*(imgs_weight_changes[i]/batch_size)
  #        biases[i] -= learning_rate*(imgs_bias_changes[i]/batch_size)
  #  return weights, biases

# #@njit()
# def test_network(weights, biases):
#    correct = 0
#    lap = 0
#    for img in zip(x_test, y_test):
#       lap += 1
#       neurons = [np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(10*[0.0], dtype=float32)]
#       for i in range(num_hidden_neurons):
#          neurons[0][i] = activation_func(sum([x[0]*x[1] for x in zip(img[0], weights[0][i*784 : 784*i+784])])+biases[0][i])
#       for i in range(num_hidden_neurons):
#          neurons[1][i] = activation_func(sum([x[0]*x[1] for x in zip(neurons[0], weights[1][num_hidden_neurons*i : num_hidden_neurons*i+num_hidden_neurons])])+biases[1][i])
#       for i in range(10):
#          neurons[2][i] = activation_func(sum([x[0]*x[1] for x in zip(neurons[1], weights[2][num_hidden_neurons*i : num_hidden_neurons*i+num_hidden_neurons])])+biases[2][i])
#       if (img[1] == np.argmax(neurons[2])):
#          correct += 1
#    print("\nAccuracy  -  ", round(correct/lap*100, 3), "%\n")
#    return correct/lap*100

#@jit(nopython=True)s
def neural_network():
   embedding_weights = [np.random.uniform(-1, 1, dictionary_size*num_hidden_neurons).astype("float"), np.random.uniform(-1, 1, num_hidden_neurons*word_dimensions).astype("float")]
   embedding_biases = [np.array([0]*num_hidden_neurons, dtype=float32),np.array([0]*word_dimensions, dtype=float32)]
   query_weights = [np.random.uniform(-1, 1, word_dimensions**2).astype("float")]
   key_weights = [np.random.uniform(-1, 1, word_dimensions**2).astype("float")]
   value_weights = [np.random.uniform(-1, 1, word_dimensions**2).astype("float")]
   last_accuracy = 0
   for i in range(epochs):
      embedding_weights, query_weights, key_weights, value_weights = train_network(x_train, embedding_weights, embedding_biases, query_weights, key_weights, value_weights)
      #new_accuracy = test_network(weights, biases)
  #     if new_accuracy - last_accuracy < 0.5:
  #        break
  #     last_accuracy = new_accuracy
  #  print("BEST ACCURACY:", round(last_accuracy,3), "%")
  #  playsound("notification.wav")
   

neural_network()