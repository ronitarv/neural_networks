from tensorflow.keras.datasets import mnist
import numpy as np
from numba import njit
import math
from numpy import float32
#from playsound import playsound
import sys
sys.path.append("/home/ronit/codes/python/machine_learning/neural_networks/mlp/stocks_predictor")
from neural_networks.components.mlp import Mlp, activation_func_derivative
from neural_networks.components.network import Network
#from neural_networks.components.mlp import Mlp

# 28*28 = 784
# 784*16 = 12544
# 784*16+16*16 = 12800

num_hidden_neurons = 20
batch_size = 100
lr = 0.1 # best 0.1
wd = 0.0001 # 1e-6 # 0.0000001 # 0.000001 # best 0.001
momentum = 0.7 # 0.9
epochs = 20
num_cost_prints = 1000

# 77.34
# 80 hn200
# 85.22 hn300
# 87,26 hn532 bs100

# best batch size for 50 hidden neurons is 80


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape = (60000, 28*28)
x_train = np.array([x/255 for x in x_train])
# x_train.shape = (int(60000/batch_size),batch_size, 28*28)
# y_train.shape = (int(60000/batch_size),batch_size)
x_test.shape = (10000, 28*28)
x_test = np.array([x/255 for x in x_test])

# #@njit()
# def activation_func_derivative(a):
#    return a*(1-a)
#    #return 1 if a > 0 else 0

# #@njit()
# def activation_func(x):
#    return 1 / (1 + math.exp(-x))
#    #return max(0.0, x)

# #@njit()
# def mlp_forward(input, layers, weights, biases):
#    neurons = [np.array([0]*l, dtype=float32) for l in layers[1::]]
#    neurons.insert(0, input)
#    for i, l in enumerate(layers[0:-1]):
#       for j in range(layers[i+1]):
#          neurons[i+1][j] = activation_func(sum([x[0]*x[1] for x in zip(neurons[i], weights[i][j*l : (j+1)*l])])+biases[i][j])
#    return neurons

# #@njit()
# def mlp_backpropagate(layers, weights, path_error_terms, neurons, weight_deltas, bias_deltas):
#    error_terms = [np.array([0]*l, dtype=float32) for l in layers[0:-1]]
#    error_terms.append(path_error_terms)
#    for i in range(len(layers)-1, 0, -1):
#       for j in range(layers[i-1]):
#          error_terms[i-1][j] = np.sum(error_terms[i]*weights[i-1][j::layers[i-1]]*activation_func_derivative(neurons[i-1][j]))
#    for i in range(len(layers)-1, 0, -1):
#       for j in range(layers[i]):
#          weight_deltas[i-1][layers[i-1]*j:(j+1)*layers[i-1]] += \
#          np.asarray([error_terms[i][j]*x for x in neurons[i-1]])
#          bias_deltas[i-1][j] += error_terms[i][j]
#    return weight_deltas, bias_deltas, error_terms

# class Mlp:
#    def __init__(self, layers, learning_rate):
#       self.layers = layers
#       self.lr = learning_rate
#       self.weights = [np.random.uniform(-1, 1, layers[i]*layers[i+1]).astype("float") for i in range(len(layers)-1)]
#       self.biases = [np.array([0]*l, dtype=float32) for l in layers[1::]]
#       self.weight_deltas = [np.array([0]*(layers[i]*layers[i+1]), dtype=float32) for i in range(len(layers)-1)]
#       self.bias_deltas = [np.array([0]*l, dtype=float32) for l in layers[1::]]
#       self.batch_size = 0
#       self.neurons = None
   
#    def forward(self, input):
#       if len(input) != self.layers[0]:
#          raise Exception("size of provided and defined input do not match")
#       neurons = mlp_forward(input, self.layers, self.weights, self.biases)
#       self.neurons = neurons
#       return neurons[-1]
   
#    def backpropagate(self, path_error_terms):
#       weight_deltas, bias_deltas, error_terms = mlp_backpropagate(self.layers, self.weights, path_error_terms, self.neurons, self.weight_deltas, self.bias_deltas)
#       self.weight_deltas = weight_deltas
#       self.bias_deltas = bias_deltas
#       self.batch_size += 1
#       self.neurons = None
#       return error_terms[0]
   
#    def update(self):
#       for i in range(len(self.layers)-1):
#          self.weights[i] -= self.lr * (self.weight_deltas[i] / self.batch_size)
#          self.biases[i] -= self.lr * (self.bias_deltas[i] / self.batch_size)
#       self.weight_deltas = [np.array([0]*(self.layers[i]*self.layers[i+1]), dtype=float32) for i in range(len(self.layers)-1)]
#       self.bias_deltas = [np.array([0]*l, dtype=float32) for l in self.layers[1::]]
#       self.batch_size = 0

##@njit()
def train_network(network: Network, x_train, y_train):
   lap = 0
   for imgs in zip(x_train, y_train):
      # imgs_weight_changes = [np.array(784*num_hidden_neurons*[0.0], dtype=float32), np.array(num_hidden_neurons**2*[0.0], dtype=float32), np.array(num_hidden_neurons*10*[0.0], dtype=float32)]
      # imgs_bias_changes = [np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(10*[0.0], dtype=float32)]
      mean_cost = []
      
      #previous_weight_changes = [np.array(784*num_hidden_neurons*[0.0], dtype=float32), np.array(num_hidden_neurons**2*[0.0], dtype=float32), np.array(num_hidden_neurons*10*[0.0], dtype=float32)]
      output = network.forward(imgs[0].astype(float32))
      answer = np.array([np.eye(1, 10, k=i)[0] for i in imgs[1]])
      error_terms = -2*(answer-output)
      #for img in zip(outputs, imgs[1]):
      #   output = img[0]
      #   answer = np.array([0.0] * 10, dtype=float32)
      #   answer[img[1]] = 1
        
        #[2*(x[0]-x[1])*x[0]*(1-x[1])*x[2]*(1-x[2]) for x in zip(output, answer, decode_neurons[2])]
        #for i in range(10):
        #error_terms.append(2*(output-answer)*activation_func_derivative(output))
        #mean_cost.append(sum([x**2 for x in answer-output]))
      mean_cost = np.mean(np.sum((answer-output)**2, axis=1))
      network.backpropagate(error_terms)
      network.update()
      lap += 1
      #print("\r", round(lap/(60000/batch_size)*100), "%   -   cost",  round(sum(mean_cost)/len(mean_cost),3))
      #if (lap%(60000/batch_size/num_cost_prints) == 0):
      print("\r", round(lap/(60000/batch_size)*100), "%   -   cost",  round(mean_cost, 3))
        #  neurons = [np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(10*[0.0], dtype=float32)]
        #  for i in range(num_hidden_neurons):
        #     neurons[0][i] = activation_func(sum([x[0]*x[1] for x in zip(img[0], weights[0][i*784 : 784*i+784])])+biases[0][i])
        #  for i in range(num_hidden_neurons):
        #     neurons[1][i] = activation_func(sum([x[0]*x[1] for x in zip(neurons[0], weights[1][num_hidden_neurons*i : num_hidden_neurons*i+num_hidden_neurons])])+biases[1][i])
        #  for i in range(10):
        #     neurons[2][i] = activation_func(sum([x[0]*x[1] for x in zip(neurons[1], weights[2][num_hidden_neurons*i : num_hidden_neurons*i+num_hidden_neurons])])+biases[2][i])
        #  answer = np.array([0.0] * 10, dtype=float32)
        #  answer[img[1]] = 1
        #  weight_changes = [np.array(784*num_hidden_neurons*[0.0], dtype=float32), np.array(num_hidden_neurons**2*[0.0], dtype=float32), np.array(num_hidden_neurons*10*[0.0], dtype=float32)]
        #  bias_changes = [np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(10*[0.0], dtype=float32)]
        #  error_terms = [np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(10*[0.0], dtype=float32)]
        #  for i in range(10):
        #     error_terms[2][i] = (2*(neurons[2][i]-answer[i]))*activation_func_derivative(neurons[2][i]) # +2*wd*sum(np.array([y for x in weights for y in x]))       -   neurons[2][i]*(1-neurons[2][i])
        #  for j in range(num_hidden_neurons):
        #     error_terms[1][j] = sum([error_terms[2][k]*weights[2][j::num_hidden_neurons][k]*activation_func_derivative(neurons[1][j]) for k in range(10)])
        #  for j in range(num_hidden_neurons):
        #     error_terms[0][j] = sum([error_terms[1][k]*weights[1][j::num_hidden_neurons][k]*activation_func_derivative(neurons[0][j]) for k in range(num_hidden_neurons)])
        #  for i in range(10):
        #     weight_changes[2][num_hidden_neurons*i:num_hidden_neurons*i+num_hidden_neurons] = \
        #     np.asarray([error_terms[2][i]*x for x in neurons[1]])
        #     bias_changes[2][i] = error_terms[2][i]
        #  for i in range(num_hidden_neurons):
        #     weight_changes[1][num_hidden_neurons*i:num_hidden_neurons*i+num_hidden_neurons] = \
        #     np.asarray([error_terms[1][i]*x for x in neurons[0]])
        #     bias_changes[1][i] = error_terms[1][i]
        #  for i in range(num_hidden_neurons):
        #     weight_changes[0][784*i:784*i+784] = np.asarray([error_terms[0][i]*x for x in img[0]])
        #     bias_changes[0][i] = error_terms[0][i]
        
        #  for i in range(3):
        #     imgs_weight_changes[i] += weight_changes[i]+2*wd*weights[i]+momentum*previous_weight_changes[i]
        #     imgs_bias_changes[i] += bias_changes[i]
        #  previous_weight_changes = weight_changes
      
      # for i in range(3):
      #    weights[i] -= learning_rate*(imgs_weight_changes[i]/batch_size)
      #    biases[i] -= learning_rate*(imgs_bias_changes[i]/batch_size)
   #return weights, biases

##@njit()
def test_network(network: Network):
   correct = 0
   lap = 0
   outputs = network.forward(x_test)
   for img in zip(outputs, y_test):
      lap += 1
      output = img[0]
      # neurons = [np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(num_hidden_neurons*[0.0], dtype=float32), np.array(10*[0.0], dtype=float32)]
      # for i in range(num_hidden_neurons):
      #    neurons[0][i] = activation_func(sum([x[0]*x[1] for x in zip(img[0], weights[0][i*784 : 784*i+784])])+biases[0][i])
      # for i in range(num_hidden_neurons):
      #    neurons[1][i] = activation_func(sum([x[0]*x[1] for x in zip(neurons[0], weights[1][num_hidden_neurons*i : num_hidden_neurons*i+num_hidden_neurons])])+biases[1][i])
      # for i in range(10):
      #    neurons[2][i] = activation_func(sum([x[0]*x[1] for x in zip(neurons[1], weights[2][num_hidden_neurons*i : num_hidden_neurons*i+num_hidden_neurons])])+biases[2][i])
      if (img[1] == np.argmax(output)):
         correct += 1
   print("\nAccuracy  -  ", round(correct/lap*100, 3), "%\n")
   return correct/lap*100

#@jit(nopython=True)s
def neural_network():
  #  weights = [np.random.uniform(-1, 1, 784*num_hidden_neurons).astype("float"), np.random.uniform(-1, 1, num_hidden_neurons**2).astype("float"), np.random.uniform(-1, 1, num_hidden_neurons*10).astype("float")]
  #  biases = [np.array([0]*num_hidden_neurons, dtype=float32),np.array([0]*num_hidden_neurons, dtype=float32),np.array([0]*10, dtype=float32)]
  last_accuracy = 0
  #layers = (Mlp((784,num_hidden_neurons,num_hidden_neurons,10), lr, wd, momentum),)
  layers = (Mlp(np.array([784,num_hidden_neurons, num_hidden_neurons, 10]), lr, wd, momentum, clipnorm=5.0), )
  network = Network(layers)
  #network = Mlp((784,num_hidden_neurons,num_hidden_neurons,10), lr, wd, momentum)
  for i in range(epochs):
    print("epoch", i+1)
    permutation = np.random.permutation(len(x_train))
    new_x_train = np.copy(x_train)[permutation]
    new_y_train = np.copy(y_train)[permutation]
    new_x_train.shape = (int(60000/batch_size),batch_size, 28*28)
    new_y_train.shape = (int(60000/batch_size),batch_size)
    train_network(network, new_x_train, new_y_train)
    new_accuracy = test_network(network)
    if new_accuracy - last_accuracy < 0.5:
        break
    last_accuracy = new_accuracy
  print("BEST ACCURACY:", round(last_accuracy,3), "%")
  #playsound("notification.wav")
  

neural_network()