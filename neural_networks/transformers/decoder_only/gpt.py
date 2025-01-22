#from tensorflow.keras.datasets import mnist
import numpy as np
from numba import njit
import math
from numpy import float64 as float32
#from playsound import playsound
import math
# from components.mlp import Mlp
from neural_networks.components.mlp import Mlp
from neural_networks.components.linear import Linear
from neural_networks.components.attention import Attention, softmax
from neural_networks.components.layer_normalization import LayerNormalization
from neural_networks.components.positional_encoding import PositionalEncoding
from neural_networks.components.network import Network
from neural_networks.components.attention import softmax_derivative
from nltk.tokenize import word_tokenize
import pickle

# 28*28 = 784
# 784*16 = 12544
# 784*16+16*16 = 12800

num_hidden_neurons = 20
batch_size = 2
lr = 0.00001
wd = 0.00001 # 0.0000001 # 0.000001
momentum = 0.7
epochs = 20000
dimensions = 5
clipnorm = 10.0
num_heads = 1

tokenize = False

if tokenize:
  with open("neural_networks/transformers/decoder_only_chatbot/train.csv", "r") as file:
    data = file.readlines()
  data = list(filter(lambda x: x != "\n", data))
  print("start")
  x_train = list(map(lambda x: word_tokenize(x), data[1:2]))
  print("end")

  with open("tiny_stories_dataset_test", "wb") as fp:
    pickle.dump(x_train, fp)

print("loading dataset...")

with open("neural_networks/transformers/decoder_only_chatbot/tiny_stories_dataset_test", "rb") as fp:
  x_train = pickle.load(fp)

x_train = list(map(lambda x: np.array(x), x_train))
x_train = list(filter(lambda x: len(x)>1, x_train))
#x_train = x_train[:1000]
x_train[0] = x_train[0][1:5]
#x_train.append(x_train[0]) # test

dictionary = list(set([y for x in x_train for y in x]))
dictionary_size = len(dictionary)

print("dataset loaded")


layers = [
  Mlp(shape=np.array([dictionary_size, dimensions]), lr=lr, weight_decay=wd, momentum=momentum, has_residual_connections=False, clipnorm=clipnorm),
  PositionalEncoding(dimensions=dimensions),
  Attention(num_heads=num_heads, dimensions=dimensions, lr=lr, has_residual_connections=True, clipnorm=clipnorm),
  LayerNormalization(dimensions, lr, clipnorm=clipnorm),
  #Mlp(layers=(dimensions, int(dimensions/2), dimensions), lr=lr, has_residual_connections=True),
  #LayerNormalization(),
  Mlp(shape=np.array([dimensions, dictionary_size]), lr=lr, weight_decay=wd, momentum=momentum, has_residual_connections=False, clipnorm=clipnorm)
]

network = Network(layers=layers)

print("gpt training started")
num_epochs = 20
batch_size = 10

def gpt():
  last_cost = 0
  counter = 0
  while True:
    for epoch in range(num_epochs):
      for i, sentence in enumerate(x_train):
    # for batch in range(int(len(x_train)/batch_size)):
    #   for sentence in x_train[batch*batch_size:(batch+1)*batch_size]:
        #for sentence in sentences:
        inputs = np.array([np.array([1 if word == x else 0 for x in dictionary], dtype=float32) for word in sentence][:-1])
        outputs = network.forward(inputs)
        softmax_outputs = [softmax(x) for x in outputs]
        expected_outputs = np.array([np.array([1 if word == x else 0 for x in dictionary], dtype=float32) for word in sentence[1:]])
        #print("cost:", np.mean([sum((x[0]-x[1])**2) for x in zip(softmax_outputs, expected_outputs)]))
        cost = np.mean([sum((x[0]-x[1])**2) for x in zip(softmax_outputs, expected_outputs)])
        #print(f'cost: {cost:.8f}  -  {cost-last_cost:+.8f}    /    mlp: {layers[0].weight_sum():.8f}     /      attention: {layers[2].weight_sum():.8f}      /        mlp: {layers[-1].weight_sum():.8f}')
        #error_terms = [2*(x[0]-x[1])*x[0]*(1-x[1])*x[2]*(1-x[2]) for x in zip(softmax_outputs, expected_outputs, outputs)]
        error_terms = -2*(expected_outputs-softmax_outputs)
        softmax_error_terms = np.array([softmax_derivative(output, error_term) for output, error_term in zip(outputs, error_terms)])
        #error_terms = [-2*(x[1]-x[0])*x[0]*(1-x[0]) for x in zip(softmax_outputs, expected_outputs, outputs)]
        #print(f'soft output: {softmax_outputs}  /  soft expected: {expected_outputs}')
        #print(outputs[-1])
        network.backpropagate(softmax_error_terms)
        counter += (np.argmax(softmax_outputs,axis=1)==np.argmax(expected_outputs,axis=1)).sum()
        if i % batch_size == 0:
          print(f'cost: {cost:.8f}    -    right: {counter}/{(len(x_train[0])-1)*batch_size}        -     update')
          counter = 0
          network.update()
        else:
          print(f'cost: {cost:.8f}    -    right: {counter}/{(len(x_train[0])-1)*batch_size}')
        #last_cost = cost
        
        
    #print("---------------------- next sentence ------------------------")
        
    #print(f'cost: {cost:.8f}    -    right: {counter}/{num_epochs*(len(x_train[0])-1)*len(x_train)}')
    counter = 0

gpt()

# expected_predictions = [1 if x == sentence[train_index+1] else 0 for x in dictionary]
# decode_error_terms[2] = [2*(x[0]-x[1])*x[0]*(1-x[1])*x[2]*(1-x[2]) for x in zip(predictions, expected_predictions, decode_neurons[2])]