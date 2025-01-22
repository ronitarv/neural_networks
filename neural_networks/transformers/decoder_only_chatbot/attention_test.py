from neural_networks.components.attention import Attention, softmax
from neural_networks.components.mlp import Mlp, activation_func_derivative
from neural_networks.components.positional_encoding import PositionalEncoding
from neural_networks.components.layer_normalization import LayerNormalization
from neural_networks.components.network import Network
import numpy as np
from numpy import float64 as float32
#layers = (Mlp((2,2),0.01,0,0), PositionalEncoding(2), Attention(1, 2, 0.01,True), Mlp((2,2),0.01,0,0))
lr = 0.01
dimensions = 6
layers = (Attention(2, dimensions, lr,True),)
#layers = (LayerNormalization(dimensions, lr),)
network = Network(layers)
#network = Network((Attention(1, 2, 0.01,True),))
#network = Mlp((4,10,4), 0.1, 0.00001, 0, False)
#inputs = [np.array([0.005, 0.49, 0.2])]
inputs = np.array([np.array([0.005,0.49, 0.35, 0.6, 0.35, 0.6]*int(dimensions/6)),np.array([0.45,0.71, 0.27, 0.2, 0.35, 0.6]*int(dimensions/6)),np.array([0.26, 0.653, 0.1, 0.1, 0.35, 0.6]*int(dimensions/6))], dtype=float32)
#inputs = np.array([np.array([2.38,1.10]*int(dimensions/2)),np.array([1.45,0.71]*int(dimensions/2)), np.array([0.43,1.34]*int(dimensions/2))], dtype=np.float32)

while True:
    output = network.forward(inputs)
    
    #error_terms = [np.array([-0.2, -0.2, -0.2, -0.2]*int(dimensions/2)),np.array([-0.2, -0.2, -0.2, -0.2]*int(dimensions/2))]
    #answer = [np.array([0.2, 0, 0.73])]
    #answer = np.array([np.array([0, 0]*int(dimensions/2)),np.array([0.3, 0]*int(dimensions/2)),np.array([0.004, 0.3]*int(dimensions/2))])
    answer = np.array([np.array([0.13, 0.54, 0.2, 0.45, 0.2, 0.45]),np.array([0.3, 0, 0.43, 0.12, 0.2, 0.45]), np.array([0.7, 0.13, 0.23, 0.5, 0.2, 0.45])], dtype=float32)
    print(f'{output[1]} / {output[2]}   -   {np.sum((answer-output)**2):.06f}')
    #print(f'{output[2]}   {np.sum((answer-output)**2):.06f}    {layers[0].weight_sum():.06f} - {layers[1].weight_sum():.06f} - {layers[2].weight_sum():.06f}')
    error_terms = -2*(answer-output)#2*(answer-output)*activation_func_derivative(output)
    gradient = network.backpropagate(error_terms)
    #print(f'[{output[0][0]: .05f} {output[0][1]: .05f}] [{output[1][0]: .05f} {output[1][1]: .05f}] [{output[2][0]: .05f} {output[2][1]: .05f}]        -      {layers[-1].weight_sum()}')
    #print(gradient)
    network.update()
    #break
print()