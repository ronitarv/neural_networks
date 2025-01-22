import math
import numpy as np

def positional_encoding(position, d_model):
 
    # Create a matrix of shape [position, d_model] where each element is the position index
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    
    # Apply sine to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Apply cosine to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding

class PositionalEncoding:
  def __init__(self, dimensions):
    self.dimensions = dimensions

  def forward(self, embeddings):
    """return embeddings"""
    embeddings += positional_encoding(len(embeddings), self.dimensions)[0]
    # for i, emb in enumerate(embeddings):
    #   embeddings[i] += np.array([math.sin(i/10000**(2*j/self.dimensions)) if j%2==0 else math.cos(i/10000**(2*j/self.dimensions)) for j, x in enumerate(emb)]) / 1
    return embeddings
  
  def backpropagate(self, path_error_terms):
    """return path_error_terms"""
    return path_error_terms
  
  def __str__(self):
    return "Positional encoding"