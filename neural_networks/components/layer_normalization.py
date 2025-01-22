import numpy as np
from numpy import float64 as float32
import torch
import torch.nn as nn

class LayerNormalization:
  def __init__(self, dimensions, lr, clipnorm=1.0):
    self.dimensions = dimensions
    self.embeddings = None
    self.lr = lr
    self.clipnorm = clipnorm
    self.batch_size = 1
    self.outputs = None
    # self.gamma = np.ones((dimensions))
    # self.beta = np.zeros((dimensions))
    self.gamma = torch.ones(dimensions, dtype=torch.float32, requires_grad=True)
    self.beta = torch.zeros(dimensions, dtype=torch.float32, requires_grad=True)
    self.delta_gamma = np.zeros((dimensions))
    self.delta_beta = np.zeros((dimensions))
    self.layer_norm = nn.LayerNorm(dimensions)
    self.optimizer = torch.optim.Adam([self.gamma, self.beta], lr=lr)
  def forward(self, embeddings):
    """return outputs"""
    embeddings = embeddings.astype(np.float32)
    tensor_embeddings = torch.from_numpy(embeddings)
    tensor_embeddings.requires_grad = True
    self.embeddings = tensor_embeddings
    self.outputs = self.layer_norm(tensor_embeddings) * self.gamma + self.beta
    # self.embeddings = np.array(embeddings)
    # eps = 1e-05
    # outputs = []
    # for emb in embeddings:
    #   outputs.append(((emb-emb.mean())/np.sqrt(emb.var()+eps))*self.gamma+self.beta)
      #print()
    #self.outputs = outputs
    #outputs = embeddings - np.mean(embeddings, axis=1).reshape(len(embeddings),1) # list(map(lambda emb: (emb-np.mean(emb))/np.std(emb), embeddings))
    #outputs = list(map(lambda emb: (emb-np.mean(emb))/np.std(emb), embeddings))
    np_outputs = self.outputs.detach().numpy()
    return np_outputs.astype(float32)
  
  # def backpropagate(self, path_error_terms):
  #   """return path_error_terms"""
  #   new_path_error_terms = []
  #   for train_index in range(len(path_error_terms)):
  #     emb = self.embeddings[train_index]
  #     N = emb.shape[0]
  #     I = np.eye(N)
  #     mean = emb.std(axis = 0)
  #     std = emb.std(axis = 0)
  #     #new_path_error_terms.append(np.mean(((N * I -1) / (N * std + 10**-100)) - (( (emb - mean) .dot((emb - mean).T) ) / (N * std**3 + 10**-100)), axis=1)*path_error_terms[train_index])
  #     new_path_error_terms.append(np.sum((((N * I -1) / (N * std + 10**-100)) - (( (emb - mean) .dot((emb - mean).T) ) / (N * std**3 + 10**-100))).transpose()*path_error_terms[train_index]*self.gamma,axis=1)/10000)
  #     self.delta_beta += path_error_terms[train_index]
  #     self.delta_gamma += path_error_terms[train_index]*((emb-emb.mean())/emb.std())
  #   self.embeddings = None
  #   #self.outputs = None
  #   # self.outputs = None
  #   #print(f'ln  old / new path error terms   -   {np.mean(np.absolute(np.absolute(path_error_terms)))} / {np.mean(np.absolute(np.absolute(new_path_error_terms)))}')
  #   return new_path_error_terms
  
  def backpropagate(self, path_error_terms):
    """return path_error_terms"""
    self.optimizer.zero_grad()
    path_error_terms = path_error_terms.astype(np.float32)
    tensor_path_error_terms = torch.from_numpy(path_error_terms)
    self.outputs.backward(tensor_path_error_terms)
    self.optimizer.step()
    np_new_path_error_terms = self.embeddings.grad.detach().numpy()
    return np_new_path_error_terms.astype(float32)
    # B, H = len(path_error_terms), self.dimensions
    # eps = 1e-05s
    # x=self.embeddings # np.array([[0.005, 0.49, 0.2]])
    # #gamma = np.array([1.0,1.0,1.0])
    # #beta = np.array([0.0,0.0,0.0])
    # dldy = np.array(path_error_terms) # -2*(0 - x)

    # #y = path_error_terms # np.array([[-1.1374,  1.2963, -0.1589]])

    # m = x.mean(axis=1)
    # mu = x - m.reshape(len(path_error_terms),1)
    # v = np.mean(mu**2, axis=1)
    # sigma = np.sqrt(v + eps)

    # self.delta_gamma += np.einsum('bq,bq,b->q', dldy, mu, 1/sigma)
    # self.delta_beta += dldy.sum(axis=0)

    # dldx = (
    #   dldy*np.expand_dims(self.gamma,axis=0) / np.expand_dims(sigma,axis=1)
    #   - 1/H * np.expand_dims(np.einsum('ph,h,p->p', dldy, self.gamma, 1/sigma),axis=1)
    #   - 1/H * mu * np.expand_dims(np.einsum('ph,h,ph,p->p', dldy, self.gamma, mu, sigma**(-3)),axis=1)
    # )

    # return dldx

  def update(self):
    pass
    #self.optimizer.step()
    # self.gamma -= self.lr * ((self.clipnorm * (self.delta_gamma / np.linalg.norm(self.delta_gamma / self.batch_size))) 
    #                                    if (np.linalg.norm(self.delta_gamma / self.batch_size) > self.clipnorm) 
    #                                    else (self.delta_gamma / self.batch_size))
    # self.beta -= self.lr * ((self.clipnorm * (self.delta_beta / np.linalg.norm(self.delta_beta / self.batch_size))) 
    #                                    if (np.linalg.norm(self.delta_beta / self.batch_size) > self.clipnorm) 
    #                                    else (self.delta_beta / self.batch_size))
    # self.delta_gamma = np.zeros((self.dimensions))
    # self.delta_beta = np.zeros((self.dimensions))

  def __str__(self):
      return "Layer normalization"
  


# ln = LayerNormalization()
# print("start")
# for i in range(100):
#   ln.forward([np.array([1,5,3], dtype=float32), np.array([5,6,7], dtype=float32)])
#   ln.backpropagation([np.array([1,2,3]), np.array([6,5,4])])
# print("end")