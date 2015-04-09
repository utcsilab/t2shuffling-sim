#!/usr/bin/env python

from __future__ import division

import numpy as np

class NN_simple:
  """ This takes an input/ output pair and tries to predict y
      given an input X. This is not a classifier. """

  def __init__(self, nodes_per_layer):
    self.theta_lst = []
    for i in range(1, len(nodes_per_layer)):
      prev, curr = nodes_per_layer[i-1], nodes_per_layer[i]
      self.theta_lst.append(np.random.rand(curr, prev).astype(np.complex64))

  def load_theta(self, theta_path):
    """ This assumes that theta list is saved as an npy file """
    theta_lst = np.load(theta_path)
    for i in range(len(theta_lst)):
      self.theta_lst[i] = theta_lst[i]

  def save_theta(self, path):
    """ This saves the theta_lst as an npy """
    np.save(path, np.array(self.theta_lst)) 

  def train(self, X, y, alpha=0.3, num_iter=100, lmbda=0, verbose=False):
    c = [self.cost(X, y, lmbda)]
    m, n = X.shape
    if verbose:
      print "Init cost: %f" % c[-1]
    for i in range(num_iter):
      grad = self.theta_gradients(X, y)
      k = len(self.theta_lst)
      for j in range(k):
        self.theta_lst[j] -= alpha * (grad[j] + (lmbda/n) * self.theta_lst[j])
      c.append(self.cost(X, y))
      if verbose:
        print "Progress: %d / %d, Cost: %f" % (i+1, num_iter, c[-1])
    return np.array(c, dtype=np.float64)

  def get_activation_layers(self, X):
    activation_layers = [X]
    for theta in self.theta_lst:
      activation_layers.append(np.dot(theta, activation_layers[-1]))
    return activation_layers

  def cost(self, X, y, lmbda=0):
    h = self.get_activation_layers(X)[-1]
    t = sum([np.linalg.norm(theta, 'fro') for theta in self.theta_lst])
    reg = (lmbda/2) * t
    return ((np.linalg.norm(y - h, 'fro')/np.linalg.norm(y, 'fro')) + reg) * 100

  def theta_gradients(self, X, y):
    activation_layers = self.get_activation_layers(X)
    delta = - (y - activation_layers[-1]);
    activation_layers.pop()
    grad = []
    past_theta = None
    for theta in self.theta_lst[::-1]:
      if past_theta is None:
        g = np.dot(delta, activation_layers[-1].conj().T)
        past_theta = theta
      else:
        g = np.dot(past_theta.conj().T, np.dot(delta, activation_layers[-1].conj().T))
        past_theta = np.dot(past_theta, theta)
      activation_layers.pop()
      grad.insert(0, g)
    return grad

if __name__ == '__main__': 
  nn = NN_simple([2, 4, 2])
  X = np.array(np.matrix('[1 2 ; 2 3 ; 3 2; 2 1; 5 4; 7 8  ; 3 1; 6  4;  7 5]').T)
  y = np.array(np.matrix('[3 -1; 5 -1; 5 1; 3 1; 9 1; 15 -1; 4 2; 10 2; 12 2]').T)
  c = nn.train(X, y, num_iter=1000, alpha=0.00005, verbose=True)
  import matplotlib.pyplot as plt
  plt.figure()
  plt.plot(abs(c))
  plt.title('Cost per iteration')
  plt.xlabel('Iteration')
  plt.ylabel('Cost')
  plt.show()
