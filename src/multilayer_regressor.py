#!/usr/bin/env python

from __future__ import division

import numpy as np
import sklearn as skl

class multilayer_regressor(skl.base.RegressorMixin):
  """ This is a multilayer regression estimator. """

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

  def train(self, X, y, num_iter=100, lmbda=0, verbose=False):
    c = [self.score(X, y)]
    m, n = X.shape
    if verbose:
      print "Init score: %f" % c[-1]
    for i in range(num_iter):
      grad = self.theta_gradients(X, y)
      k = len(self.theta_lst)
      for j in range(k):
        self.theta_lst[j] -= (grad[j] + (lmbda/n) * self.theta_lst[j])
      c.append(self.score(X, y))
      if verbose:
        print "Progress: %d / %d, Score: %f" % (i+1, num_iter, c[-1])
    return np.array(c, dtype=np.float64)

  def get_activation_layers(self, X):
    activation_layers = [X]
    for theta in self.theta_lst:
      activation_layers.append(np.dot(theta, activation_layers[-1]))
    return activation_layers

  def theta_gradients(self, X, y):
    activation_layers = self.get_activation_layers(X)
    delta = - (y - activation_layers[-1]);
    activation_layers.pop()
    grad = []
    past_theta = None
    for theta in self.theta_lst[::-1]:
      if past_theta is None:
        alpha = 0.9 * (1/(2 * np.linalg.norm(activation_layers[-1], ord=2)**2))
        g = alpha * np.dot(delta, activation_layers[-1].conj().T)
        past_theta = theta
      else:
        alpha = 0.9 * (1/(2 * np.linalg.norm(np.dot(past_theta.conj().T, activation_layers[-1]), ord=2)**2))
        g = alpha * np.dot(past_theta.conj().T, np.dot(delta, activation_layers[-1].conj().T))
        past_theta = np.dot(past_theta, theta)
      activation_layers.pop()
      grad.insert(0, g)
    return grad

  def score(self, X, y, sample_weigth=None):
    y_pred = self.get_activation_layers(X)[-1]
    u = np.linalg.norm(y - y_pred)**2
    v = np.linalg.norm(y - y.mean())**2
    return 1 - u/v

if __name__ == '__main__': 
  import matplotlib.pyplot as plt
  from sklearn import cross_validation as cv
  from sklearn import datasets

  iris = datasets.load_iris()
  X_train, X_test, y_train, y_test = cv.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
  mr = multilayer_regressor([X_train.shape[1], 1])
  c = mr.train(X_train.T, y_train.T)
  print "Score on cross-validation set: %f" % mr.score(X_test.T, y_test.T)
  plt.figure()
  plt.plot(abs(c))
  plt.title('Score per iteration')
  plt.xlabel('Iteration')
  plt.ylabel('Score')
  plt.show()
