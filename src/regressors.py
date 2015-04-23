#!/usr/bin/env python


from __future__ import division


import numpy   as np
import sklearn as skl
import sys


norm = np.linalg.norm


def sigmoid(x):
  return 1/(1 + np.exp(-x))


def sigmoid_gradient(x):
  g = sigmoid(x)
  return np.multiply(g, 1 - g)


def sigmoid_double_gradient(x):
  h = sigmoid(x)
  g = sigmoid_gradient(x)
  return g * (1 - 2 * h)


class Regressor():
  """ This is an interface regressors. """
  def __init__(self):
    raise NotImplementedError("Please implement a constructor for the regressor or use the one provided within provided within it.") 

  def save_theta(self, theta_path):
    raise NotImplementedError("Please implement a save_theta function for your regressor.")

  def load_theta(self, theta_path):
    raise NotImplementedError("Please implement a load_theta function for your regressor.") 

  def train(self, X, y, *args):
    raise NotImplementedError("Please implement a train function for your regressor.") 

  def get_prediction(self, X):
    raise NotImplementedError("Please implement a prediction function for your regressor.") 

  def score(self, X, y, sample_weigth=None):
    raise NotImplementedError("Please implement a score function for your regressor.") 

class Multilayer_Regressor(Regressor, skl.base.RegressorMixin):

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
        sys.stdout.write("\rProgress: %d%%, Score: %f" % ((i+1)/num_iter * 100, c[-1]))
        sys.stdout.flush()
    if verbose:
      print ""
    return np.array(c, dtype=np.float64)

  def get_prediction(self, X):
    return self.get_activation_layers(X)[-1]

  def get_activation_layers(self, X):
    activation_layers = [X]
    for theta in self.theta_lst:
      activation_layers.append(np.dot(theta, activation_layers[-1]))
    return activation_layers

  def theta_gradients(self, X, y):
    act = self.get_activation_layers(X)
    delta = - (y - act.pop());
    grad = []
    past_theta = None
    alpha = None
    for theta in self.theta_lst[::-1]:
      a = act.pop()
      if past_theta is None:
        alpha = 1/norm(a, ord=2)**2
        past_theta = theta
      else:
        delta = np.dot(past_theta.T, delta)
        alpha = alpha * 1/norm(past_theta, ord=2)**2
      g = np.dot(delta, a.T)
      past_theta = theta
      grad.insert(0, 0.9 * alpha * g)
    return grad

  def score(self, X, y, sample_weigth=None):
    y_pred = self.get_prediction(X)
    u = norm(y - y_pred)**2
    v = norm(y - y.mean())**2
    return 1 - u/v


class Multilayer_Logistic_Regressor(Regressor, skl.base.ClassifierMixin):

  def __init__(self, nodes_per_layer):
    self.theta_lst = []
    for i in range(1, len(nodes_per_layer)):
      prev, curr = nodes_per_layer[i-1], nodes_per_layer[i]
      self.theta_lst.append(np.zeros((curr, prev)).astype(np.complex64))

  def load_theta(self, theta_path):
    """ This assumes that theta list is saved as an npy file """
    theta_lst = np.load(theta_path)
    for i in range(len(theta_lst)):
      self.theta_lst[i] = theta_lst[i]

  def save_theta(self, path):
    """ This saves the theta_lst as an npy """
    np.save(path, np.array(self.theta_lst))

  def train(self, X, y, alpha, num_iter=100, lmbda=0, verbose=False):
    c = [self.score(X, y)]
    m, n = X.shape
    if verbose:
      print "Init score: %f" % c[-1]
    for i in range(num_iter):
      grad = self.theta_gradients(X, y, alpha)
      k = len(self.theta_lst)
      for j in range(k):
        self.theta_lst[j] -= (grad[j] + (lmbda/n) * self.theta_lst[j])
      c.append(self.score(X, y))
      if verbose:
        sys.stdout.write("\rProgress: %d%%, Score: %f" % ((i+1)/num_iter * 100, c[-1]))
        sys.stdout.flush()
    if verbose:
      print ""
    return np.array(c, dtype=np.float64)

  def get_prediction(self, X, threshold=None):
    _, _, activation_layers = self.get_layers(X)
    if threshold is None:
      return activation_layers[-1]
    return activation_layers[-1].all() > threshold

  def get_layers(self, X):
    activation_layers = [X]
    zgrad  = []
    zgrad2 = []
    for theta in self.theta_lst:
      z = theta.dot(activation_layers[-1])
      zgrad.append(sigmoid_gradient(z))
      zgrad2.append(sigmoid_double_gradient(z))
      activation_layers.append(sigmoid(z))
    return zgrad, zgrad2, activation_layers

  def theta_gradients(self, X, y, alpha):
    zgrad, zgrad2, act = self.get_layers(X)
    grad = []
    delta = -(y - act.pop())
    past_theta = None
    for theta in self.theta_lst[::-1]:
      a = act.pop(); zg = zgrad.pop();
      if past_theta == None:
        delta = delta * zg
      else:
        delta = np.dot(past_theta.T, delta) * zg
      g = np.dot(delta, a.T)
# TODO: Derive a way of finding alpha. It seems to get really complicated.
#      alpha = 0.9 * (1/(norm(theta, ord=2)**2 * norm(delta, ord=2)**2 * norm(a, ord=2)**2))
      past_theta = theta
      grad.insert(0, alpha * g)
    return grad

  def score(self, X, y, sample_weigth=None):
    y_pred = self.get_prediction(X)
    u = norm(y - y_pred)**2
    v = norm(y - y.mean())**2
    return 1 - u/v


if __name__ == '__main__': 
  import matplotlib.pyplot as plt
  from sklearn import cross_validation as cv
  from sklearn import datasets

  print "Multilayer Regressor"
  iris = datasets.load_iris()
  X_train, X_test, y_train, y_test = cv.train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
  mr = Multilayer_Regressor([X_train.shape[1], X_train.shape[1], X_train.shape[1], 1])
  c = mr.train(X_train.T, y_train.T, num_iter=1000, verbose=True)
  print "Score on cross-validation set: %f" % mr.score(X_test.T, y_test.T)

  print "Multilayer Logistic Regressor"
  digits = datasets.load_digits()
  target = np.zeros((10, digits.data.shape[0]))
  for idx in range(target.shape[1]):
    target[digits.target[idx], idx] = 1
  
  X_train, X_test, y_train, y_test = cv.train_test_split(digits.data, target.T, test_size=0.2, random_state=0)
  X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T
  nn = Multilayer_Logistic_Regressor([X_train.shape[0], X_train.shape[0]*2, 10])
  #nn = Multilayer_Logistic_Regressor([X_train.shape[0], 10])
  c = nn.train(X_train, y_train, num_iter=100, verbose=True, alpha=1e-4)
  print "Score on cross-validation set: %f" % nn.score(X_test, y_test)
