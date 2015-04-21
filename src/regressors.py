#!/usr/bin/env python


from __future__ import division


import numpy as np
import sklearn as skl


norm = np.linalg.norm


def sigmoid(x):
  return 1/(1 + np.exp(-x))


def sigmoid_gradient(x):
  return sigmoid(x) * (1 - sigmoid(x))


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
        print "Progress: %d / %d, Score: %f" % (i+1, num_iter, c[-1])
    return np.array(c, dtype=np.float64)

  def get_prediction(self, X):
    return self.get_activation_layers(X)[-1]

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
        alpha = 0.9 * (1/(norm(activation_layers[-1], ord=2)**2))
        g = alpha * np.dot(delta, activation_layers[-1].conj().T)
        past_theta = theta
      else:
        alpha = 0.9 * (1/(norm(past_theta.conj(), ord=2) * norm(activation_layers[-1], ord=2))**2)
        g = alpha * np.dot(past_theta.conj().T, np.dot(delta, activation_layers[-1].conj().T))
        past_theta = np.dot(past_theta, theta)
      activation_layers.pop()
      grad.insert(0, g)
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

  def get_prediction(self, X, threshold=None):
    if threshold is None:
      return self.get_az_layers(X)[-1][1]
    return self.get_az_layers(X)[-1][1].all() > threshold

  def get_az_layers(self, X):
    activation_layers = [X]
    z_layers = []
    for theta in self.theta_lst:
      layer = np.dot(theta, activation_layers[-1])
      z_layers.append(layer)
      activation_layers.append(sigmoid(layer))
    return z_layers, activation_layers

  def theta_gradients(self, X, y):
#TODO: Remove below line and figure out best step size.
    alpha = 0.1
    z_layers, activation_layers = self.get_az_layers(X)
    delta = - (y - activation_layers[-1]);
    activation_layers.pop()
    grad = []
    past_theta = None
    sig_grad = None
    for theta in self.theta_lst[::-1]:
      if past_theta is None:
        sig_grad = sigmoid_gradient(z_layers[-1])
#TODO: Figure out best stepsize        alpha = 0.9 * (1/(norm(activation_layers[-1], ord=2)**2))
        g = alpha * np.dot(delta * sig_grad, activation_layers[-1].conj().T)
        past_theta = theta
      else:
        sig_grad = np.dot(sig_grad, sigmoid_gradient(z_layers[-1]).T)
#TODO: Figure out best stepsize        alpha = 0.9 * (1/(norm(past_theta.conj(), ord=2) * norm(activation_layers[-1], ord=2))**2)
        g = alpha * np.dot((past_theta * sig_grad).conj().T, np.dot(delta, activation_layers[-1].conj().T))
        past_theta = np.dot(past_theta, theta)
      activation_layers.pop()
      z_layers.pop()
      grad.insert(0, g)
    return grad

  def score(self, X, y, sample_weigth=None):
    y_pred = self.get_az_layers(X)[1][-1]
    u = norm(y - y_pred)**2
    v = norm(y - y.mean())**2
    return 1 - u/v
   

if __name__ == '__main__': 
  import matplotlib.pyplot as plt
  from sklearn import cross_validation as cv
  from sklearn import datasets

  print "Multilayer Regressor"
  iris = datasets.load_iris()
  X_train, X_test, y_train, y_test = cv.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
  mr = Multilayer_Regressor([X_train.shape[1], 1])
  c = mr.train(X_train.T, y_train.T)
  print "Score on cross-validation set: %f" % mr.score(X_test.T, y_test.T)

  print "Multilayer Logistic Regressor"
  digits = datasets.load_digits()
  X_train, X_test, y_train, y_test = cv.train_test_split(digits.data, digits.target, test_size=0.4, random_state=0)
  nn = Multilayer_Logistic_Regressor([X_train.shape[1], X_train.shape[1], 10])
  c = nn.train(X_train.T, y_train.T, verbose=True, num_iter=100)
  print "Score on cross-validation set: %f" % nn.score(X_test.T, y_test.T)