from __future__           import division
from regressors           import *

import numpy as np

norm = np.linalg.norm
inv = np.linalg.inv

def svd(X):
  """ Returns the first k vectors of the SVD of X """
  U, S, V = np.linalg.svd(X, full_matrices=False)
  return np.matrix(U)

def compute_alpha(X, U):
  return np.dot(np.dot(inv(np.dot(U.H,  U)), U.H), X)

jt_models_dict = {}

def linear_regressor(X, y=None, alpha=None, reg_lambda=0, train=False, predict=True, num_iters=100, save=None, load=None, verbose=False):
  """ A simple regressor that tries to find a linear transformation
      from signal to t1-t2 pair. """
  if load is not None:
    est = Multilayer_Regressor([None])
    est.load_theta(load)
  else:
    est = Multilayer_Regressor([X.shape[0], y.shape[0]])
  if train:
    assert y.any() is not None, "Please pass in a label matrix"
    est.train(X, y, num_iter=num_iters, lmbda=reg_lambda, verbose=verbose)
    print "Test set performance: %f" % est.score(X, y)
  if save is not None:
    print "Saving estimator: " + save
    est.save_theta(save)
  if predict:
    assert load or train, "No estimator passed in."
  return est

jt_models_dict["linear_regressor"] = linear_regressor


def logistic_regressor(X, y=None, alpha=None, reg_lambda=0, train=False, predict=True, num_iters=100, save=None, load=None, verbose=False):
  assert alpha is not None, "This model needs a gradient step paramter!"
  X = X/np.max(X)
  if load is not None:
    est = Multilayer_Logistic_Regressor([None])
    est.load_theta(load)
  else:
    assert y.any() is not None, "Please pass in a label matrix"
    est = Multilayer_Logistic_Regressor([X.shape[0], y.shape[0]])
  if train:
    y = y/np.max(y)
    est.train(X, y, alpha, num_iter=num_iters, lmbda=reg_lambda, verbose=verbose)
    print "Test set performance: %f" % est.score(X, y)
  if save is not None:
    print "Saving estimator: " + save
    est.save_theta(save)
  if predict:
    assert load or train, "No estimator passed in."
  return est

jt_models_dict["logistic_regressor"] = logistic_regressor


def t2_nn_class(X, y=None, alpha=None, reg_lambda=0, train=False, predict=True, num_iters=100, save=None, load=None, verbose=False):
  """ A neural network classifier that has 4 activation layers and classifies into classes = number of y rows. """
  assert alpha is not None, "This model needs a gradient step paramter!"
  if load is not None:
    est = Multilayer_Logistic_Regressor([None])
    est.load_theta(load)
  else:
    assert y.any() is not None, "Please pass in a label matrix"
    est = Multilayer_Logistic_Regressor([X.shape[0], X.shape[0], X.shape[0], y.shape[0]])
  if train:
    est.train(X, y, alpha, num_iter=num_iters, lmbda=reg_lambda, verbose=verbose)
    print "Test set performance: %f" % est.score(X, y)
  if save is not None:
    print "Saving estimator: " + save
    est.save_theta(save)
  if predict:
    assert load or train, "No estimator passed in."
  return est

jt_models_dict["t2_nn_class"] = t2_nn_class

