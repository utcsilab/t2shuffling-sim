from __future__           import division
from regressors           import Regressor, Multilayer_Regressor

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

def linear_regressor(X, y=None, reg_lambda=0, train=False, predict=True, num_iters=100, save=None, load=None, verbose=False):
  """ A simple regressor that tries to find a linear transformation
      from signal to t1-t2 pair. """
  est = Multilayer_Regressor([X.shape[0], 2])
  if save is not None:
    est.load_theta(load)
  if train:
    assert y.any() is not None, "Please pass in a label matrix"
    est.train(X, y, num_iter=num_iters, lmbda=reg_lambda, verbose=verbose)
    print "Test set performance: %f" % est.score(X, y)
  if save is not None:
    print "Saving estimator: " + save
    est.save_theta(save)
  if predict:
    assert load or train, "No estimator passed in."
  return est.get_prediction(X)

jt_models_dict["linear_regressor"] = linear_regressor
