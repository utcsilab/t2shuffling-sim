from __future__ import division
from scipy.optimize import fmin
from nn_simple import NN_simple

import numpy as np


norm = np.linalg.norm
inv = np.linalg.inv


def svd(X):
  """ Returns the first k vectors of the SVD of X """
  U, S, V = np.linalg.svd(X, full_matrices=False)
  return np.matrix(U)


def compute_alpha(X, U, rvc=None):
   if rvc is None:
      return np.linalg.inv(U.H * U) * U.H * X
   else:
      return np.linalg.inv(U.T * U) * U.T * X

def rvc_U(U, rvc=None):
   if rvc == 'real':
      U = np.real(U)
   elif rvc == 'abs':
      U = np.abs(U)
   return U



models_dict = {}


def simple_svd(X, k=None, rvc=None):
  U = svd(X)
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, U[:, :k] * alpha[:k]

models_dict["simple_svd"] = simple_svd


def sub_col_mean(X, k=None, rvc=None):
  """ This subracts the mean along each row and returns
      the svd. """
  X_hat = X - np.mean(X, axis=1)
  U     = svd(X_hat)
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, U[:, :k] * alpha[:k]

models_dict["sub_col_mean"] = sub_col_mean


def no_reg_grad_descent(X, k=None, rvc=None):
  """ Very simple gradient descent with no regularization. Starting arbitrary U is
      arbitrary. """
  if not k:
    k=3
  U = np.matrix(np.eye(svd(X).shape[0])[:, :k])
  m, n = U.shape
  l = 1 
  
  num_iters = 10000

  for i in range(num_iters):
    alpha = inv(U.T * U) * U.T * X
    U = U - (l/m) * (U * alpha - X) * alpha.T
#    print "   " + str(i+1) + "/" + str(num_iters)
    
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)

  return U, alpha, U[:, :k] * alpha[:k]

models_dict["no_reg_grad_descent"] = no_reg_grad_descent


def reg_grad_descent(X, k=None, rvc=None):
  """ Very simple gradient descent with regularization. 
      Starting arbitrary U is arbitrary. """
  if not k:
    k=3
  U = np.matrix(np.eye(svd(X).shape[0])[:, :k])
  m, n = U.shape

  l = 1 # Rate of descent

  lmbda = 0.3

  # Scale start and end more?
  L = np.matrix(np.eye(U.shape[1]))

  num_iters = 10000

  for i in range(num_iters):
    alpha = inv(U.T * U + lmbda * L) * U.T * X
    U = U - (l/m) * (U * alpha - X) * alpha.T
#    print "   " + str(i+1) + "/" + str(num_iters)
    
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)

  return U, alpha, U[:, :k] * alpha[:k]

models_dict["reg_grad_descent"] = reg_grad_descent


def scale_col_no_power(X, k=None, rvc=None):
  """ This scales each signal by the inverse of its norm. Works
      pretty well! """
  scales = np.diag(1/np.linalg.norm(X, ord=3, axis=0))
  Xhat = X * scales
  U = svd(Xhat)
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, U[:, :k] * alpha[:k]

models_dict["scale_col_no_power"] = scale_col_no_power


def scale_col_with_power(X, k=None, rvc=None):
  """ Arbitrary raised scales to higher powers seems to work. """
  scales = np.diag(1/np.linalg.norm(X, ord=3, axis=0)**2.5)
  Xhat = X * scales
  U = svd(Xhat)
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, U[:, :k] * alpha[:k]

models_dict["scale_col_with_power"] = scale_col_with_power


def nn1(X, k=None, rvc=None):
  if k is None:
    k = 3
  nn = NN_simple([X.shape[0], k, X.shape[0]])
  
  #nn.load_theta('../saves/nn1_theta.npy')
  #c = nn.train(X, X, num_iter=2000000, alpha=1e-9, lmbda=10)
  #c = nn.train(X, X, num_iter=10, alpha=1e-20, lmbda=1e-5, verbose=True)
  c = nn.train(X, X, num_iter=2000000, alpha=1e-20, lmbda=1e-5, verbose=True)
  nn.save_theta('../saves/knee_low_res_nn1_theta.npy')
  U = nn.theta_lst[-1]
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, U * alpha

models_dict["nn1"] = nn1


def nn1_scaled_X(X, k=None, rvc=None):
  scales = np.diag(1/np.linalg.norm(X, ord=3, axis=0)**2.5)
  Xt = X * scales
  U, _, _ = nn1(Xt, k)
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, U * alpha


models_dict["nn1_scaled_X"] = nn1_scaled_X


def partition_more_low_t2(X, k=None, rvc=None):
  """ A normal SVD works really well with medium and high T2 values. 
      Can we increase the number of low T2s and therefore make it
      run better? """
  p = int(X.shape[1]/3)
  Xl = X[:, :p]
  U = svd(Xl)
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)

  return U, alpha, U[:, :k] * alpha[:k]

models_dict["partition_more_low_t2"] = partition_more_low_t2


def scale_col_with_power_low_T2(X, k=None, rvc=None):
  """ Arbitrary raised scales to higher powers seems to work. This also only used
      low T2 values. This reduces max error, but does increase Fro norm. """
  scales = np.diag(1/np.linalg.norm(X, ord=3, axis=0)**2.5)
  Xhat = X * scales
  U = svd(Xhat[:, :int(X.shape[1]/2)])
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, U[:, :k] * alpha[:k]

models_dict["scale_col_with_power_low_T2"] = scale_col_with_power_low_T2


def reg_svd(X, k=None, rvc=None):
  """ normalize each signal curve to norm 1. """
  scales = np.diag(1/np.linalg.norm(X, ord=2, axis=0))
  Xt = X * scales
  U = svd(Xt)
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, U[:, :k] * alpha[:k]

models_dict["reg_svd"] = reg_svd


def reg_svd_sq(X, k=None, rvc=None):
  scales = np.diag(1/np.linalg.norm(X, ord=2, axis=0)**2)
  Xt = X * scales
  U = svd(Xt)
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, U[:, :k] * alpha[:k]

models_dict["reg_svd_sq"] = reg_svd_sq
