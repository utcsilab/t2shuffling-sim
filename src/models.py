from __future__ import division
from scipy.optimize import fmin
from regressors import Multilayer_Regressor as mr

import numpy as np


norm = np.linalg.norm
inv = np.linalg.inv


def svd(X):
  """ Returns the first k vectors of the SVD of X """
  U, S, V = np.linalg.svd(X, full_matrices=False)
  return np.matrix(U)


def compute_alpha(X, U, rvc=None):
  if rvc is None:
    return np.dot(np.dot(inv(np.dot(U.H,  U)), U.H), X)
  else:
    return np.dot(np.dot(inv(np.dot(U.T, U)), U.T), X)


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
  return U, alpha, np.dot(U[:, :k], alpha[:k])

models_dict["simple_svd"] = simple_svd


def sub_col_mean(X, k=None, rvc=None):
  """ This subracts the mean along each row and returns
      the svd. """
  X_hat = X - np.mean(X, axis=1)
  U     = svd(X_hat)
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, np.dot(U[:, :k], alpha[:k])

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

  return U, alpha, np.dot(U[:, :k], alpha[:k])

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

  return U, alpha, np.dot(U[:, :k], alpha[:k])

models_dict["reg_grad_descent"] = reg_grad_descent


def scale_col_no_power(X, k=None, rvc=None):
  """ This scales each signal by the inverse of its norm. Works
      pretty well! """
  scales = np.diag(1/norm(X, ord=3, axis=0))
  Xhat = X * scales
  U = svd(Xhat)
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, np.dot(U[:, :k], alpha[:k])

models_dict["scale_col_no_power"] = scale_col_no_power


def scale_col_with_power(X, k=None, rvc=None):
  """ Arbitrary raised scales to higher powers seems to work. """
  scales = np.diag(1/norm(X, ord=3, axis=0)**2.5)
  Xhat = X * scales
  U = svd(Xhat)
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, np.dot(U[:, :k], alpha[:k])

models_dict["scale_col_with_power"] = scale_col_with_power


def nn1(X, k=None, rvc=None):
  if k is None:
    k = 3
  nn = mr([X.shape[0], k, X.shape[0]])
  
  #nn.load_theta('../saves/nn1_theta.npy')
  #c = nn.train(X, X, num_iter=2000000, alpha=1e-9, lmbda=10)
  #c = nn.train(X, X, num_iter=10, alpha=1e-20, lmbda=1e-5, verbose=True)
  c = nn.train(X, X, num_iter=2000000, lmbda=1e-5, verbose=True)
  nn.save_theta('../saves/knee_low_res_nn1_theta.npy')
  U = rvc_U(nn.theta_lst[-1], rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, np.dot(U[:, :k], alpha[:k])

models_dict["nn1"] = nn1


def nn1_scaled_X(X, k=None, rvc=None):
  scales = np.diag(1/norm(X, ord=3, axis=0)**2.5)
  Xt = X * scales
  U, _, _ = nn1(Xt, k)
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, np.dot(U[:, :k], alpha[:k])

models_dict["nn1_scaled_X"] = nn1_scaled_X


def batch_nn(X, k=None, rvc=None):
  """ This model gets index of the signal of the max perventage error and chooses "width" amount of
      signals to the left and right of it (which are similar since X is sorted). It then trains the
      neural net only on those 2*width values, and then repeats the process again.
      
      Tweakable parameters:
        width - The number of signals to the left and right of the signal with the max percentage 
                error. Note that the total number of signals selected is then 2 * width.
        global_iter - Number of times width is chosen and the nn is trained on. 
        nn_iter - Number of iteration nn uses to train on the 2*width signal.
        alpha - the "jump" of the descent
        lambda - the regularization paramter. """
  from metrics import get_metric
  if k is None:
    k = 4
  nn = mr([X.shape[0], k, X.shape[0]])

  nn.load_theta('../saves/batch_nn1_sim_theta.npy')

  width = 500 # Bigger the value, smaller the gobal error. Smaller the value, smaller the individual error.
  global_iter = 8
  nn_iter = 100

  U = rvc_U(nn.theta_lst[-1], rvc)
  alpha = compute_alpha(X, U, rvc)

  for i in range(global_iter):
    print "Progress: %d / %d" % (i+1, global_iter) 
    perc, fro = get_metric(X, U * alpha, disp=False)
    idx = np.argmax(perc, axis=0)
    if (idx < width):
      idx = width 
    if idx > X.shape[1]- width - 1:
      idx = X.shape[1] - width - 1
    Xhat = X[:, idx-width:idx+width]
    nn.train(Xhat, Xhat, num_iter=nn_iter, lmbda=0)
    U = rvc_U(nn.theta_lst[-1], rvc)
    alpha = compute_alpha(X, U, rvc)

#  nn.save_theta('../saves/batch_nn1_sim_theta.npy')

  return U, alpha, np.dot(U[:, :k], alpha[:k])

models_dict["batch_nn"] = batch_nn


def batch_nn_svd(X, k=None, rvc=None):
  """ This tries to do what batch_nn does, but instead of 
      doing the initial training for the thetas, it uses
      the svd as weights instead. """
  from metrics import get_metric
  
  U = svd(X)
  nn = mr([0, 0, 0])
  nn.theta_lst[0] = inv(U)[:k, :]
  nn.theta_lst[1] = U[:, :k]

  width = np.floor(X.shape[1]/64) # Bigger the value, smaller the gobal error. Smaller the value, smaller the individual error.
  #width = 8 # Bigger the value, smaller the gobal error. Smaller the value, smaller the individual error.
  global_iter = 64
  nn_iter = 100

  U = rvc_U(nn.theta_lst[-1], rvc)
  alpha = compute_alpha(X, U, rvc)

  past_perc, past_fro = get_metric(X, np.dot(U, alpha), disp=False)
  past_perc = np.max(past_perc)

  for i in range(global_iter):

    print "Progress: %d / %d" % (i+1, global_iter)

    perc, fro = get_metric(X, np.dot(U, alpha), disp=False)

    idx = np.argmax(perc, axis=0)
    if (idx < width):
      idx = width
    if idx > X.shape[1]- width - 1:
      idx = X.shape[1] - width - 1

    Xhat = X[:, idx-width:idx+width]
    nn.train(Xhat, Xhat, num_iter=nn_iter, lmbda=0)

    U = rvc_U(nn.theta_lst[-1], rvc)
    alpha = compute_alpha(X, U, rvc)

    past_perc, past_fro = np.max(perc), fro

  return U, alpha, np.dot(U[:, :k], alpha[:k])
  
models_dict["batch_nn_svd"] = batch_nn_svd


def partition_more_low_t2(X, k=None, rvc=None):
  """ A normal SVD works really well with medium and high T2 values. 
      Can we increase the number of low T2s and therefore make it
      run better? """
  p = int(X.shape[1]/3)
  Xl = X[:, :p]
  U = svd(Xl)
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, np.dot(U[:, :k], alpha[:k])

models_dict["partition_more_low_t2"] = partition_more_low_t2


def scale_col_with_power_low_T2(X, k=None, rvc=None):
  """ Arbitrary raised scales to higher powers seems to work. This also only used
      low T2 values. This reduces max error, but does increase Fro norm. """
  scales = np.diag(1/norm(X, ord=3, axis=0)**2.5)
  Xhat = X * scales
  U = svd(Xhat[:, :int(X.shape[1]/2)])
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, np.dot(U[:, :k], alpha[:k])

models_dict["scale_col_with_power_low_T2"] = scale_col_with_power_low_T2


def reg_svd(X, k=None, rvc=None):
  """ normalize each signal curve to norm 1. """
  scales = np.diag(1/norm(X, ord=2, axis=0))
  Xt = np.dot(X, scales)
  U = svd(Xt)
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, np.dot(U[:, :k], alpha[:k])

models_dict["reg_svd"] = reg_svd


def reg_svd_sq(X, k=None, rvc=None):
  scales = np.diag(1/norm(X, ord=2, axis=0)**2)
  Xt = X * scales
  U = svd(Xt)
  U = rvc_U(U, rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, np.dot(U[:, :k], alpha[:k])


models_dict["reg_svd_sq"] = reg_svd_sq

def low_TE_nn(X, k=None, rvc=None):
  if k is None:
    k = 3
  U = svd(X)
  nn = mr([0, 0, 0])
  nn.theta_lst[0] = inv(U)[:k, :]
  nn.theta_lst[1] = U[:, :k]

  feature_select = np.linspace(0, 1, X.shape[0])[::-1]**1e10

  c = nn.train(X, X, feature_selection=feature_select, num_iter=5000, verbose=True)
  U = rvc_U(nn.theta_lst[-1], rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, np.dot(U[:, :k], alpha[:k])

models_dict["low_TE_nn"] = low_TE_nn


def TE_svd(X, k=None, rvc=None):
  fs = np.diag(np.linspace(0, 1, X.shape[0])[::-1]**1e10)
  Xt = np.dot(fs, X)
  U = rvc_U(svd(Xt), rvc)
  alpha = compute_alpha(X, U, rvc)
  return U, alpha, np.dot(U[:, :k], alpha[:k])
  
models_dict["TE_svd"] = TE_svd
