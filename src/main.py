#!/usr/bin/env python


from __future__           import division
from optparse             import OptionParser
from os                   import system
from models               import models_dict
from cfl                  import readcfl, writecfl, cfl2sqcfl, sqcfl2mat, mat2sqcfl
from metrics              import get_metric
from plot                 import plot_simulation, plot_cfl_signals
from gen_FSEmatrix        import gen_FSEmatrix
from sys                  import argv
from warnings             import warn
from multilayer_regressor import multilayer_regressor as mr


import numpy as np
import scipy.io as sio


time_stamp = ""


parser = OptionParser()

# simulation options
parser.add_option("--genFSE", dest="genFSE", action="store_true", default=False, help="Set this flag if you want to generate a simulation")
parser.add_option("--loadFSE", dest="loadFSE", type=str, default=None, help="Pass in path to simulation mat FILE.")
parser.add_option("--saveFSE", dest="saveFSE", type=str, default=None, help="Pass in path (with file name) to file to save simulation.")

# estimator options
parser.add_option("--estimator", dest="estimator", action="store_true", default=False, help="Set this flag if you want to run the estimator.")
parser.add_option("--train", dest="train", action="store_true", default=False, help="Set this flag if you want to train the estimator.")
parser.add_option("--predict", dest="predict", action="store_true", default=False, help="Set this flag if you want to see the predictions.")
parser.add_option("--est-iter", dest="estiter", type=int, default=100, help="Number of iterations when training estimator.")
parser.add_option("--loadEst", dest="loadEst", type=str, default=None, help="Pass in path to estimator npy file.")
parser.add_option("--saveEst", dest="saveEst", type=str, default=None, help="Pass in path (with file name) to file to save estimator.")

# constraint options
parser.add_option("--rvc", dest="rvc", action="store_true", default=False, help="Real value constraint on basis.")
parser.add_option("--avc", dest="avc", action="store_true", default=False, help="Absolute value constraint on basis.")

#cfl options
parser.add_option("--cfl", dest="cfl", type=str, default=None, help="Path to cfl file")

# genFSE 
parser.add_option("--numT2", dest="N", type=int, default=256, help="Number of T2 values to load in.")
parser.add_option("--angles", dest="angles", type=str, default=None, help="Load in Angles in degrees.")
parser.add_option("--ETL", dest="ETL", type=int, default=None, help="Load in ETL")
parser.add_option("--TE", dest="TE", type=float, default=5.568e-3, help="Echo spacing (TE) in seconds")
parser.add_option("--T1vals", dest="T1vals", type=str, default=None, help="Load T1 values from .mat file with variable 'T1vals'")
parser.add_option("--T2vals", dest="T2vals", type=str, default=None, help="Load T2 values from .mat file with variable 'T2vals'")

# universal options
parser.add_option("--e2s", dest="e2s", type=int, default=2, help="Echoes to skip")
parser.add_option("-K", "--dim", dest="k", type=int, default=None, help="Number of basis vectors to construct. This only effects the reconstructed Xhat")
parser.add_option("--model", dest="model", type=str, default=[], action="append", help="The model you want to test")
parser.add_option("--add_control", dest="add_control", action="store_true", default=False, help="Set this flag if you want to compare said model with svd")
parser.add_option("--print-models", dest="print_models", action="store_true", default=False, help="Print all the model choices and exit.")

# saving options
parser.add_option("--save_basis", dest="save_basis", type=str, default=None, help="Pass in path to FOLDER to save basis.")
parser.add_option("--save_plots", dest="save_plots", type=str, default=None, help="Pass in path to FOLDER to save plots.")
parser.add_option("--save_imgs", dest="save_imgs", type=str, default=None, help="Pass in path to FOLDER to save images.")
parser.add_option("--set_basis_name", dest="basis_name", type=str, default=None, help="Pass this to change saved basis name. USE ONLY IF TESTING 1 MODEL.")


options, args = parser.parse_args()


if len(argv) == 1:
  parser.print_help()
  exit(0)


if options.print_models:
  for key in models_dict:
    if models_dict[key].__doc__ is not None:
      print '\t'.join(('"%s"' % key , models_dict[key].__doc__))
    else:
     print '"%s"' % key
  exit(0)


assert (options.cfl or options.loadFSE or options.genFSE), "Please pass in a cfl file XOR an angles file XOR an FSEsim."

assert int(options.cfl != None) + int(options.loadFSE != None) + int(options.genFSE), "Please pass in cfl XOR angles XOR FSEsim."

assert not (options.rvc and options.avc), "Please choose real-value NAND abs-value constraint."


if options.rvc:
    rvc = 'real'
elif options.avc:
    rvc = 'abs'
else:
    rvc = None


if options.save_imgs:
  assert options.cfl, "In order to save images, a cfl file must be passed instead of values for a simulation."

if options.basis_name != None:
  assert len(options.model) == 1, "In order to change the saved basis name, you must test a single model. "

if options.cfl:

  cfl_name = options.cfl.split('.')[-1]
  X, img_dim = sqcfl2mat(cfl2sqcfl(readcfl(options.cfl), options.e2s))

elif options.genFSE: 

  assert options.angles is not None, "In order to generate FSEmatrix, you must pass in an angles file."
  time_stamp = options.angles.split('.')[-1] 
  try:
    int(time_stamp)
    time_stamp = "." + time_stamp
  except ValueError:
    time_stamp = "" 
  except AssertionError:
    time_stamp = ""
  angles = np.loadtxt(options.angles)
  angles = angles * np.pi/180
  e2s = options.e2s
  TE  = options.TE
  N   = options.N
  if not options.T1vals:
    T1vals = np.array([500, 700, 1000, 1800]) * 1e-3
  else:
    T1vals = sio.loadmat(options.T1vals)['T1vals']
  if not options.T2vals:
    T2vals = np.linspace(20e-3, 800e-3, N)
  else:
    T2vals = sio.loadmat(options.T2vals)['T2vals']
  if T2vals.shape[0] > N:
    idx = np.random.permutation(T2vals.shape[0])
    T2vals = T2vals[idx[0:N]]
  T = len(angles)
  N = len(T2vals)
  if not options.ETL:
    ETL = T - e2s - 1
  else:
    ETL = options.ETL
  T1vals = np.ravel(T1vals)
  T2vals = np.ravel(T2vals)
  X = gen_FSEmatrix(N, angles, ETL, e2s, TE, T1vals, T2vals)
  if options.saveFSE != None:
    print "Saving as " + options.saveFSE + time_stamp
    sio.savemat(options.saveFSE + time_stamp, {"X": X, "angles": angles, "N": N, "ETL":ETL, "e2s":e2s, "TE": TE, "T1vals":T1vals, "T2vals":T2vals})

else:

  dct = sio.loadmat(options.loadFSE)
  X = dct["X"]
  angles = dct["angles"]
  N = dct["N"]
  ETL = dct["ETL"]
  e2s = dct["e2s"]
  TE = dct["TE"]
  T1vals = dct["T1vals"]
  T2vals = dct["T2vals"]


if rvc == 'real':
    print 'real value constraint'
    X = np.real(X)
elif rvc == 'abs':
    print 'abs value constraint'
    X = np.abs(X)


# TODO: Abstract this into a different file.
if options.estimator:
  #scales = np.diag(1/np.linalg.norm(X, ord=2, axis=0))
  scales = np.diag(1/np.max(X,  axis=0))
  X = np.vstack((np.ones((1, X.shape[1])), X))
  est = mr([X.shape[0], 2])
  if options.loadEst is not None:
    est.load_theta(options.loadEst)
  if options.train:
    assert options.loadFSE is not None, "To train, please pass in a simulated FSE matrix"
    y = np.zeros((2, X.shape[1]))
    c = 0
    for T2 in np.squeeze(T2vals):
      for T1 in np.squeeze(T1vals):
        y[0, c] = T1
        y[1, c] = T2
        c += 1
    est.train(X, y, num_iter=options.estiter, lmbda=0, verbose=True)
    print "Test set performance: %f" % est.score(X, y)
  if options.saveEst is not None:
    print "Saving estimator: " + options.saveEst
    est.save_theta(options.saveEst)
  if options.predict:
    assert options.loadEst or options.train, "No estimator passed in."
    estimated_map = est.get_activation_layers(X)[-1]


lst = options.model
if options.add_control:
  lst.append('simple_svd')
if options.model == 'all':
  lst = models.keys()
results = {}
for m in lst:
  print "------------------------------------------------------------"
  print "Running " + m
  print "------------------------------------------------------------"
  model = models_dict[m]
  k = options.k
  U, alpha, X_hat = model(X, options.k, rvc)
  print "Results:"
  pnorm, fro_perc_err = get_metric(X, X_hat)
  if not k:
    k = U.shape[1]
  results[m] = {'U':U, 'alpha':alpha, 'k':k, 'X_hat': X_hat, 'Percentage Error per Signal': pnorm, 'Frobenius Percentage Error': fro_perc_err}
print "------------------------------------------------------------"


if options.save_plots != None:
  for m in lst:
    mod = results[m]
    if options.cfl:
      plot_cfl_signals(mod['U'], options.k, X, mod['X_hat'], "cfl_" + m, options.e2s, options.save_plots)
    else:
      plot_simulation(mod['U'], options.k, X, mod['X_hat'], "sim_" + m, T1vals, T2vals, e2s, options.save_plots)


if options.save_imgs != None:
  # TODO
  warn('TODO: implement options.save_imgs')
  None


if options.save_basis != None:
  for m in results.keys():
    U = results[m]["U"]
    k = results[m]['k']
    if U.shape[1] < U.shape[0]:
      U = np.hstack((U, np.zeros((U.shape[0], U.shape[0] - k))))

    for i in range(5):
      U = np.expand_dims(U, axis=0)

    k_ext = '_k_%d' % k

    if options.basis_name != None:
      cfl_name = options.basis_name  
    else:
      warn('FIXME: cfl_name')
      cfl_name = 'FIXME'

    warn('FIXME: use os.path')
    writecfl(options.save_basis + cfl_name + "_" + m + k_ext + time_stamp, U)
