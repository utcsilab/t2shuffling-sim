#!/usr/bin/env python


from __future__ import division
from optparse   import OptionParser
from os         import system
from models     import models_dict
from cfl        import readcfl, writecfl, cfl2sqcfl, sqcfl2mat, mat2sqcfl
from metrics    import get_metric


import numpy as np
import scipy.io as sio


load_FSEpath = "../data/FSEmatrix.mat"


parser = OptionParser()
parser.add_option("--loadMatrix", dest="loadMatrix", action="store_true", default=False, help="Set this flag if you want to load FSEmatrix")
parser.add_option("--numT2", dest="N", type=int, default=256, help="Number of T2 values to load in.")
parser.add_option("--angles", dest="angles", type=str, default=None, help="Load in Angles in degrees.")
parser.add_option("--ETL", dest="ETL", type=int, default=None, help="Load in ETL")
parser.add_option("--TE", dest="TE", type=float, default=5.568e-3, help="Load in TE")
parser.add_option("--T1vals", dest="T1vals", type=str, default=None, help="Load in T1 values")
parser.add_option("--T2vals", dest="T2vals", type=str, default=None, help="Load in T2 values")
parser.add_option("--cfl", dest="cfl", type=str, default=None, help="Path to cfl file")
parser.add_option("--e2s", dest="e2s", type=int, default=4, help="Echos to skip")
parser.add_option("--k", dest="k", type=int, default=None, help="Number of basis vectors to construct. This only effects the reconstructed X")
parser.add_option("--model", dest="model", type=str, default='all', help="The model you want to test")
parser.add_option("--add_control", dest="add_control", action="store_true", default=False, help="Set this flag if you want to compare said model with svd")
parser.add_option("--save_cfl", dest="save_cfl", action="store_true", default=False, help="Set this flag if you want to save cfl")
#parser.add_option("--save_plots", dest="save_plots", action="store_true", default=False, help="Set this flag to save plots")
#parser.add_option("--save_imgs", dest="save_imgs", action="store_true", default=False, help="Set this flag to save imgimgss")
parser.add_option("--make_comparisson", dest="do_comp", action="store_true", default=False, help="Set this flag if you want to generate gifs")


options, args = parser.parse_args()

cfl_path = "../basis/"
time_stamp = ""

assert options.cfl or options.angles, "Please pass in a cfl file OR an angles file."""

if options.cfl:

  cfl_name = options.cfl.split('.')[-1]
  X, img_dim = sqcfl2mat(cfl2sqcfl(readcfl(options.cfl), options.e2s))

else: 

  time_stamp = "." + options.angles.split('.')[-1] 
  try:
    int(time_stamp)
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
    T1vals = np.matrix(sio.loadmat(options.T1vals)['T1vals'])
  if not options.T2vals:
    T2vals = np.linspace(20e-3, 800e-3, N)
  else:
    T2vals = np.matrix(sio.loadmat(options.T2vals)['T2vals'])
  if T2vals.shape[0] > N:
    idx = np.random.permutation(T2vals.shape[0])
    T2vals = T2vals[idx[0:N]]
  T = len(angles)
  N = len(T2vals)
  if not options.ETL:
    ETL = T - e2s - 1
  else:
    ETL = options.ETL
  if options.loadMatrix:
    X = np.matrix(sio.loadmat(load_FSEpath)["FSEmatrix"])
  else:
    X = gen_FSEmatrix(N, angles, ETL, e2s, TE, T1vals, T2vals)
    sio.savemat(data_path + "FSEmatrix", {"FSEmatrix": X})
  # TODO: add image dimension

lst = [options.model]
if options.add_control:
  lst.append['simple_svd']
if options.model == 'all':
  lst = models.keys()
results = {}
for m in lst:
  print "[--------------------------------------------------]"
  print "Running " + m
  model = models_dict[m]
  k = options.k
  U, alpha, X_hat = model(X, options.k)
  print "Results"
  pnorm, fro_perc_err = get_metric(X, X_hat)
  if not k:
    k = U.shape[1]
  results[m] = {'U':U, 'alpha':alpha, 'k':k, 'Percentage Error per Signal': pnorm, 'Frobenius Percentage Error': fro_perc_err}
  if options.do_comp:
    # TODO
    None
  print "[--------------------------------------------------]"

if options.save_cfl:
  for m in results.keys():
    U = np.array(results[m]["U"], dtype=np.complex64)
    k = results[m]['k']
    U = np.hstack((U, np.zeros((U.shape[0], U.shape[0] - k))))

    for i in range(5):
      U = np.expand_dims(U, axis=0)

    k_ext = '_k_%d' % k

    writecfl(cfl_path + cfl_name + "_" + m + k_ext + time_stamp, U)
