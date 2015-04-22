#!/usr/bin/env python

from __future__    import division
from optparse      import OptionParser
from os            import system
from regressors    import Multilayer_Regressor as mr
from jt_models     import jt_models_dict       as models
from jt_partitions import jt_partitions_dict   as partitions
from sys           import argv

import numpy    as np
import scipy.io as sio

parser = OptionParser()

parser.add_option("--model"    ,dest="models"   ,type=str ,default=[] ,action="append",help="The model you want to test")
parser.add_option("--load"     ,dest="load"     ,type=str ,default=[] ,action="append",help="Pass in path to estimator npy file. Must be in the same order as models passed in.")
parser.add_option("--save"     ,dest="save"     ,type=str ,default=[] ,action="append",help="Pass in path (with file name) to file to save estimator. Must be in the same order as models passed in.")
parser.add_option("--data"     ,dest="data_set" ,type=str ,default=None ,help="The FSE simulation the use as data.")
parser.add_option("--partition",dest="part"     ,type=str ,default=None ,help="Data_set parition method")
parser.add_option("--train"    ,dest="train"    ,default=False,action="store_true",help="Set this flag to train model.")
parser.add_option("--predict"  ,dest="predict"  ,default=False,action="store_true",help="Set this flag use model.")
parser.add_option("--print"    ,dest="pm"       ,default=False,action="store_true",help="Set this flag to print available models.")
parser.add_option("--verbose"  ,dest="verbose"  ,default=False,action="store_true",help="Models be verbose during training.")
parser.add_option("--num_iter" ,dest="num_iter" ,type=int ,default=100  ,help="Number of iterations when training estimator.")

options, args = parser.parse_args()

# TODO  Divide training so that it uses a cross validation set and a test set so as to check recall.

if len(argv) == 1:
  parser.print_help()
  exit(0)

if options.pm:
  for key in models:
    if models[key].__doc__ is not None:
      print '\t'.join(('"%s"' % key , models[key].__doc__))
    else:
      print '"%s"' % key
  exit(0)

assert len(options.models) > 0 and (options.train or options.predict) and options.data_set and options.part

if options.load:
  assert len(options.models) == len(options.load)
if options.save:
  assert len(options.models) == len(options.save)

results = {}

dct = sio.loadmat(options.data_set)
X   = dct["X"]

if options.train:
  T1vals = np.squeeze(dct["T1vals"])
  T2vals = np.squeeze(dct["T2vals"])
  y      = partitions[options.part](T1vals, T2vals)
  action = 'train'
else:
  T1vals = None
  T2vals = None
  y      = None
  action = 'predict'

print "---------------------------------------------------------------------"
print "                            JT-Classify                              "
print "---------------------------------------------------------------------"
print "Action: " + action
print "---------------------------------------------------------------------"


for i in range(len(options.models)):
  m = options.models[i]
  save = None
  load = None
  if options.save:
    save = options.save[i]
  if options.load:
    load = options.load[i]
  print "Running " + m + "."
  guess = models[m](X, y, reg_lambda=0, train=options.train, predict=options.predict, num_iters=options.num_iter, save=save, load=load, verbose=options.verbose)
  res = {'X': X, 'action': action, 'y_true': y, 'T1': T1vals, 'T2': T2vals, 'y_guess': guess}
  results[m] = res
  print "---------------------------------------------------------------------"
