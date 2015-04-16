#!/usr/bin/env python


from __future__           import division
from optparse             import OptionParser
from os                   import system
from multilayer_regressor import multilayer_regressor as mr


import numpy    as np
import scipy.io as sio


parser = OptionParser()

# estimator options
parser.add_option("--estimator", dest="estimator", action="store_true", default=False, help="Set this flag if you want to run the estimator.")
parser.add_option("--train", dest="train", action="store_true", default=False, help="Set this flag if you want to train the estimator.")
parser.add_option("--predict", dest="predict", action="store_true", default=False, help="Set this flag if you want to see the predictions.")
parser.add_option("--est-iter", dest="estiter", type=int, default=100, help="Number of iterations when training estimator.")
parser.add_option("--loadEst", dest="loadEst", type=str, default=None, help="Pass in path to estimator npy file.")
parser.add_option("--saveEst", dest="saveEst", type=str, default=None, help="Pass in path (with file name) to file to save estimator.")

options, args = parser.parse_args()


if len(argv) == 1:
  parser.print_help()
  exit(0)


print 'test'
