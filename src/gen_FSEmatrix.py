from __future__ import division
from epg import FSE_signal


import numpy as np
import sys


def gen_FSEmatrix(N, angles_rad, ETL, e2s, TE, T1vals, T2vals):
  T = len(angles_rad)

  T1 = np.sort(T1vals)
  T2 = np.sort(T2vals)

  X = np.zeros((T, N * len(T1)), dtype=np.complex64)

  c = 0
  total = N * len(T1)
  num_iter = T1.size * T2.size
  print "Generating FSE matrix"
  for T2 in T2vals:
    for T1 in T1vals:
      X[:, c] = FSE_signal(angles_rad, TE, T1, T2)[:,0]
      c = c + 1
      sys.stdout.write("\rProgress: %d%%" % (c/num_iter * 100))
      sys.stdout.flush()
  print ""

  keep = range(e2s, e2s + ETL)
  X = X[keep, :]
  return X
