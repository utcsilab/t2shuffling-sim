from __future__ import division

import numpy as np

norm = np.linalg.norm

def get_metric(X, Xhat, disp=True):
  delta = X - Xhat
  signal_norm = norm(delta, ord=2, axis=0)
  fro_norm = norm(delta, ord=2) 
  signal_perc_err = 100 * signal_norm/norm(X, ord=2, axis=0)
  TE_norm = norm(delta, ord=2, axis=1)
  TE_perc_err = 100 * TE_norm/norm(X, ord=2, axis=1)
  fro_perc_err = fro_norm/norm(X, 'fro') * 100
  if disp:
    print "Maximum percentage error for any single signal: %f" % max(signal_perc_err)
    print "Minimum percentage error for any single TE: %f" % min(TE_perc_err)
    print "Total Percentage error: %f" % fro_perc_err
  return signal_perc_err, TE_perc_err, fro_perc_err
