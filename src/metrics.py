import numpy as np

norm = np.linalg.norm

def get_metric(X, Xhat):
  delta = X - Xhat
  snorm = norm(delta, ord=1, axis=0) #Norm per signal
  fnorm = norm(delta, ord=1)         #Total norm
  pnorm = (snorm/norm(X, ord=1, axis=0)) * 100 #Percentage Error
  fro_perc_err = fnorm/norm(X) * 100
  print "Maximum percentage error for any single signal: %f" % max(pnorm)
  print "Total Percentage error: %f" % fro_perc_err
  return pnorm, fro_perc_err
