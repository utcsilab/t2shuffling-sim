# The purpose of this file is to have different return different labels

import numpy as np

jt_partitions_dict = {}

def as_is(T1vals, T2vals):
  y = np.zeros((2, len(T1vals) * len(T2vals)))
  c = 0
  for T2 in np.squeeze(T2vals):
    for T1 in np.squeeze(T1vals):
      y[0, c] = T1
      y[1, c] = T2
      c += 1
  return y

jt_partitions_dict['as_is'] = as_is

def ten(T1vals, T2vals):
  y = np.zeros((10, len(T1vals) * len(T2vals)))
  c = 0
  for T2 in np.squeeze(T2vals):
    for T1 in np.squeeze(T1vals):
      for i in range(0, 10):
        y[i, c] = int(i == (c % 10))
      c += 1
  return y

jt_partitions_dict['ten'] = ten
