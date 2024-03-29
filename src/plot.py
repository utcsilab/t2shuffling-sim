#!/usr/bin/env python


from __future__ import division
from matplotlib import gridspec
from metrics    import get_metric


import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


img_size = (20, 15)


def matrix_to_1Darray(matrix):
  return np.array(matrix).reshape(matrix.size)


def mksv_plot(ax, data, title, xlabel, ylabel, xstart=0, ylim=None, other_label=None):
  if type(data) != tuple:
    x_axis = np.arange(xstart, xstart + data.shape[0])
    ax.set_xlim(xstart, xstart + data.shape[0] - 1)
    ax.set_xticks(np.arange(xstart, xstart + data.shape[0], 2))
    ax.plot(x_axis, data)
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
      ax.set_ylim(ylim)
  elif type(data) == tuple:
    assert len(data) >= 3 and type(ax == list)
    x, t, y = [list(matrix_to_1Darray(elm)) for elm in data]
    skip = len(t)
    for i in range(skip):
      ax[i].plot(x, y[i::skip])
      ax[i].set_xlabel(xlabel)
      ax[i].set_ylabel(ylabel)
      ax[i].set_title(title + ", " + other_label + str(t[i]))
      ax[i].grid()
      if ylim:
        ax[i].set_ylim(ylim)


def plot_simulation(U, k, X, X_hat, model_name, T1vals, T2vals, e2s, path):
  span = T1vals.size
  pnorm, tnorm, fro_perc_err = get_metric(X, X_hat, disp=False)
  fig = plt.figure(figsize=img_size)
  half = int(np.floor(span/2))
  ax1 = plt.subplot2grid((4, span), (0,0), colspan=half)
  ax2 = plt.subplot2grid((4, span), (0,half), colspan=half)
  ax3 = plt.subplot2grid((4, span), (1,0), colspan=span)
  ax4 = plt.subplot2grid((4, span), (2,0), colspan=span)
  ax5 = [plt.subplot2grid((4, span), (3, i)) for i in range(span)]
  mksv_plot(ax1, X/np.max(X), 'Signals', 'TE Number', 'Signal', xstart=e2s, ylim=[0, 1])
  mksv_plot(ax2, X_hat/np.max(X_hat), 'Recovered Signals', 'TE number', 'Signal', xstart=e2s, ylim=[0, 1])
  mksv_plot(ax3, abs(X-X_hat), 'Difference', 'TE number', 'Diff', xstart=e2s)
  mksv_plot(ax4, -U[:, :k], 'Basis', 'TE number', 'Signal', xstart=e2s)
  mksv_plot(ax5, (T2vals * 1000, T1vals * 1000, pnorm), 'T2_Error', 'T2_vals (ms)', 'Percentage Error', other_label="T1 (in ms): ", ylim=[0,12])
  fig.tight_layout()
  plt.subplots_adjust(top=0.925, bottom=0.1)
  fig.text(0.1115, 0.025, "Total Percentage Error: " + str(fro_perc_err) + \
    "\nMax Percentage Error for any single signal: " + str(max(pnorm)) + "%." + \
    "Lowest Percentage Error at any TE: " + str(min(tnorm)), bbox={'alpha':0.5, 'pad':10}, fontsize=20)
  fig.text(0.1115, 1 - 0.025, "Model: " + model_name, fontsize=20, bbox={'alpha':0.5, 'pad':10})
  fig.savefig(path + model_name + '.png')

def plot_cfl_signals(U, k, X, X_hat, model_name, e2s, path):
  span = 2
  half = 1
  pnorm, fro_perc_err = get_metric(X, X_hat, disp=False)
  fig = plt.figure(figsize=img_size)
  ax1 = plt.subplot2grid((4, span), (0,0), colspan=half)
  ax2 = plt.subplot2grid((4, span), (0,half), colspan=half)
  ax3 = plt.subplot2grid((4, span), (1,0), colspan=span)
  ax4 = plt.subplot2grid((4, span), (2,0), colspan=span)
  ax5 = plt.subplot2grid((4, span), (3,0), colspan=span)
  mksv_plot(ax1, X/np.max(X), 'Signals ', 'TE Number', 'Signal', xstart=e2s, ylim=[0, 1])
  mksv_plot(ax2, X_hat/np.max(X_hat), 'Recovered Signals', 'TE number', 'Signal', xstart=e2s, ylim=[0, 1])
  mksv_plot(ax3, abs(X-X_hat), 'Difference (Abs)', 'TE number', 'Diff', xstart=e2s)
  mksv_plot(ax4, -U[:, :k], 'Basis', 'TE number', 'Signal', xstart=e2s)

  n_idx = np.argsort(np.linalg.norm(X, ord=1, axis=0))
  diff  = np.matrix(pnorm[n_idx]).T
  mksv_plot(ax5, diff, "Percentage error - Increasing norm", "TE Number", "Perc Error", xstart=e2s, ylim=[0, 20])

  fig.tight_layout()
  plt.subplots_adjust(top=0.925, bottom=0.1)
  fig.text(0.1115, 0.025, "Total Percentage Error: " + str(fro_perc_err) + \
    "%\nMax Percentage Error for any single signal: " + str(max(pnorm)) + "%", bbox={'alpha':0.5, 'pad':10}, fontsize=20)
  fig.text(0.1115, 1 - 0.025, "Model: " + model_name, fontsize=20, bbox={'alpha':0.5, 'pad':10})
  fig.savefig(path + model_name + '.png')
