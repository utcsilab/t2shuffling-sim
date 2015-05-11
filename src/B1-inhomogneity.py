
# coding: utf-8

# In[3]:

import numpy as np
from epg import *
import scipy.io


# In[4]:

TE = 4.048e-3
angles = np.loadtxt('/Users/jtamir/scan-data/feet/synth_foot/sim_for_paper/flipmod/flipangles.txt.408183520')
angles_rad = angles * np.pi / 180.
T = len(angles_rad)


# In[7]:

T1 = 1000e-3

M = 256
N = 256

B_errs = np.linspace(-.4, .4, M)
T2_vals = np.linspace(10, 1000, N) * 1e-3



A_vals = np.zeros((T, M, N))

for b in range(M):
    B_err = B_errs[b]
    print b
    
    for j in range(N):
        T2 = T2_vals[j]
    
        P = np.matrix([[0],[0],[1]])    # initially in M0

        alpha = pi/2 * (1 + B_err)
        P = epg_rf(P, alpha, pi/2)[0]    # 90 degree tip

        for i in range(T):
            alpha = angles_rad[i] * (1 + B_err)
            P = epg_relax(P, T1, T2, TE/2.)[0]
            P = epg_grad(P)
            P = epg_rf(P, alpha, 0)[0]
            P = epg_relax(P, T1, T2, TE/2.)[0]
            P = epg_grad(P)
            
            A_vals[i, b, j] = np.abs(P[0,0])


A_0 = np.zeros((T, N))
B_err = 0.

for j in range(N):
    T2 = T2_vals[j]

    P = np.matrix([[0],[0],[1]])    # initially in M0

    alpha = pi/2 * (1 + B_err)
    P = epg_rf(P, alpha, pi/2)[0]    # 90 degree tip

    for i in range(T):
        alpha = angles_rad[i] * (1 + B_err)
        P = epg_relax(P, T1, T2, TE/2.)[0]
        P = epg_grad(P)
        P = epg_rf(P, alpha, 0)[0]
        P = epg_relax(P, T1, T2, TE/2.)[0]
        P = epg_grad(P)
        
        A_0[i, j] = np.abs(P[0,0])


