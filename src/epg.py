#!/usr/bin/python

# EPG Simulation code, based off of Matlab scripts from Brian Hargreaves <bah@stanford.edu>
# 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>

import numpy as np
from numpy import pi, cos, sin, exp, conj
from warnings import warn

def epg_precomute_RR(alphas, phi):
    T = len(alphas)
    if np.any(abs(alphas) > 2 * pi):
        warn('epg_rf: Flip angle should be in radians!')

    cos_alpha = np.cos(alphas)
    cos_alpha_2 = np.cos(alphas / 2.) ** 2
    sin_alpha = np.sin(alphas)
    sin_alpha_2 = np.sin(alphas / 2.) ** 2
    jphi = 1j * phi

    a01 = exp(2 * jphi)
    a02 = -1j * exp(jphi)
    a10 = exp(-2 * jphi)
    a12 = 1j * exp(-jphi)
    a20 = -1j / 2. * exp(-jphi)
    a21 = 1j / 2. * exp(jphi)

    RR = np.empty((3,3,T), dtype=np.complex128)

    RR[0, 0, :] = cos_alpha_2
    RR[0, 1, :] = a01 * sin_alpha_2
    RR[0, 2, :] = a02 * sin_alpha
    RR[1, 0, :] = a10 * sin_alpha_2
    RR[1, 1, :] = cos_alpha_2
    RR[1, 2, :] = a12 * sin_alpha
    RR[2, 0, :] = a20 * sin_alpha
    RR[2, 1, :] = a21 * sin_alpha
    RR[2, 2, :] = cos_alpha
    return RR

def epg_rf_new(FpFmZ, alphas, alpha_id, phi, precomputed_RR):
    """ Propagate EPG states through an RF rotation of
    alpha, with phase phi (both radians).
    """
    FpFmZ = precomputed_RR[:,:,alpha_id].dot(FpFmZ)
    return FpFmZ

def epg_rf(FpFmZ, alpha, phi):
    """ Propagate EPG states through an RF rotation of 
    alpha, with phase phi (both radians).

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.

    OUTPUT:
        FpFmZ = Updated FpFmZ state.
        RR = RF rotation matrix (3x3).

    SEE ALSO:
    epg_grad, epg_grelax
    """

    # -- From Weigel at al, JMR 205(2010)276-285, Eq. 8.
    
    if abs(alpha) > 2 * pi:
        warn('epg_rf: Flip angle should be in radians!')

    RR = np.array([ [ (cos(alpha/2.))**2, exp(2*1j*phi)*(sin(alpha/2.))**2, -1j*exp(1j*phi)*sin(alpha)],
            [exp(-2*1j*phi)*(sin(alpha/2.))**2, (cos(alpha/2.))**2, 1j*exp(-1j*phi)*sin(alpha)],
            [-1j/2.*exp(-1j*phi)*sin(alpha), 1j/2.*exp(1j*phi)*sin(alpha), cos(alpha)] ])

    FpFmZ = np.dot(RR , FpFmZ)

    return FpFmZ, RR

def epg_precompute_relax(T1, T2, T):
    E2 = exp(-T/T2)
    E1 = exp(-T/T1)
    RR = 1 - E1
    EE = np.array([E2, E2, E1]).reshape((3,1))  # Decay of states due to relaxation alone.
                                 # Mz Recovery, affects only Z0 state, as
                                 # recovered magnetization is not dephased.
    return EE, RR

def epg_relax_new(FpFmZ, EE, RR):
    """ Propagate EPG states through a period of relaxation.
    """

    FpFmZ *= EE         # Apply Relaxation
    FpFmZ[2,0] += RR    # Recovery

    return FpFmZ

def epg_relax(FpFmZ, T1, T2, T):
    """ Propagate EPG states through a period of relaxation over
    an interval T.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        T1,T2 = Relaxation times (same as T)
        T = Time interval (same as T1,T2)

    OUTPUT:
        FpFmZ = updated F+, F- and Z states.
        EE = decay matrix, 3x3 = diag([E2 E2 E1]);

    SEE ALSO:
           epg_grad, epg_rf
   """

    E2 = exp(-T/T2)
    E1 = exp(-T/T1)

    EE = np.diag([E2, E2, E1])      # Decay of states due to relaxation alone.
    RR = 1 - E1                     # Mz Recovery, affects only Z0 state, as 
                                    # recovered magnetization is not dephased.


    FpFmZ = np.dot(EE, FpFmZ)       # Apply Relaxation
    FpFmZ[2,0] = FpFmZ[2,0] + RR    # Recovery  

    return FpFmZ, EE

def epg_grad_new(FpFmZ, l, noadd=False):
    """Propagate EPG states through a "unit" gradient.
    """

    # Gradient does not affect the Z states.

    if noadd == False:
        l += 1
        # Set to zero
        FpFmZ[:, l-1] = 0.0                 # add higher dephased state

    FpFmZ[0, 1:l] = FpFmZ[0, :l-1]          # shift Fp states
    FpFmZ[1, :l-1] = FpFmZ[1, 1:l]          # shift Fm states
    FpFmZ[1, l-1] = 0.                      # Zero highest Fm state
    FpFmZ[0, 0] = conj(FpFmZ[1,0])          # Fill in lowest Fp state

    return FpFmZ, l

def epg_grad(FpFmZ, noadd=False):
    """Propagate EPG states through a "unit" gradient.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        noadd = True to NOT add any higher-order states - assume
                that they just go to zero.  Be careful - this
                speeds up simulations, but may compromise accuracy!

    OUTPUT:
        Updated FpFmZ state.

    SEE ALSO:
        epg_grad, epg_grelax
    """

    # Gradient does not affect the Z states.

    if noadd == False:
        FpFmZ = np.hstack((FpFmZ, [[0],[0],[0]]))   # add higher dephased state

    FpFmZ[0,:] = np.roll(FpFmZ[0,:], 1)     # shift Fp states
    FpFmZ[1,:] = np.roll(FpFmZ[1,:], -1)    # shift Fm states
    FpFmZ[1,-1] = 0                         # Zero highest Fm state
    FpFmZ[0,0] = conj(FpFmZ[1,0])           # Fill in lowest Fp state

    return FpFmZ

def FSE_signal(angles_rad, TE, T1, T2):
    """Simulate Fast Spin Echo sequence with specific flip angle sequence.

    INPUT:
        angles_rad = array of flip angles in radians equal to echo train length
        TE = echo time/spacing
        T1 = T1 value in seconds
        T2 = T2 value in seconds

    OUTPUT:
        Transverse (complex) magnetization value at each echo time
        
    """

    T = len(angles_rad)
    S = np.zeros((T,1), dtype=complex)

    P = np.array([[0],[0],[1]])    # initially in M0

    P = epg_rf(P, pi/2, pi/2)[0]    # 90 degree tip

    for i in range(T):
        alpha = angles_rad[i]
        P = epg_relax(P, T1, T2, TE/2.)[0]
        P = epg_grad(P)
        P = epg_rf(P, alpha, 0)[0]
        P = epg_relax(P, T1, T2, TE/2.)[0]
        P = epg_grad(P)

        S[i] = P[0,0]

    return S

def FSE_signal2(angles_rad, TE, T1, T2):
    """Simulate Fast Spin Echo sequence with specific flip angle sequence.

    INPUT:
        angles_rad = array of flip angles in radians equal to echo train length
        TE = echo time/spacing
        T1 = T1 value in seconds
        T2 = T2 value in seconds

    OUTPUT:
        S1 -- Transverse (complex) magnetization value at each echo time
        S2 -- Longitudinal magnetization at each echo time

    """

    T = len(angles_rad)
    S = np.zeros((T,1), dtype=complex)
    S2 = np.zeros((T,1), dtype=complex)

    P1 = np.zeros((3, 2*T+1), dtype=complex)
    len_P = 1
    P1[:, 0] = epg_rf(np.array([0., 0., 1.]), pi / 2, pi / 2)[0]

    matrices = epg_precomute_RR(angles_rad, 0)
    EE, RR = epg_precompute_relax(T1, T2, TE/2.)

    for i in range(T):
        P1 = epg_relax_new(P1, EE, RR)
        P1, len_P = epg_grad_new(P1, len_P)
        P1 = epg_rf_new(P1, angles_rad, i, 0, precomputed_RR=matrices)
        P1 = epg_relax_new(P1, EE, RR)
        P1, len_P = epg_grad_new(P1, len_P)
        S[i] = P1[0,0]
        S2[i] = P1[2,0]

    return S, S2

def FSE_signal2_old(angles_rad, TE, T1, T2):
    """Simulate Fast Spin Echo sequence with specific flip angle sequence.

    INPUT:
        angles_rad = array of flip angles in radians equal to echo train length
        TE = echo time/spacing
        T1 = T1 value in seconds
        T2 = T2 value in seconds

    OUTPUT:
        S1 -- Transverse (complex) magnetization value at each echo time
        S2 -- Longitudinal magnetization at each echo time

    """

    T = len(angles_rad)
    S = np.zeros((T,1), dtype=complex)
    S2 = np.zeros((T,1), dtype=complex)

    P = np.array([[0],[0],[1]])    # initially in M0
    P = epg_rf(P, pi/2, pi/2)[0]    # 90 degree tip

    for i in range(T):
        alpha = angles_rad[i]
        P = epg_relax(P, T1, T2, TE/2.)[0]
        P = epg_grad(P)
        P = epg_rf(P, alpha, 0)[0]
        P = epg_relax(P, T1, T2, TE/2.)[0]
        P = epg_grad(P)
        S[i] = P[0,0]
        S2[i] = P[2,0]
    return S, S2

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T1 = 1000e-3
    T2 = 200e-3

    TE = 5e-3

    N = 100
    angles = 120 * np.ones((N,))
    angles_rad = angles * pi / 180.

    S = FSE_signal(angles_rad, TE, T1, T2)
    S2 = abs(S)
    plt.plot(TE*1000*np.arange(1, N+1), S2)
    plt.xlabel('time (ms)')
    plt.ylabel('signal')
    plt.title('T1 = %.2f ms, T2 = %.2f ms' % (T1 * 1000, T2 * 1000))
    plt.show()
    

