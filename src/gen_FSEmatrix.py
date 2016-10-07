from __future__ import division
from epg import FSE_signal2


import numpy as np
import sys


def FSE_exp_signal(T, TE, T2):
    echo_times = np.arange(T)*TE + TE
    return np.exp( -echo_times / T2)


def gen_FSEmatrix(angles_rad, ETL, e2s, TE, T1vals, T2vals, TRvals=np.array([np.inf]), driven_equil=False, disp=True):

    use_exp_decay = np.all(angles_rad == np.pi)

    T = len(angles_rad)

    LT1 = T1vals.size
    LT2 = T2vals.size
    LTR = TRvals.size

    Xfull = np.zeros((T, LT2, LT1, LTR), dtype=np.complex64)

    c = 0
    num_iter = LT1 * LT2 * LTR

    if disp:
        print "Generating FSE matrix"

    for i1, T2 in enumerate(T2vals):

        for i2, T1 in enumerate(T1vals):

            if T2 < T1:
                if use_exp_decay:
                    x = FSE_exp_signal(T, TE, T2)
                    z = np.array([0.])
                else:
                    x, z = FSE_signal2(angles_rad, TE, T1, T2)

                for i3, TR in enumerate(TRvals):

                    E1 = np.exp(-(TR - T*TE) / T1)

                    if driven_equil:
                        M0 = (1 - E1) * (1 - E1 * abs(x[-1]))
                    else:
                        M0 = (1 - E1) * (1 - E1 * abs(z[-1]))

                    Xfull[:, i1, i2, i3] = x.ravel() * M0
            else:
                Xfull[:, i1, i2, :] = 0

            c = c + LTR

        if disp:
            sys.stdout.write("\rProgress: %d%%" % (c/num_iter * 100))
            sys.stdout.flush()
    if disp:
        print ""

    keep = range(e2s, e2s + ETL)
    return Xfull[keep, :, :, :]

def gen_FSET1T2matrix(angles_rad, ETL, e2s, TE, T1T2vals, TRvals=np.array([np.inf]), driven_equil=False, disp=True):

    use_exp_decay = np.all(angles_rad == np.pi)

    T = len(angles_rad)

    L = T1T2vals.shape[0]
    LTR = TRvals.size

    Xfull = np.zeros((T, L, LTR), dtype=np.complex64)

    if disp:
        print "Generating FSE matrix"

    for i0, T1T2 in enumerate(T1T2vals):

        T1 = T1T2[0]
        T2 = T1T2[1]

        if T2 < T1:

            if use_exp_decay:
                x = FSE_exp_signal(T, TE, T2)
                z = np.array([0.])
            else:
                x, z = FSE_signal2(angles_rad, TE, T1, T2)

            for i1, TR in enumerate(TRvals):

                E1 = np.exp(-(TR - T*TE) / T1)

                if driven_equil:
                    M0 = (1 - E1) * (1 - E1 * abs(x[-1]))
                else:
                    M0 = (1 - E1) * (1 - E1 * abs(z[-1]))

                Xfull[:, i0, i1] = x.ravel() * M0
        else:
            Xfull[:, i0, :] = 0


        if disp:
            sys.stdout.write("\rProgress: %d%%" % (i0 / L * 100))
            sys.stdout.flush()

    if disp:
        print ""

    keep = range(e2s, e2s + ETL)
    return Xfull[keep, :, :]
