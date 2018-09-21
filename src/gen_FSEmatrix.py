from epg import FSE_signal2

try:
    import joblib
except ImportError:
    pass


import numpy as np
import sys


def FSE_exp_signal(T, TE, T2):
    echo_times = np.arange(T)*TE + TE
    return np.exp( -echo_times / T2)

def my_FSE_T1T2_fun(args, vars):

    angles_rad, TE, TRvals, driven_equil, fr_sign, disp = args
    T1, T2 = vars

    T = len(angles_rad)
    LTR = TRvals.size

    use_exp_decay = np.all(angles_rad == np.pi)


    x_full = np.zeros((T, LTR), dtype=np.complex64)

    if T2 < T1:
        if use_exp_decay:
            x = FSE_exp_signal(T, TE, T2)
            z = np.array([0.])
        else:
            x, z = FSE_signal2(angles_rad, TE, T1, T2)

        for i3, TR in enumerate(TRvals):

            if driven_equil:
                E1 = np.exp(-(TR - (T+1)*TE) / T1)
                M0 = (1 - E1) / (1 + fr_sign *  E1 * abs(x[-1]))
            else:
                E1 = np.exp(-(TR - T*TE) / T1)
                M0 = (1 - E1) / (1 - E1 * abs(z[-1]))

            x_full[:, i3] = x.ravel() * M0

    return x_full

def gen_FSEmatrix(angles_rad, ETL, e2s, TE, T1vals, T2vals, TRvals=np.array([np.inf]), driven_equil=False, fr_sign=-1, disp=True, par_jobs=8):

    _T1vals, _T2vals = np.meshgrid(T1vals.ravel(), T2vals.ravel())
    T1T2vals = np.vstack((_T1vals.ravel(), _T2vals.ravel())).T

    return gen_FSET1T2matrix(angles_rad, ETL, e2s, TE, T1T2vals, TRvals, driven_equil, fr_sign, disp, par_jobs)


def gen_FSET1T2matrix(angles_rad, ETL, e2s, TE, T1T2vals, TRvals=np.array([np.inf]), driven_equil=False, fr_sign=-1, disp=True, par_jobs=8):

    if disp:
        print("Generating FSE matrix")

    args = (angles_rad, TE, TRvals, driven_equil, fr_sign, disp)

    L = T1T2vals.shape[0]

    if par_jobs > 1:
        if disp:
            print("Total tasks: %d" % L)
        X_full = joblib.Parallel(n_jobs=par_jobs, verbose=disp)(joblib.delayed(my_FSE_T1T2_fun)(args, T1T2) for T1T2 in T1T2vals)
    else:
        X_full = []

        for i0, T1T2 in enumerate(T1T2vals):

            T1 = T1T2[0]
            T2 = T1T2[1]

            args = (angles_rad, TE, TRvals, driven_equil, disp)
            vars = (T1, T2)

            X_full.append(my_FSE_T1T2_fun(args, vars))

            if disp:
                sys.stdout.write("\rProgress: %d%%" % (i0 / L * 100))
                sys.stdout.flush()

    X_full = np.array(X_full).transpose((1, 0, 2))

    if disp:
        print("")

    keep = range(e2s, e2s + ETL)
    return X_full[keep, :, :]
