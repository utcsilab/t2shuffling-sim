#!/usr/bin/env python

from optparse             import OptionParser
from os                   import system
from models               import models_dict
from cfl                  import readcfl, writecfl, cfl2sqcfl, sqcfl2mat, mat2sqcfl
from metrics              import get_metric
from gen_FSEmatrix        import gen_FSET1T2matrix
from sys                  import argv, exit
from warnings             import warn
import os.path

try:
    import joblib
    use_joblib = True
except ImportError:
    use_joblib = False
    warn('JobLib not found! setting par_jobs to 1')

import imp
import t2phantom as t2p

try:
    import Phantom
except ImportError:
        pass

try:
    from plot                 import plot_simulation, plot_cfl_signals
    import matplotlib.pyplot as plt
except ImportError:
        pass


import numpy as np
import scipy.io as sio

time_stamp = ""

np.random.seed(723)

parser = OptionParser()

# simulation options
parser.add_option("--genFSE", dest="genFSE", action="store_true", default=False, help="Set this flag if you want to generate a simulation")
parser.add_option("--loadFSE", dest="loadFSE", type=str, default=None, help="Pass in path to simulation mat FILE.")
parser.add_option("--saveFSE", dest="saveFSE", type=str, default=None, help="Pass in path (with file name) to file to save simulation.")


# constraint options
parser.add_option("--rvc", dest="rvc", action="store_true", default=False, help="Real value constraint on basis.")
parser.add_option("--avc", dest="avc", action="store_true", default=False, help="Absolute value constraint on basis.")

#cfl options
parser.add_option("--cfl", dest="cfl", type=str, default=None, help="Path to cfl file")

# genFSE 
parser.add_option("--numT2", dest="N", type=int, default=256, help="Number of T2 values to load in.")
parser.add_option("--angles", dest="angles", type=str, default=None, help="Load in Angles in degrees.")
parser.add_option("--ETL", dest="ETL", type=int, default=None, help="Load in ETL")
parser.add_option("--TE", dest="TE", type=float, default=5.568e-3, help="Echo spacing (TE) in seconds")
parser.add_option("--T1", dest="T1vals", action="append", help="Specify single T1 value")
parser.add_option("--T2", dest="T2vals", action="append", help="Specify single T2 value")
parser.add_option("--TR", dest="TRvals", action="append", help="Specify single TR value")
parser.add_option("--T1vals", dest="T1vals_mat", type=str, default=None, help="Load T1 values from .mat file with variable 'T1vals'")
parser.add_option("--T2vals", dest="T2vals_mat", type=str, default=None, help="Load T2 values from .mat file with variable 'T2vals'")
parser.add_option("--T1T2vals", dest="T1T2vals_mat", type=str, default=None, help="Load T1/T2 value pairs from .mat file with variable 'T1T2vals'")
parser.add_option("--TRvals", dest="TRvals_mat", type=str, default=None, help="Load TR values from .mat file with variable 'TRvals'")
parser.add_option("--TRvals_file", dest="TRvals_file", type=str, default=None, help="Load TR values from text file")
parser.add_option("--scan_deadtime_vals_file", dest="scan_deadtime_vals_file", type=str, default=None, help="Load scan deadtime values from text file")
parser.add_option("--driveq", dest="driven_equil", action="store_true", default=False, help="Simulate driven equilibrium")
parser.add_option("--fr_sign", dest="fr_sign", type=int, default=0., help="Sign of fast recovery flip angle for driven equilibrium: 0 for -1, 1 for 1 (default: 0 == -1)")
parser.add_option("--varTR", dest="varTR", action="store_true", default=False, help="Variable TR: concatenate the TR curves to form a joint subspace")
parser.add_option("--par_jobs", dest="par_jobs", type=int, default=1, help="Run par_jobs in parallel")

# contrast synthesis
parser.add_option("--contrast_synthesis", dest="contrast_synthesis", action="store_true", default=False, help="Generate contrast synthesis matrix")
parser.add_option("--set_csynth_name", dest="csynth_name", type=str, default=None, help="Pass this to change saved csynth name. USE ONLY IF TESTING 1 MODEL.")
parser.add_option("--save_csynth", dest="save_csynth", type=str, default=None, help="Pass in path to FOLDER to save csynth matrix.")

# universal options
parser.add_option("--e2s", dest="e2s", type=int, default=2, help="Echoes to skip")
parser.add_option("-K", "--dim", dest="k", type=int, default=None, help="Number of basis vectors to construct. This only effects the reconstructed Xhat")
parser.add_option("--model", dest="model", type=str, default=[], action="append", help="The model you want to test")
parser.add_option("--add_control", dest="add_control", action="store_true", default=False, help="Set this flag if you want to compare said model with svd")
parser.add_option("--print_models", dest="print_models", action="store_true", default=False, help="Print all the model choices and exit.")
parser.add_option("--no_verbose", dest="verbose", action="store_false", default=True, help="Turn off verbose output.")

# saving options
parser.add_option("--build_phantom", dest="build_phantom", action="store_true", default=False, help="Build a phantom")
parser.add_option("--load_phantom_data", dest="phantom_data", default=None, help="Pre-computed phantom data")
parser.add_option("--save_phantom", dest="save_phantom", type=str, default=None, help="Pass in path to FOLDER to save phantom.")
parser.add_option("--set_phantom_name", dest="phantom_name", type=str, default=None, help="Pass this to change saved phantom name")
parser.add_option("--set_t2map_name", dest="t2map_name", type=str, default=None, help="Pass this to change saved t2 map name")

parser.add_option("--save_basis", dest="save_basis", type=str, default=None, help="Pass in path to FOLDER to save basis.")
parser.add_option("--set_basis_name", dest="basis_name", type=str, default=None, help="Pass this to change saved basis name. USE ONLY IF TESTING 1 MODEL.")

parser.add_option("--save_scales", dest="save_scales", type=str, default=None, help="Pass in path to FOLDER to save coefficient scaling.")
parser.add_option("--set_scale_name", dest="scale_name", type=str, default=None, help="Pass this to change saved scaling name. USE ONLY IF TESTING 1 MODEL.")

parser.add_option("--save_sim", dest="save_sim", type=str, default=None, help="Pass in path to FOLDER to save EPG simulations.")
parser.add_option("--set_sim_name", dest="sim_name", type=str, default=None, help="Pass this to change saved EPG sim name.")

parser.add_option("--save_plots", dest="save_plots", type=str, default=None, help="Pass in path to FOLDER to save plots.")
parser.add_option("--save_imgs", dest="save_imgs", type=str, default=None, help="Pass in path to FOLDER to save images.")



options, args = parser.parse_args()


if len(argv) == 1:
    parser.print_help()
    exit(0)


if options.print_models:
    for key in models_dict:
        if models_dict[key].__doc__ is not None:
            print('\t'.join(('"%s"' % key , models_dict[key].__doc__)))
        else:
            print('"%s"' % key)
    exit(0)


assert (options.cfl or options.loadFSE or options.genFSE), "Please pass in a cfl file XOR an angles file XOR an FSEsim."

assert int(options.cfl != None) + int(options.loadFSE != None) + int(options.genFSE), "Please pass in cfl XOR angles XOR FSEsim."

assert not (options.rvc and options.avc), "Please choose real-value NAND abs-value constraint."


if options.rvc:
    rvc = 'real'
elif options.avc:
    rvc = 'abs'
else:
    rvc = None


if options.save_imgs:
    assert options.cfl, "In order to save images, a cfl file must be passed instead of values for a simulation."

if options.basis_name != None:
    assert len(options.model) == 1, "In order to change the saved basis name, you must test a single model. "

if options.cfl:

    cfl_name = options.cfl.split('.')[-1]
    X, img_dim = sqcfl2mat(cfl2sqcfl(readcfl(options.cfl), options.e2s))

elif options.genFSE: 

    assert options.angles is not None, "In order to generate FSEmatrix, you must pass in an angles file."
    time_stamp = options.angles.split('.')[-1] 
    try:
        int(time_stamp)
        time_stamp = "." + time_stamp
    except ValueError:
        time_stamp = "" 
    except AssertionError:
        time_stamp = ""

    angles = np.loadtxt(options.angles) * np.pi / 180
    e2s = options.e2s
    TE  = options.TE

    if options.phantom_data != None:
        phantom_data = readcfl(options.phantom_data)
        N = phantom_data.shape[2]
    else:
        N   = options.N

    if options.T1T2vals_mat is not None:
        # keep the T1T2vals as pairs
        T1T2_mode = True
        T1T2vals = sio.loadmat(options.T1T2vals_mat)['T1T2vals']
    else:
        T1T2_mode = False
        if options.T1vals is not None:
            T1vals = np.array([float(T1) for T1 in options.T1vals])
        elif options.T1vals_mat is not None:
            T1vals = sio.loadmat(options.T1vals_mat)['T1vals']
        else:
            T1vals = np.array([500, 550, 600, 650, 700, 1000, 1800, 2000]) * 1e-3

        if options.T2vals is not None:
            T2vals = np.array([float(T2) for T2 in options.T2vals])
        elif options.T2vals_mat is not None:
            T2vals = sio.loadmat(options.T2vals_mat)['T2vals']
        else:
            T2vals = np.linspace(20e-3, 2000e-3, N)

    if options.scan_deadtime_vals_file is not None:
        # convert scan_deadtime to TR
        f = open(options.scan_deadtime_vals_file, 'r')
        scan_deadtime_vals = np.array([float(line) for line in f.readlines()]) * 1e-6 # raw units are in us
        f.close()
        TRvals = scan_deadtime_vals + len(angles) * TE
    elif options.TRvals is not None:
        TRvals = np.array([float(TR) for TR in options.TRvals])
    elif options.TRvals_mat is not None:
        TRvals = sio.loadmat(options.TRvals_mat)['TRvals']
    elif options.TRvals_file is not None:
        f = open(options.TRvals_file, 'r')
        TRvals = np.array([float(line) for line in f.readlines()]) * 1e-6 # raw units are in us
        f.close()
    else:
        TRvals = np.array([np.inf])

    print('TRvals:')
    print(TRvals)

    if T1T2_mode:
        if T1T2vals.shape[0] > N:
            idx = np.random.permutation(T1T2vals.shape[0])
            T1T2vals = T1T2vals[idx[0:N],:]
        N = T1T2vals.shape[0]
    else:
        if T2vals.shape[0] > N:
            idx = np.random.permutation(T2vals.shape[0])
            T2vals = T2vals[idx[0:N]]
        N = len(T2vals)
        T1vals = np.sort(np.ravel(T1vals))
        T2vals = np.sort(np.ravel(T2vals))

    T = len(angles)
    R = len(TRvals)
    
    if not options.ETL:
        ETL = T - e2s - 1
    else:
        ETL = options.ETL

    if not T1T2_mode:
        _T1vals, _T2vals = np.meshgrid(T1vals.ravel(), T2vals.ravel())
        T1T2vals = np.vstack((_T1vals.ravel(), _T2vals.ravel())).T
        T1T2_mode = True

    if not use_joblib:
        options.par_jobs = 1

    if options.fr_sign == 0:
        fr_sign = -1
    else:
        fr_sign = 1

    X = gen_FSET1T2matrix(angles, ETL, e2s, TE, T1T2vals, TRvals, options.driven_equil, fr_sign, options.verbose, options.par_jobs)

    if options.saveFSE != None:
        if options.verbose:
            print("Saving as " + options.saveFSE + time_stamp)
        if T1T2_mode:
            sio.savemat(options.saveFSE + time_stamp, {"X": X, "angles": angles, "N": N, "ETL": ETL, "e2s": e2s, "TE": TE, "T1T2vals": T1T2vals, "TRvals": TRvals})
        else:
            sio.savemat(options.saveFSE + time_stamp, {"X": X, "angles": angles, "N": N, "ETL": ETL, "e2s": e2s, "TE": TE, "T1vals": T1vals, "T2vals": T2vals, "TRvals": TRvals})

else:
    dct = sio.loadmat(options.loadFSE)
    X = dct["X"]
    angles = dct["angles"]
    N = dct["N"]
    ETL = dct["ETL"]
    e2s = dct["e2s"]
    TE = dct["TE"]
    if T1T2_mode:
        T1T1vals = np.sort(np.ravel(dct["T1T2vals"]))
    else:
        T1vals = np.sort(np.ravel(dct["T1vals"]))
        T2vals = np.sort(np.ravel(dct["T2vals"]))


if rvc == 'real':
    if options.verbose:
        print('real value constraint')
    X = np.real(X)
elif rvc == 'abs':
    if options.verbose:
        print('abs value constraint')
    X = np.abs(X)


lst = options.model
if options.add_control:
    lst.append('simple_svd')
if options.model == 'all':
    lst = models.keys()
results = {}
for m in lst:
    if options.verbose:
        print("------------------------------------------------------------")
        print("Running " + m)
        print("------------------------------------------------------------")
    model = models_dict[m]
    k = options.k
    if options.varTR:
        X = np.transpose(X, (2, 0, 1)).reshape((ETL * TRvals.size, -1))
    else:
        X = X.reshape((ETL, -1))

    X = X[:, ~np.all(X == 0, axis=0)] # remove all-zero signals, for cases where T2 > T1

    U, alpha, X_hat, S = model(X, options.k, rvc)

    if options.verbose:
        print("Results:")
    signal_perc_err, TE_perc_err, fro_perc_err = get_metric(X, X_hat, options.verbose)
    if not k:
        k = U.shape[1]
    results[m] = {'U': U, 'alpha': alpha, 'k': k, 'X_hat': X_hat, 'Percentage Error per Signal': signal_perc_err, 'Percentage Error per TE': TE_perc_err, 'Frobenius Percentage Error': fro_perc_err, 'S': S}
if options.verbose:
    print("------------------------------------------------------------")


if options.save_plots != None:
    for m in lst:
        mod = results[m]
        if options.cfl:
            plot_cfl_signals(mod['U'], options.k, X, mod['X_hat'], "cfl_" + m, options.e2s, options.save_plots)
        else:
            plot_simulation(mod['U'], options.k, X, mod['X_hat'], "sim_" + m, T1vals, T2vals, e2s, options.save_plots)


if options.save_imgs != None:
    # TODO
    warn('TODO: implement options.save_imgs')
    None


if options.contrast_synthesis:
    if options.verbose:
        print("------------------------------------------------------------")
        print("Computing contrast synthesis matrix")
        print("")

    Xe = gen_FSEmatrix(N, np.pi * np.ones(T), ETL, e2s, TE, T1vals, T2vals).reshape((ETL, -1))
    Xe = Xe[:, ~np.all(Xe == 0, axis=0)] # remove all-zero signals, for cases where T2 > T1

    assert (not options.varTR), "Variable TR mode not implemented for contrast synthesis!"

    for m in results.keys():
        k = results[m]['k']
        res = results[m]
        alpha = res['alpha'][:k, :]
        AA = np.kron(np.eye(ETL), np.asarray(alpha.T))
        y = np.ravel(Xe)
        Qflat = np.linalg.lstsq(AA, y)[0]
        Q = np.reshape(Qflat, (ETL, k))
        res['Q'] = Q

        if options.save_csynth != None:
            print(options.csynth_name)

            for i in range(5):
                Q = np.expand_dims(Q, axis=0)

            k_ext = '_k_%d' % k

            if options.csynth_name != None:
                cfl_name = options.csynth_name  
            else:
                cfl_name = 'csynth.' + m + k_ext + timestamp

            writecfl(os.path.join(options.save_csynth, cfl_name), Q)


if options.save_basis != None:
    if options.verbose:
        print(options.basis_name)

    for m in results.keys():
        U = results[m]["U"]
        k = results[m]['k']
        if U.shape[1] < U.shape[0]:
            U = np.hstack((U, np.zeros((U.shape[0], U.shape[0] - k))))

    for i in range(5):
        U = np.expand_dims(U, axis=0)

    k_ext = '_k_%d' % k

    if options.basis_name != None:
        cfl_name = options.basis_name  
    else:
        cfl_name = 'bas.' + m + k_ext + timestamp

    writecfl(os.path.join(options.save_basis, cfl_name), U)

if options.save_scales != None:
    if options.verbose:
        print(options.scale_name)

    for m in results.keys():
        S = results[m]["S"]

    if options.basis_name != None:
        cfl_name = options.scale_name  
    else:
        cfl_name = 'scale.' + m + timestamp

    sc = (S/S[0]).reshape((1, 1, 1, 1, 1, 1, len(S)))
    writecfl(os.path.join(options.save_scales, cfl_name), sc)

if options.save_sim != None:
    if options.verbose:
        print(options.sim_name)

    if options.sim_name != None:
        npy_name = options.sim_name  
    else:
        npy_name = 'sim.' + m + timestamp

    np.save(os.path.join(options.save_sim, npy_name), X)


if options.build_phantom:
    if options.verbose:
        print("------------------------------------------------------------")
        print("Building Phantom")
        print("")
    if options.phantom_data != None:
        if options.verbose:
            print(X.shape)
            print(ETL)
        ksp = np.zeros((phantom_data.shape[0], phantom_data.shape[1], ETL), dtype='complex64')
        if options.verbose:
            print(ksp.shape)
        imgs = Phantom.ifft2c(np.transpose(phantom_data, (1, 0, 2)))
        T2im = np.zeros(phantom_data.shape[:2])
        idx = np.random.permutation(N)
        idx = range(N)
        for i in range(N)[::-1]:
            # workaround for values stacking
            scale = .99**i
            ksp += scale * phantom_data[:, :, i].T[:, :, None] * X[None, None, :, idx[i]]
            T2im[np.where(np.abs(imgs[:, :, i]) > 1)] += scale *T2vals[idx[i]]
    else:
        FOV = (25.6, 25.6) # cm
        dims = np.array([240, 260]) # pixels
        res = FOV / dims
        P = t2p.Phantom(FOV, res)
        P.knee_objects_relax(T2vals, X)
        ksp = np.fliplr(P.build_flipmod()[None, :, :, None, None, :])
    if options.verbose:
        print("------------------------------------------------------------")

    if options.save_phantom != None:
        if options.phantom_name != None:
            cfl_name = options.phantom_name
        else:
            cfl_name = 'ksp.' + m + k_ext + timestamp

        writecfl(os.path.join(options.save_phantom, cfl_name), ksp[None, :, :, None, None, :])

        if options.t2map_name != None:
            cfl_name = options.t2map_name
        else:
            cfl_name = 't2map.' + m + k_ext + timestamp

        writecfl(os.path.join(options.save_phantom, cfl_name), T2im[None, :, :])
