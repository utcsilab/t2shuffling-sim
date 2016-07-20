#!/usr/bin/env python

from __future__           import division
from optparse             import OptionParser
from os                   import system
from models               import models_dict
from cfl                  import readcfl, writecfl, cfl2sqcfl, sqcfl2mat, mat2sqcfl
from metrics              import get_metric
from gen_FSEmatrix        import gen_FSEmatrix
from sys                  import argv, exit
from warnings             import warn
import os.path

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
parser.add_option("--TRvals", dest="TRvals_mat", type=str, default=None, help="Load TR values from .mat file with variable 'TRvals'")
parser.add_option("--driveq", dest="driven_equil", action="store_true", default=False, help="Simulate driven equilibrium")

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
parser.add_option("--save_plots", dest="save_plots", type=str, default=None, help="Pass in path to FOLDER to save plots.")
parser.add_option("--save_imgs", dest="save_imgs", type=str, default=None, help="Pass in path to FOLDER to save images.")
parser.add_option("--set_basis_name", dest="basis_name", type=str, default=None, help="Pass this to change saved basis name. USE ONLY IF TESTING 1 MODEL.")



options, args = parser.parse_args()


if len(argv) == 1:
    parser.print_help()
    exit(0)


if options.print_models:
    for key in models_dict:
        if models_dict[key].__doc__ is not None:
            print '\t'.join(('"%s"' % key , models_dict[key].__doc__))
        else:
            print '"%s"' % key
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

    if options.T1vals is not None:
        T1vals = np.array([float(T1) for T1 in options.T1vals])
    elif options.T1vals_mat is not None:
        T1vals = sio.loadmat(options.T1vals_mat)['T1vals']
    else:
        T1vals = np.array([500, 700, 1000, 1800]) * 1e-3

    if options.T2vals is not None:
        T2vals = np.array([float(T2) for T2 in options.T2vals])
    elif options.T2vals_mat is not None:
        T2vals = sio.loadmat(options.T2vals_mat)['T2vals']
    else:
        T2vals = np.linspace(20e-3, 800e-3, N)

    if options.TRvals is not None:
        TRvals = np.array([float(TR) for TR in options.TRvals])
    elif options.TRvals_mat is not None:
        TRvals = sio.loadmat(options.TRvals_mat)['TRvals']
    else:
        TRvals = np.array([np.inf])

    if T2vals.shape[0] > N:
        idx = np.random.permutation(T2vals.shape[0])
        T2vals = T2vals[idx[0:N]]

    T = len(angles)
    N = len(T2vals)
    R = len(TRvals)
    
    if not options.ETL:
        ETL = T - e2s - 1
    else:
        ETL = options.ETL

    T1vals = np.sort(np.ravel(T1vals))
    T2vals = np.sort(np.ravel(T2vals))

    X = gen_FSEmatrix(angles, ETL, e2s, TE, T1vals, T2vals, TRvals, options.driven_equil, options.verbose)

    if options.saveFSE != None:
        if options.verbose:
            print "Saving as " + options.saveFSE + time_stamp
        sio.savemat(options.saveFSE + time_stamp, {"X": X, "angles": angles, "N": N, "ETL":ETL, "e2s":e2s, "TE": TE, "T1vals":T1vals, "T2vals":T2vals})

else:
    dct = sio.loadmat(options.loadFSE)
    X = dct["X"]
    angles = dct["angles"]
    N = dct["N"]
    ETL = dct["ETL"]
    e2s = dct["e2s"]
    TE = dct["TE"]
    T1vals = np.sort(np.ravel(dct["T1vals"]))
    T2vals = np.sort(np.ravel(dct["T2vals"]))


if rvc == 'real':
    if options.verbose:
        print 'real value constraint'
    X = np.real(X)
elif rvc == 'abs':
    if options.verbose:
        print 'abs value constraint'
    X = np.abs(X)


lst = options.model
if options.add_control:
    lst.append('simple_svd')
if options.model == 'all':
    lst = models.keys()
results = {}
for m in lst:
    if options.verbose:
        print "------------------------------------------------------------"
        print "Running " + m
        print "------------------------------------------------------------"
    model = models_dict[m]
    k = options.k
    U, alpha, X_hat = model(X, options.k, rvc)
    if options.verbose:
        print "Results:"
    signal_perc_err, TE_perc_err, fro_perc_err = get_metric(X, X_hat, options.verbose)
    if not k:
        k = U.shape[1]
    results[m] = {'U':U, 'alpha':alpha, 'k':k, 'X_hat': X_hat, 'Percentage Error per Signal': signal_perc_err, 'Percentage Error per TE': TE_perc_err, 'Frobenius Percentage Error': fro_perc_err}
if options.verbose:
    print "------------------------------------------------------------"


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
        print "------------------------------------------------------------"
        print "Computing contrast synthesis matrix"
        print ""

    Xe = gen_FSEmatrix(N, np.pi * np.ones(T), ETL, e2s, TE, T1vals, T2vals)

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
            print options.csynth_name

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
        print options.basis_name

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

if options.build_phantom:
    if options.verbose:
        print "------------------------------------------------------------"
        print "Building Phantom"
        print ""
    if options.phantom_data != None:
        if options.verbose:
            print X.shape
            print ETL
        ksp = np.zeros((phantom_data.shape[0], phantom_data.shape[1], ETL), dtype='complex64')
        if options.verbose:
            print ksp.shape
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
        print "------------------------------------------------------------"

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
