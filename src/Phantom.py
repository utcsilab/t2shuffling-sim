#!/usr/bin/python

from __future__ import division

import numpy as np
import scipy.special
from matplotlib.path import Path
import progressbar

import sys

import phantom_defs as phantoms
import cfl


def gen_grid(dims):
    if len(dims) == 1:
        dims = (dims, dims)
    else:
        assert len(dims) == 2, "dims should be length one or two"

    X0, X1 = np.meshgrid(np.arange(-np.floor(dims[1] / 2),np.floor((dims[1]) / 2)), np.arange(-np.floor(dims[0]/2),np.floor((dims[0])/2)))
    return np.array(X0), np.array(X1)



def control_to_node(control, shift=0):

    if shift == 0: # fast computation
        node = np.roll(control, shift=1, axis=0) + np.roll(control, shift=2, axis=0)
        node *= 0.5
    else:
        H = np.fft.fft(bspline2(np.arange(control.shape[0]) + shift))
        node = control.copy()
        node[:, 0] = np.real(np.fft.ifft(np.fft.fft(control[:, 0]) * H))
        node[:, 1] = np.real(np.fft.ifft(np.fft.fft(control[:, 1]) * H))

    return node


def bspline2(x):
    y = np.zeros(x.shape)

    idx = np.logical_and(x > 0, x <= 1)
    y[idx] = x[idx]**2 / 2

    idx = np.logical_and(x > 1, x <= 2)
    y[idx] = -3/2 + 3 * x[idx] - x[idx]**2

    idx = np.logical_and(x > 2, x <= 3)
    y[idx] = 9/2 + 3 * x[idx] + x[idx]**2 / 2

    return y

def inside_poly(r, Y, Z):

    path = Path(r, closed=False)
    YZ = np.vstack((Y.ravel(), Z.ravel())).T
    return path.contains_points(YZ).reshape(Y.shape)

def shape_poly(shape, Y, Z):
    return inside_poly(shape['vertex'], Y, Z)

def shape_ellipse(shape, Y, Z):
    ct = np.cos(shape['angle'])
    st = np.sin(shape['angle'])

    Y0 = Y - shape['center'][0]
    Z0 = Z - shape['center'][1]

    U0 = ct * Y0 + st * Z0
    U0 *= 2 / shape['diam'][0]

    Z0 = -st * Y0 + ct * Z0
    Z0 *= 2 / shape['diam'][1]

    return np.sqrt(U0**2 + Z0**2) <= 1


def shape_bezier(shape, Y, Z):
    # get coordinates of the polygon and make mask
    control = shape['control']
    r = control_to_node(control)
    mask = inside_poly(r, Y, Z)

    c = np.roll(control, 1, axis=0)
    rp1 = np.roll(r, -1, axis=0)
    beta = c - np.roll(c, 1, axis=0)
    gamma = rp1 + r - 2 * c
    a = beta[:,0] * gamma[:,1] - beta[:,1] * gamma[:,0]

    for i in range(len(a)):
        # points inside triangle
        vtx = np.array([ r[i, :], c[i,:], rp1[i,:]])
        idx = np.where(inside_poly(vtx, Y, Z))

        b = -(Y[idx] - r[i, 0]) * gamma[i, 1] + (Z[idx] - r[i, 1]) * gamma[i, 0]
        d = -(Y[idx] - r[i, 0]) * beta[i, 1] + (Z[idx] - r[i, 1]) * beta[i, 0]

        # a>=0 for outward-pointing triangles: add the interior points
        # a<0 for inside-pointing triangles: remove the exterior points
        mask[(idx[0][b**2 < a[i]*d], idx[1][b**2 < a[i] * d])] = a[i] >= 0

    return mask

def get_shape_fun(shape_type):
    if shape_type == 'bezier':
        shape_fun = shape_bezier
    elif shape_type == 'ellipse':
        shape_fun = shape_ellipse
    elif shape_type == 'polygon':
        shape_fun = shape_poly
    else:
        assert 0, 'shape type %s not recognized!' % shape_type

    return shape_fun

def rasterize_shape(shape, Y, Z, t2relax=None, t1relax=None, TR=1.4, TE=5e-3, T=32):

    shape_type, rho, T1, T2 = shape['type'], shape['rho'], shape['T1'], shape['T2']
    shape_fun = get_shape_fun(shape_type)

    _img = rho * shape_fun(shape, Y, Z)
    
    if t2relax == 'expon':
        _img *= np.exp(-TE / T2)

    if t1relax != None:
        _img *= (1 - np.exp( (TR - T*TE) / T1))

    return _img

# returns an array of N 2D images, where the n'th element is the image of the n'th shape in phantom
def multibuild_phantom(phantom, dims, oversamp=1, verbose=False):

    N = len(phantom['shapes'])
    dims2 = [d*oversamp for d in dims]
    Y, Z = gen_grid(dims2)

    Y = Y / dims2[0] * phantom['FOV'][0]
    Z = Z / dims2[1] * phantom['FOV'][1]

    dims2.append(N)
    imgs = np.zeros(dims2)

    if verbose:
        count = 0
        bar = progressbar.ProgressBar(maxval=N, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()


    for ii, shape in enumerate(phantom['shapes']):

        imgs[:, :, ii] = rasterize_shape(shape, Y, Z, t2relax=None, t1relax=None)

        if verbose:
            bar.update(count)
            count += 1

    if verbose:
        bar.finish()


    if oversamp > 1:
        if verbose:
            print 'Downsampling...'

        cen = [np.floor(d/2) for d in dims2]
        ksp = fft2c(imgs)
        ksp2 = ksp[cen[0] - np.floor(dims[0]/2) : cen[0] + np.floor(dims[0]/2),
                cen[1] - np.floor(dims[1]/2) : cen[1] + np.floor(dims[1]/2)]
        imgs = ifft2c(ksp2)

    return imgs



# returns a 2D image corresponding to the phantom
def build_phantom(phantom, dims, oversamp=1, t2relax=None, t1relax=None, TR=1.4, TE=5e-3, T=32, verbose=False):

    N = len(phantom['shapes'])
    dims2 = [d*oversamp for d in dims]
    Y, Z = gen_grid(dims2)

    Y = Y / dims2[0] * phantom['FOV'][0]
    Z = Z / dims2[1] * phantom['FOV'][1]

    img = np.zeros(dims2)

    if verbose:
        count = 0
        bar = progressbar.ProgressBar(maxval=N, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()


    for ii, shape in enumerate(phantom['shapes']):

        _img = rasterize_shape(shape, Y, Z, t2relax, t1relax, TR, TE, T)

        img += _img

        if verbose:
            bar.update(count)
            count += 1

    if verbose:
        bar.finish()

    if verbose:
        print 'Downsampling...'

    if oversamp > 1:
        cen = [np.floor(d/2) for d in dims2]
        ksp = fft2c(img)
        ksp2 = ksp[cen[0] - np.floor(dims[0]/2) : cen[0] + np.floor(dims[0]/2),
                cen[1] - np.floor(dims[1]/2) : cen[1] + np.floor(dims[1]/2)]
        img = ifft2c(ksp2)

    return img


def fft2c(x):
    return 1 / np.sqrt(x.shape[0]*x.shape[1]) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(0,1)), axes=(0,1)), axes=(0,1))

def ifft2c(x):
    return np.sqrt(x.shape[0]*x.shape[1]) * np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x, axes=(0,1)), axes=(0,1)), axes=(0,1))

if __name__ == "__main__":

    np.random.seed(721)



    T = 32
    if len(sys.argv) > 1:
        dims = [int(sys.argv[1]), int(sys.argv[1])]
        oversamp = int(sys.argv[2])
    else:
        dims = [128, 128]
        oversamp = 1

    #brain = phantoms.simple_brain()
    brain = phantoms.brain()
    #brain = phantoms.shep()
    #mask = build_phantom(brain, dims, oversamp, t2relax='expon', t1relax=True, TE=100e-3, verbose=True).T
#
    #cfl.writecfl('img', mask)

    masks = multibuild_phantom(brain, dims, oversamp, verbose=True)
    cfl.writecfl('masks-2048', masks)



    sys.exit(0)
