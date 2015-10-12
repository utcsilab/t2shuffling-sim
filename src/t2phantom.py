#!/usr/bin/python

from __future__ import division
import sys
import numpy as np
import scipy.special


class Phantom:

    def __init__(self, FOV, res):
        self.FOV = np.array(FOV)
        self.res = np.array(res)
        self.img_dims = np.ceil(self.FOV / self.res)
        self.dk = 1 / self.FOV
        self.Wk = 1 / self.res
        self.ndims = len(self.FOV)

        self.gen_grid()

    def gen_grid(self, oversample=1):
        img_coords = np.array(np.meshgrid(*zip([np.arange(-N/2, N/2) for N in self.img_dims])))
        self.ksp_grid = img_coords / self.img_dims[:, None, None] * self.Wk[:, None, None]

    def ellipsoid(self, diam, cen=(0,0), angle_deg=0):
       assert self.ndims == 2, 'only support 2D for now'

       ksp_grid_angle = np.reshape(rot(np.reshape(self.ksp_grid, (self.ndims, -1)), angle_deg), self.ksp_grid.shape)

       K = np.sqrt(np.sum((np.array(diam)[:, None, None] * ksp_grid_angle)**2, 0))
       idx_eps = K <= .001
       idx_neps = K > .001

       j1 = scipy.special.jv(1, np.pi * K[idx_neps]) / (np.pi * K[idx_neps])
       # taylor series expansion about K ~ 0
       j1_taylor = 1 - (np.pi * K[idx_eps])**2 / 2 + (np.pi * K[idx_eps])**4 / 12

       ep = np.pi * np.prod(diam) * np.exp(-2 * np.pi * 1j * np.sum(np.array(cen)[:, None, None] * ksp_grid_angle, 0))
       ep[idx_eps] *= j1_taylor
       ep[idx_neps] *= j1

       return ep

    def disc(self, diam, width, cen=(0,0), angle_deg=0):
       assert(np.all(np.array(diam) > np.array(width)))
       ep1 = self.ellipsoid(np.array(diam), cen, angle_deg)
       ep2 = self.ellipsoid(np.array(diam) - np.array(width) / 2, cen, angle_deg)
       return ep1 - ep2

    def build_exp(self, p0, objects, te=5):
        pd_0 = p0['pd']
        ep_0 = p0['ep']
        t2_0 = p0['t2']
        ep = pd_0 * ep_0 * np.exp(-te/t2_0)
        for obj in objects:
            _ep = obj['ep']
            _pd = obj['pd']
            _t2 = obj['t2']
            # first cut out ep1 from ep
            ep -= pd_0 * _ep * np.exp(-te/t2_0)
            # next weight it by pd
            ep += _pd * _ep * np.exp(-te/_t2)
        return ep

    def build(self):
        p0, objects = self.p0, self.objects
        pd_0 = p0['pd']
        ep_0 = p0['ep']
        t2_0 = p0['t2']
        x_0 = p0['x']
        ep_bg = ep_0
        # do a first pass to make background
        for obj in objects:
            _ep = obj['ep']
            ep_bg -= _ep
        # do a second pass to fill shapes
        ep = pd_0 * ep_bg[:, :, None] * x_0[None, None, :]
        for obj in objects:
            _ep = obj['ep']
            _pd = obj['pd']
            _t2 = obj['t2']
            _x = obj['x']
            ep += _pd * _ep[:, :, None] * _x[None, None, :]
        return ep

    def populate_objects(self, p0, objects):
        self.p0 = p0
        self.objects = objects

    def knee_objects_relax(self, T2vals, X):
        self.knee_objects()

        for i in range(len(self.objects)):
            idx = np.argwhere(T2vals*1000 == self.objects[i]['t2'])[0][0]
            self.objects[i]['x'] = X[:, idx]
        self.p0['x'] = X[:, np.argwhere(T2vals*1000 == self.p0['t2'])[0][0]]


    def knee_objects(self):

        p0 = {
                'ep': self.ellipsoid((22, 22), (0, .3)),
                'pd' : 1.1,
                't2' : 180.
                }

        objects = []

        objects.append({
            'ep' : self.disc((22, 22), (2, 2), (0, .3)),
            'pd' : 1.5,
            't2' : 300.,
            })

        objects.append({
            'ep' : self.disc((8, 9), (2, 1.5), (0, 3), 10),
            'pd' : 1.5,
            't2' : 150.,
            })

        objects.append({
            'ep' : self.ellipsoid((7, 9-1.5/2), (0, 3), 10),
            'pd' : .7,
            't2' : 70.,
            })

        objects.append({
            'ep' : self.ellipsoid((2, 2.2), (3, 9), 15),
            'pd' : .7,
            't2' : 20.,
            })

        objects.append({
            'ep' : self.ellipsoid((10, 3), (3, -4), 12),
            'pd' : 1,
            't2' : 70,
            })

        objects.append({
            'ep' : self.ellipsoid((10, 3), (-3, -4), -12),
            'pd' : 1,
            't2' : 80.,
            })

        objects.append({
            'ep' : self.ellipsoid((.4, .8), (8, 2), -5),
            'pd' : .76,
            't2' : 80.,
            })

        objects.append({
            'ep' : self.ellipsoid((.4, .8), (8, 2 + .9), -5),
            'pd' : 1.2,
            't2' : 500.,
            })

        objects.append({
            'ep' : self.ellipsoid((.4, .8), (8, 2 - .9), -5),
            'pd' : 1.,
            't2' : 15.,
            })

        objects.append({
            'ep' : self.ellipsoid((.4, .8), (8, 2 + 2*.9), -5),
            'pd' : .5,
            't2' : 20.,
            })

        objects.append({
            'ep' : self.ellipsoid((.4, .8), (8, 2 - 2*.9), -5),
            'pd' : .89,
            't2' : 50.,
            })

        objects.append({
            'ep' : self.disc((6, 3), (1, 2), (0, -8)),
            'pd' : 1.2,
            't2' : 100.,
            })

        objects.append({
            'ep' : self.ellipsoid((5.5, 2), (0, -8)),
            'pd' : .8,
            't2' : 60.,
            })

        objects.append({
            'ep' : self.ellipsoid((.4, 6), (-5, 2)),
            'pd' : 1,
            't2' : 50.,
            })

        objects.append({
            'ep' : self.ellipsoid((.4, 6), (-6, 2)),
            'pd' : 1,
            't2' : 80.,
            })

        objects.append({
            'ep' : self.ellipsoid((.4, 6), (-7.5, 2)),
            'pd' : 1,
            't2' : 30.,
            })
        
        objects.append({
            'ep' : self.ellipsoid((.3, .3), (5, 0)),
            'pd' : 1,
            't2' : 800.,
            })

        objects.append({
            'ep' : self.ellipsoid((.3, .3), (-5.5, 9)),
            'pd' : 1,
            't2' : 800.,
            })

        objects.append({
            'ep' : self.ellipsoid((.3, .3), (4.5, -8)),
            'pd' : 1,
            't2' : 800.,
            })

        objects.append({
            'ep' : self.ellipsoid((.3, .3), (5, 0)),
            'pd' : 1,
            't2' : 800.,
            })

        self.p0 = p0
        self.objects = objects

def rot(x, angle_deg):
    angle = angle_deg * np.pi / 180
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.dot(R.T, x)

def fft2c(x):
    return 1 / np.sqrt(x.shape[0]*x.shape[1]) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(0,1)), axes=(0,1)), axes=(0,1))

def ifft2c(x):
    return np.sqrt(x.shape[0]*x.shape[1]) * np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x, axes=(0,1)), axes=(0,1)), axes=(0,1))


def main(FOV, dims):
    res = FOV / dims
    return Phantom(FOV, res)

if __name__ == "__main__":
    FOV = (25.6, 25.6) # cm
    dims = np.array([260, 240]) # pixels
    #res = (.1, .1) # cm
    P = main(FOV, dims)
