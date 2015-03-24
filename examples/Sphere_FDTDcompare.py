#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division, print_function

from matplotlib import pyplot as plt
import numpy as np
import os
from os.path import abspath, dirname, join, split
import sys
import zipfile

sys.path.insert(0, split(dirname(abspath(__file__)))[0])

import miefield

os.chdir(dirname(abspath(__file__)))

zf = zipfile.ZipFile("Sphere_FDTD.zip")

radius = 5.0 # radius of the cylinder in wavelengths
nmed = 1.333 # refractive index of surrounding medium
nsphe = 1.34 # refractive index of the cylinder
lD = 6.0 # distance from center of cylinder to planar detector
res = 2.0 # pixels per vacuum wavelength at the detector line

# FDTD
fdtd_real = np.loadtxt(zf.open("fdtd_real.txt"))
fdtd_imag = np.loadtxt(zf.open("fdtd_imag.txt"))
fdtd = fdtd_real + 1j*fdtd_imag

size = fdtd.shape[0] # pixel number of the planar detector

# TODO:
# Mie computation
mie     = miefield.GetFieldSphere(radius, nmed, nsphe, lD, size, res)
mieECart = miefield.io_GetCartesianField2D(mie, lD, size, res)
#mie_bg = miefield.GetFieldSphere(radius, nmed, nmed, lD, size, res)      #HHH not working

#mie_bgE=mie_bg[0]

#mieE /= mie_bg

fig, axes = plt.subplots(4,3)

axes = axes.flatten()

## FDTD plots
axes[0].imshow(np.angle(fdtd))
axes[0].set_title("FDTD phase")

axes[1].imshow(fdtd_imag)
axes[1].set_title("FDTD imag")

axes[2].imshow(fdtd_real)
#print(np.shape(fdtd_real))
axes[2].set_title("FDTD real")


## Mie plots
#print(np.shape(np.arange(size)), np.shape(  np.angle(mieECartX.flatten())   ))
axes[3].imshow(np.angle(mieECart[0]) )
axes[3].set_title("Mie phase Ex")

axes[4].imshow(mieECart[0].imag )
axes[4].set_title("Mie imag Ex")

axes[5].imshow(mieECart[0].real )
axes[5].set_title("Mie real Ex")



axes[6].imshow(np.angle(mieECart[1]) )
axes[6].set_title("Mie phase Ey")

axes[7].imshow(mieECart[1].imag )
axes[7].set_title("Mie imag Ey")

axes[8].imshow(mieECart[1].real )
axes[8].set_title("Mie real Ey")



axes[9].imshow(np.angle(mieECart[2]) )
axes[9].set_title("Mie phase Ez")

axes[10].imshow(mieECart[2].imag )
axes[10].set_title("Mie imag Ez")

axes[11].imshow(mieECart[2].real )
axes[11].set_title("Mie real Ez")

plt.show()
