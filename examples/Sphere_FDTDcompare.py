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
ncyl = 1.34 # refractive index of the cylinder
lD = 6.0 # distance from center of cylinder to planar detector
res = 13.0 # pixels per vacuum wavelength at the detector line

# FDTD
fdtd_real = np.loadtxt(zf.open("fdtd_real.txt"))
fdtd_imag = np.loadtxt(zf.open("fdtd_imag.txt"))
fdtd = fdtd_real + 1j*fdtd_imag

size = fdtd.shape[0] # pixel number of the planar detector

## TODO:
# Mie computation
#mie = miefield.GetFieldSphere(radius, nmed, ncyl, lD, size, res)
#mie_bg = miefield.GetFieldSphere(radius, nmed, nmed, lD, size, res)
#mie /= mie_bg

fig, axes = plt.subplots(2,3)

axes = axes.flatten()

## FDTD plots
axes[0].imshow(np.angle(fdtd))
axes[0].set_title("FDTD phase")

axes[1].imshow(fdtd_imag)
axes[1].set_title("FDTD imag")

axes[2].imshow(fdtd_real)
axes[2].set_title("FDTD real")

## Mie plots
axes[3].set_title("Mie phase")
axes[4].set_title("Mie imag")
axes[5].set_title("Mie real")


plt.show()
