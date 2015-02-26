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

zf = zipfile.ZipFile("Cylinder_FDTD.zip")

radius = 5.0 # radius of the cylinder in wavelengths
nmed = 1.333 # refractive index of surrounding medium
ncyl = 1.34 # refractive index of the cylinder
lD = 6.0 # distance from center of cylinder to planar detector
res = 42.0 # pixels per vacuum wavelength at the detector line

# FDTD
fdtd_real = np.loadtxt(zf.open("fdtd_real.txt"))
fdtd_imag = np.loadtxt(zf.open("fdtd_imag.txt"))
fdtd = fdtd_real + 1j*fdtd_imag

size = fdtd.shape[0] # pixel number of the planar detector

# Mie computation
mie = miefield.GetFieldCylinder(radius, nmed, ncyl, lD, size, res)
mie_bg = miefield.GetFieldCylinder(radius, nmed, nmed, lD, size, res)
mie /= mie_bg

fig, axes = plt.subplots(1,3)

axes[0].plot(np.arange(size), np.angle(mie), label="Mie angle")
axes[0].plot(np.arange(size), np.angle(fdtd), label="FDTD angle")

axes[1].plot(np.arange(size), mie.imag, label="Mie imag")
axes[1].plot(np.arange(size), fdtd_imag, label="FDTD imag")

axes[2].plot(np.arange(size), mie.real, label="Mie real")
axes[2].plot(np.arange(size), fdtd_real, label="FDTD real")

axes[0].legend()
axes[1].legend()
plt.show()
