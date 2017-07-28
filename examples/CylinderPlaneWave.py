#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division, print_function

from matplotlib import pyplot as plt
import numpy as np
from os.path import abspath, dirname, join, split
import sys

sys.path.insert(0, split(dirname(abspath(__file__)))[0])

from miefield import *
from miefield._Classes import *


# Refractive index of the cylinder
RI_cylinder = 1.1
# Refractive index of the medium
RI_background = 1.0
# Radius of the cylinder in wavelengths
radius = 20.0
# Frequency of the used light
frequency = 1.0
# Materials
cylinder = DielectricMaterial(RI_cylinder**2,0.0)
background = DielectricMaterial(RI_background**2,0.0)

reference =  DielectricMaterial(1,0.0)
wavelength = reference.getElectromagneticWavelength(frequency)
    
# The first example does not take too much time to compute
print("Please wait, this may take a minute...")
# We are looking at a rectangular observation area
minx = -radius*1.2 
maxx = +radius*3
lenx = 100
miny = -radius*1.2
maxy = +radius*1.2 
# from here we can
leny = int(np.round(lenx * (maxy-miny)/(maxx-minx)))

x = np.linspace(minx, maxx, lenx, endpoint=False)
y = np.linspace(miny, maxy, leny, endpoint=False)
X, Y = np.meshgrid(x,y)
field_measure = np.zeros((2,lenx*leny))
field_measure[0] = X.flatten()
field_measure[1] = Y.flatten()
field = getDielectricCylinderFieldUnderPlaneWave(radius*wavelength, cylinder, background, field_measure*wavelength, frequency)
print("...done.")
Ez = field.reshape(leny, lenx)

plt.subplot(211)
plt.xlabel('x [wavelengths]')
plt.ylabel('y [wavelengths]')
plt.title('amplitude |E_z| [V/m]')
plt.imshow(20*np.log10(np.abs(Ez)))


plt.subplot(212)
plt.xlabel('x [wavelengths]')
plt.ylabel('y [wavelengths]')
plt.title('phase [rad]')
plt.imshow(np.angle(Ez))

plt.tight_layout()
plt.show()
