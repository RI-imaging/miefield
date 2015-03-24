from __future__ import division, print_function

from matplotlib import pyplot as plt
import numpy as np
import os
from os.path import abspath, dirname, join, split
import sys


import matplotlib.pylab as plt
import numpy as np

sys.path.insert(0, split(dirname(abspath(__file__)))[0])

import miefield

radius = 5 # radius of the cylinder in wavelengths
nmed = 1.333 # refractive index of surrounding medium
nsphe = 1.34 # refractive index of the cylinder
lD = 6 # distance from center of cylinder to planar detector
size = 50 # pixel number of the planar detector
res = 1.0 # pixel size of the detector
mie = miefield.GetFieldSphere(radius, nmed, nsphe, lD, size, res)
print("final shape",np.shape(mie))
mieECart = miefield.io_GetCartesianField2D(mie, lD, size, res)

#import IPython
#IPython.embed()
#plt.plot(np.arange(size), finalfield[0].real.flatten(), label="real electric field")
#plt.plot(np.arange(size), finalfield[0].imag.flatten(), label="imaginary electric field")
#plt.legend()
#plt.show()
#plt.plot(np.arange(size), finalfield[3].real.flatten(), label="real magnetic field")
#plt.plot(np.arange(size), finalfield[3].imag.flatten(), label="imaginary magnetic field")
#plt.legend()
#plt.show()

fig, axes = plt.subplots(3,3)

axes = axes.flatten()

## Mie plots
axes[0].imshow(np.angle(mieECart[0]) )
axes[0].set_title("Mie phase Ex")

axes[1].imshow(mieECart[0].imag )
axes[1].set_title("Mie imag Ex")

axes[2].imshow(mieECart[0].real )
axes[2].set_title("Mie real Ex")



axes[3].imshow(np.angle(mieECart[1]) )
axes[3].set_title("Mie phase Ey")

axes[4].imshow(mieECart[1].imag )
axes[4].set_title("Mie imag Ey")

axes[5].imshow(mieECart[1].real )
axes[5].set_title("Mie real Ey")



axes[6].imshow(np.angle(mieECart[2]) )
axes[6].set_title("Mie phase Ez")

axes[7].imshow(mieECart[2].imag )
axes[7].set_title("Mie imag Ez")

axes[8].imshow(mieECart[2].real )
axes[8].set_title("Mie real Ez")

plt.show()
