import matplotlib.pylab as plt
import numpy as np
import miefield

radius = 35 # radius of the cylinder in wavelengths
nmed = 1.333 # refractive index of surrounding medium
nsphe = 1.350 # refractive index of the cylinder
lD = 12 # distance from center of cylinder to planar detector
size = 100 # pixel number of the planar detector
res = 0.5 # pixel size of the detector
efield = miefield.GetFieldSphere(radius, nmed, nsphe, lD, size, res)

#plt.plot(np.arange(size), efield.real, label="real electric field")
#plt.plot(np.arange(size), efield.imag, label="imaginary electric field")
#plt.legend()
#plt.show()


print(efield)