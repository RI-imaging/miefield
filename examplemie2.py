import matplotlib.pylab as plt
import numpy as np
import miefield

radius = 5 # radius of the cylinder in wavelengths
nmed = 1.333 # refractive index of surrounding medium
nsphe = 1.34 # refractive index of the cylinder
lD = 6 # distance from center of cylinder to planar detector
size = 10 # pixel number of the planar detector
res = 1.0 # pixel size of the detector
finalfield = miefield.GetFieldSphere(radius, nmed, nsphe, lD, size, res)
finalfieldCartesian = miefield.io_GetCartesianField2D(finalfield, lD, size, res)

#import IPython
#IPython.embed()
print("final shape",np.shape(finalfield))
#plt.plot(np.arange(size), finalfield[0].real.flatten(), label="real electric field")
#plt.plot(np.arange(size), finalfield[0].imag.flatten(), label="imaginary electric field")
#plt.legend()
#plt.show()
#plt.plot(np.arange(size), finalfield[3].real.flatten(), label="real magnetic field")
#plt.plot(np.arange(size), finalfield[3].imag.flatten(), label="imaginary magnetic field")
#plt.legend()
#plt.show()
