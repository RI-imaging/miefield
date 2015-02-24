This is an attempted pythonification of the Mie code from Guangran Kevin Zhu at 
http://www.mathworks.de/matlabcentral/fileexchange/30162-cylinder-scattering and
http://de.mathworks.com/matlabcentral/fileexchange/31119-sphere-scattering

### Developer's note
miefield requires a working installation of SciPy and NumPy.

Example Usage:

```python
import matplotlib.pylab as plt
import numpy as np
import miefield

radius = 10 # radius of the cylinder in wavelengths
nmed = 1.333 # refractive index of surrounding medium
ncyl = 1.350 # refractive index of the cylinder
lD = 12 # distance from center of cylinder to planar detector
size = 100 # pixel number of the planar detector
res = 0.5 # pixel size of the detector
efield = miefield.GetFieldCylinder(radius, nmed, ncyl, lD, size, res)

plt.plot(np.arange(size), efield.real, label="real electric field")
plt.plot(np.arange(size), efield.imag, label="imaginary electric field")
plt.legend()
plt.show()
```
