#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    Mie solution 
    
    Calculates the electric field component (Ez) for a line source that
    is scattered by a dielectric cylinder.
    
    Some of this code is a partial translation of the Matlab code from
    Guangran Kevin Zhu
    http://www.mathworks.de/matlabcentral/fileexchange/30162-cylinder-scattering
"""

from __future__ import division, print_function

import numpy as np
import os

from ._Classes import *
from ._Functions import *


__all__ = ["GetFieldCylinder", "GetSinogramCylinderRotation",
           "GetFieldCylinderDisplaced",
           "getDielectricCylinderFieldUnderPlaneWave"]


def GetFieldCylinder(radius, nmed, ncyl, lD, size, res):
    """ Computes the Mie solution  for a dielectric cylinder.
        
    Calculate the field scattered by a dielectric cylinder due to
    an incident TM_y plane wave at a certain distance `lD` from the 
    center of the cylinder on a line.
    
        ^ x  
        |  
        ----> z  
              
         cylinder     detector   
           ___           . (xmax,lD)   
         /     \         .  
        | (0,0) |        .  
         \ ___ /         .  
                         . (xmin,lD)  
    
    E0 = exp(-ikz)


    Parameters
    ----------
    radius : float
        Radius of the cylinder in vacuum wavelengths.
    nmed : float
        Refractive index of the surrounding medium.
    ncyl : float
        Refractive index of the cylinder.
    lD : float
        Distance lD from the detector to the center of the cylinder
        in vacuum wavelengths.
    size : float
        Detector size in pixels
    res : float
        Resolution of detector in pixels per vacuum wavelength.


    Returns
    -------
    out : one-dimensional ndarray of length `size`, complex
        Electric field at the detector.


    Notes
    --------
    This code is based on a translation of the Matlab code from
    Guangran Kevin Zhu:
    http://www.mathworks.de/matlabcentral/fileexchange/30162-cylinder-scattering
    """
    cylinder = DielectricMaterial(ncyl**2,0.0)
    background = DielectricMaterial(nmed**2,0.0)
    reference =  DielectricMaterial(1,0.0)
    lambref = reference.getElectromagneticWavelength(1.0)
    xmax = size / res / 2.0
    # the detector resolution is not dependent on the medium
    detector = np.linspace(-xmax, xmax, size, endpoint=True) * lambref
    sensor_location = np.zeros((2,size))
    sensor_location[0] = lD*lambref    # optical path length to detector
    sensor_location[1] = detector
    return getDielectricCylinderFieldUnderPlaneWave(radius*lambref, 
             cylinder, background, sensor_location).flatten()



def GetSinogramCylinderRotation(radius, nmed, ncyl, lD, lC, size, A,
                                res):
    """ Computes sinogram with Mie solution for displaced cylinder
    
    
    
    Parameters
    ----------
    radius : float
        Radius of the cylinder in vacuum wavelengths.
    nmed : float
        Refractive index of the surrounding medium.
    ncyl : float
        Refractive index of the cylinder.
    lD : float
        Distance lD from the detector to the rotational center
        in vacuum wavelengths.
    lC : float
        Distance from the center of the cylinder to the rotational
        center in vacuum waveengths.
    size : float
        Detector size in pixels
    A : int
        Number of angles from zero to 2PI
    res : float
        Resolution of detector in pixels per vacuum wavelength.


    Returns
    -------
    out : one-dimensional ndarray of length `size`, complex
        Electric field at the detector.
    
    
    See Also
    --------
    `GetFieldCylinderDisplaced` for a sketch
    
    """
    warnings.warn("This functions has not been verified!")
    sino = np.zeros((A,size), dtype=np.complex)
    angles = np.linspace(0, 2*np.pi, A, endpoint=False)
    for i in range(A):
        ang = angles[i]
        zc = lC*np.cos(ang)
        xc = lC*np.sin(ang)

        pos = (xc, zc)
        print(i, pos, norm(pos)) 
        sino[i] = GetFieldCylinderDisplaced(radius, nmed, ncyl, lD,
                                            size, pos, res)
    return sino


def GetFieldCylinderDisplaced(radius, nmed, ncyl, lD, size, pos, res):
    """ Computes the Mie solution for a dielectric cylinder.
        
    Calculate the field scattered by a dielectric cylinder due to
    an incident TM_y plane wave at a certain distance `lD` from the 
    center of the cylinder on a line.
    
        ^ x  
        |  
        ----> z  
              
         origin      cylinder     detector   
                       ___           
                     /     \         . (xmax,lD)    
                    |(xc,zc)|        .  
         (0,0)       \ ___ /         .  
                                     .
                                     . (xmin,lD)  
    
    E0 = exp(-ikz)


    Parameters
    ----------
    radius : float
        Radius of the cylinder in vacuum wavelengths.
    nmed : float
        Refractive index of the surrounding medium.
    ncyl : float
        Refractive index of the cylinder.
    lD : float
        Distance lD from the detector to the origin
        in vacuum wavelengths.
    size : float
        Detector size in pixels
    pos : tuple of floats
        Cylinder position (xc, zc) in vacuum wavelengths.
    res : float
        Resolution of detector in pixels per vacuum wavelength.


    Returns
    -------
    out : one-dimensional ndarray of length `size`, complex
        Electric field at the detector.


    Notes
    --------
    This code is based on a translation of the Matlab code from
    Guangran Kevin Zhu:
    http://www.mathworks.de/matlabcentral/fileexchange/30162-cylinder-scattering
    """
    warnings.warn("This functions has not been verified!")
    (xc, zc) = pos
    #zc = 0
    cylinder = DielectricMaterial(ncyl**2,0.0)
    background = DielectricMaterial(nmed**2,0.0)
    reference =  DielectricMaterial(1,0.0)
    lambref = reference.getElectromagneticWavelength(1.0)
    # compute xmax in wavelengths
    xmax = size / res / 2.0
    # the detector resolution is not dependent on the medium
    detector = np.linspace(-xmax-xc, xmax-xc, size, endpoint=False) * lambref
    sensor_location = np.zeros((2,size))
    print(lD-zc)
    sensor_location[0,:] = (lD-zc)*lambref    # real distance to detector
    sensor_location[1,:] = detector
    return getDielectricCylinderFieldUnderPlaneWave(radius*lambref, 
             cylinder, background, sensor_location).flatten()



def getDielectricCylinderFieldUnderPlaneWave(radius, cylinder,
                              background, sensor_location, frequency=1):
    """ Compute plane wave dielectric cyllinder solution
    
    Calculate the field scattered by a dielectric cylinder due to
    an incident TM_z plane wave. If the sensor_location is outside
    the cylinder, this function returns the scattered field. If the
    sensor_location is inside the cylinder, it returns the total
    field inside the cylinder. The incident TM_z plane wave is
    assumed propagating from left (neg X) to right (pos X).
   
    Parameters
    ----------
    radius : float
        radius of the cylinder in wavelengths.
    cylinder : DielectricMaterial
        dielectric cylinder.
    background : DielectricMaterial
        medium in which the cylinder is embedded.
    sensor_location : ndarray, shape (N,2)
        Coordinates of the detector.
    frequency : float, optional
        frequency of the incoming wave. 
   
    Returns
    -------
    Ez : ndarray, length N
        electric field at the `sensor_location` in [V/m]

    Notes
    -----
    This problem is solved in Problem 11-26 in [Balanis1989].   
                 
    This function is a translation to Python from a matlab script
    by Guangran Kevin Zhu.

    See Also
    --------
    `line_scat_diel_cyl_pw`
        a line detector wrapper for this function.
    """
    # [Ez, H_rho, H_phi] = getDielectricCylinderFieldUnderPlaneWave(radius, cylinder,
    #                                               background,
    #                                               sensor_location,
    #                                               frequency)
    phi, rho = cart2pol(sensor_location[0], sensor_location[1])
    phi.resize(len(phi),1)
    rho.resize(len(rho),1)
    Ez = np.zeros(rho.shape, dtype=np.complex128)
    #H_rho = np.zeros(rho.shape, dtype=np.complex128)
    #H_phi = np.zeros(rho.shape, dtype=np.complex128)
    
    omega = 2 * np.pi * frequency
    k = background.getElectromagneticWaveNumber(frequency)
    eta = background.getIntrinsicImpedance(frequency)
    k_d = cylinder.getElectromagneticWaveNumber(frequency)
    eta_d = cylinder.getIntrinsicImpedance(frequency)
    N_max = getN_max(radius, cylinder, background, frequency)
    
    n = np.arange(-N_max, N_max, dtype=np.int16).reshape(1,-1)
    
    x = k_d * radius
    y = k * radius
    temp = np.arange(-N_max-1, N_max+1, dtype=np.int16).reshape(1,-1)

    a = besselj(temp, x)
    d = 0.5 * (a[:,: -2] - a[:,2:])
    b = besselj(temp, y)
    e = 0.5 * (b[:,: -2] - b[:,2:])
    c = besselh(temp, 2, y)
    f = 0.5 * (c[:,: -2] - c[:,2:])
    a = a[:,1: -1]
    b = b[:,1: -1]
    c = c[:,1: -1]
    del temp


    #if (rho > radius):
    ## outside
    # Matlab code:
    # num = 1/eta_d.*d.*b-1/eta.*e.*a;
    # den = 1/eta.*a.*f -1/eta_d.*d.*c;
    # a_n = num./den.*j.^(-n);
    #
    # Ez(iFreq)    =              sum(a_n.*besselh(n,2,k*rho).*exp(j*n*phi));
    # H_rho(iFreq) = -1./(j*eta).*sum(a_n.*besselh(n,2,k*rho)./(k*rho).*j.*n.*exp(j*n*phi));
    # H_phi(iFreq) = +1./(j*eta).*sum(a_n.*besselh_derivative(n,2,k*rho).*exp(j*n*phi));
    out = np.where(rho > radius)
    num = 1/eta_d * d * b  -  1/eta * e * a
    den = 1/eta * a * f  -  1/eta_d * d * c
    a_n = num / den * 1j**(-n)

    Ez[out] =                          np.sum(a_n * besselh(n, 2, k * rho[out].reshape(-1,1)) * np.exp(1j * n * phi[out].reshape(-1,1)), axis=-1)
    #H_rho[out] = - 1.0 / (1j * eta) *  np.sum(a_n * besselh(n, 2, k * rho[out].reshape(-1,1)) / (k * rho[out].reshape(-1,1)) * 1j * n * np.exp(1j * n * phi[out].reshape(-1,1)), axis=-1)
    #H_phi[out] = + 1.0 / (1j * eta) *  np.sum(a_n * besselh_derivative(n, 2, k * rho[out].reshape(-1,1)) * np.exp(1j * n * phi[out].reshape(-1,1)), axis=-1)



    ## Add plane wave component
    ## In Zhu's scripts this is done in
    ## getPlaneWaveUsingCylindricalExpansion.m
    pwk_m   = background.getElectromagneticWaveNumber(frequency)
    pwN_max = getN_max(rho[out], background, background, frequency)
    pwN_max = np.max(pwN_max)
    pwn     = np.arange(-pwN_max,+pwN_max).reshape(1,-1)
    pwtemp  = 1j**(-pwn) * besselj(pwn, pwk_m * rho[out].reshape(-1,1)) * np.exp(1j*pwn*phi[out].reshape(-1,1))

    Ez[out] += np.sum(pwtemp, axis=-1)
    del pwtemp


    #else:
    ## inside
    # Matlab code:
    # num = 1/eta.*f.*b-1/eta.*e.*c;
    # den = 1/eta.*f.*a-1/eta_d.*d.*c;
    # c_n = num./den.*j.^(-n);
    #
    # Ez(iFreq)    =                sum(c_n.*besselj(n,k_d*rho).*exp(j*n*phi));
    # H_rho(iFreq) = -1./(j*eta_d).*sum(c_n.*besselj(n,k_d*rho)./(k_d*rho).*j.*n.*exp(j*n*phi));
    # H_phi(iFreq) = +1./(j*eta_d).*sum(c_n.*besselj_derivative(n,k_d*rho).*exp(j*n*phi));
    ins = np.where(rho <= radius)
    num = 1/eta * f * b  -  1/eta * e * c
    den = 1/eta * f * a  -  1/eta_d * d * c
    c_n = num/den * 1j**(-n)
    Ez[ins] =                            np.sum(c_n * besselj(n, k_d * rho[ins].reshape(-1,1))  * np.exp(1j * n * phi[ins].reshape(-1,1)), axis=-1)
    #H_rho[ins] = - 1.0 / (1j * eta_d) *  np.sum(c_n * besselj(n, k_d * rho[ins].reshape(-1,1)) / (k_d * rho[ins].reshape(-1,1)) * 1j * n * np.exp(1j * n * phi[ins].reshape(-1,1)), axis=-1)
    #H_phi[ins] = + 1.0 / (1j * eta_d) *  np.sum(c_n * besselj_derivative(n, k_d * rho[ins].reshape(-1,1)) * np.exp(1j * n * phi[ins].reshape(-1,1)), axis=-1)
    return np.conjugate(Ez) #, H_rho, H_phi




if __name__ == "__main__":
    import IPython
    from matplotlib import pyplot as plt
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

    wavelength = background.getElectromagneticWavelength(frequency)
        
    # The first example does not take too much time to compute
    if True:
        print("Please wait, this may take a minute...")
        # We are looking at a rectangular observation area
        minx = -radius*1.2 
        maxx = +radius*3
        lenx = 100
        miny = -radius*1.2
        maxy = +radius*1.2 
        # from here we can
        leny = np.round(lenx * (maxy-miny)/(maxx-minx))
        
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


    if False:
        print("Please wait, this may take a couple of minutes...")
        # We are looking at a rectangular observation area
        minx = -25
        maxx = 40
        resolution = 5 # px per wavelength
        miny = -25
        maxy = 25
        # from here we can
        
        x = np.arange(minx, maxx, 1./resolution)
        y = np.arange(miny, maxy, 1./resolution)
        
        lenx = len(x)
        leny = len(y)

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

    IPython.embed()
