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

from __future__ import division
from __future__ import print_function

import numpy as np

from ._Classes import *
from ._Functions import *


__all__ = ["GetFieldCylinderLineSource", "getDielectricCylinderFieldUnderLineSource"]


def GetFieldCylinderLineSource(radius, nmed, ncyl, s_loc, lD, size, res):
    """ Computes the Mie solution  for a dielectric cylinder.
        
    Calculate the field scattered by a dielectric cylinder due to
    an incident TM_y plane wave at a certain distance `lD` from the 
    center of the cylinder on a line.
    
        ^ x  
        |  
        ----> z  
              
         source      cylinder     detector   
                       ___           . (xmax,lD)   
        (xS, zS)     /     \         .  
           .        | (0,0) |        .  
                     \ ___ /         .  
                                     . (xmin,lD)  
    

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
    s_loc : tuple of floats
        Source location in vacuum wavelengths (xS, zS)
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
    detector = np.linspace(-xmax, xmax, size, endpoint=False) * lambref
    
    sensor_location = np.zeros(2)
    sensor_location[0] = lD*lambref    # real path length to detector
    #sensor_location[1] = detector
    (xS, zS) = s_loc
    source_location = np.zeros(2)
    source_location[0] = zS * lambref
    source_location[1] = xS * lambref
    ln = len(detector)
    E = np.zeros(ln)
    
    for i in range(ln):
        sensor_location[1] = detector[i]
        # Compute scattered field
        ret = getDielectricCylinderFieldUnderLineSource(radius*lambref, 
                        cylinder, background,
                        sensor_location,
                        source_location)
        E[i] = ret[0]
        
        # Compute incoming plane wave
        if norm(sensor_location) > radius:
            E[i] += getCylindricalWaveUsingCylindricalExpansion(
                                             background,
                                             source_location,
                                             sensor_location)[0]

    return E


def getDielectricCylinderFieldUnderLineSource(radius, cylinder,
                              background, sensor_location,
                              source_location, frequency=1):
    """ Compute line source dielectric cyllinder solution
    
    Calculate the field scattered by a dielectric cylinder due to
    an incident spherical wave. If the sensor_location is outside
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
    sensor_location : ndarray, shape (2,1)
        Coordinates of the detector.
    source_location : ndarray, shape (2,1)
        Coordinates of the source.
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

    omega = 2 * np.pi * frequency
    EPS_0  = 8.854187817620389e-12
    epsilon = EPS_0 * background.getPermittivity(frequency)[0]

    phi_s, rho_s = cart2pol(source_location[0], source_location[1])
    phi, rho = cart2pol(sensor_location[0], sensor_location[1])


    phi.resize(len(np.atleast_1d(phi)),1)
    rho.resize(len(np.atleast_1d(rho)),1)
    Ez = np.zeros(rho.shape, dtype=np.complex128)
    #H_rho = np.zeros(rho.shape, dtype=np.complex128)
    #H_phi = np.zeros(rho.shape, dtype=np.complex128)
    
    
    k_1 = background.getElectromagneticWaveNumber(frequency)
    k_2 = cylinder.getElectromagneticWaveNumber(frequency)
    
    x_1 = k_1 * radius
    x_2 = k_2 * radius
    
    
    #eta = background.getIntrinsicImpedance(frequency)
    #k_d = cylinder.getElectromagneticWaveNumber(frequency)
    #eta_d = cylinder.getIntrinsicImpedance(frequency)
    N_max = getN_max(radius, cylinder, background, frequency)
    
    nu = np.arange(-N_max, N_max, dtype=np.int16).reshape(1,-1)
    
    #x = k_d * radius
    #y = k * radius
    #temp = np.arange(-N_max-1, N_max+1, dtype=np.int16).reshape(1,-1)

    factor = (-k_1**2/(4*omega*epsilon))

    if rho < rho_s: #inside source
        if rho < radius: # inside cylinder
            dn_num = ( k_1 * besselj(nu,x_1)   * besselh_derivative(nu,2,x_1) 
                      -k_1 * besselh(nu,2,x_1) * besselj_derivative(nu,x_1)  )
            dn_den = ( k_1 * besselj(nu,x_2)   * besselh_derivative(nu,2,x_1)
                      -k_2 * besselh(nu,2,x_1) * besselj_derivative(nu,x_2)  )
            dn     =  dn_num/dn_den
                
            Ez = factor*np.sum(dn * besselh(nu,2,k_1*rho_s)*besselj(nu,k_2*rho)*np.exp(1j*nu*(phi-phi_s)))
            #H_rho(iFreq) = 1/(4*j)/rho.*sum(dn.*besselh(nu,2,k_1*rho_s).*besselj(nu,k_2*rho).*(j*nu).*exp(j*nu*(phi-phi_s)));
            #H_phi(iFreq) =-k_2/(4*j).*sum(dn.*besselh(nu,2,k_1*rho_s).*besselj(nu,k_2*rho).*exp(j*nu*(phi-phi_s)));
        else: # outside cylinder
            cn_num = ( k_2 * besselj(nu,x_1)   * besselj_derivative(nu,x_2)
                      -k_1 * besselj(nu,x_2)   * besselj_derivative(nu,x_1)  )
            cn_den = ( k_1 * besselj(nu,x_2)   * besselh_derivative(nu,2,x_1)
                      -k_2 * besselh(nu,2,x_1) * besselj_derivative(nu,x_2)  )
            cn     =  cn_num/cn_den
            
            Ez = factor*np.sum(cn*besselh(nu,2,k_1*rho_s)*besselh(nu,2,k_1*rho)*np.exp(1j*nu*(phi-phi_s)))
            #H_rho(iFreq) = 1/(4*j)/rho.*sum(cn.*besselh(nu,2,k_1*rho_s).*besselh(nu,2,k_1*rho).*(j*nu).*exp(j*nu*(phi-phi_s)));
            #H_phi(iFreq) =-k_1/(4*j).*sum(cn.*besselh(nu,2,k_1*rho_s).*besselh_derivative(nu,2,k_1*rho).*exp(j*nu*(phi-phi_s)));
    else: # outside source radius
        cn_num = ( k_2 * besselj(nu,x_1)   * besselj_derivative(nu,x_2)
                  -k_1 * besselj(nu,x_2)   * besselj_derivative(nu,x_1)  )
        cn_den = ( k_1 * besselj(nu,x_2)   * besselh_derivative(nu,2,x_1)
                  -k_2 * besselh(nu,2,x_1) * besselj_derivative(nu,x_2)  )
        cn     =  cn_num/cn_den
        
        Ez = factor*np.sum(cn*besselh(nu,2,k_1*rho)*besselh(nu,2,k_1*rho_s)*np.exp(1j*nu*(phi-phi_s)))
        #H_rho(iFreq) = 1/(4*j)/rho.*sum(cn.*besselh(nu,2,k_1*rho).*besselh(nu,2,k_1*rho_s).*(j*nu).*exp(j*nu*(phi-phi_s)));
        #H_phi(iFreq) =-k_1/(4*j).*sum(cn.*besselh(nu,2,k_1*rho).*besselh_derivative(nu,2,k_1*rho_s).*exp(j*nu*(phi-phi_s)));

    return np.conjugate(Ez) #, H_rho, H_phi


def getCylindricalWaveUsingCylindricalExpansion(background, source_location, sensor_location, frequency=1):
    """ Calculate cylindrical wave
    
    Calculate the cylindrical wave due to an infinitely-long current
    source with a unity current of 1A. This is the numerical
    implementation of (5-119) based on (5-103) in [Harrington2001]

    Input:
    
    background         DielectricMaterial
    source_location    2x1 vector in the form of [x; y] (m)
    sensor_location    2x1 vector in the form of [x; y] (m)
    frequency          Nx1 vector in (Hz)

    Output:
    Ez                  Nx1 vector (V/m)
    H_rho               Nx1 vector (A/m)
    H_phi               Nx1 vector (A/m)
    """
    #omega          = 2*np.pi*frequency
    EPS_O          = 8.854187817620389e-12
    
    [phi_s, rho_s] = cart2pol(source_location[0], source_location[1])
    [phi, rho]     = cart2pol(sensor_location[0], sensor_location[1])
    
    Ez        = np.zeros(np.atleast_1d(frequency).shape)
    k = background.getElectromagneticWaveNumber(frequency)
    epsilon= EPS_O*background.getPermittivity(frequency)[0]
    N_max = getN_max(rho, background, background, frequency)
    nu = np.arange(-N_max, N_max, dtype=np.int16).reshape(1,-1)
    factor= -k**2/(4*omega*epsilon)
    x     = k*rho
    x_s   = k*rho_s

    if (rho < rho_s): # inside source
        Ez    = factor*np.sum(besselh(nu,2,x_s)*besselj(nu,x)*np.exp(1j*nu*(phi-phi_s)))
        #H_rho(iFreq) = 1/(4*j)/rho.*sum(besselh(nu,2,x_s).*besselj(nu,x).*(j*nu).*exp(j*nu*(phi-phi_s)));
        #H_phi(iFreq) =-k/(4*j).*sum(besselh(nu,2,x_s).*besselj_derivative(nu,x).*exp(j*nu*(phi-phi_s)));
    else: # outside source circle
        Ez    = factor*np.sum(besselh(nu,2,x)*besselj(nu,x_s)*np.exp(1j*nu*(phi-phi_s)))
        #H_rho(iFreq) = 1/(4*j)/rho.*sum(besselh(nu,2,x).*besselj(nu,x_s).*(j*nu).*exp(j*nu*(phi-phi_s)));
        #H_phi(iFreq) =-k/(4*j).*sum(besselh_derivative(nu,2,x).*besselj(nu,x_s).*exp(j*nu*(phi-phi_s)));
    
    return Ez
    
    


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
