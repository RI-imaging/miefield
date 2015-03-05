#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    Mie solution 
    
    Calculates the electric field component (Ex) for a plane wave that
    is scattered by a dielectric sphere.
    
    Some of this code is a partial translation of the Matlab code from
    Guangran Kevin Zhu
    http://www.mathworks.de/matlabcentral/fileexchange/30162-cylinder-scattering
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import os

import scipy

from ._Classes import *
from ._Functions import *


__all__ = ["GetFieldSphere"]

def GetFieldSphere(radius, nmed, nsphe, lD, size, res):

    sphere = DielectricMaterial(nsphe**2,0.0)
    background = DielectricMaterial(nmed**2,0.0)
    reference =  DielectricMaterial(1,0.0)
    lambref = reference.getElectromagneticWavelength(1.0)
    xmax = size / res / 2.0
    # the detector resolution is not dependent on the medium
    detector = np.linspace(-xmax, xmax, size, endpoint=True) * lambref
    sensor_location = np.zeros((3,size))
    sensor_location[0] = lD*lambref    # optical path length to detector
    sensor_location[1] = detector
    sensor_location[2] = detector  #HHH 3D experience
    return getDielectricSphereFieldUnderPlaneWave(radius*lambref, 
             sphere, background, sensor_location)
    


def getDielectricSphereFieldUnderPlaneWave(radius, sphere, background,
                                  sensor_location, frequency=1):
    """
        Calculate the field scattered by a dielectric sphere centered at
        the origine due to an incident x-polarized plane wave. The
        scattered field is in (11-239) in [Balanis1989].  see the notes
        on 2008-05-24 for the coefficients, a_n, b_n, and c_n.
        
        See Fig. 11-25 in [Balanis1989] for the exact geometry.
        
        Input:
        
        radius             scalar to denote the radius of the sphere (m)
        sphere             object of DielectricMaterial
        background         object of DielectricMaterial
        sensor_location    3x1 vector in the form of [x; y; z] (m)
        frequency          Nx1 vector in (Hz)
        
        Output:
        
        E_r               Nx1 vector (V/m)
        E_phi             Nx1 vector (V/m)
        E_theta           Nx1 vector (V/m)
        H_r               Nx1 vector (A/m)
        H_phi             Nx1 vector (A/m)
        H_theta           Nx1 vector (A/m)
        
        This function is a translation to Python from a matlab script
        by Guangran Kevin Zhu.
    """
    #p = inputParser
    #p.addRequired[int('radius')-1,int(isnumeric)-1]
    #p.addRequired[int('sphere')-1,int(isobject)-1]
    #p.addRequired[int('background')-1,int(isobject)-1]
    #p.addRequired[int('sensor_location')-1,int(isvector)-1]
    #p.addRequired[int('frequency')-1,int(isnumeric)-1]
    #p.addParamValue[int('debug')-1,-1,lambda x: x == 0. or x == 1.]
    #p.parse[int(radius)-1,int(sphere)-1,int(background)-1,int(sensor_location)-1,int(frequency)-1,varargin.cell[:]]
    #if p.Results.debug:
    #    np.disp((p.Results))
    
    # Compute all intrinsic variables
    omega = 2.*np.pi*frequency
    eta = background.getIntrinsicImpedance(frequency)
    k = background.getElectromagneticWaveNumber(frequency)
    mu = background.getComplexPermeability(frequency)
    eps = background.getComplexPermittivity(frequency)
    eta_d = sphere.getIntrinsicImpedance(frequency)
    k_d = sphere.getElectromagneticWaveNumber(frequency)
    mu_d = sphere.getComplexPermeability(frequency)
    eps_d = sphere.getComplexPermittivity(frequency)
    

        
    
    N = getN_max(radius, sphere, background, frequency)    
    #N = getN_max((p.Results.radius), cellarray(np.hstack((p.Results.sphere))), (p.Results.background), (p.Results.frequency))
    #N = matcompat.max(N)

    nu = np.arange(N) + 1 
    [r, theta, phi] = cart2sph(sensor_location[0], sensor_location[1], sensor_location[2])
    r.resize(len(r),1)
    theta.resize(len(theta),1)
    phi.resize(len(phi),1)
    # Compute coefficients 
    a_n = 1j**(-nu) * (2*nu+1) / (nu*(nu+1))
    
    #a_n = np.dot(np.ones(nFreq, 1.), a_n)
    
    # temp2 denotes the expression
    # kzlegendre(nu,1,cos(theta))/sin(theta). Here I am using a
    # recursive relation to compute temp2, which avoids the numerical
    # difficulty when theta == 0 or PI.
    temp2 = np.zeros((len(theta),len(nu))) #HHH matlab original: temp2 = np.zeros((len(nu), len(theta))) ###### changed to akin to matlab
    temp2[:,0] = -1                        #HHH all in first column
    temp2[:,1] = (-3*np.cos(theta)).T      #HHH matlab original: temp2(2) = -3*cos(theta) ##### Transverse or it doens't work. You need to replace a column with a row, figure that.
    # if N = 10, then nu = [1,2,3,4,5,6,7,8,9,19]
    for n in np.arange(len(nu)-2)+1:
        # matlab: [2,3,4,5,6,7,8,9]
        # python: [1,2,3,4,5,6,7,8]
        temp2[:,n+1] = (2*n+1)/n * np.cos(theta).T*temp2[:,n] - (n+1)/n * temp2[:,n-1]     #HHH matlab original: temp2(n+1) = (2*n+1)/n*cos(theta)*temp2(n) - (n+1)/n*temp2(n-1)   ####selecting whole columns, using transverses properly
        
    # temp1 denotes the expression
    # sin(theta)*kzlegendre_derivative(nu,1,cos(theta)).  Here I am
    # also using a recursive relation to compute temp1 from temp2,
    # which avoids numerical difficulty when theta == 0 or PI.
    temp1 = np.zeros((len(theta), len(nu)))  #HHH changed to keep matlab's structure.
    temp1[:,0] = np.cos(theta).T
    for n in np.arange(len(nu)-1)+1:
        # matlab: [2,3,4,5,6,7,8,9,10]  (index starts at 1)
        # python: [1,2,3,4,5,6,7,8,9]   (index starts at 0)  
        temp1[:,n-1] = (n+1) * temp2[:,n-1]-n*np.cos(theta).T*temp2[:,n]
        
    #temp1 = np.dot(np.ones(nFreq, 1.), temp1)
    #temp2 = np.dot(np.ones(nFreq, 1.), temp2)
    
    iNU = 10
    
    #if p.Results.debug:
    #    A = np.array(np.vstack((np.hstack((ric_besselh_derivative(iNU, 2., np.dot(k, radius)), np.dot(matdiv(-np.sqrt(np.dot(eps, mu)), np.sqrt(np.dot(eps_d, mu_d))), ric_besselj_derivative(iNU, np.dot(k_d, radius))))), np.hstack((ric_besselh(iNU, 2., np.dot(k, radius)), np.dot(matdiv(-mu, mu_d), ric_besselj(iNU, np.dot(k_d, radius))))))))
    #    rhs = np.dot(-a_n[int(iNU)-1], np.array(np.vstack((np.hstack((ric_besselj_derivative(iNU, np.dot(k, radius)))), np.hstack((ric_besselj(iNU, np.dot(k, radius))))))))
    #    x = linalg.solve(A, rhs)
    #    np.disp(np.array(np.hstack(('b_n ', num2str(x[0]), d_n, num2str(x[1])))))
    #    A = np.array(np.vstack((np.hstack((ric_besselh(iNU, 2., np.dot(k, radius)), np.dot(matdiv(-np.sqrt(np.dot(eps, mu)), np.sqrt(np.dot(eps_d, mu_d))), ric_besselj(iNU, np.dot(k_d, radius))))), np.hstack((ric_besselh_derivative(iNU, 2., np.dot(k, radius)), np.dot(matdiv(-mu, mu_d), ric_besselj_derivative(iNU, np.dot(k_d, radius))))))))
    #    rhs = np.dot(-a_n[int(iNU)-1], np.array(np.vstack((np.hstack((ric_besselj(iNU, np.dot(k, radius)))), np.hstack((ric_besselj_derivative(iNU, np.dot(k, radius))))))))
    #    x = linalg.solve(A, rhs)
    #    np.disp(np.array(np.hstack(('c_n ', num2str(x[0]), e_n, num2str(x[1])))))
    #    np.disp('------')
    
    #alpha = np.zeros((len(theta),len(nu)))          #HHH In matlab, alpha is a row, with nu number values. since here r,theta,phi is a column, alpha has to be an array the size of (theta,nu), so it can include all the nus (in row) per value of r (in colum)
    #print("alpha shape",np.shape(alpha))
    
    
    #HHH initializing final result, and adding 0j so it's imaginary from the start
    E_r        = np.zeros(np.shape(theta)) + 0j
    E_theta    = np.zeros(np.shape(theta)) + 0j
    E_phi      = np.zeros(np.shape(theta)) + 0j
    H_r        = np.zeros(np.shape(theta)) + 0j
    H_theta    = np.zeros(np.shape(theta)) + 0j
    H_phi      = np.zeros(np.shape(theta)) + 0j   
    
    for elem in range (0,np.size(r)):         #HHH gotta evaluate element by element in r (which is a column array)
        #print("elem: ",elem, "is r: ",r[elem],"/",radius,"out of ", np.size(r))
    
        if r[elem] < radius:
            #num = j.*mu_d/sqrt(mu)*sqrt(eps_d);
            num = 1j*mu_d/np.sqrt(mu)*np.sqrt(eps_d)
            #den =  - sqrt(mu.  *eps_d)    *ones(1,N).       *transpose(ric_besselj(nu,k_d*radius)).    *transpose(ric_besselh_derivative(nu,2,k*radius))...
            #       + sqrt(mu_d.*eps)      *ones(1,N).       *transpose(ric_besselh(nu,2,k*radius)).    *transpose(ric_besselj_derivative(nu,k_d*radius));
            den = ( - (np.sqrt(mu   * eps_d)*np.ones((1,N))) * np.transpose(ric_besselj(nu,k_d*radius)) * np.transpose(ric_besselh_derivative(nu,2,k*radius))
                    + (np.sqrt(mu_d * eps  )*np.ones((1,N))) * np.transpose(ric_besselh(nu,2,k*radius)) * np.transpose(ric_besselj_derivative(nu,k_d*radius))    )
            #d_n = num*ones(1,N)./den.*a_n;
            d_n = num*np.ones((1, N))/den*a_n
        
            #den =  + sqrt(mu.*eps_d)       *ones(1,N).      *transpose(ric_besselh(nu,2,k*radius)).    *transpose(ric_besselj_derivative(nu,k_d*radius))...
            #       - sqrt(mu_d.*eps)       *ones(1,N).      *transpose(ric_besselj(nu,k_d*radius)).    *transpose(ric_besselh_derivative(nu,2,k*radius));
            den = ( + (np.sqrt(mu   * eps_d)*np.ones((1,N))) * np.transpose(ric_besselh(nu,2,k*radius)) * np.transpose(ric_besselj_derivative(nu,k_d*radius))
                    - (np.sqrt(mu_d * eps  )*np.ones((1,N))) * np.transpose(ric_besselj(nu,k_d*radius)) * np.transpose(ric_besselh_derivative(nu,2,k*radius))     )
            #e_n = num*ones(1,N)./den.*a_n;
            e_n =  num*np.ones((1, N))/den*a_n
        
            x = k_d * r[elem]               #HHH x of the current r[elem]
            x=x[0]                      #HHH x should be integer... or problems

            ## Implement (11-239a) in [Balanis1989] 
            #alpha = (transpose(ric_besselj_derivative(nu,x,2))+transpose(ric_besselj(nu,x)))...
            #        .*transpose(kzlegendre(nu,1,cos(theta))*ones(1,nFreq));   

            alpha = (  (np.transpose(ric_besselh_derivative(nu, 2, x, 2)) + np.transpose(ric_besselh(nu, 2, x))) * 
                              np.transpose(kzlegendre(nu, 1, np.cos(theta[elem])))         )       #HHH obviously, specific theta[elem] is used for alpha            
                                       
            # E_r = -j*cos(phi)*sum(d_n.*alpha, 2);
            E_r[elem] = -1j*np.cos(phi[elem]) * np.sum(d_n*alpha, 1)                    #HHH use specific row of phi to get a single number
            #H_r  = -j*sin(phi)*sum(e_n.*alpha, 2)./eta_d;
            H_r[elem] = -1j*np.sin(phi[elem]) * np.sum(e_n*alpha, 1)/eta_d              #HHH use specific row of phi to get a single number

            ## Implement (11-239b) in [Balanis1989]
            #alpha = transpose(ric_besselj_derivative(nu,x)).*temp1;
            alpha = np.transpose(ric_besselj_derivative(nu, x))*temp1[elem]
            #beta = transpose(ric_besselj(nu,x)).*temp2;
            beta = np.transpose(ric_besselj(nu, x))*temp2[elem]
            # summation = j*d_n.*alpha - e_n.*beta;
            summation = 1j*d_n*alpha-e_n*beta
            # E_theta   = cos(phi)./x.*sum(summation,2);
            E_theta[elem] = np.cos(phi[elem])/x*np.sum(summation, 1)
            # summation = j*e_n.*alpha - d_n.*beta;
            summation = 1j*e_n*alpha - d_n*beta
            # H_theta = sin(phi)./x.*sum(summation,2)./eta_d;
            H_theta[elem] = np.sin(phi[elem])/x*np.sum(summation, 1)/eta_d

            ## Implement (11-239c) in [Balanis1989]
            # alpha = transpose(ric_besselj_derivative(nu,x)).*temp2;
            alpha = np.transpose(ric_besselj_derivative(nu, x))*temp2[elem]
            # beta = transpose(ric_besselj(nu,x)).*temp1;
            beta = np.transpose(ric_besselj(nu, x))*temp1[elem]
            # summation = j*d_n.*alpha - e_n.*beta;
            summation = 1j*d_n*alpha - e_n*beta
            # E_phi = sin(phi)./x.*sum(summation,2);
            E_phi[elem] = np.sin(phi[elem])/x*np.sum(summation, 1)
            # summation = j*e_n.*alpha - d_n.*beta;
            summation = 1j*e_n*alpha - d_n*beta
            # H_phi     =-cos(phi)./x.*sum(summation,2)./eta_d;
            H_phi[elem] = -np.cos(phi[elem])/x*np.sum(summation, 1)/eta_d
        
        else:

            # num =  + sqrt(mu_d.*eps)*ones(1,N).      *transpose(ric_besselj(nu,k*radius))  .     *transpose(ric_besselj_derivative(nu,k_d*radius)) ...
            #        - sqrt(mu.*eps_d)*ones(1,N).      *transpose(ric_besselj(nu,k_d*radius)).     *transpose(ric_besselj_derivative(nu,k*radius));
            num = ( (np.sqrt(mu_d*eps)*np.ones((1, N))) * np.transpose(ric_besselj(nu, k*radius))  *np.transpose(ric_besselj_derivative(nu, k_d*radius))
                   -(np.sqrt(mu*eps_d)*np.ones((1, N))) * np.transpose(ric_besselj(nu, k_d*radius))*np.transpose(ric_besselj_derivative(nu, k*radius))   )

            #den =  + sqrt(mu.*eps_d)*ones(1,N).        *transpose(ric_besselj(nu,k_d*radius))       *transpose(ric_besselh_derivative(nu,2,k*radius))...
            #       - sqrt(mu_d.*eps)*ones(1,N).        *transpose(ric_besselh(nu,2,k*radius)).      *transpose(ric_besselj_derivative(nu,k_d*radius));
            den = ( (np.sqrt(mu*eps_d)*np.ones((1, N))) * np.transpose(ric_besselj(nu, k_d*radius))  * np.transpose(ric_besselh_derivative(nu, 2, k*radius))
                   -(np.sqrt(mu_d*eps)*np.ones((1, N))) * np.transpose(ric_besselh(nu, 2, k*radius)) * np.transpose(ric_besselj_derivative(nu, k_d*radius)))
               
            #b_n = num./den.*a_n;
            b_n = num/den*a_n
            
            
            #num =  + sqrt(mu_d.*eps)*ones(1,N).        *transpose(ric_besselj(nu,k_d*radius)).     *transpose(ric_besselj_derivative(nu,k*radius))...
            #       - sqrt(mu.*eps_d)*ones(1,N).        *transpose(ric_besselj(nu,k*radius))  .     *transpose(ric_besselj_derivative(nu,k_d*radius));
            num = ( (np.sqrt(mu_d*eps)*np.ones((1, N))) * np.transpose(ric_besselj(nu, k_d*radius)) * np.transpose(ric_besselj_derivative(nu, k*radius))
                   -(np.sqrt(mu*eps_d)*np.ones((1, N))) * np.transpose(ric_besselj(nu, k*radius))   * np.transpose(ric_besselj_derivative(nu, k_d*radius))  )

            #den = + sqrt(mu.*eps_d)*ones(1,N).         *transpose(ric_besselh(nu,2,k*radius)).      *transpose(ric_besselj_derivative(nu,k_d*radius))...
            #      - sqrt(mu_d.*eps)*ones(1,N).         *transpose(ric_besselj(nu,k_d*radius)).      *transpose(ric_besselh_derivative(nu,2,k*radius));
            den = ( (np.sqrt(mu*eps_d)*np.ones((1, N))) * np.transpose(ric_besselh(nu, 2, k*radius)) * np.transpose(ric_besselj_derivative(nu, k_d*radius))
                   -(np.sqrt(mu_d*eps)*np.ones((1, N))) * np.transpose(ric_besselj(nu, k_d*radius))  * np.transpose(ric_besselh_derivative(nu, 2, k*radius))  )

            # c_n = num./den.*a_n;
            c_n = num/den*a_n
        
            #if p.Results.debug:
            #    np.disp(np.array(np.hstack(('b_n ', num2str(b_n[int(iNU)-1]), c_n, num2str(c_n[int(iNU)-1])))))
            #    return []

            x = k*r[elem]               #HHH x of the current r[elem]
            x=x[0]                      #HHH x should be integer... or problems
            
            ## Implement (11-239a) in [Balanis1989]
            #alpha = (transpose(ric_besselh_derivative(nu,2,x,2))      +transpose(ric_besselh(nu,2,x)))...
            #    .*transpose(kzlegendre(nu,1,cos(theta))*ones(1,nFreq));

            alpha = (  (np.transpose(ric_besselh_derivative(nu, 2, x, 2)) + np.transpose(ric_besselh(nu, 2, x))) * 
                              np.transpose(kzlegendre(nu, 1, np.cos(theta[elem])))         )       #HHH obviously, specific theta[elem] is used for alpha[elem]             
            
            #E_r = -j*cos(phi)*sum(b_n.*alpha, 2);
            E_r[elem] = -1j*np.cos(phi[elem]) * np.sum(b_n*alpha, 1)                    #HHH use specific row of phi to get a single number
            #H_r = -j*sin(phi)*sum(c_n.*alpha, 2)./eta;
            H_r[elem] = -1j*np.sin(phi[elem]) * np.sum(c_n*alpha, 1)/eta                #HHH use specific row of phi to get a single number
        
            ## Implement (11-239b) in [Balanis1989]
            #alpha = transpose(ric_besselh_derivative(nu,2,x)).*temp1;
            alpha = np.transpose(ric_besselh_derivative(nu, 2, x))*temp1[elem]
            #beta = transpose(ric_besselh(nu,2,x)).*temp2;
            beta = np.transpose(ric_besselh(nu, 2, x))*temp2[elem]
            #summation = j*b_n.*alpha - c_n.*beta;
            summation = 1j*b_n*alpha - c_n*beta
            #E_theta = cos(phi)./x.*sum(summation,2);
            E_theta[elem] = np.cos(phi[elem])/x*np.sum(summation, 1)
            #summation = j*c_n.*alpha - b_n.*beta;
            summation = 1j*c_n*alpha - b_n*beta
            #H_theta = sin(phi)./x.*sum(summation,2)./eta;
            H_theta[elem] = np.sin(phi[elem])/x*np.sum(summation, 1)/eta
            
            ## Implement (11-239c) in [Balanis1989]
            #alpha     = transpose(ric_besselh_derivative(nu,2,x)).*temp2;
            alpha = np.transpose(ric_besselh_derivative(nu, 2, x))*temp2[elem]
            #beta = transpose(ric_besselh(nu,2,x)).*temp1;
            beta = np.transpose(ric_besselh(nu, 2, x))*temp1[elem]
            #summation = j*b_n.*alpha - c_n.*beta;
            summation = 1j*b_n*alpha - c_n*beta
            #E_phi = sin(phi)./x.*sum(summation,2);
            E_phi[elem] = np.sin(phi[elem])/x*np.sum(summation, 1)
            #summation = j*c_n.*alpha - b_n.*beta;
            summation = 1j*c_n*alpha - b_n*beta
            #H_phi =-cos(phi)./x.*sum(summation,2)./eta;
            H_phi[elem] = -np.cos(phi[elem])/x*np.sum(summation, 1)/eta
        
    
    return [E_r, E_theta, E_phi, H_r, H_theta, H_phi]