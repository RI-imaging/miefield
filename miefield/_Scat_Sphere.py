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


def getDielectricSphereFieldUnderPlaneWave(radius, sphere, background,
                                  sensor_location, frequency, varargin):
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
    mu_d = sphere.getComplexPermeability(MU_O)
    eps_d = sphere.getComplexPermittivity(frequency)
    

    N = getN_max(radius, sphere, background, frequency)
    #N = getN_max((p.Results.radius), cellarray(np.hstack((p.Results.sphere))), (p.Results.background), (p.Results.frequency))
    #N = matcompat.max(N)

    nu = np.arange(N) + 1 

    [r, theta, phi] = cart2sph(sensor_location[0], sensor_location[1], sensor_location[2])
    # Compute coefficients 
    a_n = 1j**(-nu) * (2*nu+1) / (nu*(nu+1))
    
    #a_n = np.dot(np.ones(nFreq, 1.), a_n)
    
    # temp2 denotes the expression
    # kzlegendre(nu,1,cos(theta))/sin(theta). Here I am using a
    # recursive relation to compute temp2, which avoids the numerical
    # difficulty when theta == 0 or PI.
    temp2 = np.zeros((len(nu), len(theta)))
    temp2[0] = -1
    temp2[1] = -3 * np.cos(theta)
    # if N = 10, then nu = [1,2,3,4,5,6,7,8,9,19]
    for n in np.arange(len(nu)-2)+1:
        # matlab: [2,3,4,5,6,7,8,9]
        # python: [1,2,3,4,5,6,7,8]
        temp2[n+1] = (2*n+1)/n * np.cos(theta)*temp2[n] - (n+1)/n * temp2[n-1]
        
    # temp1 denotes the expression
    # sin(theta)*kzlegendre_derivative(nu,1,cos(theta)).  Here I am
    # also using a recursive relation to compute temp1 from temp2,
    # which avoids numerical difficulty when theta == 0 or PI.
    temp1 = np.zeros((length(nu), len(theta)))
    temp1[0] = np.cos(theta)
    for n in np.arange(len(nu)-1)+1:
        # matlab: [2,3,4,5,6,7,8,9,10]  (index starts at 1)
        # python: [1,2,3,4,5,6,7,8,9]   (index starts at 0)  
        temp1[n-1] = (n+1) * temp2[n-1]-n*np.cos(theta)*temp2[n]
        
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
    
    
    if r<radius:
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
        
        x = k_d * r
        
        ## Implement (11-239a) in [Balanis1989] 
        #alpha = (transpose(ric_besselj_derivative(nu,x,2))+transpose(ric_besselj(nu,x)))...
        #        .*transpose(kzlegendre(nu,1,cos(theta))*ones(1,nFreq));        
        alpha = ( (np.transpose(ric_besselj_derivative(nu, x, 2))+np.transpose(ric_besselj(nu, x)))
                  *np.transpose(kzlegendre(nu, 1, np.cos(theta)))              )
        # E_r = -j*cos(phi)*sum(d_n.*alpha, 2);
        E_r = -1j*np.cos(phi) * np.sum(d_n*alpha, 1)
        #H_r  = -j*sin(phi)*sum(e_n.*alpha, 2)./eta_d;
        H_r = -1j*np.sin(phi) * np.sum(e_n*alpha, 1)/eta_d
        
        ## Implement (11-239b) in [Balanis1989]
        #alpha = transpose(ric_besselj_derivative(nu,x)).*temp1;
        alpha = np.transpose(ric_besselj_derivative(nu, x))*temp1
        #beta = transpose(ric_besselj(nu,x)).*temp2;
        beta = np.transpose(ric_besselj(nu, x))*temp2
        # summation = j*d_n.*alpha - e_n.*beta;
        summation = 1j*d_n*alpha-e_n*beta
        # E_theta   = cos(phi)./x.*sum(summation,2);
        E_theta = np.cos(phi)/x*np.sum(summation, 1)
        # summation = j*e_n.*alpha - d_n.*beta;
        summation = 1j*e_n*alpha - d_n*beta
        # H_theta = sin(phi)./x.*sum(summation,2)./eta_d;
        H_theta = np.sin(phi)/x*np.sum(summation, 1)/eta_d
        
        ## Implement (11-239c) in [Balanis1989]
        # alpha = transpose(ric_besselj_derivative(nu,x)).*temp2;
        alpha = np.transpose(ric_besselj_derivative(nu, x))*temp2
        # beta = transpose(ric_besselj(nu,x)).*temp1;
        beta = np.transpose(ric_besselj(nu, x))*temp1
        # summation = j*d_n.*alpha - e_n.*beta;
        summation = 1j*d_n*alpha - e_n*beta
        # E_phi = sin(phi)./x.*sum(summation,2);
        E_phi = np.sin(phi)/x*np.sum(summation, 1)
        # summation = j*e_n.*alpha - d_n.*beta;
        summation = 1j*e_n*alpha - d_n*beta
        # H_phi     =-cos(phi)./x.*sum(summation,2)./eta_d;
        H_phi = -np.cos(phi)/x*np.sum(summation, 1)/eta_d
        
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
        
        x = k*r

        ## Implement (11-239a) in [Balanis1989]
        #alpha = (transpose(ric_besselh_derivative(nu,2,x,2))      +transpose(ric_besselh(nu,2,x)))...
        #    .*transpose(kzlegendre(nu,1,cos(theta))*ones(1,nFreq));
        alpha = ( (np.transpose(ric_besselh_derivative(nu, 2, x, 2)) + np.transpose(ric_besselh(nu, 2, x))) 
                 *(np.transpose(kzlegendre(nu, 1, np.cos(theta))))                                           )
        #E_r = -j*cos(phi)*sum(b_n.*alpha, 2);
        E_r = -1j*np.cos(phi) * np.sum(b_n*alpha, 1)
        #H_r = -j*sin(phi)*sum(c_n.*alpha, 2)./eta;
        H_r = -1j*np.sin(phi) * np.sum(c_n*alpha, 1)/eta
        
        ## Implement (11-239b) in [Balanis1989]
        #alpha = transpose(ric_besselh_derivative(nu,2,x)).*temp1;
        alpha = np.transpose(ric_besselh_derivative(nu, 2, x))*temp1
        #beta = transpose(ric_besselh(nu,2,x)).*temp2;
        beta = np.transpose(ric_besselh(nu, 2, x))*temp2
        #summation = j*b_n.*alpha - c_n.*beta;
        summation = 1j*b_n*alpha - c_n*beta
        #E_theta = cos(phi)./x.*sum(summation,2);
        E_theta = np.cos(phi)/x*np.sum(summation, 1)
        #summation = j*c_n.*alpha - b_n.*beta;
        summation = 1j*c_n*alpha - b_n*beta
        #H_theta = sin(phi)./x.*sum(summation,2)./eta;
        H_theta = np.sin(phi)/x*np.sum(summation, 1)/eta
        
        ## Implement (11-239c) in [Balanis1989]
        #alpha     = transpose(ric_besselh_derivative(nu,2,x)).*temp2;
        alpha = np.transpose(ric_besselh_derivative(nu, 2, x))*temp2
        #beta = transpose(ric_besselh(nu,2,x)).*temp1;
        beta = np.transpose(ric_besselh(nu, 2, x))*temp1
        #summation = j*b_n.*alpha - c_n.*beta;
        summation = 1j*b_n*alpha - c_n*beta
        #E_phi = sin(phi)./x.*sum(summation,2);
        E_phi = np.sin(phi)/x*np.sum(summation, 1)
        #summation = j*c_n.*alpha - b_n.*beta;
        summation = 1j*c_n*alpha - b_n*beta
        #H_phi =-cos(phi)./x.*sum(summation,2)./eta;
        H_phi = -np.cos(phi)/x*np.sum(summation, 1)/eta
        
    
    return [E_r, E_theta, E_phi, H_r, H_theta, H_phi]


def GetFieldSphere():
    pass