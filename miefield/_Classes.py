#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module mie

Analytical solution (series) for scattering on spheres and cylinders

"""
from __future__ import division
from __future__ import print_function

import numpy as np

__all__ = ["DielectricMaterial"]

class DielectricMaterial(object):
    def __init__(self, epsilon_r, sigma_e):
        self.epsilon_r = epsilon_r
        self.sigma_e = sigma_e
        self.mu_r = 1
        self.sigma_m = 0


    def getComplexPermeability(self, frequency):
        MU_O      = 4*np.pi*1e-7
        mu_r      = self.mu_r + self.sigma_m / (1j*2*np.pi*frequency*MU_O)
        return mu_r
        
                
    def getComplexPermittivity(self, frequency):
        EPS_O          = 8.8541878176e-12 
        epsilon_r      = self.epsilon_r + self.sigma_e / (1j*2*np.pi*frequency*EPS_O)
        return epsilon_r


    def getComplexRefractiveIndex(self, frequency):
        eps_r = self.getComplexPermittivity(frequency)
        mu_r  = self.getComplexPermeability(frequency)
        ref_idx =  np.sqrt(eps_r * mu_r)
        return ref_idx
    

    def getElectromagneticWaveNumber(self, frequency):
        C_O = 2.997924580003452e+08
        permittivity = self.getComplexPermittivity(frequency)
        permeability = self.getComplexPermeability(frequency)
        wave_number  = 2*np.pi*frequency*np.sqrt(permittivity * permeability) / C_O
        return wave_number


    def getElectromagneticWavelength(self, frequency):
        wave_number = self.getElectromagneticWaveNumber(frequency)
        wavelength = 2*np.pi / np.real(wave_number)
        return wavelength


    def getIntrinsicImpedance(self, frequency):
        EPS_O = 8.8541878176e-12
        MU_O  = 4*np.pi*1e-7
        ETA_O = np.sqrt(MU_O/EPS_O)
        eta   = ETA_O*np.sqrt(self.getComplexPermeability(frequency) /
                              self.getComplexPermittivity(frequency))
        return eta
        
        
    def getPermittivity(self, frequency):
        epsilon_r = np.ones(np.atleast_1d(frequency).shape)*self.epsilon_r
        sigma_e = np.ones(np.atleast_1d(frequency).shape)*self.sigma_e
        return np.array([epsilon_r, sigma_e])
