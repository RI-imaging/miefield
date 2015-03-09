#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    extract_fields

    Extract the field from Zhu saved
    ( MYDielectricSphereTotalFieldUnderPlaneWave.m )
    
    Plot phase and amplitude
    
    PROBLEM:
    This script takes the Er,Ephi,Etheta fields from Zhus script and
    computes the Ex,Ey,Ez field for the backpropagation.
    
    This is a possible source of error. Somehow the reconstruction 
    does not work.
    
    Reonstruct the three-dimensional refractive index distribution for
    a sphere. The Thomson problem must be solved for the particular
    number of projections.

"""

from __future__ import division

import csv
import IPython
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['text.latex.unicode']=True
matplotlib.rcParams['font.family']='serif'
matplotlib.rcParams['text.latex.preamble']=[r"""\usepackage{amsmath}
                                            \usepackage[utf8x]{inputenc}
                                            \usepackage{amssymb}"""] 
from matplotlib import pylab as plt
from matplotlib import pylab as plt
import numpy as np
import os
import sys

scatriddir=os.path.realpath("../tomography-reconstruction/")
sys.path.append(scatriddir)

from scatrid import born_rytov as br
from scatrid import filters

from tool import *


def io_GetCartesianField2D(field, lD, size, res):
    """
        Fields are saved in spherical coordinates
        convert them to cartesian coordinates.
        fname must contain the "phi" component.
        
        plotDielectricSphereTotalFieldUnderPlaneWave from Zhu:
        Wave propagates in +z-direction.
        
        Data is in spherical coordinates.
        
        y
        ^ x
        |/
        ----> z  
              
         sphere       detector   
                           ,  (xD,yD,zD=lD)   
           ___            /|
         /     \         | | 
        |(0,0,0)|        | | 
         \ ___ /         | | 
                         |/  (xD,yD,zD=lD)  
                         ´ 
        
        
    """
    #Er = io_OpenDatField2D(DIR, fname[:-7]+"r.dat")
    Ephi = field[0].flatten()
    #Etheta = io_OpenDatField2D(DIR, fname[:-7]+"theta.dat")
    Ephi = field[1].flatten()
    #Ephi = io_OpenDatField2D(DIR, fname)
    Ephi = field[2].flatten()
    # Define cartesian coordinates.
    #info = io_GetInfoZhu(fname)
    
    
    #sd = info["size of detector [px]"]
    sd = res
    #xlim = sd/2*info["effective pixel size [wavelengths]"]
    xlim = sd/2*size
    car = np.linspace(-xlim, xlim, sd, endpoint=True)

 #   x = info["axial measurement position [wavelengths]"]
 #   z = car.reshape(-1,1)
 #   y = car.reshape(1,-1)
 #
 #   r_plane = np.sqrt(x**2 + y**2)
 #   phi = np.arctan2(y,x)
 #   theta = np.arctan2(r_plane,z)
    

    # NOTE: Zhu coordinates are in wavelengths
    #z = info["axial measurement position [wavelengths]"]
    z = lD 
    #x,y = np.meshgrid(-car,car)
    x = car.reshape(1,-1)
    y = car.reshape(-1,1)
    #x,y = np.meshgrid(car,car)
    # Define spherical coordinates
 
 
    # [rmin, rmax]
    r_plane = np.sqrt(x**2 + y**2)
 
    r_full = np.sqrt(x**2 + y**2 +z**2)
    # [0, 2pi]
    phi = np.arctan2(y,x)
 
    # [0, theta_max]
    theta = np.arctan2(z,r_plane)
    # z = r cos(theta) 
    
    #theta = np.arccos(z/r_full)    
    
    
    # E = Er * er + Ephi * ephi + Etheta * etheta
    # er =       sin(theta)cos(phi)   ex
    #           +sin(theta)sin(phi)   ey
    #           +cos(theta)           ez
    #
    # ephi =   -sin(phi)              ex
    #          +cos(phi)              ey
    #
    # etheta =  cos(theta)cos(phi)    ex
    #          +cos(theta)sin(phi)    ey
    #          -sin(theta)            ez
    
    # Therefore:
    # Ex =   Er     *  sin(theta)cos(phi)
    #       +Ephi   * -sin(phi)
    #       +Etheta *  cos(theta)cos(phi)
    #
    # Ey =   Er     * +sin(theta)sin(phi)
    #       +Ephi   *  cos(phi)
    #       +Etheta * +cos(theta)sin(phi)
    #
    # Ez =   Er     *  cos(theta)
    #       +Etheta * -sin(theta)

    
    Ex =(   Er     *  np.sin(theta)*np.cos(phi)
           +Ephi   * -np.sin(phi)
           +Etheta *  np.cos(theta)*np.cos(phi)    )
    
    Ey =(   Er     *  np.sin(theta)*np.sin(phi)
           +Ephi   *  np.cos(phi)
           +Etheta *  np.cos(theta)*np.sin(phi)    )

    Ez =(   Er     *  np.cos(theta)
           +Etheta * -np.sin(theta)    )

    import IPython
    IPython.embed()
    return [Ex,Ey,Ez]
    
    
def io_OpenDatField2D(DIR, fname):
    """ Opens dat file and returns
        returns: Field in file
    """
    f = open(os.path.join(DIR,fname),"r")
    r = csv.reader(f, delimiter='\t')
    Ey = list()
    for line in r:
        Ex = list()
        for item in line:
            item = item.replace("i","j")
            Ex.append(eval(item))
        Ey.append(Ex)
    return np.transpose(np.array(Ey))
    #return np.transpose(np.array(Ey))
    #return np.array(Ey)
    

def io_GetDatFields2D(DIR, fname):
    """ Calls io_OpenDatField for Field with background.
        Background filename is "B"+fname.
        returns: EX, BEx
    """
    Ex = io_GetCartesianField2D(DIR, fname)
    BEx = io_GetCartesianField2D(DIR, "B"+fname)
    
    #Ex = np.transpose( np.transpose(Ex[:-1])[:-1] )
    #BEx = np.transpose( np.transpose(BEx[:-1])[:-1] )
    
    return Ex, BEx


def io_GetInfoZhu(filename="filename.txt"):
    """ Get all info from the filename which looks like this:
    
        Ez_RI1.001000_RImed1.000000_r10.000000_yD15.000000_Xsize128.000000_px0.500000__Ephi
    
    """
    info = dict()
    data = filename.split("_")
    for item in data:
        if item[:5] == "RImed":
            info["refractive index of medium"] = float(item[5:])
        elif item[:2] == "RI":
            info["refractive index of sphere"] = float(item[2:])
        if item[:1] == "r":
            info["radius of sphere [wavelengths]"] = float(item[1:])
        if item[:2] == "yD":
            info["axial measurement position [wavelengths]"] = float(item[2:])
        if item[:5] == "Xsize":
            info["size of detector [px]"] = float(item[5:])
        if item[:2] == "px":
            info["effective pixel size [wavelengths]"] = float(item[2:])
        if item[:1] == "E":
            if item[-4:] == ".dat":
                #remove file type
                item = item[:-4]
            info["Field component"] = item[1:]
    info["axial measurement position [px]"] = info["axial measurement position [wavelengths]"]/info["effective pixel size [wavelengths]"]
    return info
    
def LoadThomsonTopology(filename):
    """
        Load text data that was created using 
        http://thomson.phy.syr.edu/thomsonapplet.php
        
        Do "Data Quick Search" with "Global Minima" for the number
        of particles A that you want.
        
        Returns: List of cartesian coordinates of shape (A,3)
    """
    f = open(filename,"r")
    r = csv.reader(f, delimiter=' ')
    Number = r.next()
    one = r.next()
    Energy = r.next()
    coords = list()
    for line in r:
        if len(line) != 0:
            line[:] = [p for p in line if p != '']
            row = list()
            for item in line:
                row.append(eval(item))
            coords.append(row)
    res = np.array(coords)
    return res



Subdir = "sphere_data/"
#Subdir = "sphere4/"
ODIR = "/home/paul/repos/paulm/programming/Mie/"
A=16

THOMSONFILE = ODIR + "Thomson/{}.txt".format(A)
DIR = ODIR+Subdir
OUTDIR = DIR
INTERPOLATION = False
PREBGCORR = False
backmeth = "born"

#angles = np.arange(A)/A*360.
# In 4PI we now have an equidistant distribution of projections along
# the unit sphere.
# This data we import from files that we obtained from
# http://thomson.phy.syr.edu/thomsonapplet.php
print("Using Thomson file: {}".format(THOMSONFILE))
angles = LoadThomsonTopology(THOMSONFILE)
A = len(angles)

## Find file
print "Working directory: {}".format(DIR)
files = os.listdir(DIR)
fname = None
for f in files:
    if f[-9:] == "_Ephi.dat" and (f[:3] == "Ez_" or f[:2] == "E_"):
        fname=f
if fname is None:
    raise ValueError("No matching .dat file found.")
else:
    print "Using stem file: {}".format(fname)
    
### Get data
info = io_GetInfoZhu(fname)
Ex,BEx = io_GetDatFields2D(DIR,fname)

# Add some information
info["wavelength [nm]"] = 500
w2px = info["wavelength [nm]"] / 1e3
nmed = info["refractive index of medium"]
info["effective pixel size [um]"] = info["effective pixel size [wavelengths]"] * w2px
info["axial measurement position [um]"] = info["axial measurement position [wavelengths]"] * w2px / nmed

#measurement = dict()
#measurement["u field sinogram"] = u
#measurement["u0 field"] = u0
#measurement["acquisition angles"] = angles

setup = dict()
setup["refractive index of medium"] = info["refractive index of medium"]
setup["wavelength [nm]"] = info["wavelength [nm]"]
setup["effective pixel size [um]"] = info["effective pixel size [um]"]
setup["axial measurement position [um]"] = info["axial measurement position [um]"]

methods = dict()
methods["approximation type"] = backmeth
methods["postprocessing"] = None

system = dict()
system["copy arrays"] = True # overwrites arrays as they come in


# Until here everything should be the same as in Zhu_backpropagate.py
phase = np.angle(Ex)
Bphase = np.angle(BEx)

Int = Ex.real**2 + Ex.imag**2
BInt = BEx.real**2 + BEx.imag**2

myfig, axes = plt.subplots(2,2)
axes[0][0].imshow(Bphase, aspect="equal")
axes[0][1].imshow(phase, aspect="equal")
axes[1][0].imshow(BInt, aspect="equal")
axes[1][1].imshow(Int, aspect="equal")

plt.savefig(os.path.join(DIR,fname+".png"))


IPython.embed()
