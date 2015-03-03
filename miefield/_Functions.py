#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module mie

Analytical solution (series) for scattering on spheres and cylinders

"""
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import special

__all__ = ["besselh", "besselj", "besselh_derivative", 
           "besselj_derivative", "cart2pol", "getN_max",
           "hypot", "ric_besselh", "ric_besselh_derivative",
           "ric_besselj", "ric_besselj_derivative", "legendre",
           "ric_bessely", "ric_bessely_derivative", "kzlegendre",
           "norm",
           "cart2sph"]


def besselh(n, k, z):
    """ Hankel function
        n - real order
        k - kind (1,2)
        z - complex argument
    """
    if k == 2:
        return special.hankel2(n,z)
    elif 0 == 1:
        return special.hankel1(n,z)
    else:
        raise TypeError("Only second or first kind.")


def besselj(n, z):
    """ Bessel function of first kind
        n - real order
        z - complex argument
    """
    return special.jv(n,z)


def besselh_derivative(n, k, z):
    """ Derivative of besselh
    """
    if k == 2:
        return special.h2vp(n,z)
    elif 0 == 1:
        return special.h1vp(n,z)
    else:
        raise TypeError("Only second or first kind.")
    
        
def besselj_derivative(n, z):
    """ Derivative of besselj
    """
    return special.jvp(n,z)
    

def cart2pol(x,y):
    """ Convert cartesian to polar coordinates 
        X,Y input array arrays
        Output: theta, rho
    """
    rho = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
    return theta, rho


def cart2sph(x,y,z):
    """ Convert cartesian to spherical coordinates 
        x,y,z input array arrays
        Output: r, theta, phi
         - r      in [0 \infty)
         - theta  in [0 pi]
         - phi    in [0 2*pi)
    """

    for elem in range (0,np.size(x)):
        if x[elem]==0 and y[elem]==0 and z[elem]==0:
            r = 0
            theta = 0
            phi = 0
        else:
            hypotxy = hypot(x,y)
            r       = hypot(hypotxy,z)
            theta   = np.arccos(z/r)
            phi     = 0;
            #print(r)
            if x[elem]==0 and y[elem]==0:
                phi = 0
            elif x[elem] >= 0 and y[elem] >= 0:
                phi = np.arcsin(y/hypotxy)
            elif x[elem] <= 0 and y[elem] >= 0:
                phi = np.pi - np.arcsin(y/hypotxy)
            elif x[elem] <= 0 and y[elem] <= 0:
                phi = np.pi - np.arcsin(y/hypotxy)
            elif x[elem] >= 0 and y[elem] <= 0:
                phi = 2*np.pi + np.arcsin(y/hypotxy)
    return [r, theta, phi]


def getN_max(radius, sphere, background, frequency):
    """ 
        Determine the number of Bessel functions to be evaluated.
        [Wiscombe1980]
        
        Input:
        *radius*      radius of the cylinder or sphere *array*
        *sphere*      object of *DielectricMaterial*, Debye material
        *background*  object of *DielectricMaterial*, Debye material
        *frequency*   frequency of the wave *float*

        Output:
        *N_max*       number of Bessel functions, *array* same shape as *radius*
    """
    k_m = np.conj(background.getElectromagneticWaveNumber(frequency))
    x = np.atleast_1d(k_m * radius)
    x = np.abs(x)
    # x may be complex the background is lossy
    N_m = np.conj(background.getComplexRefractiveIndex(frequency))
    #m = np.zeros(max(sphere.shape))
    # Need to take the conjugate of the refractive index since I use
    # exp(j\omega t) as the time-harmonic factor while [Yang2003]
    # uses exp(-j\omega t).
    m = np.conj(sphere.getComplexRefractiveIndex(frequency)) / N_m
    #idx = np.flatnonzero(0.02 <= x[:, (end -1)] & x[:, (end -1)] < 8)
    #N_stop[(idx -1)] = round_(x[(idx -1), (end -1)] + 4 * x[(idx -1), (end -1)] ** (1 / 3) + 1)
    #idx = np.flatnonzero(8.0 <= x[:, (end -1)] & x[:, (end -1)] < 4200)
    #N_stop[(idx -1)] = round_(x[(idx -1), (end -1)] + 4.05 * x[(idx -1), (end -1)] ** (1 / 3) + 2)
    #idx = np.flatnonzero(4200 <= x[:, (end -1)] & x[:, (end -1)] < 20000)
    #N_stop[(idx -1)] = round_(x[(idx -1), (end -1)] + 4 * x[(idx -1), (end -1)] ** (1 / 3) + 2)
    #N_max = np.max(np.array([N_stop, round_(abs(m.dot(x))), round_(abs(m[:, (2 -1):end].dot(x[:, (1 -1):end - 1])))]).reshape(1, -1), np.array([]), 2) + 15

    N_stop = np.ones(x.shape)
    idsmall = np.where((0.02 <= x) * (x < 8))
    idmiddle = np.where((8 <= x) * (x < 4200))
    idlarge = np.where((4200 <= x) * (x < 20000))

    N_stop[idsmall] = np.ceil(x[idsmall] + 4 * x[idsmall]**(1/3) + 1)
    N_stop[idmiddle] = np.ceil(x[idmiddle] + 4.05 * x[idmiddle]**(1/3) + 2)
    N_stop[idlarge] = np.ceil(x[idlarge] + 4 * x[idlarge]**(1/3) + 2)

    N_max = np.max( np.array([N_stop, np.ceil(np.abs(m*x))]), axis=0) + 15
    
   
    #if 0.02 <= x and x < 8:
    #    N_stop = np.ceil(x + 4 * x**(1/3) + 1)
    #elif 8 <= x and x < 4200:
    #    N_stop = np.ceil(x + 4.05 * x ** (1/3) + 2)
    #elif 4200 <= x and x < 20000:
    #    N_stop = np.ceil(x + 4 * x**(1/3) + 2)
    #N_max = np.max(np.array([N_stop, np.ceil(np.abs(m*x)), 2])) + 15

    return N_max


def hypot(x,y):
    " Square root of sum of squares "
    return np.sqrt(x**2 + y**2)


def ric_besselj(nu,x):
    """
    J = ric_besselj(nu, x) implements the Riccati-Bessel functions.
    
    J_{nu}(x) = \sqrt{\frac{\pi x}{2}} J_{nu+1/2}(x)
    
    where
    nu  order of the Bessel's function. Must be a column vector.
    x   must  be a row complex vector
    """
    x  = x.reshape(1, -1)
    nu = nu.reshape(-1 ,1)

    a = besselj(nu+0.5,x)
    
    ##if (len(x) == 1):
    ##    a = a.reshape(-1, 1)
    ##elif (len(nu) == 1):
    ##    a = a.reshape(1, -1)
    ##else
    ##    a = a.transpose()

    J = np.dot( np.sqrt(np.pi/2.*(np.ones((len(nu),1))*x)), a.transpose() )
    return J

    ## We could also use the scipy function
    ## But there is a problem: x is can only be a scalar.
    #return special.riccati_jn(n[-1], x)


def ric_besselh(nu,K,x):
    """
    H = ric_besselh(nu, K, x) implements the Riccati-Bessel function, which is
    defined as
    
    H_{nu}(x) = \sqrt{\frac{\pi x}{2}} H_{nu+1/2}(x)
    
    where
    nu  order of the spherical Bessel's function. Must be a column vector.
    K   1 for Hankel's function of the first kind; 2 for Hankel's
        function of the second kind.
    x   Must be a row vector.
    """

    if (K != 1 and K != 2):
        raise Exception('Improper kind of Hankel function')
    if (K==1):
        H = ric_besselj(nu,x) + 1j*ric_bessely(nu,x)
    else:
        H = ric_besselj(nu,x) - 1j*ric_bessely(nu,x)

    return H


def ric_bessely(nu,x):
    """
    Y = ric_bessely(nu, x) implements the Riccati-Neumann's functions
    
    Y_{nu}(x) = \sqrt{\frac{\pi x}{2}} Y_{nu+1/2}(x)
    
    where
    nu  order of the Bessel's function. Must be a column vector.
    x   must  be a row complex vector
    """
    x  = x.reshape(1, -1)
    nu = nu.reshape(-1 ,1)

    a = besselj(nu+0.5,x)
    
    ##if (len(x) == 1):
    ##    a = a.reshape(-1, 1)
    ##elif (len(nu) == 1):
    ##    a = a.reshape(1, -1)
    ##else
    ##    a = a.transpose()

    Y = np.sqrt(np.pi/2.*(np.ones((len(nu),1))*x)) * a.transpose()

    if (np.sum(x==0) != 0):
        print('ric_bessely evaluated at x=0. Return -inf')

    temp2 = np.ones((len(nu),1))*x
    Y[np.where(temp2==0)] = -np.inf
    temp1 = nu*np.ones((1,len(x)))
    Y[np.where( (temp1==0) * (temp2==0) )] = -1

    return Y

    ## We could also use the scipy function
    ## But there is a problem: x is can only be a scalar.
    #return special.riccati_jn(n[-1], x)


def ric_besselh_derivative(nu, K, x, flag=1):
    """
    H = ric_besselh_derivative(nu, K, x) using the recursive relationship to
    calculate the first derivative of the Riccati-Bessel function
    
    
    H_{nu}(x) = \sqrt{\frac{\pi x}{2}} H_{nu+1/2}(x)
    
    nu         order of the riccati-Hankel's function. Must be a columne vector.
    K = 1      if it is Hankel's function of the first kind; K=2 if it is Hankel's function of the
                second kind. 
    x          Must be a row evector
    
    flag      1 for the first order derivative; 2 for the second order derivative
    
    """
    if (K!=1 and K!=2):
        raise Exception('Improper kind of Hankel function. K must be either 1 or 2.')
    #print(np.shape(x))
    if (K==1):
        H = ric_besselj_derivative(nu,x,flag) + 1j*ric_bessely_derivative(nu,x,flag)
    else:
        H = ric_besselj_derivative(nu,x,flag) - 1j*ric_bessely_derivative(nu,x,flag)
    return H


def ric_besselj_derivative(nu, x, flag=1):
    """
    J = ric_besselj_derivative(nu, x, flag) using the recursive relationship to
    calculate the first derivative of the Riccati-Bessel funtion.
    
    The Riccati-Bessel's function is defined as
    
    J_{nu}(x) = \sqrt{\frac{\pi x}{2}} J_{nu+1/2}(x)
    
    nu         order of the Riccati-Bessel's function.  Must be a column vector.
    x          Must be a row vector.
    flat       1, first order derivative order; 2, second order derivative
    """
    #print(np.shape(x))
    x    = x.reshape(1, -1)
    nu   = nu.reshape(-1 ,1)

    temp = np.ones((len(nu), 1))*x

    #print("1")
    #print(np.shape(ric_besselj(nu-1, x)))
    #print(np.shape(ric_besselj(nu,   x)))
    #print("2")
    #print(np.shape(np.dot(  ric_besselj(nu,   x), 1/temp)))
    #print(np.shape(1/temp))
    #print("3")
    #print(np.shape(ric_besselj(nu+1, x)           ))
    #print("4")
    
    if (flag == 1):
        J = 0.5*(           ric_besselj(nu-1, x)
                 + np.dot(  ric_besselj(nu,   x), 1/temp)
                 -          ric_besselj(nu+1, x)           )
    elif (flag ==2):
        J = 0.5*(         ric_besselj_derivative(nu-1, x)
                 + np.dot(ric_besselj_derivative(nu,   x),  1/temp    )
                 - np.dot(ric_besselj(nu, x             ),  temp**(-2))
                 -        ric_besselj_derivative(nu+1, x)               )
    else:
        raise Exception('This script only handles first and second derivative.')

    temp2 = np.ones((len(nu), 1))*x
    J[np.where(temp2==0)] = 0         # x = 0, all zeros
    temp1 = nu*np.ones((1, len(x)))
    if (flag ==1):
        J[np.where((temp1==0)*(temp2==0))] = 1
    return J


def ric_bessely_derivative(nu, x, flag=1):
    """
    Y = ric_bessely_derivative(nu, x, flag) using the recursive relationship to
    calculate the first derivative of the Riccati-Neumann's function.
    
    The Riccati-Neumann's function is defined as
    
    Y_{nu}(x) = \sqrt{\frac{\pi x}{2}} Y_{nu+1/2}(x)
    
    nu         order of the riccati Bessel's function.  Must be a column vector.
    x          Must be a row vector.
    flat       1, first order derivative order; 2, second order derivative
    """


    x    = x.reshape(1, -1)
    nu   = nu.reshape(-1 ,1)

    temp = np.ones((len(nu), 1))*x
    if (flag ==1):
        Y = 0.5*(              ric_bessely(nu-1, x)
                 + temp**(-1) *ric_bessely(nu,   x)
                 -             ric_bessely(nu+1, x)    )
    elif (flag ==2):
        Y = 0.5*(              ric_bessely_derivative(nu-1, x)
                 + temp**(-1) *ric_bessely_derivative(nu,   x)
                 - temp**(-2) *ric_bessely(nu, x             )
                 -             ric_bessely_derivative(nu+1, x)  )
    else:
        raise Exception('This script only handles first and second derivative.')

    temp2 = np.ones((len(nu),1))*x
    Y[np.where(temp2==0)] = -np.inf       # x = 0, all zeros
    temp1 = nu*np.ones((1, len(x)))
    if (flag ==1):
        Y[np.where((temp1==0)*(temp2==0))] = 1
    else:
        Y[np.where((temp1==0)*(temp2==0))] = -1

    return Y


def kzlegendre(n,m,x):
    """
    P = kzlegendre(n,m,x) computes the associated Legendre's function of
    degree n and order m. It is a wrapper function of MATLAB legendre(n,x) function.
    Note that Matlab use the definition of the associated Legendre's
    function with Condon-Shortly phase factor.
     
    n        an integer vector denoting the degree.
    m        an integer vector denoting the order. When m==0, we compute the
             legendre's polynomial.  m must statisfy that m <= n.
    x        contain real values in [-1, 1] and be a column vector.
    
    n and m cannot simultaneously be vectors.
    """
        
    x = x.reshape(1, -1)

    if (len(n) > 1 and len(m) > 1):
        raise Exception('Both n and m have length greater than ONE.'+
                        'Only one, either the order or the degree can'+
                        'be a vector.')

    if (len(n) == 1 and len(m) >= 1):
        if n > np.max(m):
            a = legendre(n,x)
            #P = a(m+1,:);
            P = a[m+1,:]
        else:
            P = np.zeros((len(m), len(x)))
            a = legendre(n,x)
            idx = np.where(m <= n)
            if np.sum(idx) != 0:
                #P = a(m(idx)+1,:);
                P = a[m[idx]+1,:]
    else:
        if m > np.max(m):
            P = np.zeros((len(n), len(x)))
        else:
            P = np.zeros((len(n), len(x)))
            for i in range(len(n)):
                P[i,:] = kzlegendre(n[i], m, x)
    
    return P


def legendre(n,x):
    """ This function behaves as the the matlab legendre function.
    
        The statement legendre(2,0:0.1:0.2) returns the matrix
                x = 0   x = 0.1  x = 0.2
        m = 0  -0.5000  -0.4850	 -0.4400
        m = 1   0       -0.2985  -0.5879
        m = 2   3.0000   2.9700  2.8800
    """
    x = np.atleast_1d(x)
    result = np.zeros((len(x),n+1))
    for i in range(len(x)):
        # Gives us row vector
        a = special.lpmn(n,n,x[i])[0].transpose()[-1]
        result[i] = a
    return result.transpose()
       
    
    
def norm(vec):
    """ returns the length of an N-dim vector """
    return np.sqrt(np.sum(np.array(vec)**2))
