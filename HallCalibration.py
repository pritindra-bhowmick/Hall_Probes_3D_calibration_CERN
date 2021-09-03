#importing packages
import numpy as np
import scipy as sp
from math import *
from scipy.special import sph_harm
import csv

# *******************************************************************************************
#       BASIC FUNCTIONS 
# *******************************************************************************************

# Roots and weights of legendre polynomials for gauss quadrature
wi=[[0.6521451548625461, 0.3478548451374539], [0.3607615730481386, 0.4679139345726910, 0.1713244923791703], 
    [0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763], 
    [ 0.2955242247147529, 0.2692667193099964, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881], 
    [ 0.2491470458134028, 0.2334925365383548, 0.2031674267230659, 0.1600783285433462, 0.1069393259953184,  0.0471753363865118]]
xi=[[0.3399810435848563, 0.8611363115940526], [0.6612093864662645, 0.2386191860831969, 0.9324695142031520], 
    [0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975362], 
    [ 0.1488743389816312, 0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717], 
    [ 0.1252334085114689, 0.3678314989981802, 0.5873179542866174, 0.7699026741943047, 0.9041172563704749,  0.9815606342467193]]

#Azimuthal and Polar angles
def theta(x): return 6*x
def phi(x):   return 5*x+pi/12

# Sign function to find the sign of sin^m(theta)
def mysign(x,m):
    if x==0: return 1
    if m%2==0: return 1
    return np.sign(sin(theta(x)))

# Rotation matrices
def Rot(x): return np.array([[cos(phi(x))*cos(theta(x)),-sin(phi(x)),sin(theta(x))*cos(phi(x))],
                             [sin(phi(x))*cos(theta(x)),cos(phi(x)),sin(theta(x))*sin(phi(x))],
                             [-sin(theta(x)),0,cos(theta(x))]])

# Lagrange's interpolation method for even functions 
def lagrange_even(x,y,xx):
    n = len(x)
    sum = 0
    for i in range(n):
        product = y[i]
        for j in range(n):
            if i!=j:
                product *= (xx**2-x[j]**2)/(x[i]**2-x[j]**2)
        sum += product
    return sum

# Simpson's 1/3 integration algorithm
def Integrate(y,ba):
    n=len(y)
    integral=2*y[0]
    for i in range(1,n):
        integral += (2+2*(i%2))*y[i]
    return (ba/3)*integral/n

# function that gives azimuthal and polar angles for a given vector
def getAngle(x):
    normvec = np.linalg.norm(x)
    if abs(normvec-abs(x[2]))<1.e-7: return 0,0
    thvec = np.arccos(x[2]/normvec)
    cosphi = x[0]/sqrt(x[0]**2+x[1]**2)
    sinphi = x[1]/sqrt(x[0]**2+x[1]**2)
    if cosphi==0 and sinphi>0: return thvec.real, pi/2
    elif cosphi==0 and sinphi<0: return thvec.real, 3*pi/2
    elif sinphi<0: return thvec.real,(2*pi-np.arccos(cosphi)).real
    else : return thvec.real,(np.arccos(cosphi)).real
