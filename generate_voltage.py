# Work done by Pritindra Bhowmick, EP-DT-DT, CERN during Summer Studentship 2021 at CERN.
# Project name: High precision calibration of 3D hall probes for magnet mapping apparatuses.
# Find me at pritindra2001@gmail.com, pritindra.bhowmick@cern.ch
# Find the project report at CERN CDS url coming soon
# History : Version 1: 04-09-2021


# ------------------------------------------------------------------------------
# 1 : importing packages
# ------------------------------------------------------------------------------

import numpy as np
from datetime import datetime

# ------------------------------------------------------------------------------
# 2 : Setting up angles and rotation arrays and matrices
# ------------------------------------------------------------------------------

#Number of points on the surface 
iter = 240

#Azimuthal and Polar angles
def theta(x): return 6*x
def phi(x):   return 5*x+np.pi/12

#Rotation matrix
def Rot(x): return np.array([[np.cos(phi(x))*np.cos(theta(x)),-np.sin(phi(x)),np.sin(theta(x))*np.cos(phi(x))],
                             [np.sin(phi(x))*np.cos(theta(x)),np.cos(phi(x)),np.sin(theta(x))*np.sin(phi(x))],
                             [-np.sin(theta(x)),0,np.cos(theta(x))]])

#angle array
angle = np.linspace(0,2*np.pi,iter,endpoint=False)   # values of angle t

# ------------------------------------------------------------------------------
# 3 : Making a model for hall voltage
# ------------------------------------------------------------------------------

I=1*10**(-2)   #current, we know this
R=200          #hall coefficient
G=1            #planar hall coefficient
tau = 4        #Non linearity parameter
mT = -0.006    #Temperature dependence

#Hall voltage
def V_hall(n,J,B,T): return  2*tau*(1-np.exp(-R*I*np.dot(n,B)/tau))/(1+np.exp(-R*I*np.dot(n,B)/tau)) 

#Planar Hall Voltage
def V_phe(n,J,B) : return 2*G*I * np.dot(J,B) * np.dot(np.cross(n,J),B) 

#output voltage from the hall probe
def V_out(n,J,B,T): return (1+np.sin(mT*T))*(V_hall(n,J,B,T)+V_phe(n,J,B))

# ------------------------------------------------------------------------------
# 4 : Introducing non-orthogonality
# ------------------------------------------------------------------------------

# Arbitrary angles of rotation
alpha = np.array([np.pi/51,np.pi/49,np.pi/55])
beta  = np.array([np.pi/44,np.pi/57,np.pi/55])
gamma = np.array([np.pi/45,np.pi/47,np.pi/53])

Rott = [np.array([[np.cos(alpha[i])*np.cos(beta[i]) , np.cos(alpha[i])*np.sin(beta[i])*np.sin(gamma[i]) - np.sin(alpha[i])*np.cos(gamma[i]) , np.cos(alpha[i])*np.sin(beta[i])*np.cos(gamma[i]) + np.sin(alpha[i])*np.sin(gamma[i])], 
                [np.sin(alpha[i])*np.cos(beta[i]) , np.sin(alpha[i])*np.sin(beta[i])*np.sin(gamma[i]) + np.cos(alpha[i])*np.cos(gamma[i]) , np.sin(alpha[i])*np.sin(beta[i])*np.cos(gamma[i]) - np.cos(alpha[i])*np.sin(gamma[i])],  
                [-np.sin(beta[i]) , np.cos(beta[i])*np.sin(gamma[i]) , np.cos(beta[i])*np.cos(gamma[i])]]) for i in range(3)]

# n and j values for 3 hall probes
n0 = [np.matmul([1,0,0],Rott[0]),np.matmul([0,1,0],Rott[1]),np.matmul([0,0,1],Rott[2])]
j0 = [np.matmul([0,1,0],Rott[0]),np.matmul([0,0,1],Rott[1]),np.matmul([1,0,0],Rott[2])]

# ------------------------------------------------------------------------------
# 5 : Setting up magnetic field values for 3D scan
# ------------------------------------------------------------------------------

xi=[[0.3399810435848563, 0.8611363115940526], [0.6612093864662645, 0.2386191860831969, 0.9324695142031520], 
    [0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975362], 
    [ 0.1488743389816312, 0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717], 
    [ 0.1252334085114689, 0.3678314989981802, 0.5873179542866174, 0.7699026741943047, 0.9041172563704749,  0.9815606342467193]]

# If type1 =='A' linear magnetic field scaling
# If type1 =='B' Legendre Roots fields
type1 = 'A'

if type1 =='A': 
    # List of magnetic field values to scan
    B0_list = [0.5,1.5,2.5]
    order=len(B0_list)
    B_max = B0_list[-1]

if type1 =='B':
    # number of Roots of Legendre polynomials in positive x axis 
    order = 5
    # Maximum magnetic field to be scaled
    B_max = 2.5
    B0_list = B_max * np.array(xi[order-2])
    
# B0_list os the listof magnetic fields used for 3D scanning
B0_list = np.array([[0,0,B0] for B0 in B0_list])

# Two temperatures for scanning
T0 = 20
T1 = 25

# ------------------------------------------------------------------------------
# 6 : Wrting output voltages on a file for reading
# ------------------------------------------------------------------------------

filename = 'hallprobe2'

sourceFile = open(filename+'Voltages.csv', 'w')

print(type1+'\n'+str(iter)+','+str(I),file = sourceFile)
for B0 in B0_list:
    print(str(B0[2])+',',file = sourceFile,end = '')
print('\n'+str(T0)+','+str(T1),file = sourceFile)
print(str(order)+','+str(B_max),file = sourceFile)

for B0 in B0_list:
    print('#  B = '+str(B0[2])+' \t T = '+str(T0)+' \t Date and time '+str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")), file = sourceFile)
    for t in angle:
        Rr=Rot(t)
        n1,n2,n3 = np.matmul(n0,Rr)
        j1,j2,j3 = np.matmul(j0,Rr)
        string = str(V_out(n1,j1,B0,T0))+','+str(V_out(n2,j2,B0,T0))+','+str(V_out(n3,j3,B0,T0))
        print(string, file = sourceFile)

print('#  B = '+str(B0[2])+' \t T = '+str(T1)+' \t Date and time '+str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")), file = sourceFile)
for t in angle:
    Rr=Rot(t)
    n1,n2,n3 = np.matmul(n0,Rr)
    j1,j2,j3 = np.matmul(j0,Rr)
    string = str(V_out(n1,j1,B0,T1))+','+str(V_out(n2,j2,B0,T1))+','+str(V_out(n3,j3,B0,T1))
    print(string, file = sourceFile)
