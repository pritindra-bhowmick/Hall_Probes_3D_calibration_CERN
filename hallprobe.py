#1
#importing packages

import numpy as np
import scipy as sp
from math import *
from scipy.special import sph_harm
import csv

#############################################################################

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

#Lagrange's interpolation method for even functions 
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
def Integrate(y,x0):
    n=len(y)
    integral=2*y[0]
    for i in range(1,n):
        integral += (2+2*(i%2))*y[i]
    return (x0/3)*integral/n

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
    
    
############################################################################

def scan_3D(filename):
    
    # ------------------------------------------------------------------------------
    # 1 : Scanning volatages from file along with other parameters
    # ------------------------------------------------------------------------------
    
    V_list = []

    with open(filename+'Voltages.csv', 'r') as file:
        read = csv.reader(file)
        i=0
        for row in read:

            if i==0: # scanning type of magnetic field
                type1 = str(row[0])
            if i==1: # number of points on the parametrisation and hall current
                iter = int(row[0])
                I = float(row[1])
            if i==2: # values of magnetic field
                B0_list = [float(r) for r in row[:-1]]
            if i==3: # values of temperature
                T0 = float(row[0])
                T1 = float(row[1])
            if i==4: # order = number of B fields
                order = int(row[0])
                B_max = float(row[1])
            
            i=i+1
            
            # Scanning the voltages 
            if i>5: 
                if (i-6)%(iter+1)==0 : continue
                V_list.append(float(row[0]))
                V_list.append(float(row[1]))
                V_list.append(float(row[2]))

            
        
    # Reshaping voltage values to a desirable form 
    V_list = np.array(V_list).reshape(order+1,iter,3)
    Vlist=np.zeros([order+1,3,iter])
    for i in range(order+1):
        Vlist[i] = V_list[i].transpose()

    del V_list
    
    # ------------------------------------------------------------------------------
    # 2 : Opening output files    
    # ------------------------------------------------------------------------------
    
    sourceFile = open(filename+'Parameters.csv', 'w')
    
    if type1 == 'A':
        print('A',file = sourceFile)
        # Printing the magnetic field values 
        for B0 in B0_list: print(str(B0)+',',file = sourceFile,end = '')
        print('',file = sourceFile)
    
    if type1 == 'B':
        print('B\n'+str(B_max)+','+str(order),file = sourceFile)

    # ------------------------------------------------------------------------------
    # 3 : Setting up angle and spherical harmonic arrays
    # ------------------------------------------------------------------------------
    
    # angle values and sin(angle)
    angle = np.linspace(0,2*pi,iter,endpoint=False)   # values of t
    sinthetalist = np.array([abs(sin(theta(x))) for x in angle]) #values of sin(t)
    
    #lim = l_max : for spherical harmonics
    lim=10

    Ylmlist=[]
    
    # NB: python calculates spherical harmonic function as a function of just cos of the angle and 
    # sin(theta) = sqrt(1-cos^2(theta)) is used for sine components. This does not preserve the sign
    # of sin(theta). Hence we are introduncing this mysign function which gives sign of sin^m(x)

    for l in range(lim+1):
        for m in range(-l,l+1):
            Ylmlist.append(np.array([sph_harm(m, l, phi(x),theta(x))*mysign(x,m) for x in angle]))

    # Arrays with the values of spherical harmonics
    def Ylmarray(l,m): return Ylmlist[l*l+l+m]
    
    # ------------------------------------------------------------------------------
    # 4 : Treating the mixing of spherical harmonics
    # ------------------------------------------------------------------------------
    
    # matrix with mixing of Ylm functions
    sph_orth_coeff = np.zeros([lim**2,lim**2])+1.j*np.zeros([lim**2,lim**2])
    
    def checking(l0,m0):
        for l in range(l0+1):
            for m in range(-l,l+1):
                if l==0 and l0==11: continue
                inte = Integrate(sinthetalist*Ylmarray(l0,m0)*Ylmarray(l,-m),2*pi)*pi
                if abs(inte) > 1e-10:
                    sph_orth_coeff[l*l+l+m][l0*l0+l0+m0]=inte
                    sph_orth_coeff[l0*l0+l0+m0][l*l+l+m]=np.conj(inte)
    
    # Filling the matrix with mixing of Ylm functions
    for l0 in range(lim):
        for m0 in range(-l0,l0+1):
            checking(l0,m0)
            
    # Inverse of matrix with mixing of Ylm functions
    inv_sph_orth_coeff = np.linalg.inv(sph_orth_coeff)
    
    # The values of l scanning is done
    # l=0 -> Noise
    # l=1 -> Hall effect
    # l=2 -> Planar hall effect
    # l=odd -> non-linearity
    l_list = [0,1,2,3,5,7,9]
    
    # printing l values on a file
    for l in l_list: print(str(l)+',',file = sourceFile,end ='')
    print('',file = sourceFile)
    
    # ------------------------------------------------------------------------------
    # 5 : Defining 3D scan functions for one particular magnetic field
    # ------------------------------------------------------------------------------
    
    # Function that gives c_lm values for l=1,2
    # Takes input as an array of volates
    def SphericalHarmonicCoeff00(V_array):

        clm=[]
        
        # loop for finding c_lm values
        for l in range(lim):
            for m in range(-l,l+1):
                inte = Integrate(sinthetalist*V_array*Ylmarray(l,-m)*pi,2*pi)
                clm.append(inte)
        
        # correcting for mixing of spherical harmonics
        clm = np.matmul(inv_sph_orth_coeff,clm)
        return clm[1:9]

    # Funtion that gives c_l = sqrt(sum_m(c_lm^2)) values
    # Takes input as an array of volates
    def SphericalHarmonicCoeffmag(V_array):

        clm=[]
        
        # loop for finding c_lm values
        for l in range(lim):
            for m in range(-l,l+1):
                inte = Integrate(sinthetalist*V_array*Ylmarray(l,-m)*pi,2*pi)
                clm.append(inte)
        
        # correcting for mixing of spherical harmonics
        clm = np.matmul(inv_sph_orth_coeff,clm)
        
        # calculating c_l = sqrt(sum_m(c_lm^2))
        clm_mag = []
        for l in range(0,lim): clm_mag.append(np.sqrt(np.sum(np.square(np.abs(clm[(l)**2:(l+1)**2])))))
        return(clm_mag)
    
    #############################################################################
    
    def SolveforJ(n0,xy,yz,zx):

        i_max = list(n0).index(max(n0))
        nx,ny,nz = n0
    #     print(xy,yz,zx)

    #     print(i_max)
        def func(jj): 
            jx,jy,jz = jj
            return np.array([-jx*jz*nx+jy*jz*ny+jx**2 *nz-jy**2 *nz - xy, jy**2 *nx- jz**2 *nx-jx*jy*ny+jx*jz*nz -yz, jx*jy*nx-jx**2 *ny+jz**2 *ny-jy*jz*nz-zx])
        def dfunc(jx,jy,nz):return np.linalg.inv(np.array([[-(jz*nx) + 2*jx*nz,jz*ny - 2*jy*nz,-(jx*nx) + jy*ny],[-(jy*ny) + jz*nz,2*jy*nx - jx*ny,-2*jz*nx + jx*nz],[jy*nx - 2*jx*ny,jx*nx - jz*nz,2*jz*ny - jy*nz]]).transpose())        

        jj1 = np.array([nz,nz,-(nx+ny)])/np.linalg.norm([nz,nz,-(nx+ny)])
        jj2 = np.cross(n0,jj1)
    #     print(jj1,jj2)
    #     print(func(jj1),func(jj2))

        def funcbi(a,i_max):
            jj = cos(a)*jj1+sin(a)*jj2
            fff = func(jj)
    #         print(np.linalg.norm(jj),jj,fff)
            return fff[i_max]

        xi = func(jj1)*func(jj2)
    #  
        for i in range(6):

            i_max = i//2
            xl = 0+(i%2)*pi/2
            xu = pi/2+(i%4)*pi/2
            fu = funcbi(xu,i_max)
            fl = funcbi(xl,i_max)
    #         print(fu,fl,fu*fl)
            if fu*fl<0:
                xm = (xl+xu)/2

                for iiiii in range(100):

                    fu = funcbi(xu,i_max)
                    fl = funcbi(xl,i_max)
                    fm = funcbi(xm,i_max)
    #                 print(iiiii,xu,xl,xm,fu,fl,fm)
                    if fu*fm<=0: xl = xm
                    if fl*fm<=0: xu = xm
                    if (abs(xu-xl)/xm<1.e-12) and abs(fm)<1.e-12:break
                    xm = (xl+xu)/2

                jj = cos(xm)*jj1+sin(xm)*jj2
                if np.linalg.norm(func(jj))<1.e-3: return jj

        print('ERROR ERROR')
        return [0,0,0]
    
    nMat = []
    jMat = []

    for j in range(3):
#         print('----------------------------------------------------------------------------\n',j+1)
        clm = SphericalHarmonicCoeff00(Vlist[0][j])
        nz = clm[1].real
        nx = - sqrt(2) * clm[2].real
        ny = sqrt(2) * clm[2].imag
        n = np.array([nx,ny,nz])/np.linalg.norm([nx,ny,nz])
        l2 = clm[3:8]
        l2 = sqrt(2) *np.array(l2)/np.linalg.norm(l2)
        jfout = SolveforJ(n,l2[0].imag,l2[1].imag,l2[1].real)
        print('n,j = ',n,jfout)
        nMat.append(n)
        jMat.append(list(jfout))

    nMat = np.array(nMat)/np.linalg.norm(nMat[0])
    jMat = np.array(jMat)/np.linalg.norm(jMat[0])

    for ii in range(3):
        for jj in range(3):
            print(str(nMat[ii][jj])+',', file = sourceFile,end='')
        for jj in range(3):
            print(str(jMat[ii][jj])+',', file = sourceFile,end='')
        print('',file = sourceFile)
            
    ###########################################################################
    
    B0 = B0_list[-1]

    Ra = SphericalHarmonicCoeffmag(Vlist[-1][2])[1]/I/B0/2.046653415892977
    Rb = SphericalHarmonicCoeffmag(Vlist[-2][2])[1]/I/B0/2.046653415892977
    R1 = (Rb-Ra)/(T1-T0)
    R0 = 0.5*(Ra+Rb-R1*(T1+T0))
    print(str(R0)+','+str(R1)+','+str(T0)+','+str(Ra*I),file = sourceFile)
    
    ###########################################################################
    
    clm=[[],[],[]]

    for j in range(3):

        for l in range(lim):
            for m in range(-l,l+1):
                inte = Integrate(sinthetalist*Vlist[-2][j]*Ylmarray(l,-m)*pi,2*pi)
                clm[j].append(inte)

        clm[j] = list(np.matmul(inv_sph_orth_coeff,clm[j]))

    for l in range(l_list[-1]+1):
#         print('\nl = '+str(l)+'\n', file = sourceFile1)
        div0 = np.linalg.norm(clm[0][l**2:(l+1)**2])
        div1 = np.linalg.norm(clm[1][l**2:(l+1)**2])
        div2 = np.linalg.norm(clm[2][l**2:(l+1)**2])
        for m in range(-l,l+1):
            string = str(clm[0][l*l+l+m]/div0)+','+str(clm[1][l*l+l+m]/div1)+','+str(clm[2][l*l+l+m]/div2)
            clm[0][l*l+l+m] /= div0
            clm[1][l*l+l+m] /= div1
            clm[2][l*l+l+m] /= div2
#             print(string, file = sourceFile1)
            print(string, file = sourceFile)
            
    #########################################################################

    xarray = np.array(B0_list)/B_max
    warray = np.array(wi[order-2])
    legend = np.array([sp.special.legendre(n)(xarray) for n in range(2*order)])
    B0 = np.array([0.,0.,B_max])
    B0mag = np.linalg.norm(B0)
    Blist = [[0.,0.,B] for B in B0_list]

    Vmodelcoeff = []

    integrand = [[] for i in l_list]
    xarrayint = [[] for i in l_list]

    for j in range(len(Blist)):
        print(Blist[j][2])
        clm_mag = SphericalHarmonicCoeffmag(Vlist[j][0])
        for i in range(len(l_list)): 
            l = l_list[i]
            if abs(clm_mag[l])>1.e-8: 
                integrand[i].append(abs(clm_mag[l]))
                xarrayint[i].append(Blist[j][2]/B0[2])

#     print(integrand)
#     print(xarrayint)



    for i in range(len(l_list)): 

        l = l_list[i]

        integrand1 = np.array(integrand[i])/(np.abs(np.array(xarrayint[i])*B0mag)**abs(l))

        r = order-len(integrand1)
        if r!=0:integrand1 = np.concatenate((np.array([lagrange_even(xarrayint[i],integrand1,x1) for x1 in xarray[:r]]),integrand1))

        if type1 == 'A':
#             print(B0_list)
#             print(integrand1)
            for inte in integrand1:    
                print(str(inte)+',',file = sourceFile, end='')
                print(str(inte)+',', end='')
        
        if type1 == 'B':
            for k in range(0,2*order,2):    
                inte = (2*k+1)*np.sum(integrand1*warray*legend[k])
                print(str(inte)+',',file = sourceFile, end='')
                print(str(inte)+',', end='')
#                 Vmodelcoeff.append([l,k,a_lk])
        
        print("",file = sourceFile)


################################################################################

def findB(filename,Vgiven,T):

    clm=[[],[],[]]
    alk=[]
    err = 1.e-9

    with open(filename+'Parameters.csv', 'r') as file:
        read = csv.reader(file)
        i=0
        for row in read:

            if i==0: type1 = str(row[0])
            if i==1: 
                if type1 == 'A':
                    B0_list = [float(r) for r in row[:-1]]
                    order = len(B0_list)
                if type1 == 'B':
                    B_max = float(row[0])
                    order = int(row[1])
                for jj in range(order): alk.append([])
            if i==2:
                l_list = [int(l) for l in row[:-1]]
                l_m = (l_list[-1]+1)**2
                l_l = len(l_list)
            if i==6: 
                R0 = float(row[0])
                R1 = float(row[1])
                T0 = float(row[2])
                div = float(row[3])
            if i >= 7: 
                if i < l_m+7:
                    for jj in range(3):
                        clm[jj].append(complex(row[jj]))
                if i>=l_m+7 and i<l_m+l_l+7:
                    for jj in range(order):
                        alk[jj].append(float(row[jj]))

            i = i+1

    clm = np.array(clm)
    alk = np.array(alk)
    if type1 =='A': alk = alk.transpose()

    def V_out_model(B,n):

        th, ph = getAngle(B)
        Bmag = np.linalg.norm(B)
        Vmodel = 0

        if type1=='A':
            for i in range(l_l):
                l = l_list[i]
                p1= lagrange_even(B0_list,alk[i],Bmag)
                p2=0
                for m in range(-l,l+1):
                    p2 += clm[n][l*l+l+m]*sph_harm(m, l, ph,th)
                Vmodel += Bmag**l*p1*p2

        if type1=='B':
            leg_array = np.array([sp.special.legendre(2*k)(Bmag/B_max) for k in range(order)])
            for i in range(l_l):
                l = l_list[i]
                p1=0
                p2=0
                for k in range(0,order):
                    if abs(alk[k][i])>err : p1+=alk[k][i]* leg_array[k]
                for m in range(-l,l+1):
                    p2 += clm[n][l*l+l+m]*sph_harm(m, l, ph,th)
                Vmodel += Bmag**l*p1*p2

        return Vmodel.real * (T*R1+R0)/(T0*R1+R0)


    dt=0.001000
    V_from_dev = np.array(Vgiven)/div

    def VMat(X1): return np.array([V_out_model(X1,i)/div - V_from_dev[i] for i in range(3)])

    X0 = V_from_dev
    x0,y0,z0 = X0

    def Jac(x,y,z,VMatxyz):  return np.array([(VMat([x+dt,y,z])-VMatxyz),(VMat([x,y+dt,z])-VMatxyz),(VMat([x,y,z+dt])-VMatxyz)]).transpose()/dt

    def Jac_Inv(x,y,z,VMatxyz): return np.linalg.inv(Jac(x,y,z,VMatxyz))

    for i in range(20):
        V = VMat(X0)
        jac = Jac(x0,y0,z0,V)
        sol = X0 - np.matmul(V,np.linalg.inv(jac))
#         print(X0)
        if (np.linalg.norm(X0-sol)/np.linalg.norm(sol)<1.e-9 and np.linalg.norm(V)<1.e-9): break
        X0 = sol
        x0,y0,z0 = X0

    return X0
                
            
                
