# -*- coding: utf-8 -*-
'''
Created on 30 mars 2014

@author: Mohamed
'''

import sys
import numpy as np
import matplotlib.pylab as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import scipy.sparse.linalg as la
import math
from opencavity.utilsfunc import UtilsFunc

class Help(object):
    """Class containing functions that launch help/docs html files"""
    def __init__(self):
        print ("launching open_cavity documentation ... ")
        import webbrowser
        url=""
        webbrowser.open(url, new=1, autoraise=True)
    

class AmpMask2D(object):
    """Class containing definitions of amplitude masks as aperture and losses like absorbers in the cavity
    ::Args:
        - grid_x(float), grid_y(float):  Squared grid (matrix) vectors in which the shape of the mask is defined these tow vectors are important because usually in cavity eienvalues problem they don't follow a linear spacingbut Lgendre-Gauss spacing scheme (for the integral calculation). 
        
    .. Note::
        the unit of the dimensions are normalized to the wavelength's unit (grid_x=1000 means that it is =1000 the wavelength unit. 
    
    Example of use 
            >>> apert=solver.AmpMask2D(x1,y1) # create a mask object   
            >>> apert.add_circle(100)#create a circular aperture in x1,y1 coordinates with radius=100
            >>> apert.add_rectangle(3, 50) # add a rectangle 
            >>> apert.add_rectangle(50, 3)
    
    """
    def __init__(self,grid_x,grid_y): 
        """
       constructor
        """
        print("creating mask object...")
        self.Msk=np.array([])
        self.grid_x=grid_x
        self.grid_y=grid_y
        
    def add_circle(self,radius,x_center=0,y_center=0,positive=True):
        """Create a circular aperture function and add it to the mask object.
        
        ::Args: 
            - radius: the radius of the circular aperture
            - x_center, y_center: coordinates of the shape center, default values (0,0)
            - positive: is a boolean flag, default value =True, means the amplitude inside the shape ='1' and '0' outside
        
        .. Note::
        
        :Returns:
            - none.
        """
     
        if positive==True:
            amp1=1
            amp2=0
        else:
            amp1=0
            amp2=1
        
        Nx=np.size(self.grid_x)
        Ny=np.size(self.grid_y)
        Msk0=np.zeros((Nx,Ny))+np.zeros((Nx,Ny))*1j
        #self.Msk=np.zeros((Nx,Ny))+np.zeros((Nx,Ny))*1j
        for i in range(Nx):
            for j in range(Ny):
                if (self.grid_x[i]-x_center)**2+(self.grid_y[j]-y_center)**2<radius**2:
                    Msk0[i,j]=amp1
                else:
                    Msk0[i,j]=amp2
        #assigning the mask
        if self.Msk.size ==0:  
            self.Msk=Msk0  #if this the first mask we only assign it 
        else:
            self.Msk=self.Msk*Msk0  #otherwise we merge it with the existing mask 
        
        return 
    
    def add_rectangle(self,x_dim,y_dim,x_center=0,y_center=0,positive=True):
        """Create a rectangular aperture function and add it to the mask object.
        
        :Args: 
            - x_dim, y_dim: dimensions of the rectangle. 
            - x_center, y_center: coordinates of the shape center, default values (0,0)
            - positive: is a boolean flag, default value =True, means the amplitude inside the shape ='1' and '0' outside
        
        .. Note::
        
        :Returns:
            - none.
        """
        if positive==True:
            amp1=1
            amp2=0
        else:
            amp1=0
            amp2=1
        
        Nx=np.size(self.grid_x)
        Ny=np.size(self.grid_y)
        Msk0=np.zeros((Nx,Ny))+np.zeros((Nx,Ny))*1j
        #self.Msk=np.zeros((Nx,Ny))+np.zeros((Nx,Ny))*1j
        #self.Msk=np.zeros((Nx,Ny))

        for i in range(Nx):
            for j in range(Ny):
                #if grid_x[i]<x_dim  and grid_y[j]<y_dim:
                if (np.abs(self.grid_x[i]-y_center) <x_dim) and (np.abs(self.grid_y[j]-x_center) <y_dim) :
                    Msk0[i,j]=amp1
                else:
                    Msk0[i,j]=amp2
        #assigning the mask
        if self.Msk.size ==0:  
            self.Msk=Msk0  #if this the first mask we only assign it 
        else:
            self.Msk=self.Msk*Msk0  #otherwise we merge it with the existing mask 
                  
        return             

    def show_msk2D(self,what='amplitude'):
        """Show the amplitude/phase of the mask in 2D plot.
        
        :Args: 
            - what : string= (amplitude/phase) to choose what to plot.
        
        .. Note::
        This function needs matplotlib package
        
        :Returns:
            - none.
        """
        if what=='amplitude':
            msk=np.abs(self.Msk)
        elif what=='phase':
            msk=np.angle(self.Msk)
        else:
            print('"what" must be "amplitude" or "phase"')
            sys.exit(1)
        
        plt.figure()
        plt.pcolor(self.grid_x,self.grid_y,msk)
        #plt.show()
        return
    
    def show_msk3D(self,what='amplitude'):
        """Show the amplitude/phase of the mask in 3D plot.
        
        :Args: 
            - what : string= (amplitude/phase) to choose what to plot.
        
        .. Note::
        This function needs matplotlib package
        
        :Returns:
            - none.
        """
        if what=='amplitude':
            msk=np.abs(self.Msk)
        elif what=='phase':
            msk=np.angle(self.Msk)
        else:
            print('"what" must be "amplitude" or "phase"')
            sys.exit(1)
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X,Y=np.meshgrid(self.grid_x,self.grid_y) #we create a meshgrid for the plot
        #surf = ax.plot_surface(X, Y, np.abs(msk2), rstride=1,cmap=cm.jet ,cstride=1, linewidth=0)
        surf = ax.plot_surface(X, Y, msk, rstride=1 ,cstride=1, linewidth=0)
        plt.set_cmap('hot')
        #plt.show()
            
class CavEigenSys(object):
    '''
    classdocs
    '''
    


    def __init__(self,wavelength=1):
        '''
        Constructor
        '''
        self.dim_flag=''    #Flag indicating whether the system is 1D or 2D
        self.wavelength=wavelength  # all distances are relative to wavelength unit, for cavity_example_2D L=1000 means 1000*lambda unit
                                    # if lambda is in micron L=1000 micron =1mm
        self.k=2*math.pi/wavelength; #wave-number
        self.x1=[]  #first axis (of starting complex field)
        self.x2=[]  #second axis (of calculated complex field)
        self.Kt=[]  #the matrix kernel
        self.l=[]   #the eigenvalue vector
        self.v=[]   #the eigenvectors
        
        self.utils=UtilsFunc()
        
    def fresnel1D_cav(self,x1,x2,d,R1,R2):
        """"Fresnel Kernel formulation for 1D systems, this function is used internally in the solver, to construct the cavity Kernel matrix.
       
        :Args: 
            - x1,x2 : 1D vectors of real, defining the calculation zones of 1st and 2nd mirrors forming the optical cavity.
            - d : (positive real) the cavity length.
            - R1, R2: (reals) Radius of curvature of the two mirrors forming the cavity 
        
        .. Note::
        this function is appropriate for 1D systems, of optical cavities composed of 2 mirrors 
        
        :Returns:
            - none. the Fresnel kernel of the system is build and stored directly in the class attribute: 'self.K' '
        
        """
        wavelength=self.wavelength
        
        g1=1-d/R1;
        g2=1-d/R2;
        A=2*g1*g2-1;
        B=2*g2*d;
        D=A;
        y=-1j/(wavelength*np.sqrt(B))*np.exp((-1j*math.pi/wavelength)*(A*x1**2+D*x2**2-2*x1*x2)/B);    
        
        return y     
    
    def fresnel1D_ABCD_cav(self,x1,x2,A,B,C,D):
        """Fresnel Kernel formulation for a general ABCD optical 1D systems, this function is used internally in the solver, to construct the cavity Kernel matrix.
       
        :Args: 
            - x1,x2 : 1D vectors of real, defining the calculation zones of 1st and 2nd mirrors forming the optical cavity.
            - A,B,C,D: (reals) elements of the optical matrix defining the paraxial optical system. 
        
        .. Note::
        this function is appropriate for 1D systems, for a general case, optical cavity (multi-elements) 
        
        :Returns:
            - none. the Fresnel kernel of the system is build and stored directly in the class attribute: 'self.K' '
        
        """
        if B==0:
            print("Propagation distance can not be '0' please change the B element in the ABCD matrix.")
            sys.exit(1)
        else:
            wavelength=self.wavelength
            #y=-1j/(wavelength*np.sqrt(B))*np.exp((+1j*math.pi/wavelength)*(A*x1**2+D*x2**2-2*x1*x2)/B);
            y=-1j/(wavelength*np.sqrt(B))*np.exp((-1j*math.pi/wavelength)*(A*x1**2+D*x2**2-2*x1*x2)/B);
            
        return y
    
    def fresnel2DC(self,x1,x2,y1,y2,d,R1,R2):
        """"Fresnel Kernel formulation for 2D systems, this function is used internally in the solver, to construct the cavity Kernel matrix.
       
        :Args: 
            - x1,y1,x2,y2 : 1D vectors of real, defining the calculation zones of 1st (x1,y1) and 2nd (x2,y2) mirrors forming the optical cavity.
            - d : (positive real) the cavity length.
            - R1, R2: (reals) Radius of curvature of the two mirrors forming the cavity 
        
        .. Note::
        this function is appropriate for 2D systems, of optical cavities composed of 2 mirrors 
        
        :Returns:
            - none. the Fresnel kernel of the system is build and stored directly in the class attribute: 'self.K' '
        
        """
        wavelength=self.wavelength
        
        g1=1-d/R1;
        g2=1-d/R2;
        A=2*g1*g2-1;
        B=2*g2*d;
        D=A;
        yx=-1j/(wavelength*np.sqrt(B))*np.exp((-1j*math.pi/wavelength)*(A*x1**2+D*x2**2-2*x1*x2)/B);
        yy=-1j/(wavelength*np.sqrt(B))*np.exp((-1j*math.pi/wavelength)*(A*y1**2+D*y2**2-2*y1*y2)/B);
        
        return yx*yy
    
    def fresnel2D_ABCD(self,x1,x2,y1,y2,A,B,C,D):
        """Fresnel Kernel formulation for a general ABCD optical 2D systems, this function is used internally in the solver, to construct the cavity Kernel matrix.
       
        :Args: 
            - x1,y1,x2,y2 : 1D vectors of real, defining the calculation zones of 1st (x1,y1) and 2nd (x2,y2) mirrors forming the optical cavity.
            - A,B,C,D: (reals) elements of the optical matrix defining the paraxial optical system. 
        
        .. Note::
        this function is appropriate for 2D systems, for a general case, optical cavity (multi-elements) 
        
        :Returns:
            - none. the Fresnel kernel of the system is build and stored directly in the class attribute: 'self.K' '
        
        """
        wavelength=self.wavelength
        
        yx=-1j/(wavelength*np.sqrt(B))*np.exp((-1j*math.pi/wavelength)*(A*x1**2+D*x2**2-2*x1*x2)/B);
        yy=-1j/(wavelength*np.sqrt(B))*np.exp((-1j*math.pi/wavelength)*(A*y1**2+D*y2**2-2*y1*y2)/B);
        return yx*yy
    
    def build_2D_cav(self,a,n_pts,R1,R2,d):
        self.dim_flag="2D"
        #utils=UtilsFunc()
        abscissa,weight=self.utils.gauss_legendre(n_pts) # generate Legendre Gauss abscisas and weight for integration
        order=np.argsort(abscissa)
        abscissa=abscissa[order]
        weight=weight[order]
        
        self.x1=a*abscissa       
        self.x2=np.copy(self.x1)
        self.y1=np.copy(self.x1)
        self.y2=np.copy(self.x1)
        
        Nn=n_pts
        Mn=Nn
        self.Kt=np.zeros((Nn*Mn,Nn*Mn))+np.zeros((Nn*Mn,Nn*Mn))*1j
        line_block=np.zeros((Nn,Nn*Mn))+np.zeros((Nn,Nn*Mn))*1j
        Absc_j,Absc_v=np.meshgrid(self.x2,self.y2);
        W_j=np.diag(weight)
        
        print("Building the  kernel matrix ...")
        for u in range(Nn):
            for i in range(Nn):
                K=a*a*weight[i]*W_j.dot(self.fresnel2DC(self.x1[i],self.y1[u],Absc_j, Absc_v, d, R1, R2))
                #print(np.shape(K),np.shape(line_block),np.shape(self.Kt))
                
                #line_block[:,(i-1)*Nn+1:i*Nn]=K
                line_block[:,i*Nn:(i+1)*Nn]=K
                
            #self.Kt[(u-1)*Nn+1:u*Nn,:]=line_block
            self.Kt[u*Nn:(u+1)*Nn,:]=line_block
            line_block=np.zeros((Nn,Nn*Mn))+np.zeros((Nn,Nn*Mn))*1j
            #adv=u/Nn*100
            #print("Building the  kernel matrix "+repr(u+1)+ " / "+repr(Nn))

        print("Building the  kernel matrix done.")       
        return
    
            
    def build_2D_cav_ABCD(self,a,n_pts,A,B,C,D):
        """Build the  Fresnel-Kernel for a general ABCD optical 2D systems, this function construct the cavity Kernel matrix and stores it in the class attribute 'self.K'.
       
        :Args: 
            - a : (positive, real) Size of calculation zone (squared zone) 
            - n_pts: number of points used in discretization of the calculation zone, the step will be 'a/n_pts'
            - A,B,C,D: (reals) elements of the optical matrix defining the paraxial optical system. 
            
        .. Note::
            - this function is appropriate for 2D systems, for a general case, optical cavity (multi-elements).
            - All distances are normalized to the wavelength unit.
        
        :Returns:
            - none. the Fresnel kernel of the system is build and stored directly in the class attribute: 'self.K'.
        
        """
        
        self.dim_flag="2D"
        #utils=UtilsFunc()
        abscissa,weight=self.utils.gauss_legendre(n_pts) # generate Legendre Gauss abscisas and weight for integration
        order=np.argsort(abscissa)
        abscissa=abscissa[order]
        weight=weight[order]
        
        self.x1=a*abscissa       
        self.x2=np.copy(self.x1)
        self.y1=np.copy(self.x1)
        self.y2=np.copy(self.x1)
        
        Nn=n_pts
        Mn=Nn
        self.Kt=np.zeros((Nn*Mn,Nn*Mn))+np.zeros((Nn*Mn,Nn*Mn))*1j
        line_block=np.zeros((Nn,Nn*Mn))+np.zeros((Nn,Nn*Mn))*1j
        Absc_j,Absc_v=np.meshgrid(self.x2,self.y2);
        W_j=np.diag(weight)
        
        print("Building the  kernel matrix ...")
        for u in range(Nn):
            for i in range(Nn):
                #K=a*a*weight[i]*W_j.dot(self.fresnel2DC(self.x1[i],self.y1[u],Absc_j, Absc_v, d, R1, R2))
                K=a*a*weight[i]*W_j.dot(self.fresnel2D_ABCD(self.x1[i],self.y1[u],Absc_j, Absc_v, A,B,C,D))
                #print(np.shape(K),np.shape(line_block),np.shape(self.Kt))
                
                #line_block[:,(i-1)*Nn+1:i*Nn]=K
                line_block[:,i*Nn:(i+1)*Nn]=K
                
            #self.Kt[(u-1)*Nn+1:u*Nn,:]=line_block
            self.Kt[u*Nn:(u+1)*Nn,:]=line_block
            line_block=np.zeros((Nn,Nn*Mn))+np.zeros((Nn,Nn*Mn))*1j
            #adv=u/Nn*100
            #print("Building the  kernel matrix "+repr(u+1)+ " / "+repr(Nn))

        print("Building the  kernel matrix done.")       
        return

            
    def build_1D_cav(self,a,n_pts,R1,R2,d):
        """
        - calculate the matrix kernel of the cavity.
        - R1, R2 are given in wavelength units (normalized).
        - return the matrix kernel and the x axis (lengendre-Gauss distribution).
        
        Fresnel Kernel calculation for 1D 2 mirrors optical systems, this function is used internally in the solver, to construct the cavity Kernel matrix.
       
        :Args: 
            - a: (positive real)the size of the calculation area.
            - n_pts: (positive integer) the number of point used in discretization.
            - R1,R2: (reals) the radius of curvature of the 1st and 2nd mirror.
            - d : (positive real) the length of the cavity (distance between the two mirrors)   
        
        .. Note::
            - this function is appropriate for 1D systems, for a 2 mirrors, optical cavity.
            - x1 : the vector representing the initial plane is generated inside the function rather than getting it as an argument because it follows a legendre polynomials distribution and not linear spacing.
            - The kernel matrix elements spacing follows Legendre polynomials distribution rather than linear spacing, this is needed to replace the Fresnel integral by a sum (Legendre-Gauss quadrature scheme)
            - All the distances are in the wavelength unit.
        
        :Returns:
            - none. the Fresnel kernel of the system and x1,x2 the 2 planes (initial and propagated) are build and stored directly in the class attribute: 'self.K', 'self.x1','self.x2'
            
        """
        print("Building the  kernel matrix ...")
        self.dim_flag="1D"
        #utils=UtilsFunc()
        abscissa,weight=self.utils.gauss_legendre(n_pts) # generate Legendre Gauss abscisas and weight for integration
        order=np.argsort(abscissa)
        abscissa=abscissa[order]
        weight=weight[order]
        
        #self.x1=np.linspace(-a, a, n_pts)
        
        self.x1=a*abscissa       
        self.x2=np.copy(self.x1)
        self.Kt=np.zeros((n_pts,n_pts))+np.zeros((n_pts,n_pts))*1j
        for i in range(n_pts): 
            for j in range(n_pts):
                self.Kt[i,j]=a*weight[j]*self.fresnel1D_cav(a*abscissa[i],a*abscissa[j],d,R1,R2)
        
        print("Building the  kernel matrix done.")
        return 
    
    def build_1D_cav_ABCD(self,a,n_pts,A,B,C,D):
        """Build the  Fresnel-Kernel for a general ABCD optical 1D systems, this function construct the cavity Kernel matrix and stores it in the class attribute 'self.K'.
       
        :Args: 
            - a : (positive, real) Size of calculation zone (squared zone) 
            - n_pts: number of points used in discretization of the calculation zone, the step will be 'a/n_pts'
            - A,B,C,D: (reals) elements of the optical matrix defining the paraxial optical system. 
            
        .. Note::
            - this function is appropriate for 1D systems, for a general case, optical cavity (multi-elements).
            - All distances are in the wavelength unit.
        
        :Returns:
            - none. the Fresnel kernel of the system is build and stored directly in the class attribute: 'self.K'.
        
        """
        self.dim_flag="1D"
        #utils=UtilsFunc()
        abscissa,weight=self.utils.gauss_legendre(n_pts) # generate Legendre Gauss abscisas and weight for integration
        order=np.argsort(abscissa)
        abscissa=abscissa[order]
        weight=weight[order]
        
        #self.x1=np.linspace(-a, a, n_pts)
        
        self.x1=a*abscissa       
        self.x2=np.copy(self.x1)
        self.Kt=np.zeros((n_pts,n_pts))+np.zeros((n_pts,n_pts))*1j
        for i in range(n_pts): 
            for j in range(n_pts):
                self.Kt[i,j]=a*weight[j]*self.fresnel1D_ABCD_cav(a*abscissa[i],a*abscissa[j], A, B, C, D)
        
        return 

    def  solve_modes(self,n_modes=30):
        """Calculate the eigenvalues and eigenfunctions of the matrix-Kernel of the optical cavity defined in class attribute 'self.K'.
       
        :Args: 
            - n_modes: number of eigenvalues and eigenfunctions to calculate 
        
        :Returns:
            - none. the eigenvalues and eigenfunctions are stored directly in the class attribute: 'self.l' and 'self.v' respectively.
        
        .. Note::
            - The i'th eigenvalue corresponds to: losses (amplitude) and phase-shift (phase) per round-trip of the i'th mode of the cavity.
            - The i'th eigenfunction corresponds to the complex field distribution function of the i'th mode of the cavity. 
            - eigenvalues and modes can be obtained using the function 'get_mode(n)'.
            - the eigenfunctions (modes of the cavity) can be shwon using 'show_mode(n)' to show the n'th mode.
        
        
        """
        print("running the eigenvalues solver...")
        if self.dim_flag=='':
            print('The matrix kernel is empty')
            sys.exit(1)
        else:
            
            #self.l,self.v=la.eigs(self.Kt,n_modes, which="LM") #solving the eigenvalue problem
            if self.dim_flag=='2D':
                
                "with initial values vector"
                npts=np.size(self.x1)
                v00=np.random.rand(npts**2)
#                v00=np.ones(npts**2) #changed here 11-06-2015 by Seghil
                self.l,self.v=la.eigs(self.Kt,n_modes, which="LM", v0=v00) #solving the eigenvalue problem
            
            elif self.dim_flag=='1D':
                self.l,self.v=la.eigs(self.Kt,n_modes, which="LM") #solving the eigenvalue problem

            self.l,self.v=self.eig_sort(self.l, self.v) #sorting eigenvalues & eigenvectors
            self.normalize_modes1D() # normalize the amplitude of the mode to have a max=1
        return
    
    def get_mode1D(self,n):
        """Fetch the n'th eigenvalue and eigenfunctions from the solved eigenbasis of 1D system.
       
        :Args: 
            - n: order of eigenvalues and eigenfunctions to fetch 
        
        :Returns:
            - self.l[n]: (complex) the n'th eigenvalue. 
            - self.v[:,n]:  (complex vector) the n'th eigenfunction of the system. (complex field distribution of the n'th mode of the cavity)
        .. Note::
            - this function is used with 1D systems.
            - The i'th eigenvalue corresponds to: losses (amplitude) and phase-shift (phase) per round-trip of the i'th mode of the cavity.
            - The i'th eigenfunction corresponds to the complex field distribution function of the i'th mode of the cavity. 
        
        """
        if self.dim_flag=='':
            print("There are no modes in the system yet")
            sys.exit(1)
        elif self.dim_flag=='2D':
            print("This function is for 1D systems please use the 2D one")
            sys.exit(1)
        else:
            return self.l[n], self.v[:,n] 
        
    def normalize_modes1D(self):
        """Normalize the amplitude of all calculated modes, this function is used internally.
        """
        max_v=np.amax(np.abs(self.v),axis=0)
        self.v=self.v/max_v
            
        return
    
    def normalize_beam(self, beam):
        """Normalize the amplitude a beam this function is used internally.
        :Args: 
            - beam : 1D or 2D field 
        :Returns:
            - beam : normalized to the maximum value of the entered beam
        
        """
        beam=beam/beam.max()
        
        return beam
    
    
    def get_mode2D(self,n):
        """Fetch the n'th eigenvalue and eigenfunctions from the solved eigenbasis of 2D system.
       
        :Args: 
            - n: order of eigenvalues and eigenfunctions to fetch 
        
        :Returns:
            - self.l[n]: (complex) the n'th eigenvalue. 
            - tem :  2D (complex vector) the n'th eigenfunction of the system. (complex field distribution of the n'th mode of the cavity)
        .. Note::
            - this function is used with 2D systems.
            - out of the solver the eigenfunction is a complex 1D vector, it has to be reshaped to get the 2D field distribution called tem. 
            - The i'th eigenvalue corresponds to: losses (amplitude) and phase-shift (phase) per round-trip of the i'th mode of the cavity.
            - The i'th eigenfunction corresponds to the complex field distribution function of the i'th mode of the cavity. 
        
        
        """
        
        if self.dim_flag=='':
            print("There are no modes in the system yet")
            sys.exit(1)
        elif self.dim_flag=='1D':
            print("This function is for 2D systems please use the 1D one")
            sys.exit(1)
        else:
            npts=np.size(self.x1)
            tem=self.v[:,n].reshape(npts,npts) #the mode out of the solver is 1 column vector it must be reshaped to npts x npts
        
        return self.l[n], tem
    
    def show_mode(self,n,what='amplitude'):
        """Show the amplitude/phase of the n'th mode.
        
        :Args: 
            - what : string= (amplitude/phase) to choose what to plot.
            - n : (positive integer) the order of the mode to show.
        .. Note::
        This function needs matplotlib package
        
        :Returns:
            - none.
        """
        if self.dim_flag=='':
            print("the system kernel is empty ")
            sys.exit(1)
        elif self.dim_flag=='1D':
            plt.figure()
            if what=='amplitude':
                plt.plot(self.x1,np.abs(self.v[:,n]))
            elif what=='phase':
                plt.plot(self.x1,np.angle(self.v[:,n]))
            elif what=='intensity':
                plt.plot(self.x1,np.abs((self.v[:,n])**2))
            else:
                print("what must be 'amplitude','intensity' or 'phase'")
                
        elif self.dim_flag=='2D':
            npts=np.size(self.x1)
            tem=self.v[:,n].reshape(npts,npts);
            plt.figure()
            if what=='amplitude': 
                plt.pcolor(self.x1,self.y1,np.abs(tem))
                plt.colorbar()
            elif what=='phase':
                plt.pcolor(self.x1,self.y1,np.angle(tem))
                plt.colorbar()
            elif what=='intensity':
                plt.pcolor(self.x1,self.y1,np.abs(tem**2))
                plt.colorbar()
            else:
                print("what must be 'amplitude','intensity' or 'phase'")
            
            
            
        return
    
    def apply_mask1D(self,MaskObj):
        """Applay a phase and amplitude mask to the matrix kernel of 1D systems.
        
        :Args: 
            - MaskObj : object of the class AmpMask2D which contains the mask matrix in self.Msk
        .. Note::
            - this function is to use with 1D systems
            - This function multiply each coulumn of the matrix Kernel and the mask (phas & amplitude)
        
        :Returns:
            - none. the modifications are applyied directly on the kernel (self.Kt)
        """
        print("Applying 1D Mask...")
        #Mask=MaskObj.Msk
        Mask=MaskObj 
        for i in range(np.size(Mask)):
                self.Kt[i,:]=self.Kt[i,:]*Mask[i]
        
        print("Mask applied.")                  
        return 
    
    def apply_mask2D(self,MaskObj):
        """Apply a phase and amplitude mask to the matrix kernel of 2D systems.
        
        :Args: 
            - MaskObj : object of the class AmpMask2D which contains the mask matrix in self.Msk
        .. Note::
            - this function is to use with 1D systems
            - This function multiply each coulumn of the matrix Kernel and the mask (phas & amplitude)
        
        :Returns:
            - none. the modifications are applyied directly on the kernel (self.Kt)
        """
        print("Applying 2D Mask...")
        Mask=MaskObj.Msk
        Nn=np.size(Mask)
        Mask_alpha=Mask.reshape(1,Nn)
        for u in range(Nn):
            self.Kt[u,:]=self.Kt[u,:]*Mask_alpha
        print("Mask applied.")
        return 
    
    def cascade_subsystem(self,SysObj,order=1):
        """Cascade 2 systems (2 objects 'MatEigenSolv') each one containig its Matrix kernel (self.Kt).
        
        :Args: 
            - SysObj : (object of the classMatEigenSolv) contains the kernel matrix in 'self.Kt' and all elements of the system.
            - order: (1/-1) corresponds to the order of cascading (order=1 :sys1 --> sys2); (order=-1 :sys2 --> sys1)
        .. Note::
            - this function is to use with 1D & 2D systems
            - This function multiply each coulumn of the matrix Kernel and the mask (phas & amplitude)
        
        :Returns:
            - none. the modifications are applyied directly on the kernel (self.Kt)
        """
        if self.dim_flag=='':
            print("the kernel matrix is empty!")
            sys.exit(1)
        else:
            Kt2=SysObj.Kt
            if order==1:
                self.Kt=np.dot(self.Kt,Kt2)
            elif order==-1:
                self.Kt=np.dot(Kt2,self.Kt)
            else:
                print("order can take '1' or '-1' values only ")
                sys.exit(1)
                
            print("systems cascaded.")
        
        return
    
    def eig_sort(self,l,v):
        """l :eigenvalue; v:eigenvector.
        sorting eigenvalues & eigenvectors.
        this function is used internally after solving the eigenvalue problem to sort the modes from the fundamental (0) to the (n'th)
        """
        idx=np.argsort(np.abs(l))
        l=l[idx]
        v=v[:,idx]
        l=l[::-1]
        v=v[:,::-1]
        
        return l,v
    
    def find_waist(self,beam,x,value=0.36):
        """find the waist at 36% of the maximum amlitude 
        
        
        """
        beam=self.normalize_beam(beam)
        
        idx=self.utils.find_nearest(np.abs(beam),value)
        return np.abs(x[idx])
    
    def find_mode_waist(self,n):
        pass
        # function to write its uses find_waist
        
        return
    
if __name__ == '__main__':
    print("hiii there")
    
    