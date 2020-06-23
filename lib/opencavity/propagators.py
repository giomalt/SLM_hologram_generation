'''
Created on 23 mars 2014

@author: Mohamed seghilani
'''
import math
import numpy as np
import sys
import matplotlib.pylab as plt
from opencavity.modesolver import AmpMask2D

class FresnelProp(object):
    '''
        Class for Fresnel propagation kernel construction and integrals calculation
        this class contains all informations about the optical system: 
            - wavelength =1 (default value) 
            - U1 : the initial field (complex 1D or 2D vector)
            - x1,y1 : starting plane coordinates (1D vector of float)
            - U2 : resulting field   ((complex 1D or 2D vector)
            - x2,y2 : result plane coordinates (1D vector of float)
        The optical system can be 1D or 2D, this information is stored in the attribute
        'dimension_flag' 
            
            
        This class contains a method to calculate the propagated field at several distances  
        to follow the propagation. It is called yz_prop_chart()
        
        The class contains plotting methods to show the start/result field 
        
        Steps to use this class: 
        
        >>> opSys=FresnelProp() # create a propagation system (object)
        >>> opSys.set_start_beam(U1, x) # set the starting field 
        >>> opSys.set_ABCD(M1) # M1 is an ABCD optical system matrix
        >>> opSys.propagate1D_ABCD(x2=30*x) # calculate the propagation integral
        >>> opSys.show_result_beam(what='intensity') # plot the resulting field 
        >>> opSys.show_result_beam(what='phase') 
        
    
    '''


    def __init__(self,wavelength=1):
        '''
        Constructor
        '''
        self.wavelength=wavelength  # all distances are relative to wavelength unit, for cavity_example_2D L=1000 means 1000*lambda unit
                                    # if lambda is in micron L=1000 micron =1mm
        self.k=2*math.pi/wavelength; #wave-number
        self.M=[]  #ABCD matrix
        self.x1=[]
        self.x2=[]
        self.y1=[]
        self.y2=[]
        self.U1=[]  #start field
        self.U2=[]  #result field
        
        self.Uyz=[] #result matrix for Uyz propagation
        self.zz=[]  #z axis for Uyz chart
        
        self.X1=[]  #for the meshgrid
        self.Y1=[]
        self.dim_flag=''
        
    def set_start_beam(self,U1,x1,y1=[]):
        """Assign initial value to self.U1 taken as the start beam for propagation functions in the class.
        
        Args:
            - U1 (complex 1D or 2D matrix): the initial field 
            - x1 (vector of float): abscissa of the starting plane
            - y1 (vector of float): ordinate of the starting plane (for 2D case)
        
        .. Note::
            - default value of y1 is void vector, this assumes 1D system.
            - For 2D system if y1 is not given it will be taken equals to x1.
        
        Returns:
            - none.
        
        """
        if np.size(U1.shape)==1:
            self.dim_flag='1D'
            self.U1=U1
            self.x1=x1
            self.y1=y1
            
        else:
            self.dim_flag='2D'
            self.U1=U1
            self.x1=x1
            if y1==[]:
                print("2D initial beam but y1 is missing..")
                print("y1 is set = x1.")
                self.y1=x1
        
        return
    
    def get_start_beam(self):
        """
        Fetch the initial beam contained in 'self.U1' and the corresponding abscissa self.x1 it returns U2,x2 and y2 if 2D.
        Args:
            -none
            
        Returns: 
            An array containing the following elements:
            - U1 (complex 1D or 2D matrix): the initial field "entered by the user"
            - x1 (vector of float): abscissa of the starting plane "entered by the user"
            - y1 (vector of float): ordinate of the starting plane (for 2D case)
        
        .. Note::
            - This function returns the field entered by the user using the function 'set_start_beam'.
            - The same result can be obtained by directly accessing 'self.U1','self.x1' and 'self.y1' of the class
            
        """
        if self.dim_flag=='1D':
            return self.U1, self.x1
        elif self.dim_flag=='2D':
            return self.U1,self.x1, self.y1
        else:
            print("Empty system!")
            sys.exit(1)
            
    def get_result_beam(self):
        """
        Fetch the propagation result contained in 'self.U2' and the corresponding abscissa self.x2 it returns U2,x2 and y2 if 2D.
        Args:
            -none
            
        Returns: 
            An array containing the following elements:
            - U2 (complex 1D or 2D matrix): the propagation result field 
            - x2 (vector of float): abscissa of the starting plane
            - y2 (vector of float): ordinate of the starting plane (for 2D case)
        
        .. Note::
            - The same result can be obtained by directly accessing 'self.U2','self.x2' and 'self.y2' of the class after calculation. 
        
        """
        
        if self.dim_flag=='1D':
            return self.U2, self.x2
        elif self.dim_flag=='2D':
            return  self.U2,self.x2,self.y2
        else:
            print("Empty system!")
            sys.exit(1)
            

    def apply_mask1D(self,Mask):
        """Apply phase and amplitude mask given as an argument to the initial field.
        Args:
            -Mask (complex 1D matrix): the initial field will be multiplied by this matrix (element by element).
            
        Returns: 
            -none.
        
        .. Note::
            - The same result can be obtained by directly multiplying  'self.U1' by Mask, but using this function is preferred for the clarity of the code.
        
        Example of use 
            >>> opSys=FresnelProp() # creating propagator object    
            >>> T_lens=np.exp((1j*opSys.k/(2*f))*(x)**2) # creating phase mask of thin lens with FL=f 
            >>> opSys.set_start_beam(tem00, x) # setting the initial, see the function documentation for more information 
            >>> opSys.set_ABCD(M1)  # set the ABCD propagation matrix 
            >>> opSys.apply_mask1D(T_lens) # Applying the phase mask
        
        """
        print("Applying 1D Mask...")
        #Mask=MaskObj.Msk
        n_pts_msk=np.size(Mask)
        n_pts_beam=np.size(self.U1)
        
        if n_pts_beam == n_pts_msk:
            self.U1=self.U1*Mask
            print("Mask applied.")                  
             
        else:
            print("The phase mask and the initial field must have the same length!")
            sys.exit(1)
        
        return
    
    def set_ABCD(self,M):
        """assign an ABCD matrix the system (the ABCD matrix is self.M : an attribute of the class holding the system)

        Args:
            - M (2x2) real matrix : Paraxial propagation system.  
            
        Returns: 
            -none.
        
        .. Note::
            - The same result can be obtained by directly assigning 'self.M=M' by Mask, but using this function is preferred for the clarity of the code.
        
        Example of use 
            >>> #  definition of the ABCD matrices L1, L2, f are real variables     
            >>> M1=np.array([[1, L1],[0, 1]]); # propagation distance L1 
            >>> M2=np.array([[1, 0],[-1/f, 1]]); # thin lens with EFL=f
            >>> M3=np.array([[1, L2],[0, 1]]) # propagation distance L2
            >>> M=M3.dot(M2).dot(M1) # calculating the global matrix 

            >>> opSys=FresnelProp() # creating propagation system (object)     
            >>> opSys.set_ABCD(M)  # set the ABCD propagation matrix
        
        
        """
        if not M.shape[0]==2 & M.shape[1]==2:
            print("ABCD matrix must be 2x2!")
            sys.exit(1)
        else:
            self.M=M
        
        return
    
    def cascade_subsystem(self,M2):
        """Cascade subsytem does dot product with the initial ABCD matrix (inverted order).
        Args:
            - M2 (2x2) real matrix : Paraxial propagation system.  
            
        Returns: 
            -none.
        
        .. Note::
            - The same result can be obtained by directly doing the dot product 'self.M=np.dot(M2,self.M) but using this function is preferred for the clarity of the code. 
            - another way to do the same thing (preferred one) is to calculate the complete system matrix and then assign it 
            (see 'set_ABCD()' function doc )
            - Matrix with propagation distance ='0' can not be assigned this causes division by '0' in the propagation Kernel  
        
        Example of use 
        
            >>> #  definition of the ABCD matrices L1, L2, f are real variables     
            >>> M1=np.array([[1, L1],[0, 1]]); # propagation distance L1 
            >>> M2=np.array([[1, 0],[-1/f, 1]]); # thin lens with EFL=f
            >>> M3=np.array([[1, L2],[0, 1]]) # propagation distance L2
            >>> M=M3.dot(M2).dot(M1) # calculating the global matrix 

            >>> opSys=FresnelProp() # creating propagation system (object)     
            >>> opSys.set_ABCD(M)  # set the ABCD propagation matrix
        
        
        """
        if not M2.shape[0]==2 & M2.shape[1]==2:
            print("ABCD matrix must be 2x2!")
            sys.exit(1)
        else:
            if self.M==[]:
                self.M=M2
            else:
                self.M=np.dot(M2,self.M)
        
        return
    
    def kernel1D_ABCD(self,x1,x2,A,B,C,D):
        """
        Fresnel Kernel 1D this function is used internally in Fresnel integral calculation
        """
        y=np.sqrt(1j/(self.wavelength*B))*np.exp((-1j*self.k/(2*B))*(A*x1**2+D*x2**2-2*x1*x2))
        return y
    
    def kernel2D_ABCD(self,x1,x2,y1,y2,Ax,Bx,Cx,Dx,Ay=0,By=0,Cy=0,Dy=0):
        """
        fresnel Kernel 2D this function is used internally in Fresnel integral calculation
        """
        if Ay==0:
            Ay=Ax
        if By==0:
            By=Bx
        if Cy==0:
            Cy=Cx
        if Dy==0:
            Dy=Dx
        
        y=1j/(self.wavelength*np.sqrt(Bx*By))*np.exp((-1j*self.k/2)*((Ax*x1**2+Dx*x2**2-2*x1*x2)/Bx+(Ay*y1**2+Dy*y2**2-2*y1*y2)/By))
        return y
    
    def propagate1D_ABCD(self,x2=[]):
        """Fresnel propagation (1D case) of a complex field U1 one iteration through ABCD optical system
        from the plane x1 to the plane x2 .
        
        Args:
            - x2 (real 1D vector) : vector defining the propagation plane coordinates can be assimilated to a detector surface for example. by default it is a void vector, this means that the result plane is taken equal to the startingone (same size).   
            
        Returns: 
            -none. the propagation result is stored in 'self.U2' to get it use the function 'self.get_result_beam()' 
        
        .. Note::
            - x2 size must to satisfy Fresnel condition (Paraxial optics condition) ...to be explained later...
        
        Example of use 
        
            >>> opSys=FresnelProp() # creating propagation system (object)    
            >>> opSys.set_start_beam(tem00, x)# tem00 is a complex field , 'x' (real) is starting plane  
            >>> opSys.set_ABCD(M)  # set the ABCD propagation matrix
            >>> opSys.propagate2D_ABCD() # propagation through the ABCD system, the result plane is equal the starting one
            >>> opSys.show_result_beam() # plots the result field 
            >>> opSys.show_result_beam() # plots the result field 
        
        """
        if self.dim_flag=='':
            print('the initial field is empty!')
            sys.exit(1)
        elif self.dim_flag=='2D':
            print('please use 2D propagation function.')
            sys.exit(1)
        else:
            if x2==[]:
                self.x2=self.x1
            else:
                self.x2=x2
            
            A=self.M[0,0]; B=self.M[0,1]; C=self.M[1,0]; D=self.M[1,1]
            self.U2=np.zeros(np.size(self.x2))+np.zeros(np.size(self.x2))*1j
            for i in range(np.size(self.x2)):
                Mi=self.U1*np.exp(-1j*self.k*(B))*self.kernel1D_ABCD(self.x1,self.x2[i],A,B,C,D);
                self.U2[i]=np.trapz(Mi, self.x1)
                #self.U2[i]=np.sum(Mi)
        return   
     
    
    def propagate1Dfft(self,x2=[]):
          
        if self.dim_flag=='':
            print('the initial field is empty!')
            sys.exit(1)
        elif self.dim_flag=='2D':
            print('please use 2D propagation function.')
            sys.exit(1)
        else:
            if x2==[]:
                self.x2=self.x1
            else:
                self.x2=x2
          
        
        npts=self.x1.size
        x_max=self.x1.max()
        dx=x_max*2/npts
            
           
        A=self.M[0,0]; B=self.M[0,1]; C=self.M[1,0]; D=self.M[1,1]
        fx=self.x1/(self.wavelength*B)
#         #kk=np.sqrt(1j/(self.wavelength*B))*np.exp((-1j*self.k/(2*B))*(A*x1**2+D*x2**2-2*x1*x2))
#         kk1=self.U1*np.exp((-1j*self.k/(2*B))*(A*self.x1**2))
        kk2=np.sqrt(1j/(self.wavelength*B))*np.exp((-1j*self.k/(2*B))*(D*fx**2))
#         self.U2=np.zeros(np.size(self.x2))+np.zeros(np.size(self.x2))*1j
#          
        Mi=np.fft.fftshift(np.fft.fft(self.U1*np.exp(1j*np.pi/(self.wavelength*B)*self.x1)))
#         Mi2=np.fft.fftshift(np.fft.fft(kk2)) 
#         #H = np.exp(1i*k*z).*exp(-1i*pi*lambda*z*(u.^2+v.^2)); 
        Mi2=np.exp(1j*self.k*(B))*np.exp(-1j*np.pi/(self.wavelength*B)*(fx**2))
#         Mi2=np.exp(-1j*self.k*(B))*self.kernel1D_ABCD(fx,fx,A,B,C,D)
        #self.U2=np.fft.ifft(np.fft.fftshift(Mi*Mi2))
        self.U2=Mi*Mi2
        self.x2=fx
        return
     
# #     def cpropagate1D_ABCD(self,U1,x1,x2,A,B,C,D):
#         """
#         propagator using c library
#         """
#         import ctypes
#         import os
#         chemin=os.getcwd()+"\libpropagator_c.dll"
#         mydll = ctypes.cdll.LoadLibrary(chemin)
#         
#         
#         size_x=np.size(x1);
#         
#         tab1= ctypes.c_double*size_x #(ctypes.c_int*taille)()  equivalent aux 2 lignes
#         U2_real=tab1()            
#         
#         tab2= ctypes.c_double*size_x #(ctypes.c_int*taille)()  equivalent aux 2 lignes
#         U2_imag=tab2() 
#         
#         
#         
# #         c_float_p = ctypes.POINTER(ctypes.c_float)
# #         data = numpy.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
# #         data = data.astype(numpy.float32)
# #         data_p = data.ctypes.data_as(c_float_p)
#          
#         U1_real=np.array(np.real(U1))
#         U1_imag=np.array(np.imag(U1))
#             
#         c_double_ptr = ctypes.POINTER(ctypes.c_double)
#         
#         U1_real=U1_real.astype(np.double)
#         U1_imag=U1_imag.astype(np.double)
#         
#         U1_real_ptr=U1_real.ctypes.data_as(c_double_ptr)
#         U1_imag_ptr=U1_imag.ctypes.data_as(c_double_ptr)
#         
#         x1=x1.astype(np.double)
#         x2=x2.astype(np.double)
#         
#         x1_ptr=x1.ctypes.data_as(c_double_ptr)
#         x2_ptr=x2.ctypes.data_as(c_double_ptr)
#         
#         wavelength_c=ctypes.c_float() #pointeur sur un c_float
#         Ac=ctypes.c_float()
#         Bc=ctypes.c_float()
#         Cc=ctypes.c_float()
#         Dc=ctypes.c_float()
#         
#         wavelength_c.value=self.wavelength
#         Ac.value=A; Bc.value=B; Cc.value=C; Dc.value=D; 
#         
#         mydll.propagate1D(wavelength_c,U1_real_ptr,U1_imag_ptr,U2_real,U2_imag,x1_ptr,x2_ptr,size_x,Ac,Bc,Cc,Dc)    
#         #propagate1D(float wavelenth,double* U1_real,double* U1_imag,double* U2_real,double* U2_imag,double* x1, double* x2,int size_x,float A,float B, float C, float D){        
#         #U2=U2_real+1j*U2_imag
#         U2=np.zeros(np.size(x2))+np.zeros(np.size(x2))*1j
#         for i in range(size_x):
#             U2[i]=U2_real[i]+1j*U2_imag[i]
#         
#         return U2
#     
    def propagate2D_ABCD(self,x2=[],y2=[]):
        """Fresnel propagation (2D case) of a complex field U1 one iteration through ABCD optical system from the plane x1,y1 to the plane x2,y2 .
        
        Args:
            - x2,y2 (real 1D vectors) : vectors defining the propagation plane coordinates, can be assimilated to a detector surface for example. by default it is a void vector, this means that the result plane is taken equal to the starting one (same size).   
            
        Returns: 
            -none. the propagation result is stored in 'self.U2' to get it use the function 'self.get_result_beam()' 
        
        .. Note::
            - x2,y2 size must to satisfy Fresnel condition (Paraxial optics condition) ...to be explained later...
        
        Example of use 
        
            >>> opSys=FresnelProp() # creating propagation system (object)    
            >>> opSys.set_start_beam(tem00, x)# tem00 is a complex 2D field , 'x' (real) is starting plane  
            >>> opSys.set_ABCD(M)  # set the ABCD propagation matrix
            >>> opSys.propagate1D_ABCD(x2=30*x) # propagation through the ABCD system, the result plane is 30 times the starting one
        
        
        """
        if self.dim_flag=='':
            print('the initial field is empty!')
            sys.exit(1)
        elif self.dim_flag=='1D':
            print('please use 1D propagation function.')
            sys.exit(1)
        else:
            print('calculating Fresnel propagation...')
            #it's ok, start porpagation calcul 
            if x2==[]:
                self.x2=self.x1
                #self.y2=self.y1
            else:
                self.x2=x2
            
            if y2==[]:
                self.y2=x2
            else:
                self.y2=y2
            
            Ax=self.M[0,0]; Bx=self.M[0,1]; Cx=self.M[1,0]; Dx=self.M[1,1]
            Ay=Ax; By=Bx; Cy=Cx; Dy=Dx 
            
            
            self.U2=np.zeros((np.size(self.x2),np.size(self.y2)))+np.zeros((np.size(self.x2),np.size(self.y2)))*1j
            for i in range(np.size(self.x2)):
                for j in range(np.size(self.y2)):
                    self.X1,self.Y1=np.meshgrid(self.x1,self.y1);
                    
                    Mi=self.U1*np.exp(-1j*self.k*(np.sqrt(Bx*By)))*self.kernel2D_ABCD(self.X1,self.x2[i],self.Y1,self.y2[j],Ax,Bx,Cx,Dx,Ay,By,Cy,Dy);
                    integral1=np.trapz(Mi, self.x1)
                    integral2=np.trapz(integral1,self.y1)
                    #integral1=np.sum(Mi)
                    #integral2=np.sum(integral1)
                    
                    #self.U2[i]=np.sum(Mi)
                    self.U2[i,j]=integral2
        
        return       
    
    def cpropagate2D_ABCD(self,U1,x1,x2,y1,y2,Ax,Bx,Cx,Dx,Ay=0,By=0,Cy=0,Dy=0):
        """
        propagator using c library
        """
        import ctypes
        import os
        
        if Ay==0:
            Ay=Ax
        if By==0:
            By=Bx
        if Cy==0:
            Cy=Cx
        if Dy==0:
            Dy=Dx
            
        chemin=os.getcwd()+"\libpropagator_c.dll"
        mydll = ctypes.cdll.LoadLibrary(chemin)
        
        size_x=np.size(x1);
        size_y=np.size(y1);            
         
        # An array of double* can be passed to your function as double**.
        U2_real=(ctypes.POINTER(ctypes.c_double)*size_x)()  #pointer on an array of double*
        for i in range(size_x): #we fill each int* with an int* 
            U2_real[i]=(ctypes.c_double*size_x)()
            
        U2_imag=(ctypes.POINTER(ctypes.c_double)*size_x)()
        for i in range(size_x): 
            U2_imag[i]=(ctypes.c_double*size_x)()   
            
        U1_real_ptr=(ctypes.POINTER(ctypes.c_double)*size_x)()  #pointer on an array of double*
        for i in range(size_x): #we fill each int* with an int* 
            U1_real_ptr[i]=(ctypes.c_double*size_x)()
            
        U1_imag_ptr=(ctypes.POINTER(ctypes.c_double)*size_x)()
        for i in range(size_x): 
            U1_imag_ptr[i]=(ctypes.c_double*size_x)() 
        
        
#         c_float_p = ctypes.POINTER(ctypes.c_float)
#         data = numpy.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
#         data = data.astype(numpy.float32)
#         data_p = data.ctypes.data_as(c_float_p)
         
        U1_real=np.array(np.real(U1)) #separation of real an imag parts for the function
        U1_imag=np.array(np.imag(U1))
        
        for i in range(size_x):
            for j in range(size_y):
                U1_real_ptr[i][j]=U1_real[i][j]
                U1_imag_ptr[i][j]=U1_imag[i][j]
        
    
        c_double_ptr = ctypes.POINTER(ctypes.c_double)
#        c_2D_double_ptr=ctypes.POINTER(c_double_ptr)
#         
#         U1_real=U1_real.astype(np.double)
#         U1_imag=U1_imag.astype(np.double)
#         
#         U1_real_ptr=U1_real.ctypes.data_as(c_2D_double_ptr)
#         U1_imag_ptr=U1_imag.ctypes.data_as(c_2D_double_ptr)
        
        x1=x1.astype(np.double)
        x2=x2.astype(np.double)
        y1=y1.astype(np.double)
        y2=y2.astype(np.double)
        
        x1_ptr=x1.ctypes.data_as(c_double_ptr)
        x2_ptr=x2.ctypes.data_as(c_double_ptr)
        y1_ptr=y1.ctypes.data_as(c_double_ptr)
        y2_ptr=y2.ctypes.data_as(c_double_ptr)
        
        wavelength_c=ctypes.c_float() #pointeur sur un c_float
        Ac=ctypes.c_float()
        Bc=ctypes.c_float()
        Cc=ctypes.c_float()
        Dc=ctypes.c_float()
        
        wavelength_c.value=self.wavelength
        Ac.value=Ax; Bc.value=Bx; Cc.value=Cx; Dc.value=Dx;
         
        mydll.propagate2D(wavelength_c,U1_real_ptr,U1_imag_ptr,U2_real,U2_imag,x1_ptr,x2_ptr,y1_ptr,y2_ptr,size_x,size_y,Ac,Bc,Cc,Dc)
        
        U2=np.zeros((np.size(x2),np.size(y2)))+np.zeros((np.size(x2),np.size(y2)))*1j
        for i in range(size_x):
            for j in range(size_y):
                U2[i][j]=U2_real[i][j]+1j*U2_imag[i][j]
        
        return U2
    
    def yz_prop_chart(self,Lmin,Lmax,nstep,x2=[]):
        """Propagate the 1D complex field to several planes and store the results in a matrix to follow the propagation as a function of distance    .
        
        Args:
            - Lmin (real) : initial distance from which propagations calculation starts.
            - Lmax (real)> Lmax : Stop distance until which propagations are calculated.
            - nstep (integer) : number of planes where the propagated field is calculated.
            - x2(real 1D vectors) : vector defining the propagation plane coordinates, can be assimilated to a detector surface for example. by default it is a void vector, this means that the result plane is taken equal to the starting one (same size)   
            
        Returns: 
            -none. the propagation result is stored in 'self.Uyz' :2D complex matrix containing result field at several planes stored in 'self.zz'. 
                    
        .. Note::
            
        Example of use 
        
            >>> opSys=FresnelProp() # creating propagation system (object)    
            >>> opSys.set_start_beam(tem00, x)# tem00 is a complex 1D field , 'x' (real) is initial plane  
            >>> #opSys.yz_prop_chart(5e3,50e3,100,30*x) # propagate the start field from Lmin=5mm to Lmax=50mm at 100 intermediate planes (linearly spaced ), result plane is 30x times the start one.
            >>> #opSys.show_prop_yz() # do the calculations
            >>> #opSys.show_prop_yz(what='intensity') # show the result 
            >>> #plt.show()

        
        """
        if self.dim_flag=='':
            print('the initial field is empty!')
            sys.exit(1)
        elif self.dim_flag=='2D':
            print('please use 2D propagation function.')
            sys.exit(1)
        else:
            if x2==[]:
                self.x2=self.x1
            else:
                self.x2=x2
        
            dz=(Lmax-Lmin)/nstep 
            if dz<=0:
                print('Lmin must be < Lmax.')
                sys.exit(1)
            else:
                zz=np.zeros(nstep)  #the z vector contains all propagation distances
                self.Uyz=np.zeros((np.size(self.U1),nstep))+np.zeros((np.size(self.U1),nstep))*1j
                zi=Lmin
                
                for i in range (nstep):
                    zi=zi+dz
                    zz[i]=zi
                    self.M=np.array([[1, zi],[0, 1]])
                    self.propagate1D_ABCD(self.x2) #the result is in U2 attribute of the class (self) it takes the ABCD matrix from self.M
                    self.Uyz[:,i]=self.U2
                self.zz=zz
            
        return
    
    def show_result_beam(self,what='amplitude'):
        """shows 'self.U2'  result of propagation calculation.
        Args:
            - what (string): flag to indicate what to plot (amplitude,phase or intensity) of the result field, by default is amplitude.
        Returns: 
            -none. 
        
        .. Note::
            - the function plots the resulting field using 'matplotlib'.
            - it plots 'self.U2' as a function of 'self.x2' for 1D case.
            - it shows 2D map of 'self.U2' in the plane defined by 'self.x2', 'self.y2'.
            - sometimes (when not using Ipython) the function 'matplotlib.pylab.show()' must be used to show the plot result.
        Example of use 
        
            >>> opSys=FresnelProp() # creating propagation system (object)   
            >>> opSys.set_start_beam(tem00, x)# tem00 is a complex 2D field , 'x' (real) is initial plane  
            >>> opSys.set_ABCD(M)  # set the ABCD propagation matrix
            >>> opSys.propagate1D_ABCD(x2=30*x) # propagation through the ABCD system, the result plane is 30 times the initial one
            >>> opSys.show_result_beam(what='intensity') # show result of propagation 
            >>> opSys.show_result_beam(what='phase')
        
        
        """
        if self.dim_flag=='':
            print("the system is empty ")
            sys.exit(1)
        elif self.dim_flag=='1D':
            plt.figure()
            if what=='amplitude':
                plt.plot(self.x2,np.abs(self.U2))
            elif what=='phase':
                plt.plot(self.x2,np.angle(self.U2))
            elif what=='intensity':
                plt.plot(self.x2,np.abs(self.U2)**2)
            else:
                print("what must be 'amplitude','intensity' or 'phase'")
                
        elif self.dim_flag=='2D':
            plt.figure()
            if what=='amplitude': 
                plt.pcolor(self.x2,self.y2,np.abs(self.U2))
            elif what=='phase':
                plt.pcolor(self.x2,self.y2,np.angle(self.U2))
            elif what=='intensity':
                plt.pcolor(self.x2,self.y2,np.abs(self.U2)**2)
            else:
                print("what must be 'amplitude','intensity' or 'phase'")
        return
        
    def show_start_beam(self,what='amplitude'):
        """
        shows self.U1 the start beam assigned by the user
        Args:
            - what (string): flag to indicate what to plot (amplitude,phase or intensity) of the result field, by default is amplitude.
        Returns: 
            -none. 
        
        .. Note::
            - the function plots the resulting field using 'matplotlib'.
            - it plots 'self.U1' as a function of 'self.x1' for 1D case.
            - it shows 2D map of 'self.U2' in the plane defined by 'self.x1', 'self.y1'.
            - sometimes (when not using Ipython) the function 'matplotlib.pylab.show()' must be used to show the plot result.
        Example of use 
        
            >>> opSys=FresnelProp() # creating propagation system (object)   
            >>> opSys.set_start_beam(tem00, x)# tem00 is a complex 2D field , 'x' (real) is initial plane  
            >>> opSys.show_start_beam(what='intensity') # show initial field assigned by the user
            >>> opSys.show_start_beam(what='phase')
        
        """
        if self.dim_flag=='':
            print("the system is empty ")
            sys.exit(1)
        elif self.dim_flag=='1D':
            plt.figure()
            if what=='amplitude':
                plt.plot(self.x1,np.abs(self.U1))
            elif what=='phase':
                plt.plot(self.x1,np.angle(self.U1))
            elif what=='intensity':
                plt.plot(self.x1,np.abs(self.U1)**2)
            else:
                print("what must be 'amplitude','intensity' or 'phase'")
                
        elif self.dim_flag=='2D':
            plt.figure()
            if what=='amplitude': 
                plt.pcolor(self.x1,self.y1,np.abs(self.U1))
            elif what=='phase':
                plt.pcolor(self.x1,self.y1,np.angle(self.U1))
            elif what=='intensity':
                plt.pcolor(self.x1,self.y1,np.abs(self.U1)**2)
            else:
                print("what must be 'amplitude','intensity' or 'phase'")
                
    def show_prop_yz(self,what='amplitude'):
        """shows self.Uyz : result of propagations at successive planes to follow the propagation
        Args:
            - what (string): flag to indicate what to plot (amplitude,phase or intensity) of the result field, by default is amplitude.
        Returns: 
            -none. 
        
        .. Note::
            - the function plots the resulting field using 'matplotlib'.
            - it shows 2D map of 'self.U2' representing propagation result at several planes defined in 'self.zz'.
            - sometimes (when not using Ipython) the function 'matplotlib.pylab.show()' must be used to show the plot result.
        Example of use 
        
            >>> opSys=FresnelProp() # creating propagation system (object)    
            >>> opSys.set_start_beam(tem00, x)# tem00 is a complex 1D field , 'x' (real) is initial plane  
            >>> #opSys.yz_prop_chart(5e3,50e3,100,30*x) # propagate the start field from Lmin=5mm to Lmax=50mm at 100 intermediate planes (linearly spaced ), result plane is 30x times the start one.
            >>> #opSys.show_prop_yz() # do the calculations
            >>> #opSys.show_prop_yz(what='intensity') # show intensity of resulting fields 
            >>> #plt.show()
        
        
        """
        if self.Uyz ==[]:
            print('the propagation chart is empty!')
            sys.exit(1)
        else:
            plt.figure()
            if what=='amplitude': 
                plt.pcolor(self.zz,self.x2,np.abs(self.Uyz))
            elif what=='phase':
                plt.pcolor(self.zz,self.x2,np.angle(self.Uyz))
            elif what=='intensity':
                plt.pcolor(self.zz,self.x2,np.abs(self.Uyz)**2)
            else:
                print("what must be 'amplitude','intensity' or 'phase'")
            
        
if __name__ == '__main__':
    pass
    
    