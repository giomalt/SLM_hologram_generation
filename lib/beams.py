'''
Created on 9 mars 2014

@author: Mohamed seghilani
'''

import numpy as np 
import math
from opencavity import utilsfunc
class HgBasis(object):
    '''
    Generation of Hermite Gauss beams 
    '''
    
    def __init__(self,wavelength,w0x,w0y=0):
        '''
        Constructor
        '''
        self.k=2*np.pi/wavelength
        self.w0x=w0x
        self.w0y=w0y
        self.wavelength=wavelength
        self.zrx=np.pi*w0x**2/wavelength
        self.zry=np.pi*w0y**2/wavelength
        
    def greet(self):
        print ('hii')
        
    def Wx(self,z):
        "x waist evolution in z"
        if self.w0x==0:
            print("W0x=0 ! division by '0'")
        else:
            y=self.w0x*np.sqrt(1+(z/self.zrx)**2)
            return y
    
    def Wy(self,z):
        "y waist evolution in z"
        if self.w0y ==0:
            print("W0y=0 ! division by '0'")
        else:
            y=self.w0x*np.sqrt(1+(z/self.zry)**2)
            return y
        
    def Rx(self,z):
        "radius of curvature of the beam evolution"
        if self.w0x==0:
            print("W0x=0 ! division by '0'")
        else:
            y=z+(self.zrx**2)/z
            return y        
    
    def Ry(self,z):
        "radius of curvature of the beam evolution"
        if self.w0y==0:
            print("W0y=0 ! division by '0'")
        else:
            y=z+(self.zry**2)/z
            return y
    
    def Phix(self,z):
        if self.w0x==0:
            print("W0x=0 ! division by '0'")
        else:
            y=np.arctan(z/self.zrx)
            return y

    def Phiy(self,z):
        if self.w0y==0:
            print("W0y=0 ! division by '0'")
        else:
            y=np.arctan(z/self.zry)
            return y
    
    def Zb(self,r,z):
        y=z-self.Rx(z)+np.sqrt(self.Rx(z)**0-r**2)
        return y      

    def generate_hg(self,m,p,x,y,z):
        'generate a Hermite Gauss beam of order m,p'
        w0x=self.w0x
        #w0y=self.w0y
        wavelength=self.wavelength
        k=2*np.pi/wavelength
        Func=utilsfunc.UtilsFunc()
        
        
        y1=w0x*np.sqrt(np.exp(1j*(2*m+1)*self.Phix(z)+1j*(2*p+1)*self.Phiy(z))/((2**m)*math.factorial(m)*self.Wx(z)*(2**p)*math.factorial(p)*self.Wy(z)))
        y2=Func.hermite(m, np.sqrt(2)*(x/self.Wx(z)))
        y3=Func.hermite(p, np.sqrt(2)*(y/self.Wy(z)))
        y4=np.exp(-1j*k*z-1j*k*((x**2)/(2*self.Rx(z))+(y**2)/(2*self.Ry(z)))-((x**2)/self.Wx(z)**2+(y**2)/self.Wy(z)**2))
        print(y2.shape,y3.shape,y4.shape)
        #print(np.size(y))
        if m==0 and np.size(x)>1 and np.size(y)>1:
            y2=y2.reshape(y4.shape[0],y4.shape[1])
        if p==0 and np.size(y)>1 and np.size(x)>1:
            y3=y3.reshape(y4.shape[0],y4.shape[1])
            
        print(y2.shape,y3.shape,y4.shape)
        
        y=y1*y2*y3*y4
        return y               
    
    def generate_h(self,m,p,x,y,z):
        'generate a Hermite Gauss beam of order m,p'
        w0x=self.w0x
        #w0y=self.w0y
        wavelength=self.wavelength
        k=2*np.pi/wavelength
        Func=utilsfunc.UtilsFunc()
        
        
        y2=Func.hermite(m, np.sqrt(2)*(x/self.Wx(z)))
        y3=Func.hermite(p, np.sqrt(2)*(y/self.Wy(z)))
        #print(y2.shape,y3.shape,y4.shape)
        #print(np.size(y))
        if m==0 and np.size(x)>1 and np.size(y)>1:
            y2=y2.reshape(x.shape[0],x.shape[1])
        if p==0 and np.size(y)>1 and np.size(x)>1:
            y3=y3.reshape(x.shape[0],x.shape[1]) #it could be used the dimension of y, it does not matter
            
        print(y2.shape,y3.shape)
        
        y=y2*y3
        return y               

    def generate_hybrid(self,m,p,x,y,z,winx):
        'generate a Hermite polynomial times a Gauss beam with different waist of order m,p'

        w0x=self.w0x
        winy=winx
        #w0y=self.w0y
        wavelength=self.wavelength
        k=2*np.pi/wavelength
        Func=utilsfunc.UtilsFunc()
        
        
        y1=w0x*np.sqrt(np.exp(1j*(2*m+1)*self.Phix(z)+1j*(2*p+1)*self.Phiy(z))/((2**m)*math.factorial(m)*self.Wx(z)*(2**p)*math.factorial(p)*self.Wy(z)))
        y2=Func.hermite(m, np.sqrt(2)*(x/self.Wx(z)))
        y3=Func.hermite(p, np.sqrt(2)*(y/self.Wy(z)))
        y4=np.exp(-1j*k*z-1j*k*((x**2)/(2*self.Rx(z))+(y**2)/(2*self.Ry(z)))-((x**2)/winx**2+(y**2)/winy**2))
        #print(np.size(y))
        if m==0 and np.size(x)>1 and np.size(y)>1:
            y2=y2.reshape(y4.shape[0],y4.shape[1])
        if p==0 and np.size(y)>1 and np.size(x)>1:
            y3=y3.reshape(y4.shape[0],y4.shape[1])
            
        
        y=y1*y2*y3*y4
        return y               

    
class LgBasis(object):
    '''
    Generation of Hermite Gauss beams 
    '''
    
    def __init__(self,wavelength,w0):
        '''
        Constructor
        '''
        self.k=2*np.pi/wavelength
        self.w0=w0
        self.wavelength=wavelength
        self.zr=np.pi*w0**2/wavelength
    
    def generate_lg(self,p,m,x,y,z):
        'generate a Hermite Gauss beam of order m,p'
        
        r2=x**2+y**2
        phi0=0
        theta = np.arctan2(y,x) + phi0;
        
        w0=self.w0
        k=self.k
        w=self.w0*np.sqrt(1+(z/self.zr)**2)
        R=z+(self.zr**2)/z
        phi=np.arctan(z/self.zr)
        
        if m==0:
            delta=1
        else:
            delta=0
        
        Xr=2*r2/self.w0**2
        
        Lpm=0
        
        for i in range(p+1):
            A=((-1)**i)*math.factorial(p+m)*(Xr**i)/(math.factorial(p-i)*math.factorial(m+i)*math.factorial(i))
            Lpm=Lpm+A
          
        E=np.sqrt((2*math.factorial(p))/ ((1+delta)*np.pi*math.factorial(p+m)))*(w0/w**2)*np.exp(-(r2/w**2))*Lpm*((np.sqrt(2*r2)/w)**m)*np.exp(1j*(m*theta-k*z+phi*(1+m+2*p)-(k*r2/(2*R))))             
        #E=sqrt((2*factorial(p))/((1+delta)*pi*factorial(p+m)))*(w0/w)*exp(-(r2/w^2)).*Lpm.*((sqrt(2*r2)/w).^m).*exp(1i*(m*theta-k*z+phi*(1+m+2*p)-(k*r2/(2*R))));
        return E
    
if __name__ == '__main__':
    import matplotlib.pylab as plt

        
    H=HgBasis(1,30,30)
    z=0.01
    x=np.linspace(-100, 100, 200)
    y=x
    #in 1D
    tem00=H.generate_hg(0,1, 0,y, z)
    print(tem00.shape)
    plt.plot(x,abs(tem00))
    plt.show()
    
     
#     X,Y=np.meshgrid(x,y)
#     tem00=H.generate_hg(0, 1, X, Y, z)+H.generate_hg(1, 0, X, Y, z)*np.exp(-1j*math.pi/2)
#     plt.set_cmap('hot')
#     plt.imshow(np.abs(tem00),extent=[-100,100,-100,100])
#     plt.show()
#     plt.Figure
#     plt.imshow(np.angle(tem00),extent=[-100,100,-100,100])
#     plt.show()
    
    
