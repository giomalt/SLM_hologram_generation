'''
Created on 11 mars 2014

@author: Mohamed seghilani
'''
import opencavity
import numpy as np
import math

class UtilsFunc(object):
    '''
    Utility functions to be called from other classes
    '''


    def __init__(self):
        '''
        Constructor
        '''
    def greet(self):
        '''
        test funtion
        '''
        print('hiii from utils fnctions')
        
        
    def hermite_poly(self,n):
        '''
    for more info see 'http://suinotes.wordpress.com/2010/05/26/hermite-polynomials-with-matlab/
        function h = hermite_rec(n)
        if( 0==n ), h = 1;
        elseif( 1==n ), h = [2 0];
        else
           h1 = zeros(1,n+1);
           h1(1:n) = 2*hermite_rec(n-1);
        
           h2 = zeros(1,n+1);
           h2(3:end) = 2*(n-1)*hermite_rec(n-2);
        
           h = h1 - h2;
        end
    '''    
        if (n==0):
            h=1
        elif (n==1):
            h=np.array([2,0])
        else:
            h1=np.zeros(n+1)
            h1[0:n]=2*self.hermite_poly(n-1);
            
            h2=np.zeros(n+1)
            h2[2:np.size(h2)]=2*(n-1)*self.hermite_poly(n-2)
            
            h=h1-h2
                 
        
        return h
    
    def hermite(self,n,x):
        'evaluate the hermite polynomial at a given x'
        """
        h = hermite_rec(n);
        y = h(end);
        p = 1;
        for i=length(h)-1:-1:1
            y = y + h(i) * x(:).^p;
            p = p+1;
        end
        
        % restore the shape of y, the same as x
        y = reshape(y,size(x));
        """
        'or simply by calling numpy.polyval'
        if( n<0 ):
            print('The order of Hermite polynomial must be greater than or equal to 0.')
        elif(0!=int(n)-n):
            print('The order of Hermite polynomial must be an integer.')
        elif(n==0):
            y=np.ones(np.size(x))
            return y
        else:
            h=self.hermite_poly(n)
            y=np.polyval(h, x)
            return y

    def gauss_legendre(self,m,tol=10e-9):
        """
        x,A = gauss_legendre(m,tol=10e-9)
        Returns nodal abscissas {x} and weights {A} of
        Gauss-Legendre m-point quadrature.
        http://w3mentor.com/learn/python/scientific-computation/gauss-legendre-m-point-quadrature-in-python/
        """
 
        def legendre(t,m):
            p0 = 1.0; p1 = t
            for k in range(1,m):
                p = ((2.0*k + 1.0)*t*p1 - k*p0)/(1.0 + k )
                p0 = p1; p1 = p
            dp = m*(p0 - t*p1)/(1.0 - t**2)
            return p,dp
     
        A = np.zeros(m)   
        x = np.zeros(m)   
        nRoots = (m + 1)/2          # Number of non-neg. roots
        for i in range(nRoots):
            t = math.cos(math.pi*(i + 0.75)/(m + 0.5))  # Approx. root
            for j in range(30): 
                p,dp = legendre(t,m)          # Newton-Raphson
                dt = -p/dp; t = t + dt        # method         
                if abs(dt) < tol:
                    x[i] = t; x[m-i-1] = -t
                    A[i] = 2.0/(1.0 - t**2)/(dp**2) # Eq.(6.25)
                    A[m-i-1] = A[i]
                    j # just to remove the warning: unused variable j
                    break
        return x,A
    
    
    def find_nearest(self,array,value):
        idx = (np.abs(array-value)).argmin()
        
        return idx
    
        
    def tic(self):
        #Homemade version of matlab tic and toc functions
        import time
        global startTime_for_tictoc
        startTime_for_tictoc = time.time()

    def toc(self):
        import time
        if 'startTime_for_tictoc' in globals():
            print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        else:
            print ("Toc: start time not set")


if __name__=='__main__':
    'test the functions'
    c=UtilsFunc()
    c.greet()
    d=c.hermite_poly(0)
    e=c.hermite(0,2)
    x=np.array([1,2,3,4])
    f=c.hermite(3, x)
    g,w=c.gauss_legendre(6)
    print(d)
    print(e)
    print(f)
    print(g*6)
    print(w)
    
else:
    #print("utilsfunction module imported")
    pass
