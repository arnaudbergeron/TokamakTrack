#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy import integrate

class Wire:
    '''
    Implements an arbitrary shaped wire
    '''
    #coordz = np.array([])
    '''Coordinates of the vertex of the wire in the form [X,Y,Z]'''
    #I = 1
    '''Complex current carried by the wire'''

    def __init__(self):
        '''
        By default initited nothing
        '''
        return

    def Set_Current(self, current):
        '''Sets current in wire'''
        self.I = current
        return

    def Create_Toroidal_Coil(self, R1, R2, N, step):
        '''
        Create_Toroidal_Coil( R1 , R2 , N , step )
        Creates a toroidal coil of major radius R1, minor radius R2 with N
         turns and a step step
         Initiates coordz
        '''
        a = R1
        b = R2
        c = N

        t = np.r_[0:2 * np.pi:step]

        X = (a + b * np.sin(c * t)) * np.cos(t);
        Y = (a + b * np.sin(c * t)) * np.sin(t);
        Z = b * np.cos(c * t);

        self.coordz = np.array([X, Y, Z])

        return

    def Create_Solenoid(self, R, N, l, step):
        '''
        Create_Solenoid(self, R , N , l , step )
        Creates a solenoid whose length is l with radius R, N turns with step
        step along the z axis
        '''
        a = R;
        b = l / (2 * np.pi * N);
        T = l / b;

        t = np.r_[0:T:step]

        X = a * np.cos(t);
        Y = a * np.sin(t);
        Z = b * t;
        
        self.coordz = np.array([X, Y, Z])
        
        return

    def Create_Loop(self, center, radius, NOP, theta, Orientation='xy'):
        '''
        Create_Loop(self,center,radius,NOP)
        a circle with center defined as
        a vector CENTER, radius as a scaler RADIS. NOP is 
        the number of points on the circle.
        '''
        t = np.linspace(0, 2 * np.pi, NOP)

        if Orientation == 'xy':
            X = center[0] + radius * np.sin(t)
            Y = center[1] + radius * np.cos(t)
            Z = np.zeros(NOP)
        elif Orientation == 'xz':
            X = center[0] + radius * np.sin(t)
            Z = center[1] + radius * np.cos(t)
            Y = np.zeros(NOP)
        elif Orientation == 'yz':
            Y = center[0] + radius * np.sin(t)
            Z = center[1] + radius * np.cos(t)
            X = np.zeros(NOP)
        
        XYZ_add = self.Rotation_Z(np.array([X, Y, Z]), theta)
        
        try:
            self.coordz
        except AttributeError:
            self.coordz = XYZ_add
        else:
            XYZ = self.coordz
            self.coordz = np.concatenate((XYZ, XYZ_add), axis=1)
        
        return
    
    def Create_D_Shape_coil(self, theta):
        
        def create_d_shaped_coil(r_in, r_out, x_in):
            #returns 1 branch of the d shaped electromagnetic coil with set inner and outer radii
            #https://lss.fnal.gov/conf/C720919/p240.pdf

            k = np.log(r_out/r_in)

            def d_coil_func(x):
                return(2*np.log(x)/(((k**2)-4*(np.log(x)**2))**0.5))

            def calculate_y(x):
                return integrate.quad(d_coil_func, r_in, x)

            vect_y = np.vectorize(calculate_y)
            return vect_y(x_in)[0]
        
        def Rotation_Z(theta):
            return np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta), np.cos(theta), 0],
                             [0, 0, 1]])
        
        x = np.linspace(0.5,2,1000)
        z = create_d_shaped_coil(1,4, x)

        X = np.concatenate((x,np.flip(x),np.zeros(100)+0.5))
        Z = np.concatenate((-z+2*z[-1],np.flip(z),np.linspace(z[0],-z[0]+2*z[-1],100))) - max(z)
        Y = np.zeros(len(X))
        
        self.coordz = Rotation_Z(theta) @ np.array([X,Y,Z])
        
        return
        
    
    def Transform_Shift(self, x_shift, y_shift, z_shift):
        
        X = self.coordz[0] + x_shift
        Y = self.coordz[1] + y_shift
        Z = self.coordz[2] + z_shift
        
        self.coordz = np.array([X, Y, Z])
        
        return
    
    def Rotation_Z(self, vect, theta):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        return rotation_matrix @ vect

