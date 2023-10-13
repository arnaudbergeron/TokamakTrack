#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from BSSolver import B_field_Solver

class Particle_Path_Solver(object):
    
    def __init__(self, object):
        self.charge = object[0]
        self.mass = 1
        
        self.B_field_Solver_for_object = []
        for myWire in object[1:]:
            self.B_field_Solver_for_object.append(B_field_Solver(myWire))
        return
    
    def acceleration_in_B_field(self, x, v):
        """ Acceleration in B-field due to Lorentz Force
        
        Parameters
        ----------
        x : np.ndarray
            position
        v : np.ndarray
            velocity 
        
        Returns
        -------
        a : np.ndarray
            acceleration
        """
        
        B_field_at_x_in_object = []
        for Solver in self.B_field_Solver_for_object:
            B_field_at_x_in_object.append(Solver.Solve_B_field_at_a_point(x))
        
        B_field_at_x = np.sum(B_field_at_x_in_object, axis=0)
        
        F = self.charge * np.cross(v, B_field_at_x)
        
        return F / self.mass
    
    
    def rk4(self, x0, v0, dt):
        """ Runge-kutta.

        Parameters
        ----------
        x0 : np.ndarray
            position 
        v0 : np.ndarray
            velocity 
        dt : float
            time step
        
        Returns
        -------
        x : np.ndarray, v : np.ndarray
        
        Source
        ------
        http://doswa.com/2009/01/02/fourth-order-runge-kutta-numerical-integration.html
        """
        a0 = self.acceleration_in_B_field(x0, v0)
        
        x1 = x0 + 0.5 * v0 * dt
        v1 = v0 + 0.5 * a0 * dt
        a1 = self.acceleration_in_B_field(x1, v1)
        
        x2 = x0 + 0.5 * v1 * dt
        v2 = v0 + 0.5 * a1 * dt
        a2 = self.acceleration_in_B_field(x2, v2)
        
        x3 = x0 + v2 * dt
        v3 = v0 + a2 * dt
        a3 = self.acceleration_in_B_field(x3, v3)
        
        x4 = x0 + (dt / 6) * (v0 + 2*v1 + 2*v2 + v3)
        v4 = v0 + (dt / 6) * (a0 + 2*a1 + 2*a2 + a3)
        
        return x4, v4
    
    
    def trajectory(self, t0, x0, v0, dt, max_iterations):
        """ step-by-step 3D particle trajectory
        
        Parameters
        ----------
        t0 : float
            initial time
        x0 : np.ndarray
            initial position 
        v0 : np.ndarray
            initial velocity 
        dt : float
            time step
        max_iterations : int
            number of steps
        
        Returns
        -------
        np.array([['time', 'x', 'y', 'z']])
        """
        # initialise
        i = 1
        t = t0
        x = np.array(x0)
        v = np.array(v0)
        result = [[t, x[0], x[1], x[2]]]
        
        # step-by-step trajectory
        while i < max_iterations:
            x, v = self.rk4(x, v, dt)
            t += dt
            
            
            # If the particle escapes from the inside of tore, cease calculations
            distance_xy = np.linalg.norm(x[:2])
            distance_z = x[2]
            if (distance_xy > 1.5) or (distance_xy < 0.5) or (distance_z > 0.5):
                print('collision')
                break
            
            
            # record
            result.append([t, x[0], x[1], x[2]])
            # next step
            i += 1
        
        # output
        return np.array(result)

