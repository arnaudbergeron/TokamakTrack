#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class B_field_Solver(object):
    
    def __init__(self, object):
        """ Initiate the solver with the important paramaters depending on the object
        
        Parameters
        ----------
        self.vect_dL : np.ndarray
            sequence of vectors along the wire
        self.mid_point_dL : np.ndarray
            reference point for each self.vect_dL
        """
        
        self.current = object.I
        self.curve = object.coordz.T
        
        self.vect_dL = self.curve_dSegment_vector(self.curve)
        self.mid_point_dL = self.curve_dSegment_midpoint(self.curve)
        
        return
    
    def curve_dSegment_vector(self, curve):
        """ Convert a discretize curve define with point in the sequences of vector that constitute the curve
        
        Parameters
        ----------
        curve : np.ndarray of shape (n,3)
            sequence of points

        Returns
        -------
        vect_dSegment : np.ndarray
            sequence of vectors
        """
        vect_dSegment = np.diff(curve, axis=0)
        
        return vect_dSegment


    def curve_dSegment_midpoint(self, curve):
        """ Take a discretize curve and return the midpoint of each segment that constitute the curve
    
        Parameters
        ----------
        curve : np.ndarray of shape (n,3)
            sequence of points
    
        Returns
        -------
        mid_point_dSegment : np.ndarray
        sequence of midpoints
        """
        mid_point_dSegment = (curve[:-1] + curve[1:])/2
    
        return mid_point_dSegment
    
    def Solve_B_field_at_a_point(self, point): 
        """ Solves B field at a given point
    
        Parameters
        ----------    
        point : np.ndarray

        Returns
        -------
        B_field : np.ndarray
            B-field for at the given point
        """
        
        #vector from the infinitesimal displacement vector along the wire to the data point
        vect_r = point - self.mid_point_dL
    
        dB = self.solve_dB(vect_r)
        B_field = np.sum(dB, axis=0)
    
        return B_field
    
    def solve_dB(self, vect_r):
        """ Solves dB according to Biot Savart
    
        Parameters
        ----------    
        vect_r : np.ndarray
            sequence of vectors from the data point to the reference point on the wire

        Returns
        -------
        dB : np.ndarray of shape (n,3)
            infinitesimal B-field for each reference point
        """
        norm_r3 = np.linalg.norm(vect_r, axis=1)**3
        vect_dL_cross_vect_r = np.cross(self.vect_dL, vect_r)
        
        dB = self.current*(vect_dL_cross_vect_r.T/norm_r3).T
        
        return dB
    
    def Solve_B_field_for_data_points(self, data):
        """ Solves B field for a set of data point
    
        Parameters
        ----------    
        data : np.ndarray

        Returns
        -------
        B_field : np.ndarray of shape (n,3)
            B-field at each data points
        """
        B_field = np.apply_along_axis(self.Solve_B_field_at_a_point, axis=1, arr=data)
        return B_field

