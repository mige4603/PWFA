#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:21:59 2019

@author: michael
"""

import sys
sys.path.append('/home/michael/Documents/PWFA/Decay_Model/')

import h5py as h5
import numpy as np
import scipy.interpolate as sci

import global_variables as var

def initial_condition(x, y, z):
    """ Returns initial plasma/neutral gas density.
    
    Parameters
    ----------
    x : array
        x value within each cell
        
    y : array
        y value within each cell
        
    z : array
        z value within each cell
    """
    r2 = x**2 + y**2
    
    n = np.exp(-.5 * var.delta * r2)
    
    plasma_den = n
    neutral_den = 1. - plasma_den
    
    return plasma_den, neutral_den

def import_initial_conditions(fileName):
    """ Returns initial plasma/neutral gas density.
    
    Parameters
    ----------
    fileName : str
        location of h5 file where initial density is stored
    """
    myFile = h5.File(fileName, 'r')
    
    den = myFile['density data'][:]
    X_dom = myFile['X Domain'][:] / var.s
    Z_dom = myFile['Z Domain'][:] / var.s
    
    myFile.close()
    
    density = sci.RectBivariateSpline(Z_dom, X_dom, den)
    
    plasma_density = np.empty((var.Rpts, var.Rpts, var.Zpts))
    for x_ind, x in enumerate(X_dom):
        for y_ind, y in enumerate(X_dom):
            r = np.sqrt(x**2 + y**2)
            for z_ind, z in enumerate(Z_dom):
                plasma_density[x_ind, y_ind, z_ind] = density(z, r)[0][0]
                
    neutral_density = np.full((var.Rpts, var.Rpts, var.Zpts), 1.) - plasma_density     
        
    return plasma_density, neutral_density  

def calculate_sigma_psi(rel_vel_x, rel_vel_y, rel_vel_z, neut_den):
    """ Calculate the sigma and psi values need at each step of the 
    integrable function.
    
    Parameters
    ----------
    rel_vel : array
        relative fluid velocities (V_n - V_p) in each grid point
        
    neut_den : array
        neutral gas density at each grid point
    """
    vel_in = - np.dstack((rel_vel_x, rel_vel_y, rel_vel_z))
    vel_in_2 = np.linalg.norm(vel_in)
    
    vel_cx = 2.54648 * var.v_thermal_sq + (vel_in_2)
    sigma = var.sigma_func(vel_cx, var.v, var.source)

    psi_base = - (var.v_thermal_sq * neut_den) / np.sqrt( 7.06858 * var.v_thermal_sq + 2 * (vel_cx*vel_cx + vel_in_2) )
    
    psi_x = rel_vel_x * psi_base
    psi_y = rel_vel_y * psi_base
    psi_z = rel_vel_z * psi_base
    
    return sigma, psi_x, psi_y, psi_z
    
def periodic_boundary_den(in_put):    
    """ Returns spliced density domain for periodic boundary conditions.
    
    Parameters
    ----------
    in_put : array
        plasma/neutral gas density
    """    
    out_put = np.empty(in_put.shape)
    out_put[1:-1, 1:-1, 1:-1] = in_put[1:-1, 1:-1, 1:-1]    
    
    out_put[0::, 0, 0::] = in_put[0::, -2, 0::]
    out_put[0::, -1, 0::] = in_put[0::, 1, 0::]
    
    out_put[0, 0::, 0::] = in_put[-2, 0::, 0::]
    out_put[-1, 0::, 0::] = in_put[1, 0::, 0::]
    
    return out_put
    
def periodic_boundary_vel_x(in_put):  
    """ Returns spliced x velocity domain for periodic boundary conditions.
    
    Parameters
    ----------
    in_put : array
        fluid velocity in x direction
    """
    out_put = np.zeros(in_put.shape)
    out_put[1:-1, 1:-1, 1:-1] = in_put[1:-1, 1:-1, 1:-1]
    
    out_put[0::, 0, 0::] = in_put[0::, -2, 0::]
    out_put[0::, -1, 0::] = in_put[0::, 1, 0::]
    
    out_put[0, 0::, 0::] = -in_put[-2, 0::, 0::]
    out_put[-1, 0::, 0::] = -in_put[1, 0::, 0::]
    
    return out_put

def periodic_boundary_vel_y(in_put):
    """ Returns spliced y velocity domain for periodic boundary conditions.
    
    Parameters
    ----------
    in_put : array
        fluid velocity in y direction
    """
    out_put = np.zeros(in_put.shape)
    out_put[1:-1, 1:-1, 1:-1] = in_put[1:-1, 1:-1, 1:-1]
    
    out_put[0::, 0, 0::] = -in_put[0::, -2, 0::]
    out_put[0::, -1, 0::] = -in_put[0::, 1, 0::]
    
    out_put[0, 0::, 0::] = in_put[-2, 0::, 0::]
    out_put[-1, 0::, 0::] = in_put[1, 0::, 0::]
    
    return out_put
    
def periodic_boundary_vel_z(in_put):
    """ Returns spliced z velocity domain for periodic boundary conditions.
    
    Parameters
    ----------
    in_put : array
        fluid velocity in z direction
    """
    out_put = np.zeros(in_put.shape)
    out_put[1:-1, 1:-1, 1:-1] = in_put[1:-1, 1:-1, 1:-1]
    
    out_put[0::, 0, 0::] = in_put[0::, -2, 0::]
    out_put[0::, -1, 0::] = in_put[0::, 1, 0::]
    
    out_put[0, 0::, 0::] = in_put[-2, 0::, 0::]
    out_put[-1, 0::, 0::] = in_put[1, 0::, 0::]
        
    return out_put

def calculate_density(den, vel_x, vel_y, vel_z, den_sh, Zsize, periodic):
    """ Returns time derivative of fluid density.
    
    Parameters
    ----------
    den : array
        fluid density
        
    vel_x : array
        fluid velocity in x direction    
    
    vel_y : array
        fluid velocity in y direction    

    vel_z : array
        fluid velocity in z direction    

    den_sh : array
        source/sink terms in continuity equation
        
    Zsize : int
        number of dimensions along axis=2
    """
    if periodic:
        # Define Data Size
        hv = np.zeros(den.shape)
        den_out = np.zeros(den.shape)
        
        # Calculate Central Differences
        hv[1:-1, 1:-1, 1:-1] = var.hv_den_r * ( (den[2::, 1:-1, 1:-1] - 2*den[1:-1, 1:-1, 1:-1] + den[0:-2, 1:-1, 1:-1]) + (den[1:-1, 2::, 1:-1] - 2*den[1:-1, 1:-1, 1:-1] + den[1:-1, 0:-2, 1:-1]) ) + var.hv_den_z * var.size_ratio * (den[1:-1, 1:-1, 2::] - 2*den[1:-1, 1:-1, 1:-1] + den[1:-1, 1:-1, 0:-2]) 
        den_out[1:-1, 1:-1, 1:-1] = vel_x[1:-1, 1:-1, 1:-1] * (den[2::, 1:-1, 1:-1] - den[0:-2, 1:-1, 1:-1]) + vel_y[1:-1, 1:-1, 1:-1] * (den[1:-1, 2::, 1:-1] - den[1:-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[1:-1, 1:-1, 1:-1] * (den[1:-1, 1:-1, 2::] - den[1:-1, 1:-1, 0:-2]) + den[1:-1, 1:-1, 1:-1] * (vel_x[2::, 1:-1, 1:-1] - vel_x[0:-2, 1:-1, 1:-1] + vel_y[1:-1, 2::, 1:-1] - vel_y[1:-1, 0:-2, 1:-1] + var.size_ratio * (vel_z[1:-1, 1:-1, 2::] - vel_z[1:-1, 1:-1, 0:-2]) )

        # Out Put Calculation
        den_out[1:-1, 1:-1, 1:-1] = den_out[1:-1, 1:-1, 1:-1] - hv[1:-1, 1:-1, 1:-1]
        den_out[1:-1, 1:-1, 1:-1] = den[1:-1, 1:-1, 1:-1] - 0.5 * var.step_ratio * den_out[1:-1, 1:-1, 1:-1] - den_sh[1:-1, 1:-1, 1:-1]
        
        # Periodic Boundaries
        den_out = periodic_boundary_den(den_out)
     
    else:
        # Define Data Size
        hv = np.empty(den.shape)
        den_out = np.empty(den.shape)
        
        # Central hv
        hv[1:-1, 1:-1, 1:-1] = var.hv_den_r * ( (den[2::, 1:-1, 1:-1] - 2*den[1:-1, 1:-1, 1:-1] + den[0:-2, 1:-1, 1:-1]) + (den[1:-1, 2::, 1:-1] - 2*den[1:-1, 1:-1, 1:-1] + den[1:-1, 0:-2, 1:-1]) ) + var.hv_den_z * var.size_ratio * (den[1:-1, 1:-1, 2::] - 2*den[1:-1, 1:-1, 1:-1] + den[1:-1, 1:-1, 0:-2]) 
        
        # X Boundary hv
        hv[0, 1:-1, 1:-1] = var.hv_den_r * ( var.hv_den_bnd * (den[0, 1:-1, 1:-1] - 2*den[1, 1:-1, 1:-1] + den[2, 1:-1, 1:-1]) + (den[0, 2::, 1:-1] - 2*den[0, 1:-1, 1:-1] + den[0, 0:-2, 1:-1]) ) + var.hv_den_z * var.size_ratio * (den[0, 1:-1, 2::] - 2*den[0, 1:-1, 1:-1] + den[0, 1:-1, 0:-2]) 
        hv[-1, 1:-1, 1:-1] = var.hv_den_r * ( var.hv_den_bnd * (den[-1, 1:-1, 1:-1] - 2*den[-2, 1:-1, 1:-1] + den[-3, 1:-1, 1:-1]) + (den[-1, 2::, 1:-1] - 2*den[-1, 1:-1, 1:-1] + den[-1, 0:-2, 1:-1]) ) + var.hv_den_z * var.size_ratio * (den[-1, 1:-1, 2::] - 2*den[-1, 1:-1, 1:-1] + den[-1, 1:-1, 0:-2])
        
        # Y Boundary hv
        hv[1:-1, 0, 1:-1] = var.hv_den_r * ( (den[2::, 0, 1:-1] - 2*den[1:-1, 0, 1:-1] + den[0:-2, 0, 1:-1]) + var.hv_den_bnd * (den[1:-1, 0, 1:-1] - 2*den[1:-1, 1, 1:-1] + den[1:-1, 2, 1:-1]) ) + var.hv_den_z * var.size_ratio * (den[1:-1, 0, 2::] - 2*den[1:-1, 0, 1:-1] + den[1:-1, 0, 0:-2])
        hv[1:-1, -1, 1:-1] = var.hv_den_r * ( (den[2::, -1, 1:-1] - 2*den[1:-1, -1, 1:-1] + den[0:-2, -1, 1:-1]) + var.hv_den_bnd * (den[1:-1, -1, 1:-1] - 2*den[1:-1, -2, 1:-1] + den[1:-1, -3, 1:-1]) ) + var.hv_den_z * var.size_ratio * (den[1:-1, -1, 2::] - 2*den[1:-1, -1, 1:-1] + den[1:-1, -1, 0:-2]) 
        
        # Z Boundary hv
        hv[1:-1, 1:-1, 0] = var.hv_den_r * ( (den[2::, 1:-1, 0] - 2*den[1:-1, 1:-1, 0] + den[0:-2, 1:-1, 0]) + (den[1:-1, 2::, 0] - 2*den[1:-1, 1:-1, 0] + den[1:-1, 0:-2, 0]) ) + var.hv_den_z * var.size_ratio * var.hv_den_bnd * (den[1:-1, 1:-1, 0] - 2*den[1:-1, 1:-1, 1] + den[1:-1, 1:-1, 2])
        hv[1:-1, 1:-1, -1] = var.hv_den_r * ( (den[2::, 1:-1, -1] - 2*den[1:-1, 1:-1, -1] + den[0:-2, 1:-1, -1]) + (den[1:-1, 2::, -1] - 2*den[1:-1, 1:-1, -1] + den[1:-1, 0:-2, -1]) ) + var.hv_den_z * var.size_ratio * var.hv_den_bnd * (den[1:-1, 1:-1, -1] - 2*den[1:-1, 1:-1, -2] + den[1:-1, 1:-1, -3])
        
        # Edge Boundaries hv
        hv[0, 0, 1:-1] = var.hv_den_r * ( var.hv_den_edg * (den[0, 0, 1:-1] - 2*den[1, 0, 1:-1] + den[2, 0, 1:-1]) + var.hv_den_edg * (den[0, 0, 1:-1] - 2*den[0, 1, 1:-1] + den[0, 2, 1:-1]) ) + var.hv_den_z * var.size_ratio * (den[0, 0, 2::] - 2*den[0, 0, 1:-1] + den[0, 0, 0:-2]) 
        hv[0, -1, 1:-1] = var.hv_den_r * ( var.hv_den_edg * (den[0, -1, 1:-1] - 2*den[1, -1, 1:-1] + den[2, -1, 1:-1]) + var.hv_den_edg * (den[0, -1, 1:-1] - 2*den[0, -2, 1:-1] + den[0, -3, 1:-1]) ) + var.hv_den_z * var.size_ratio * (den[0, -1, 2::] - 2*den[0, -1, 1:-1] + den[0, -1, 0:-2]) 
        hv[-1, 0, 1:-1] = var.hv_den_r * ( var.hv_den_edg * (den[-1, 0, 1:-1] - 2*den[-2, 0, 1:-1] + den[-3, 0, 1:-1]) + var.hv_den_edg * (den[-1, 0, 1:-1] - 2*den[-1, 1, 1:-1] + den[-1, 2, 1:-1]) ) + var.hv_den_z * var.size_ratio * (den[-1, 0, 2::] - 2*den[-1, 0, 1:-1] + den[-1, 0, 0:-2]) 
        hv[-1, -1, 1:-1] = var.hv_den_r * ( var.hv_den_edg * (den[-1, -1, 1:-1] - 2*den[-2, -1, 1:-1] + den[-3, -1, 1:-1]) + var.hv_den_edg * (den[-1, -1, 1:-1] - 2*den[-1, -2, 1:-1] + den[-1, -3, 1:-1]) ) + var.hv_den_z * var.size_ratio * (den[-1, -1, 2::] - 2*den[-1, -1, 1:-1] + den[-1, -1, 0:-2]) 
        
        hv[0, 1:-1, 0] = var.hv_den_r * ( var.hv_den_edg * (den[0, 1:-1, 0] - 2*den[1, 1:-1, 0] + den[2, 1:-1, 0]) + (den[0, 2::, 0] - 2*den[0, 1:-1, 0] + den[0, 0:-2, 0]) ) + var.hv_den_z * var.size_ratio * var.hv_den_edg * (den[0, 1:-1, 0] - 2*den[0, 1:-1, 1] + den[0, 1:-1, 2]) 
        hv[-1, 1:-1, 0] = var.hv_den_r * ( var.hv_den_edg * (den[-1, 1:-1, 0] - 2*den[-2, 1:-1, 0] + den[-3, 1:-1, 0]) + (den[-1, 2::, 0] - 2*den[-1, 1:-1, 0] + den[-1, 0:-2, 0]) ) + var.hv_den_z * var.size_ratio * var.hv_den_edg * (den[-1, 1:-1, 0] - 2*den[-1, 1:-1, 1] + den[-1, 1:-1, 2]) 
        hv[1:-1, 0, 0] = var.hv_den_r * ( (den[2::, 0, 0] - 2*den[1:-1, 0, 0] + den[0:-2, 0, 0]) + var.hv_den_edg * (den[1:-1, 0, 0] - 2*den[1:-1, 1, 0] + den[1:-1, 2, 0]) ) + var.hv_den_z * var.size_ratio * var.hv_den_edg * (den[1:-1, 0, 0] - 2*den[1:-1, 0, 1] + den[1:-1, 0, 2]) 
        hv[1:-1, -1, 0] = var.hv_den_r * ( (den[2::, -1, 0] - 2*den[1:-1, -1, 0] + den[0:-2, -1, 0]) + var.hv_den_edg * (den[1:-1, -1, 0] - 2*den[1:-1, -2, 0] + den[1:-1, -3, 0]) ) + var.hv_den_z * var.size_ratio * var.hv_den_edg * (den[1:-1, -1, 0] - 2*den[1:-1, -1, 1] + den[1:-1, -1, 2]) 
        
        hv[0, 1:-1, -1] = var.hv_den_r * ( var.hv_den_edg * (den[0, 1:-1, -1] - 2*den[1, 1:-1, -1] + den[2, 1:-1, -1]) + (den[0, 2::, -1] - 2*den[0, 1:-1, -1] + den[0, 0:-2, -1]) ) + var.hv_den_z * var.size_ratio * var.hv_den_edg * (den[0, 1:-1, -1] - 2*den[0, 1:-1, -2] + den[0, 1:-1, -3]) 
        hv[-1, 1:-1, -1] = var.hv_den_r * ( var.hv_den_edg * (den[-1, 1:-1, -1] - 2*den[-2, 1:-1, -1] + den[-3, 1:-1, -1]) + (den[-1, 2::, -1] - 2*den[-1, 1:-1, -1] + den[-1, 0:-2, -1]) ) + var.hv_den_z * var.size_ratio * var.hv_den_edg * (den[-1, 1:-1, -1] - 2*den[-1, 1:-1, -2] + den[-1, 1:-1, -3]) 
        hv[1:-1, 0, -1] = var.hv_den_r * ( (den[2::, 0, -1] - 2*den[1:-1, 0, -1] + den[0:-2, 0, -1]) + var.hv_den_edg * (den[1:-1, 0, -1] - 2*den[1:-1, 1, -1] + den[1:-1, 2, -1]) ) + var.hv_den_z * var.size_ratio * var.hv_den_edg * (den[1:-1, 0, -1] - 2*den[1:-1, 0, -2] + den[1:-1, 0, -3]) 
        hv[1:-1, -1, -1] = var.hv_den_r * ( (den[2::, -1, -1] - 2*den[1:-1, -1, -1] + den[0:-2, -1, -1]) + var.hv_den_edg * (den[1:-1, -1, -1] - 2*den[1:-1, -2, -1] + den[1:-1, -3, -1]) ) + var.hv_den_z * var.size_ratio * var.hv_den_edg * (den[1:-1, -1, -1] - 2*den[1:-1, -1, -2] + den[1:-1, -1, -3]) 
        
        # Corner Boundaries hv
        hv[0, 0, 0] = var.hv_den_bnd * ( var.hv_den_r * ( (den[0,0,0] - 2*den[1,0,0] + den[2,0,0]) + (den[0,0,0] - 2*den[0,1,0] + den[0,2,0]) ) + var.hv_den_z * var.size_ratio * (den[0,0,0] - 2*den[0,0,1] + den[0,0,2]) )
        hv[-1, 0, 0] = var.hv_den_bnd * ( var.hv_den_r * ( (den[-1,0,0] - 2*den[-2,0,0] + den[-3,0,0]) + (den[-1,0,0] - 2*den[-1,1,0] + den[-1,2,0]) ) + var.hv_den_z * var.size_ratio * (den[-1,0,0] - 2*den[-1,0,1] + den[-1,0,2]) )
        hv[0, -1, 0] = var.hv_den_bnd * ( var.hv_den_r * ( (den[0,-1,0] - 2*den[1,-1,0] + den[2,-1,0]) + (den[0,-1,0] - 2*den[0,-2,0] + den[0,-3,0]) ) + var.hv_den_z * var.size_ratio * (den[0,-1,0] - 2*den[0,-1,1] + den[0,-1,2]) )
        hv[-1, -1, 0] = var.hv_den_bnd * ( var.hv_den_r * ( (den[-1,-1,0] - 2*den[-2,-1,0] + den[-3,-1,0]) + (den[-1,-1,0] - 2*den[-1,-2,0] + den[-1,-3,0]) ) + var.hv_den_z * var.size_ratio * (den[-1,-1,0] - 2*den[-1,-1,1] + den[-1,-1,2]) )
        
        hv[0, 0, -1] = var.hv_den_bnd * ( var.hv_den_r * ( (den[0,0,-1] - 2*den[1,0,-1] + den[2,0,-1]) + (den[0,0,-1] - 2*den[0,1,-1] + den[0,2,-1]) ) + var.hv_den_z * var.size_ratio * (den[0,0,-1] - 2*den[0,0,-2] + den[0,0,-3]) )
        hv[-1, 0, -1] = var.hv_den_bnd * ( var.hv_den_r * ( (den[-1,0,-1] - 2*den[-2,0,-1] + den[-3,0,-1]) + (den[-1,0,-1] - 2*den[-1,1,-1] + den[-1,2,-1]) ) + var.hv_den_z * var.size_ratio * (den[-1,0,-1] - 2*den[-1,0,-2] + den[-1,0,-3]) )
        hv[0, -1, -1] = var.hv_den_bnd * ( var.hv_den_r * ( (den[0,-1,-1] - 2*den[1,-1,-1] + den[2,-1,-1]) + (den[0,-1,-1] - 2*den[0,-2,-1] + den[0,-3,-1]) ) + var.hv_den_z * var.size_ratio * (den[0,-1,-1] - 2*den[0,-1,-2] + den[0,-1,-3]) )
        hv[-1, -1, -1] = var.hv_den_bnd * ( var.hv_den_r * ( (den[-1,-1,-1] - 2*den[-2,-1,-1] + den[-3,-1,-1]) + (den[-1,-1,-1] - 2*den[-1,-2,-1] + den[-1,-3,-1]) ) + var.hv_den_z * var.size_ratio * (den[-1,-1,-1] - 2*den[-1,-1,-2] + den[-1,-1,-3]) )
            
        # Central den_out
        den_out[1:-1, 1:-1, 1:-1] = vel_x[1:-1, 1:-1, 1:-1] * (den[2::, 1:-1, 1:-1] - den[0:-2, 1:-1, 1:-1]) + vel_y[1:-1, 1:-1, 1:-1] * (den[1:-1, 2::, 1:-1] - den[1:-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[1:-1, 1:-1, 1:-1] * (den[1:-1, 1:-1, 2::] - den[1:-1, 1:-1, 0:-2]) + den[1:-1, 1:-1, 1:-1] * (vel_x[2::, 1:-1, 1:-1] - vel_x[0:-2, 1:-1, 1:-1] + vel_y[1:-1, 2::, 1:-1] - vel_y[1:-1, 0:-2, 1:-1] + var.size_ratio * (vel_z[1:-1, 1:-1, 2::] - vel_z[1:-1, 1:-1, 0:-2]) )
        
        # X Boundary den_out
        den_out[0, 1:-1, 1:-1] = vel_x[0, 1:-1, 1:-1] * (den[1, 1:-1, 1:-1] - den[0, 1:-1, 1:-1]) + vel_y[0, 1:-1, 1:-1] * (den[0, 2::, 1:-1] - den[0, 0:-2, 1:-1]) + var.size_ratio * vel_z[0, 1:-1, 1:-1] * (den[0, 1:-1, 2::] - den[0, 1:-1, 0:-2]) + den[0, 1:-1, 1:-1] * (vel_x[1, 1:-1, 1:-1] - vel_x[0, 1:-1, 1:-1] + vel_y[0, 2::, 1:-1] - vel_y[0, 0:-2, 1:-1] + var.size_ratio * (vel_z[0, 1:-1, 2::] - vel_z[0, 1:-1, 0:-2]) )
        den_out[-1, 1:-1, 1:-1] = vel_x[-1, 1:-1, 1:-1] * (den[-1, 1:-1, 1:-1] - den[-2, 1:-1, 1:-1]) + vel_y[-1, 1:-1, 1:-1] * (den[-1, 2::, 1:-1] - den[-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[-1, 1:-1, 1:-1] * (den[-1, 1:-1, 2::] - den[-1, 1:-1, 0:-2]) + den[-1, 1:-1, 1:-1] * (vel_x[-1, 1:-1, 1:-1] - vel_x[-2, 1:-1, 1:-1] + vel_y[-1, 2::, 1:-1] - vel_y[-1, 0:-2, 1:-1] + var.size_ratio * (vel_z[-1, 1:-1, 2::] - vel_z[-1, 1:-1, 0:-2]) )
    
        # Y Boundary den_out
        den_out[1:-1, 0, 1:-1] = vel_x[1:-1, 0, 1:-1] * (den[2::, 0, 1:-1] - den[0:-2, 0, 1:-1]) + vel_y[1:-1, 0, 1:-1] * (den[1:-1, 1, 1:-1] - den[1:-1, 0, 1:-1]) + var.size_ratio * vel_z[1:-1, 0, 1:-1] * (den[1:-1, 0, 2::] - den[1:-1, 0, 0:-2]) + den[1:-1, 0, 1:-1] * (vel_x[2::, 0, 1:-1] - vel_x[0:-2, 0, 1:-1] + vel_y[1:-1, 1, 1:-1] - vel_y[1:-1, 0, 1:-1] + var.size_ratio * (vel_z[1:-1, 0, 2::] - vel_z[1:-1, 0, 0:-2]) )
        den_out[1:-1, -1, 1:-1] = vel_x[1:-1, -1, 1:-1] * (den[2::, -1, 1:-1] - den[0:-2, -1, 1:-1]) + vel_y[1:-1, -1, 1:-1] * (den[1:-1, -1, 1:-1] - den[1:-1, -2, 1:-1]) + var.size_ratio * vel_z[1:-1, -1, 1:-1] * (den[1:-1, -1, 2::] - den[1:-1, -1, 0:-2]) + den[1:-1, -1, 1:-1] * (vel_x[2::, -1, 1:-1] - vel_x[0:-2, -1, 1:-1] + vel_y[1:-1, -1, 1:-1] - vel_y[1:-1, -2, 1:-1] + var.size_ratio * (vel_z[1:-1, -1, 2::] - vel_z[1:-1, -1, 0:-2]) )
    
        # Z Boundary den_out
        den_out[1:-1, 1:-1, 0] = vel_x[1:-1, 1:-1, 0] * (den[2::, 1:-1, 0] - den[0:-2, 1:-1, 0]) + vel_y[1:-1, 1:-1, 0] * (den[1:-1, 2::, 0] - den[1:-1, 0:-2, 0]) + var.size_ratio * vel_z[1:-1, 1:-1, 0] * (den[1:-1, 1:-1, 1] - den[1:-1, 1:-1, 0]) + den[1:-1, 1:-1, 0] * (vel_x[2::, 1:-1, 0] - vel_x[0:-2, 1:-1, 0] + vel_y[1:-1, 2::, 0] - vel_y[1:-1, 0:-2, 0] + var.size_ratio * (vel_z[1:-1, 1:-1, 1] - vel_z[1:-1, 1:-1, 0]) )
        den_out[1:-1, 1:-1, -1] = vel_x[1:-1, 1:-1, -1] * (den[2::, 1:-1, -1] - den[0:-2, 1:-1, -1]) + vel_y[1:-1, 1:-1, -1] * (den[1:-1, 2::, -1] - den[1:-1, 0:-2, -1]) + var.size_ratio * vel_z[1:-1, 1:-1, -1] * (den[1:-1, 1:-1, -1] - den[1:-1, 1:-1, -2]) + den[1:-1, 1:-1, -1] * (vel_x[2::, 1:-1, -1] - vel_x[0:-2, 1:-1, -1] + vel_y[1:-1, 2::, -1] - vel_y[1:-1, 0:-2, -1] + var.size_ratio * (vel_z[1:-1, 1:-1, -1] - vel_z[1:-1, 1:-1, -2]) )
        
        # Edge Boundary den_out
        den_out[0, 0, 1:-1] = vel_x[0, 0, 1:-1] * (den[1, 0, 1:-1] - den[0, 0, 1:-1]) + vel_y[0, 0, 1:-1] * (den[0, 1, 1:-1] - den[0, 0, 1:-1]) + var.size_ratio * vel_z[0, 0, 1:-1] * (den[0, 0, 2::] - den[0, 0, 0:-2]) + den[0, 0, 1:-1] * (vel_x[1, 0, 1:-1] - vel_x[0, 0, 1:-1] + vel_y[0, 1, 1:-1] - vel_y[0, 0, 1:-1] + var.size_ratio * (vel_z[0, 0, 2::] - vel_z[0, 0, 0:-2]) )
        den_out[0, -1, 1:-1] = vel_x[0, -1, 1:-1] * (den[1, -1, 1:-1] - den[0, -1, 1:-1]) + vel_y[0, -1, 1:-1] * (den[0, -1, 1:-1] - den[0, -2, 1:-1]) + var.size_ratio * vel_z[0, -1, 1:-1] * (den[0, -1, 2::] - den[0, -1, 0:-2]) + den[0, -1, 1:-1] * (vel_x[1, -1, 1:-1] - vel_x[0, -1, 1:-1] + vel_y[0, -1, 1:-1] - vel_y[0, -2, 1:-1] + var.size_ratio * (vel_z[0, -1, 2::] - vel_z[0, -1, 0:-2]) )
        den_out[-1, 0, 1:-1] = vel_x[-1, 0, 1:-1] * (den[-1, 0, 1:-1] - den[-2, 0, 1:-1]) + vel_y[-1, 0, 1:-1] * (den[-1, 1, 1:-1] - den[-1, 0, 1:-1]) + var.size_ratio * vel_z[-1, 0, 1:-1] * (den[-1, 0, 2::] - den[-1, 0, 0:-2]) + den[-1, 0, 1:-1] * (vel_x[-1, 0, 1:-1] - vel_x[-2, 0, 1:-1] + vel_y[-1, 1, 1:-1] - vel_y[-1, 0, 1:-1] + var.size_ratio * (vel_z[-1, 0, 2::] - vel_z[-1, 0, 0:-2]) )
        den_out[-1, -1, 1:-1] = vel_x[-1, -1, 1:-1] * (den[-1, -1, 1:-1] - den[-2, -1, 1:-1]) + vel_y[-1, -1, 1:-1] * (den[-1, -1, 1:-1] - den[-1, -2, 1:-1]) + var.size_ratio * vel_z[-1, -1, 1:-1] * (den[-1, -1, 2::] - den[-1, -1, 0:-2]) + den[-1, -1, 1:-1] * (vel_x[-1, -1, 1:-1] - vel_x[-2, -1, 1:-1] + vel_y[-1, -1, 1:-1] - vel_y[-1, -2, 1:-1] + var.size_ratio * (vel_z[-1, -1, 2::] - vel_z[-1, -1, 0:-2]) )
    
        den_out[0, 1:-1, 0] = vel_x[0, 1:-1, 0] * (den[1, 1:-1, 0] - den[0, 1:-1, 0]) + vel_y[0, 1:-1, 0] * (den[0, 2::, 0] - den[0, 0:-2, 0]) + var.size_ratio * vel_z[0, 1:-1, 0] * (den[0, 1:-1, 1] - den[0, 1:-1, 0]) + den[0, 1:-1, 0] * (vel_x[1, 1:-1, 0] - vel_x[0, 1:-1, 0] + vel_y[0, 2::, 0] - vel_y[0, 0:-2, 0] + var.size_ratio * (vel_z[0, 1:-1, 1] - vel_z[0, 1:-1, 0]) )
        den_out[0, 1:-1, -1] = vel_x[0, 1:-1, -1] * (den[1, 1:-1, -1] - den[0, 1:-1, -1]) + vel_y[0, 1:-1, -1] * (den[0, 2::, -1] - den[0, 0:-2, -1]) + var.size_ratio * vel_z[0, 1:-1, -1] * (den[0, 1:-1, -1] - den[0, 1:-1, -2]) + den[0, 1:-1, -1] * (vel_x[1, 1:-1, -1] - vel_x[0, 1:-1, -1] + vel_y[0, 2::, -1] - vel_y[0, 0:-2, -1] + var.size_ratio * (vel_z[0, 1:-1, -1] - vel_z[0, 1:-1, -2]) )
        den_out[-1, 1:-1, 0] = vel_x[-1, 1:-1, 0] * (den[-1, 1:-1, 0] - den[-2, 1:-1, 0]) + vel_y[-1, 1:-1, 0] * (den[-1, 2::, 0] - den[-1, 0:-2, 0]) + var.size_ratio * vel_z[-1, 1:-1, 0] * (den[-1, 1:-1, 1] - den[-1, 1:-1, 0]) + den[-1, 1:-1, 0] * (vel_x[-1, 1:-1, 0] - vel_x[-2, 1:-1, 0] + vel_y[-1, 2::, 0] - vel_y[-1, 0:-2, 0] + var.size_ratio * (vel_z[-1, 1:-1, 1] - vel_z[-1, 1:-1, 0]) )
        den_out[-1, 1:-1, -1] = vel_x[-1, 1:-1, -1] * (den[-1, 1:-1, -1] - den[-2, 1:-1, -1]) + vel_y[-1, 1:-1, -1] * (den[-1, 2::, -1] - den[-1, 0:-2, -1]) + var.size_ratio * vel_z[-1, 1:-1, -1] * (den[-1, 1:-1, -1] - den[-1, 1:-1, -2]) + den[-1, 1:-1, -1] * (vel_x[-1, 1:-1, -1] - vel_x[-2, 1:-1, -1] + vel_y[-1, 2::, -1] - vel_y[-1, 0:-2, -1] + var.size_ratio * (vel_z[-1, 1:-1, -1] - vel_z[-1, 1:-1, -2]) )
    
        # Corner Boundary den_out
        den_out[0,0,0] = vel_x[0,0,0] * (den[1,0,0] - den[0,0,0]) + vel_y[0,0,0] * (den[0,1,0] - den[0,0,0]) + var.size_ratio * vel_z[0,0,0] * (den[0,0,1] - den[0,0,0]) + den[0,0,0] * (vel_x[1,0,0] - vel_x[0,0,0] + vel_y[0,1,0] - vel_y[0,0,0] + var.size_ratio * (vel_z[0,0,1] - vel_z[0,0,0]) )
        den_out[-1,0,0] = vel_x[-1,0,0] * (den[-1,0,0] - den[-2,0,0]) + vel_y[-1,0,0] * (den[-1,1,0] - den[-1,0,0]) + var.size_ratio * vel_z[-1,0,0] * (den[-1,0,1] - den[-1,0,0]) + den[-1,0,0] * (vel_x[-1,0,0] - vel_x[-2,0,0] + vel_y[-1,1,0] - vel_y[-1,0,0] + var.size_ratio * (vel_z[-1,0,1] - vel_z[-1,0,0]) )
        den_out[0,-1,0] = vel_x[0,-1,0] * (den[1,-1,0] - den[0,-1,0]) + vel_y[0,-1,0] * (den[0,-1,0] - den[0,-2,0]) + var.size_ratio * vel_z[0,-1,0] * (den[0,-1,1] - den[0,-1,0]) + den[0,-1,0] * (vel_x[1,-1,0] - vel_x[0,-1,0] + vel_y[0,-1,0] - vel_y[0,-2,0] + var.size_ratio * (vel_z[0,-1,1] - vel_z[0,-1,0]) )
        den_out[-1,-1,0] = vel_x[-1,-1,0] * (den[-1,-1,0] - den[-2,-1,0]) + vel_y[-1,-1,0] * (den[-1,-1,0] - den[-1,-2,0]) + var.size_ratio * vel_z[-1,-1,0] * (den[-1,-1,1] - den[-1,-1,0]) + den[-1,-1,0] * (vel_x[-1,-1,0] - vel_x[-2,-1,0] + vel_y[-1,-1,0] - vel_y[-1,-2,0] + var.size_ratio * (vel_z[-1,-1,1] - vel_z[-1,-1,0]) )
        
        den_out[0,0,-1] = vel_x[0,0,-1] * (den[1,0,-1] - den[0,0,-1]) + vel_y[0,0,-1] * (den[0,1,-1] - den[0,0,-1]) + var.size_ratio * vel_z[0,0,-1] * (den[0,0,-1] - den[0,0,-2]) + den[0,0,-1] * (vel_x[1,0,-1] - vel_x[0,0,-1] + vel_y[0,1,-1] - vel_y[0,0,-1] + var.size_ratio * (vel_z[0,0,-1] - vel_z[0,0,-2]) )
        den_out[-1,0,-1] = vel_x[-1,0,-1] * (den[-1,0,-1] - den[-2,0,-1]) + vel_y[-1,0,-1] * (den[-1,1,-1] - den[-1,0,-1]) + var.size_ratio * vel_z[-1,0,-1] * (den[-1,0,-1] - den[-1,0,-2]) + den[-1,0,-1] * (vel_x[-1,0,-1] - vel_x[-2,0,-1] + vel_y[-1,1,-1] - vel_y[-1,0,-1] + var.size_ratio * (vel_z[-1,0,-1] - vel_z[-1,0,-2]) )
        den_out[0,-1,-1] = vel_x[0,-1,-1] * (den[1,-1,-1] - den[0,-1,-1]) + vel_y[0,-1,-1] * (den[0,-1,-1] - den[0,-2,-1]) + var.size_ratio * vel_z[0,-1,-1] * (den[0,-1,-1] - den[0,-1,-2]) + den[0,-1,-1] * (vel_x[1,-1,-1] - vel_x[0,-1,-1] + vel_y[0,-1,-1] - vel_y[0,-2,-1] + var.size_ratio * (vel_z[0,-1,-1] - vel_z[0,-1,-2]) )
        den_out[-1,-1,-1] = vel_x[-1,-1,-1] * (den[-1,-1,-1] - den[-2,-1,-1]) + vel_y[-1,-1,-1] * (den[-1,-1,-1] - den[-1,-2,-1]) + var.size_ratio * vel_z[-1,-1,-1] * (den[-1,-1,-1] - den[-1,-1,-2]) + den[-1,-1,-1] * (vel_x[-1,-1,-1] - vel_x[-2,-1,-1] + vel_y[-1,-1,-1] - vel_y[-1,-2,-1] + var.size_ratio * (vel_z[-1,-1,-1] - vel_z[-1,-1,-2]) )

        # Out Put Calculation        
        den_out = den_out - hv  
        den_out = den - 0.5 * var.step_ratio * den_out - den_sh
           
    return den_out

def calculate_velocity_x(vel_x, vel_y, vel_z, den, den_inv, vel_sh, k, Zsize, periodic):
    """ Returns time derivative of fluid velocity in x direction.
    
    Parameters
    ----------
    vel_x : array
        fluid velocity in x direction    
    
    vel_y : array
        fluid velocity in y direction    

    vel_z : array
        fluid velocity in z direction    

    den : array
        fluid density    
    
    den_inv : array
        inverse fluid density   

    vel_sh : array
        source/sink terms in momentum equation
        
    k : float
        scalar for density gradient
    
    Zsize : int
        number of dimensions along axis=2
    """
    if periodic:
        # Define Data Size
        hv = np.zeros(vel_x.shape)
        vel_x_out = np.zeros(vel_x.shape)
        
        # Calculate Central Differences
        hv[1:-1, 1:-1, 1:-1] = var.hv_vel_r * ( (vel_x[2::, 1:-1, 1:-1] - 2*vel_x[1:-1, 1:-1, 1:-1] + vel_x[0:-2, 1:-1, 1:-1]) + (vel_x[1:-1, 2::, 1:-1] - 2*vel_x[1:-1, 1:-1, 1:-1] + vel_x[1:-1, 0:-2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_x[1:-1, 1:-1, 2::] - 2*vel_x[1:-1, 1:-1, 1:-1] + vel_x[1:-1, 1:-1, 0:-2]) 
        vel_x_out[1:-1, 1:-1, 1:-1] = vel_x[1:-1, 1:-1, 1:-1] * (vel_x[2::, 1:-1, 1:-1] - vel_x[0:-2, 1:-1, 1:-1]) + vel_y[1:-1, 1:-1, 1:-1] * (vel_x[1:-1, 2::, 1:-1] - vel_x[1:-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[1:-1, 1:-1, 1:-1] * (vel_x[1:-1, 1:-1, 2::] - vel_x[1:-1, 1:-1, 0:-2]) + 2 * k * den_inv[1:-1, 1:-1, 1:-1] * (den[2::, 1:-1, 1:-1] - den[0:-2, 1:-1, 1:-1])

        # Out Put Calculation
        vel_x_out[1:-1, 1:-1, 1:-1] = vel_x_out[1:-1, 1:-1, 1:-1] - hv[1:-1, 1:-1, 1:-1]
        vel_x_out[1:-1, 1:-1, 1:-1] = vel_x[1:-1, 1:-1, 1:-1] - .5 * var.step_ratio * vel_x_out[1:-1, 1:-1, 1:-1] - var.dT * vel_sh[1:-1, 1:-1, 1:-1]
        
        # Periodic Boundaries
        vel_x_out = periodic_boundary_vel_x(vel_x_out)
                
    else:
        hv = np.empty(vel_x.shape)
        vel_x_out = np.empty(vel_x.shape)
        
        # Central hv
        hv[1:-1, 1:-1, 1:-1] = var.hv_vel_r * ( (vel_x[2::, 1:-1, 1:-1] - 2*vel_x[1:-1, 1:-1, 1:-1] + vel_x[0:-2, 1:-1, 1:-1]) + (vel_x[1:-1, 2::, 1:-1] - 2*vel_x[1:-1, 1:-1, 1:-1] + vel_x[1:-1, 0:-2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_x[1:-1, 1:-1, 2::] - 2*vel_x[1:-1, 1:-1, 1:-1] + vel_x[1:-1, 1:-1, 0:-2]) 
        
        # X Boundary hv
        hv[0, 1:-1, 1:-1] = var.hv_vel_r * ( var.hv_vel_bnd * (vel_x[0, 1:-1, 1:-1] - 2*vel_x[1, 1:-1, 1:-1] + vel_x[2, 1:-1, 1:-1]) + (vel_x[0, 2::, 1:-1] - 2*vel_x[0, 1:-1, 1:-1] + vel_x[0, 0:-2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_x[0, 1:-1, 2::] - 2*vel_x[0, 1:-1, 1:-1] + vel_x[0, 1:-1, 0:-2]) 
        hv[-1, 1:-1, 1:-1] = var.hv_vel_r * ( var.hv_vel_bnd * (vel_x[-1, 1:-1, 1:-1] - 2*vel_x[-2, 1:-1, 1:-1] + vel_x[-3, 1:-1, 1:-1]) + (vel_x[-1, 2::, 1:-1] - 2*vel_x[-1, 1:-1, 1:-1] + vel_x[-1, 0:-2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_x[-1, 1:-1, 2::] - 2*vel_x[-1, 1:-1, 1:-1] + vel_x[-1, 1:-1, 0:-2])
        
        # Y Boundary hv
        hv[1:-1, 0, 1:-1] = var.hv_vel_r * ( (vel_x[2::, 0, 1:-1] - 2*vel_x[1:-1, 0, 1:-1] + vel_x[0:-2, 0, 1:-1]) + var.hv_vel_bnd * (vel_x[1:-1, 0, 1:-1] - 2*vel_x[1:-1, 1, 1:-1] + vel_x[1:-1, 2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_x[1:-1, 0, 2::] - 2*vel_x[1:-1, 0, 1:-1] + vel_x[1:-1, 0, 0:-2])
        hv[1:-1, -1, 1:-1] = var.hv_vel_r * ( (vel_x[2::, -1, 1:-1] - 2*vel_x[1:-1, -1, 1:-1] + vel_x[0:-2, -1, 1:-1]) + var.hv_vel_bnd * (vel_x[1:-1, -1, 1:-1] - 2*vel_x[1:-1, -2, 1:-1] + vel_x[1:-1, -3, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_x[1:-1, -1, 2::] - 2*vel_x[1:-1, -1, 1:-1] + vel_x[1:-1, -1, 0:-2])
        
        # Z Boundary hv
        hv[1:-1, 1:-1, 0] = var.hv_vel_r * ( (vel_x[2::, 1:-1, 0] - 2*vel_x[1:-1, 1:-1, 0] + vel_x[0:-2, 1:-1, 0]) + (vel_x[1:-1, 2::, 0] - 2*vel_x[1:-1, 1:-1, 0] + vel_x[1:-1, 0:-2, 0]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_bnd * (vel_x[1:-1, 1:-1, 0] - 2*vel_x[1:-1, 1:-1, 1] + vel_x[1:-1, 1:-1, 2])
        hv[1:-1, 1:-1, -1] = var.hv_vel_r * ( (vel_x[2::, 1:-1, -1] - 2*vel_x[1:-1, 1:-1, -1] + vel_x[0:-2, 1:-1, -1]) + (vel_x[1:-1, 2::, -1] - 2*vel_x[1:-1, 1:-1, -1] + vel_x[1:-1, 0:-2, -1]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_bnd * (vel_x[1:-1, 1:-1, -1] - 2*vel_x[1:-1, 1:-1, -2] + vel_x[1:-1, 1:-1, -3])
        
        # Edge Boundaries hv
        hv[0, 0, 1:-1] = var.hv_vel_r * var.hv_vel_edg * ( (vel_x[0, 0, 1:-1] - 2*vel_x[1, 0, 1:-1] + vel_x[2, 0, 1:-1]) + (vel_x[0, 0, 1:-1] - 2*vel_x[0, 1, 1:-1] + vel_x[0, 2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_x[0, 0, 2::] - 2*vel_x[0, 0, 1:-1] + vel_x[0, 0, 0:-2])
        hv[0, -1, 1:-1] = var.hv_vel_r * var.hv_vel_edg * ( (vel_x[0, -1, 1:-1] - 2*vel_x[1, -1, 1:-1] + vel_x[2, -1, 1:-1]) + (vel_x[0, -1, 1:-1] - 2*vel_x[0, -2, 1:-1] + vel_x[0, -3, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_x[0, -1, 2::] - 2*vel_x[0, -1, 1:-1] + vel_x[0, -1, 0:-2])
        hv[-1, 0, 1:-1] = var.hv_vel_r * var.hv_vel_edg * ( (vel_x[-1, 0, 1:-1] - 2*vel_x[-2, 0, 1:-1] + vel_x[-3, 0, 1:-1]) + (vel_x[-1, 0, 1:-1] - 2*vel_x[-1, 1, 1:-1] + vel_x[-1, 2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_x[-1, 0, 2::] - 2*vel_x[-1, 0, 1:-1] + vel_x[-1, 0, 0:-2])
        hv[-1, -1, 1:-1] = var.hv_vel_r * var.hv_vel_edg * ( (vel_x[-1, -1, 1:-1] - 2*vel_x[-2, -1, 1:-1] + vel_x[-3, -1, 1:-1]) + (vel_x[-1, -1, 1:-1] - 2*vel_x[-1, -2, 1:-1] + vel_x[-1, -3, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_x[-1, -1, 2::] - 2*vel_x[-1, -1, 1:-1] + vel_x[-1, -1, 0:-2])
        
        hv[0, 1:-1, 0] = var.hv_vel_r * ( var.hv_vel_edg * (vel_x[0, 1:-1, 0] - 2*vel_x[1, 1:-1, 0] + vel_x[2, 1:-1, 0]) + (vel_x[0, 2::, 0] - 2*vel_x[0, 1:-1, 0] + vel_x[0, 0:-2, 0]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_x[0, 1:-1, 0] - 2*vel_x[0, 1:-1, 1] + vel_x[0, 1:-1, 2])
        hv[-1, 1:-1, 0] = var.hv_vel_r * ( var.hv_vel_edg * (vel_x[-1, 1:-1, 0] - 2*vel_x[-2, 1:-1, 0] + vel_x[-3, 1:-1, 0]) + (vel_x[-1, 2::, 0] - 2*vel_x[-1, 1:-1, 0] + vel_x[-1, 0:-2, 0]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_x[-1, 1:-1, 0] - 2*vel_x[-1, 1:-1, 1] + vel_x[-1, 1:-1, 2])
        hv[0, 1:-1, -1] = var.hv_vel_r * ( var.hv_vel_edg * (vel_x[0, 1:-1, -1] - 2*vel_x[1, 1:-1, -1] + vel_x[2, 1:-1, -1]) + (vel_x[0, 2::, -1] - 2*vel_x[0, 1:-1, -1] + vel_x[0, 0:-2, -1]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_x[0, 1:-1, -1] - 2*vel_x[0, 1:-1, -2] + vel_x[0, 1:-1, -3])
        hv[-1, 1:-1, -1] = var.hv_vel_r * ( var.hv_vel_edg * (vel_x[-1, 1:-1, -1] - 2*vel_x[-2, 1:-1, -1] + vel_x[-3, 1:-1, -1]) + (vel_x[-1, 2::, -1] - 2*vel_x[-1, 1:-1, -1] + vel_x[-1, 0:-2, -1]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_x[-1, 1:-1, -1] - 2*vel_x[-1, 1:-1, -2] + vel_x[-1, 1:-1, -3])    
        
        hv[1:-1, 0, 0] = var.hv_vel_r * ( (vel_x[2::, 0, 0] - 2*vel_x[1:-1, 0, 0] + vel_x[0:-2, 0, 0]) + var.hv_vel_edg * (vel_x[1:-1, 0, 0] - 2*vel_x[1:-1, 1, 0] + vel_x[1:-1, 2, 0]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_x[1:-1, 0, 0] - 2*vel_x[1:-1, 0, 1] + vel_x[1:-1, 0, 2])
        hv[1:-1, -1, 0] = var.hv_vel_r * ( (vel_x[2::, -1, 0] - 2*vel_x[1:-1, -1, 0] + vel_x[0:-2, -1, 0]) + var.hv_vel_edg * (vel_x[1:-1, -1, 0] - 2*vel_x[1:-1, -2, 0] + vel_x[1:-1, -3, 0]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_x[1:-1, -1, 0] - 2*vel_x[1:-1, -1, 1] + vel_x[1:-1, -1, 2])
        hv[1:-1, 0, -1] = var.hv_vel_r * ( (vel_x[2::, 0, -1] - 2*vel_x[1:-1, 0, -1] + vel_x[0:-2, 0, -1]) + var.hv_vel_edg * (vel_x[1:-1, 0, -1] - 2*vel_x[1:-1, 1, -1] + vel_x[1:-1, 2, -1]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_x[1:-1, 0, -1] - 2*vel_x[1:-1, 0, -2] + vel_x[1:-1, 0, -3])
        hv[1:-1, -1, -1] = var.hv_vel_r * ( (vel_x[2::, -1, -1] - 2*vel_x[1:-1, -1, -1] + vel_x[0:-2, -1, -1]) + var.hv_vel_edg * (vel_x[1:-1, -1, -1] - 2*vel_x[1:-1, -2, -1] + vel_x[1:-1, -3, -1]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_x[1:-1, -1, -1] - 2*vel_x[1:-1, -1, -2] + vel_x[1:-1, -1, -3])
        
        # Corner Boundaries hv
        hv[0, 0, 0] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_x[0,0,0] - 2*vel_x[1,0,0] + vel_x[2,0,0]) + (vel_x[0,0,0] - 2*vel_x[0,1,0] + vel_x[0,2,0]) ) + var.hv_vel_z * var.size_ratio * (vel_x[0,0,0] - 2*vel_x[0,0,1] + vel_x[0,0,2]) )
        hv[-1, 0, 0] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_x[-1,0,0] - 2*vel_x[-2,0,0] + vel_x[-3,0,0]) + (vel_x[-1,0,0] - 2*vel_x[-1,1,0] + vel_x[-1,2,0]) ) + var.hv_vel_z * var.size_ratio * (vel_x[-1,0,0] - 2*vel_x[-1,0,1] + vel_x[-1,0,2]) )
        hv[0, -1, 0] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_x[0,-1,0] - 2*vel_x[1,-1,0] + vel_x[2,-1,0]) + (vel_x[0,-1,0] - 2*vel_x[0,-2,0] + vel_x[0,-3,0]) ) + var.hv_vel_z * var.size_ratio * (vel_x[0,-1,0] - 2*vel_x[0,-1,1] + vel_x[0,-1,2]) )
        hv[-1, -1, 0] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_x[-1,-1,0] - 2*vel_x[-2,-1,0] + vel_x[-3,-1,0]) + (vel_x[-1,-1,0] - 2*vel_x[-1,-2,0] + vel_x[-1,-3,0]) ) + var.hv_vel_z * var.size_ratio * (vel_x[-1,-1,0] - 2*vel_x[-1,-1,1] + vel_x[-1,-1,2]) )
        
        hv[0, 0, -1] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_x[0,0,-1] - 2*vel_x[1,0,-1] + vel_x[2,0,-1]) + (vel_x[0,0,-1] - 2*vel_x[0,1,-1] + vel_x[0,2,-1]) ) + var.hv_vel_z * var.size_ratio * (vel_x[0,0,-1] - 2*vel_x[0,0,-2] + vel_x[0,0,-3]) )
        hv[-1, 0, -1] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_x[-1,0,-1] - 2*vel_x[-2,0,-1] + vel_x[-3,0,-1]) + (vel_x[-1,0,-1] - 2*vel_x[-1,1,-1] + vel_x[-1,2,-1]) ) + var.hv_vel_z * var.size_ratio * (vel_x[-1,0,-1] - 2*vel_x[-1,0,-2] + vel_x[-1,0,-3]) )
        hv[0, -1, -1] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_x[0,-1,-1] - 2*vel_x[1,-1,-1] + vel_x[2,-1,-1]) + (vel_x[0,-1,-1] - 2*vel_x[0,-2,-1] + vel_x[0,-3,-1]) ) + var.hv_vel_z * var.size_ratio * (vel_x[0,-1,-1] - 2*vel_x[0,-1,-2] + vel_x[0,-1,-3]) )
        hv[-1, -1, -1] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_x[-1,-1,-1] - 2*vel_x[-2,-1,-1] + vel_x[-3,-1,-1]) + (vel_x[-1,-1,-1] - 2*vel_x[-1,-2,-1] + vel_x[-1,-3,-1]) ) + var.hv_vel_z * var.size_ratio * (vel_x[-1,-1,-1] - 2*vel_x[-1,-1,-2] + vel_x[-1,-1,-3]) )
            
        # Central vel_x_out
        vel_x_out[1:-1, 1:-1, 1:-1] = vel_x[1:-1, 1:-1, 1:-1] * (vel_x[2::, 1:-1, 1:-1] - vel_x[0:-2, 1:-1, 1:-1]) + vel_y[1:-1, 1:-1, 1:-1] * (vel_x[1:-1, 2::, 1:-1] - vel_x[1:-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[1:-1, 1:-1, 1:-1] * (vel_x[1:-1, 1:-1, 2::] - vel_x[1:-1, 1:-1, 0:-2]) + 2 * k * den_inv[1:-1, 1:-1, 1:-1] * (den[2::, 1:-1, 1:-1] - den[0:-2, 1:-1, 1:-1])
        
        # X Boundary vel_x_out
        vel_x_out[0, 1:-1, 1:-1] = vel_x[0, 1:-1, 1:-1] * (vel_x[1, 1:-1, 1:-1] - vel_x[0, 1:-1, 1:-1]) + vel_y[0, 1:-1, 1:-1] * (vel_x[0, 2::, 1:-1] - vel_x[0, 0:-2, 1:-1]) + var.size_ratio * vel_z[0, 1:-1, 1:-1] * (vel_x[0, 1:-1, 2::] - vel_x[0, 1:-1, 0:-2]) + 2 * k * den_inv[0, 1:-1, 1:-1] * (den[1, 1:-1, 1:-1] - den[0, 1:-1, 1:-1])
        vel_x_out[-1, 1:-1, 1:-1] = vel_x[-1, 1:-1, 1:-1] * (vel_x[-1, 1:-1, 1:-1] - vel_x[-2, 1:-1, 1:-1]) + vel_y[-1, 1:-1, 1:-1] * (vel_x[-1, 2::, 1:-1] - vel_x[-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[-1, 1:-1, 1:-1] * (vel_x[-1, 1:-1, 2::] - vel_x[-1, 1:-1, 0:-2]) + 2 * k * den_inv[-1, 1:-1, 1:-1] * (den[-1, 1:-1, 1:-1] - den[-2, 1:-1, 1:-1])
        
        # Y Boundarty vel_x_out
        vel_x_out[1:-1, 0, 1:-1] = vel_x[1:-1, 0, 1:-1] * (vel_x[2::, 0, 1:-1] - vel_x[0:-2, 0, 1:-1]) + vel_y[1:-1, 0, 1:-1] * (vel_x[1:-1, 1, 1:-1] - vel_x[1:-1, 0, 1:-1]) + var.size_ratio * vel_z[1:-1, 0, 1:-1] * (vel_x[1:-1, 0, 2::] - vel_x[1:-1, 0, 0:-2]) + 2 * k * den_inv[1:-1, 0, 1:-1] * (den[2::, 0, 1:-1] - den[0:-2, 0, 1:-1])
        vel_x_out[1:-1, -1, 1:-1] = vel_x[1:-1, -1, 1:-1] * (vel_x[2::, -1, 1:-1] - vel_x[0:-2, -1, 1:-1]) + vel_y[1:-1, -1, 1:-1] * (vel_x[1:-1, -1, 1:-1] - vel_x[1:-1, -2, 1:-1]) + var.size_ratio * vel_z[1:-1, -1, 1:-1] * (vel_x[1:-1, -1, 2::] - vel_x[1:-1, -1, 0:-2]) + 2 * k * den_inv[1:-1, -1, 1:-1] * (den[2::, -1, 1:-1] - den[0:-2, -1, 1:-1])
        
        # Z Boundary vel_x_out
        vel_x_out[1:-1, 1:-1, 0] = vel_x[1:-1, 1:-1, 0] * (vel_x[2::, 1:-1, 0] - vel_x[0:-2, 1:-1, 0]) + vel_y[1:-1, 1:-1, 0] * (vel_x[1:-1, 2::, 0] - vel_x[1:-1, 0:-2, 0]) + var.size_ratio * vel_z[1:-1, 1:-1, 0] * (vel_x[1:-1, 1:-1, 1] - vel_x[1:-1, 1:-1, 0]) + 2 * k * den_inv[1:-1, 1:-1, 0] * (den[2::, 1:-1, 0] - den[0:-2, 1:-1, 0])
        vel_x_out[1:-1, 1:-1, -1] = vel_x[1:-1, 1:-1, -1] * (vel_x[2::, 1:-1, -1] - vel_x[0:-2, 1:-1, -1]) + vel_y[1:-1, 1:-1, -1] * (vel_x[1:-1, 2::, -1] - vel_x[1:-1, 0:-2, -1]) + var.size_ratio * vel_z[1:-1, 1:-1, -1] * (vel_x[1:-1, 1:-1, -1] - vel_x[1:-1, 1:-1, -2]) + 2 * k * den_inv[1:-1, 1:-1, -1] * (den[2::, 1:-1, -1] - den[0:-2, 1:-1, -1])
        
        # Edge Boundary vel_x_out
        vel_x_out[0, 0, 1:-1] = vel_x[0, 0, 1:-1] * (vel_x[1, 0, 1:-1] - vel_x[0, 0, 1:-1]) + vel_y[0, 0, 1:-1] * (vel_x[0, 1, 1:-1] - vel_x[0, 0, 1:-1]) + var.size_ratio * vel_z[0, 0, 1:-1] * (vel_x[0, 0, 2::] - vel_x[0, 0, 0:-2]) + 2 * k * den_inv[0, 0, 1:-1] * (den[1, 0, 1:-1] - den[0, 0, 1:-1])
        vel_x_out[0, -1, 1:-1] = vel_x[0, -1, 1:-1] * (vel_x[1, -1, 1:-1] - vel_x[0, -1, 1:-1]) + vel_y[0, -1, 1:-1] * (vel_x[0, -1, 1:-1] - vel_x[0, -2, 1:-1]) + var.size_ratio * vel_z[0, -1, 1:-1] * (vel_x[0, -1, 2::] - vel_x[0, -1, 0:-2]) + 2 * k * den_inv[0, -1, 1:-1] * (den[1, -1, 1:-1] - den[0, -1, 1:-1])
        vel_x_out[-1, 0, 1:-1] = vel_x[-1, 0, 1:-1] * (vel_x[-1, 0, 1:-1] - vel_x[-2, 0, 1:-1]) + vel_y[-1, 0, 1:-1] * (vel_x[-1, 1, 1:-1] - vel_x[-1, 0, 1:-1]) + var.size_ratio * vel_z[-1, 0, 1:-1] * (vel_x[-1, 0, 2::] - vel_x[-1, 0, 0:-2]) + 2 * k * den_inv[-1, 0, 1:-1] * (den[-1, 0, 1:-1] - den[-2, 0, 1:-1])
        vel_x_out[-1, -1, 1:-1] = vel_x[-1, -1, 1:-1] * (vel_x[-1, -1, 1:-1] - vel_x[-2, -1, 1:-1]) + vel_y[-1, -1, 1:-1] * (vel_x[-1, -1, 1:-1] - vel_x[-1, -2, 1:-1]) + var.size_ratio * vel_z[-1, -1, 1:-1] * (vel_x[-1, -1, 2::] - vel_x[-1, -1, 0:-2]) + 2 * k * den_inv[-1, -1, 1:-1] * (den[-1, -1, 1:-1] - den[-2, -1, 1:-1])
    
        vel_x_out[0, 1:-1, 0] = vel_x[0, 1:-1, 0] * (vel_x[1, 1:-1, 0] - vel_x[0, 1:-1, 0]) + vel_y[0, 1:-1, 0] * (vel_x[0, 2::, 0] - vel_x[0, 0:-2, 0]) + var.size_ratio * vel_z[0, 1:-1, 0] * (vel_x[0, 1:-1, 1] - vel_x[0, 1:-1, 0]) + 2 * k * den_inv[0, 1:-1, 0] * (den[1, 1:-1, 0] - den[0, 1:-1, 0])
        vel_x_out[0, 1:-1, -1] = vel_x[0, 1:-1, -1] * (vel_x[1, 1:-1, -1] - vel_x[0, 1:-1, -1]) + vel_y[0, 1:-1, -1] * (vel_x[0, 2::, -1] - vel_x[0, 0:-2, -1]) + var.size_ratio * vel_z[0, 1:-1, -1] * (vel_x[0, 1:-1, -1] - vel_x[0, 1:-1, -2]) + 2 * k * den_inv[0, 1:-1, -1] * (den[1, 1:-1, -1] - den[0, 1:-1, -1])
        vel_x_out[-1, 1:-1, 0] = vel_x[-1, 1:-1, 0] * (vel_x[-1, 1:-1, 0] - vel_x[-2, 1:-1, 0]) + vel_y[-1, 1:-1, 0] * (vel_x[-1, 2::, 0] - vel_x[-1, 0:-2, 0]) + var.size_ratio * vel_z[-1, 1:-1, 0] * (vel_x[-1, 1:-1, 1] - vel_x[-1, 1:-1, 0]) + 2 * k * den_inv[-1, 1:-1, 0] * (den[-1, 1:-1, 0] - den[-2, 1:-1, 0])
        vel_x_out[-1, 1:-1, -1] = vel_x[-1, 1:-1, -1] * (vel_x[-1, 1:-1, -1] - vel_x[-2, 1:-1, -1]) + vel_y[-1, 1:-1, -1] * (vel_x[-1, 2::, -1] - vel_x[-1, 0:-2, -1]) + var.size_ratio * vel_z[-1, 1:-1, -1] * (vel_x[-1, 1:-1, -1] - vel_x[-1, 1:-1, -2]) + 2 * k * den_inv[-1, 1:-1, -1] * (den[-1, 1:-1, -1] - den[-2, 1:-1, -1])
    
        # Corner Boundary vel_x_out
        vel_x_out[0,0,0] = vel_x[0,0,0] * (vel_x[1,0,0] - vel_x[0,0,0]) + vel_y[0,0,0] * (vel_x[0,1,0] - vel_x[0,0,0]) + var.size_ratio * vel_z[0,0,0] * (vel_x[0,0,1] - vel_x[0,0,0]) + 2 * k * den_inv[0,0,0] * (den[1,0,0] - den[0,0,0])
        vel_x_out[0,-1,0] = vel_x[0,-1,0] * (vel_x[1,-1,0] - vel_x[0,-1,0]) + vel_y[0,-1,0] * (vel_x[0,-2,0] - vel_x[0,-1,0]) + var.size_ratio * vel_z[0,-1,0] * (vel_x[0,-1,1] - vel_x[0,-1,0]) + 2 * k * den_inv[0,-1,0] * (den[1,-1,0] - den[0,-1,0])
        vel_x_out[-1,0,0] = vel_x[-1,0,0] * (vel_x[-1,0,0] - vel_x[-2,0,0]) + vel_y[-1,0,0] * (vel_x[-1,1,0] - vel_x[-1,0,0]) + var.size_ratio * vel_z[-1,0,0] * (vel_x[-1,0,1] - vel_x[-1,0,0]) + 2 * k * den_inv[-1,0,0] * (den[-1,0,0] - den[-2,0,0])
        vel_x_out[-1,-1,0] = vel_x[-1,-1,0] * (vel_x[-1,-1,0] - vel_x[-2,-1,0]) + vel_y[-1,-1,0] * (vel_x[-1,-2,0] - vel_x[-1,-1,0]) + var.size_ratio * vel_z[-1,-1,0] * (vel_x[-1,-1,1] - vel_x[-1,-1,0]) + 2 * k * den_inv[-1,-1,0] * (den[-1,-1,0] - den[-2,-1,0])
        
        vel_x_out[0,0,-1] = vel_x[0,0,-1] * (vel_x[1,0,-1] - vel_x[0,0,-1]) + vel_y[0,0,-1] * (vel_x[0,1,-1] - vel_x[0,0,-1]) + var.size_ratio * vel_z[0,0,-1] * (vel_x[0,0,-1] - vel_x[0,0,-2]) + 2 * k * den_inv[0,0,-1] * (den[1,0,-1] - den[0,0,-1])
        vel_x_out[0,-1,-1] = vel_x[0,-1,-1] * (vel_x[1,-1,-1] - vel_x[0,-1,-1]) + vel_y[0,-1,-1] * (vel_x[0,-2,-1] - vel_x[0,-1,-1]) + var.size_ratio * vel_z[0,-1,-1] * (vel_x[0,-1,-1] - vel_x[0,-1,-2]) + 2 * k * den_inv[0,-1,-1] * (den[1,-1,-1] - den[0,-1,-1])
        vel_x_out[-1,0,-1] = vel_x[-1,0,-1] * (vel_x[-1,0,-1] - vel_x[-2,0,-1]) + vel_y[-1,0,-1] * (vel_x[-1,1,-1] - vel_x[-1,0,-1]) + var.size_ratio * vel_z[-1,0,-1] * (vel_x[-1,0,-1] - vel_x[-1,0,-2]) + 2 * k * den_inv[-1,0,-1] * (den[-1,0,-1] - den[-2,0,-1])
        vel_x_out[-1,-1,-1] = vel_x[-1,-1,-1] * (vel_x[-1,-1,-1] - vel_x[-2,-1,-1]) + vel_y[-1,-1,-1] * (vel_x[-1,-2,-1] - vel_x[-1,-1,-1]) + var.size_ratio * vel_z[-1,-1,-1] * (vel_x[-1,-1,-1] - vel_x[-1,-1,-2]) + 2 * k * den_inv[-1,-1,-1] * (den[-1,-1,-1] - den[-2,-1,-1])
        
        # Out Put Calculation
        vel_x_out = vel_x_out - hv    
        vel_x_out = vel_x - .5 * var.step_ratio * vel_x_out - var.dT * vel_sh

    return vel_x_out
    
def calculate_velocity_y(vel_x, vel_y, vel_z, den, den_inv, vel_sh, k, Zsize, periodic) :
    """ Returns time derivative of fluid velocity in y direction.
    
    Parameters
    ----------
    vel_x : array
        fluid velocity in x direction    
    
    vel_y : array
        fluid velocity in y direction    

    vel_z : array
        fluid velocity in z direction    

    den : array
        fluid density    
    
    den_inv : array
        inverse fluid density   

    vel_sh : array
        source/sink terms in momentum equation
        
    k : float
        scalar for density gradient
    
    Zsize : int
        number of dimensions along axis=2
    """
    if periodic:
        # Define Data Sizes
        hv = np.zeros(vel_y.shape)
        vel_y_out = np.zeros(vel_y.shape)
    
        # Calculate Central Differences
        hv[1:-1, 1:-1, 1:-1] = var.hv_vel_r * ( (vel_y[2::, 1:-1, 1:-1] - 2*vel_y[1:-1, 1:-1, 1:-1] + vel_y[0:-2, 1:-1, 1:-1]) + (vel_y[1:-1, 2::, 1:-1] - 2*vel_y[1:-1, 1:-1, 1:-1] + vel_y[1:-1, 0:-2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_y[1:-1, 1:-1, 2::] - 2*vel_y[1:-1, 1:-1, 1:-1] + vel_y[1:-1, 1:-1, 0:-2]) 
        vel_y_out[1:-1, 1:-1, 1:-1] = vel_x[1:-1, 1:-1, 1:-1] * (vel_y[2::, 1:-1, 1:-1] - vel_y[0:-2, 1:-1, 1:-1]) + vel_y[1:-1, 1:-1, 1:-1] * (vel_y[1:-1, 2::, 1:-1] - vel_y[1:-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[1:-1, 1:-1, 1:-1] * (vel_y[1:-1, 1:-1, 2::] - vel_y[1:-1, 1:-1, 0:-2]) + 2 * k * den_inv[1:-1, 1:-1, 1:-1] * (den[1:-1, 2::, 1:-1] - den[1:-1, 0:-2, 1:-1])
        
        # Out Put Calculation
        vel_y_out[1:-1, 1:-1, 1:-1] = vel_y_out[1:-1, 1:-1, 1:-1] - hv[1:-1, 1:-1, 1:-1]
        vel_y_out[1:-1, 1:-1, 1:-1] = vel_y[1:-1, 1:-1, 1:-1] - .5 * var.step_ratio * vel_y_out[1:-1, 1:-1, 1:-1] - var.dT * vel_sh[1:-1, 1:-1, 1:-1]
        
        # Periodic Boundaries
        vel_y_out = periodic_boundary_vel_y(vel_y_out)
        
    else:
        # Define Data Size
        hv = np.empty(vel_y.shape)
        vel_y_out = np.empty(vel_y.shape)
        
        # Central hv    
        hv[1:-1, 1:-1, 1:-1] = var.hv_vel_r * ( (vel_y[2::, 1:-1, 1:-1] - 2*vel_y[1:-1, 1:-1, 1:-1] + vel_y[0:-2, 1:-1, 1:-1]) + (vel_y[1:-1, 2::, 1:-1] - 2*vel_y[1:-1, 1:-1, 1:-1] + vel_y[1:-1, 0:-2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_y[1:-1, 1:-1, 2::] - 2*vel_y[1:-1, 1:-1, 1:-1] + vel_y[1:-1, 1:-1, 0:-2]) 
    
        # X Boundary hv
        hv[0, 1:-1, 1:-1] = var.hv_vel_r * ( var.hv_vel_bnd * (vel_y[0, 1:-1, 1:-1] - 2*vel_y[1, 1:-1, 1:-1] + vel_y[2, 1:-1, 1:-1]) + (vel_y[0, 2::, 1:-1] - 2*vel_y[0, 1:-1, 1:-1] + vel_y[0, 0:-2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_y[0, 1:-1, 2::] - 2*vel_y[0, 1:-1, 1:-1] + vel_y[0, 1:-1, 0:-2])
        hv[-1, 1:-1, 1:-1] = var.hv_vel_r * ( var.hv_vel_bnd * (vel_y[-1, 1:-1, 1:-1] - 2*vel_y[-2, 1:-1, 1:-1] + vel_y[-3, 1:-1, 1:-1]) + (vel_y[-1, 2::, 1:-1] - 2*vel_y[-1, 1:-1, 1:-1] + vel_y[-1, 0:-2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_y[-1, 1:-1, 2::] - 2*vel_y[-1, 1:-1, 1:-1] + vel_y[-1, 1:-1, 0:-2])
        
        # Y Boundary hv
        hv[1:-1, 0, 1:-1] = var.hv_vel_r * ( (vel_y[2::, 0, 1:-1] - 2*vel_y[1:-1, 0, 1:-1] + vel_y[0:-2, 0, 1:-1]) + var.hv_vel_bnd * (vel_y[1:-1, 0, 1:-1] - 2*vel_y[1:-1, 1, 1:-1] + vel_y[1:-1, 2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_y[1:-1, 0, 2::] - 2*vel_y[1:-1, 0, 1:-1] + vel_y[1:-1, 0, 0:-2])
        hv[1:-1, -1, 1:-1] = var.hv_vel_r * ( (vel_y[2::, -1, 1:-1] - 2*vel_y[1:-1, -1, 1:-1] + vel_y[0:-2, -1, 1:-1]) + var.hv_vel_bnd * (vel_y[1:-1, -1, 1:-1] - 2*vel_y[1:-1, -2, 1:-1] + vel_y[1:-1, -3, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_y[1:-1, -1, 2::] - 2*vel_y[1:-1, -1, 1:-1] + vel_y[1:-1, -1, 0:-2])
        
        # Z Boundary hv
        hv[1:-1, 1:-1, 0] = var.hv_vel_r * ( (vel_y[2::, 1:-1, 0] - 2*vel_y[1:-1, 1:-1, 0] + vel_y[0:-2, 1:-1, 0]) + (vel_y[1:-1, 2::, 0] - 2*vel_y[1:-1, 1:-1, 0] + vel_y[1:-1, 0:-2, 0]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_bnd * (vel_y[1:-1, 1:-1, 0] - 2*vel_y[1:-1, 1:-1, 1] + vel_y[1:-1, 1:-1, 2])
        hv[1:-1, 1:-1, -1] = var.hv_vel_r * ( (vel_y[2::, 1:-1, -1] - 2*vel_y[1:-1, 1:-1, -1] + vel_y[0:-2, 1:-1, -1]) + (vel_y[1:-1, 2::, -1] - 2*vel_y[1:-1, 1:-1, -1] + vel_y[1:-1, 0:-2, -1]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_bnd * (vel_y[1:-1, 1:-1, -1] - 2*vel_y[1:-1, 1:-1, -2] + vel_y[1:-1, 1:-1, -3])
        
        # Edge Boundaries hv
        hv[0, 0, 1:-1] = var.hv_vel_r * var.hv_vel_edg * ( (vel_y[0, 0, 1:-1] - 2*vel_y[1, 0, 1:-1] + vel_y[2, 0, 1:-1]) + (vel_y[0, 0, 1:-1] - 2*vel_y[0, 1, 1:-1] + vel_y[0, 2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_y[0, 0, 2::] - 2*vel_y[0, 0, 1:-1] + vel_y[0, 0, 0:-2])
        hv[0, -1, 1:-1] = var.hv_vel_r * var.hv_vel_edg * ( (vel_y[0, -1, 1:-1] - 2*vel_y[1, -1, 1:-1] + vel_y[2, -1, 1:-1]) + (vel_y[0, -1, 1:-1] - 2*vel_y[0, -2, 1:-1] + vel_y[0, -3, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_y[0, -1, 2::] - 2*vel_y[0, -1, 1:-1] + vel_y[0, -1, 0:-2])
        hv[-1, 0, 1:-1] = var.hv_vel_r * var.hv_vel_edg * ( (vel_y[-1, 0, 1:-1] - 2*vel_y[-2, 0, 1:-1] + vel_y[-3, 0, 1:-1]) + (vel_y[-1, 0, 1:-1] - 2*vel_y[-1, 1, 1:-1] + vel_y[-1, 2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_y[-1, 0, 2::] - 2*vel_y[-1, 0, 1:-1] + vel_y[-1, 0, 0:-2])
        hv[-1, -1, 1:-1] = var.hv_vel_r * var.hv_vel_edg * ( (vel_y[-1, -1, 1:-1] - 2*vel_y[-2, -1, 1:-1] + vel_y[-3, -1, 1:-1]) + (vel_y[-1, -1, 1:-1] - 2*vel_y[-1, -2, 1:-1] + vel_y[-1, -3, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_y[-1, -1, 2::] - 2*vel_y[-1, -1, 1:-1] + vel_y[-1, -1, 0:-2])
        
        hv[0, 1:-1, 0] = var.hv_vel_r * ( var.hv_vel_edg * (vel_y[0, 1:-1, 0] - 2*vel_y[1, 1:-1, 0] + vel_y[2, 1:-1, 0]) + (vel_y[0, 2::, 0] - 2*vel_y[0, 1:-1, 0] + vel_y[0, 0:-2, 0]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_y[0, 1:-1, 0] - 2*vel_y[0, 1:-1, 1] + vel_y[0, 1:-1, 2])
        hv[-1, 1:-1, 0] = var.hv_vel_r * ( var.hv_vel_edg * (vel_y[-1, 1:-1, 0] - 2*vel_y[-2, 1:-1, 0] + vel_y[-3, 1:-1, 0]) + (vel_y[-1, 2::, 0] - 2*vel_y[-1, 1:-1, 0] + vel_y[-1, 0:-2, 0]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_y[-1, 1:-1, 0] - 2*vel_y[-1, 1:-1, 1] + vel_y[-1, 1:-1, 2])
        hv[1:-1, 0, 0] = var.hv_vel_r * ( (vel_y[2::, 0, 0] - 2*vel_y[1:-1, 0, 0] + vel_y[0:-2, 0, 0]) + var.hv_vel_edg * (vel_y[1:-1, 0, 0] - 2*vel_y[1:-1, 1, 0] + vel_y[1:-1, 2, 0]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_y[1:-1, 0, 0] - 2*vel_y[1:-1, 0, 1] + vel_y[1:-1, 0, 2])
        hv[1:-1, -1, 0] = var.hv_vel_r * ( (vel_y[2::, -1, 0] - 2*vel_y[1:-1, -1, 0] + vel_y[0:-2, -1, 0]) + var.hv_vel_edg * (vel_y[1:-1, -1, 0] - 2*vel_y[1:-1, -2, 0] + vel_y[1:-1, -3, 0]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_y[1:-1, -1, 0] - 2*vel_y[1:-1, -1, 1] + vel_y[1:-1, -1, 2])
        
        hv[0, 1:-1, -1] = var.hv_vel_r * ( var.hv_vel_edg * (vel_y[0, 1:-1, -1] - 2*vel_y[1, 1:-1, -1] + vel_y[2, 1:-1, -1]) + (vel_y[0, 2::, -1] - 2*vel_y[0, 1:-1, -1] + vel_y[0, 0:-2, -1]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_y[0, 1:-1, -1] - 2*vel_y[0, 1:-1, -2] + vel_y[0, 1:-1, -3])
        hv[-1, 1:-1, -1] = var.hv_vel_r * ( var.hv_vel_edg * (vel_y[-1, 1:-1, -1] - 2*vel_y[-2, 1:-1, -1] + vel_y[-3, 1:-1, -1]) + (vel_y[-1, 2::, -1] - 2*vel_y[-1, 1:-1, -1] + vel_y[-1, 0:-2, -1]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_y[-1, 1:-1, -1] - 2*vel_y[-1, 1:-1, -2] + vel_y[-1, 1:-1, -3])
        hv[1:-1, 0, -1] = var.hv_vel_r * ( (vel_y[2::, 0, -1] - 2*vel_y[1:-1, 0, -1] + vel_y[0:-2, 0, -1]) + var.hv_vel_edg * (vel_y[1:-1, 0, -1] - 2*vel_y[1:-1, 1, -1] + vel_y[1:-1, 2, -1]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_y[1:-1, 0, -1] - 2*vel_y[1:-1, 0, -2] + vel_y[1:-1, 0, -3])
        hv[1:-1, -1, -1] = var.hv_vel_r * ( (vel_y[2::, -1, -1] - 2*vel_y[1:-1, -1, -1] + vel_y[0:-2, -1, -1]) + var.hv_vel_edg * (vel_y[1:-1, -1, -1] - 2*vel_y[1:-1, -2, -1] + vel_y[1:-1, -3, -1]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_y[1:-1, -1, -1] - 2*vel_y[1:-1, -1, -2] + vel_y[1:-1, -1, -3])
        
        # Corner Boundaries hv
        hv[0, 0, 0] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_y[0,0,0] - 2*vel_y[1,0,0] + vel_y[2,0,0]) + (vel_y[0,0,0] - 2*vel_y[0,1,0] + vel_y[0,2,0]) ) + var.hv_vel_z * var.size_ratio * (vel_y[0,0,0] - 2*vel_y[0,0,1] + vel_y[0,0,2]) )
        hv[-1, 0, 0] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_y[-1,0,0] - 2*vel_y[-2,0,0] + vel_y[-3,0,0]) + (vel_y[-1,0,0] - 2*vel_y[-1,1,0] + vel_y[-1,2,0]) ) + var.hv_vel_z * var.size_ratio * (vel_y[-1,0,0] - 2*vel_y[-1,0,1] + vel_y[-1,0,2]) )
        hv[0, -1, 0] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_y[0,-1,0] - 2*vel_y[1,-1,0] + vel_y[2,-1,0]) + (vel_y[0,-1,0] - 2*vel_y[0,-2,0] + vel_y[0,-3,0]) ) + var.hv_vel_z * var.size_ratio * (vel_y[0,-1,0] - 2*vel_y[0,-1,1] + vel_y[0,-1,2]) )
        hv[-1, -1, 0] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_y[-1,-1,0] - 2*vel_y[-2,-1,0] + vel_y[-3,-1,0]) + (vel_y[-1,-1,0] - 2*vel_y[-1,-2,0] + vel_y[-1,-3,0]) ) + var.hv_vel_z * var.size_ratio * (vel_y[-1,-1,0] - 2*vel_y[-1,-1,1] + vel_y[-1,-1,2]) )
        
        hv[0, 0, -1] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_y[0,0,-1] - 2*vel_y[1,0,-1] + vel_y[2,0,-1]) + (vel_y[0,0,-1] - 2*vel_y[0,1,-1] + vel_y[0,2,-1]) ) + var.hv_vel_z * var.size_ratio * (vel_y[0,0,-1] - 2*vel_y[0,0,-2] + vel_y[0,0,-3]) )
        hv[-1, 0, -1] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_y[-1,0,-1] - 2*vel_y[-2,0,-1] + vel_y[-3,0,-1]) + (vel_y[-1,0,-1] - 2*vel_y[-1,1,-1] + vel_y[-1,2,-1]) ) + var.hv_vel_z * var.size_ratio * (vel_y[-1,0,-1] - 2*vel_y[-1,0,-2] + vel_y[-1,0,-3]) )
        hv[0, -1, -1] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_y[0,-1,-1] - 2*vel_y[1,-1,-1] + vel_y[2,-1,-1]) + (vel_y[0,-1,-1] - 2*vel_y[0,-2,-1] + vel_y[0,-3,-1]) ) + var.hv_vel_z * var.size_ratio * (vel_y[0,-1,-1] - 2*vel_y[0,-1,-2] + vel_y[0,-1,-3]) )
        hv[-1, -1, -1] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_y[-1,-1,-1] - 2*vel_y[-2,-1,-1] + vel_y[-3,-1,-1]) + (vel_y[-1,-1,-1] - 2*vel_y[-1,-2,-1] + vel_y[-1,-3,-1]) ) + var.hv_vel_z * var.size_ratio * (vel_y[-1,-1,-1] - 2*vel_y[-1,-1,-2] + vel_y[-1,-1,-3]) )
            
        # Central vel_y_out
        vel_y_out[1:-1, 1:-1, 1:-1] = vel_x[1:-1, 1:-1, 1:-1] * (vel_y[2::, 1:-1, 1:-1] - vel_y[0:-2, 1:-1, 1:-1]) + vel_y[1:-1, 1:-1, 1:-1] * (vel_y[1:-1, 2::, 1:-1] - vel_y[1:-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[1:-1, 1:-1, 1:-1] * (vel_y[1:-1, 1:-1, 2::] - vel_y[1:-1, 1:-1, 0:-2]) + 2 * k * den_inv[1:-1, 1:-1, 1:-1] * (den[1:-1, 2::, 1:-1] - den[1:-1, 0:-2, 1:-1])
        
        # X Boundary vel_y_out
        vel_y_out[0, 1:-1, 1:-1] = vel_x[0, 1:-1, 1:-1] * (vel_y[1, 1:-1, 1:-1] - vel_y[0, 1:-1, 1:-1]) + vel_y[0, 1:-1, 1:-1] * (vel_y[0, 2::, 1:-1] - vel_y[0, 0:-2, 1:-1]) + var.size_ratio * vel_z[0, 1:-1, 1:-1] * (vel_y[0, 1:-1, 2::] - vel_y[0, 1:-1, 0:-2]) + 2 * k * den_inv[0, 1:-1, 1:-1] * (den[0, 2::, 1:-1] - den[0, 0:-2, 1:-1])
        vel_y_out[-1, 1:-1, 1:-1] = vel_x[-1, 1:-1, 1:-1] * (vel_y[-1, 1:-1, 1:-1] - vel_y[-2, 1:-1, 1:-1]) + vel_y[-1, 1:-1, 1:-1] * (vel_y[-1, 2::, 1:-1] - vel_y[-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[-1, 1:-1, 1:-1] * (vel_y[-1, 1:-1, 2::] - vel_y[-1, 1:-1, 0:-2]) + 2 * k * den_inv[-1, 1:-1, 1:-1] * (den[-1, 2::, 1:-1] - den[-1, 0:-2, 1:-1])
        
        # Y Boundarty vel_y_out
        vel_y_out[1:-1, 0, 1:-1] = vel_x[1:-1, 0, 1:-1] * (vel_y[2::, 0, 1:-1] - vel_y[0:-2, 0, 1:-1]) + vel_y[1:-1, 0, 1:-1] * (vel_y[1:-1, 1, 1:-1] - vel_y[1:-1, 0, 1:-1]) + var.size_ratio * vel_z[1:-1, 0, 1:-1] * (vel_y[1:-1, 0, 2::] - vel_y[1:-1, 0, 0:-2]) + 2 * k * den_inv[1:-1, 0, 1:-1] * (den[1:-1, 1, 1:-1] - den[1:-1, 0, 1:-1])
        vel_y_out[1:-1, -1, 1:-1] = vel_x[1:-1, -1, 1:-1] * (vel_y[2::, -1, 1:-1] - vel_y[0:-2, -1, 1:-1]) + vel_y[1:-1, -1, 1:-1] * (vel_y[1:-1, -1, 1:-1] - vel_y[1:-1, -2, 1:-1]) + var.size_ratio * vel_z[1:-1, -1, 1:-1] * (vel_y[1:-1, -1, 2::] - vel_y[1:-1, -1, 0:-2]) + 2 * k * den_inv[1:-1, -1, 1:-1] * (den[1:-1, -1, 1:-1] - den[1:-1, -2, 1:-1])
        
        # Z Boundary vel_y_out
        vel_y_out[1:-1, 1:-1, 0] = vel_x[1:-1, 1:-1, 0] * (vel_y[2::, 1:-1, 0] - vel_y[0:-2, 1:-1, 0]) + vel_y[1:-1, 1:-1, 0] * (vel_y[1:-1, 2::, 0] - vel_y[1:-1, 0:-2, 0]) + var.size_ratio * vel_z[1:-1, 1:-1, 0] * (vel_y[1:-1, 1:-1, 1] - vel_y[1:-1, 1:-1, 0]) + 2 * k * den_inv[1:-1, 1:-1, 0] * (den[1:-1, 2::, 0] - den[1:-1, 0:-2, 0])
        vel_y_out[1:-1, 1:-1, -1] = vel_x[1:-1, 1:-1, -1] * (vel_y[2::, 1:-1, -1] - vel_y[0:-2, 1:-1, -1]) + vel_y[1:-1, 1:-1, -1] * (vel_y[1:-1, 2::, -1] - vel_y[1:-1, 0:-2, -1]) + var.size_ratio * vel_z[1:-1, 1:-1, -1] * (vel_y[1:-1, 1:-1, -1] - vel_y[1:-1, 1:-1, -2]) + 2 * k * den_inv[1:-1, 1:-1, -1] * (den[1:-1, 2::, -1] - den[1:-1, 0:-2, -1])
        
        # Edge Boundary vel_y_out
        vel_y_out[0, 0, 1:-1] = vel_x[0, 0, 1:-1] * (vel_y[1, 0, 1:-1] - vel_y[0, 0, 1:-1]) + vel_y[0, 0, 1:-1] * (vel_y[0, 1, 1:-1] - vel_y[0, 0, 1:-1]) + var.size_ratio * vel_z[0, 0, 1:-1] * (vel_y[0, 0, 2::] - vel_y[0, 0, 0:-2]) + 2 * k * den_inv[0, 0, 1:-1] * (den[0, 1, 1:-1] - den[0, 0, 1:-1])
        vel_y_out[0, -1, 1:-1] = vel_x[0, -1, 1:-1] * (vel_y[1, -1, 1:-1] - vel_y[0, -1, 1:-1]) + vel_y[0, -1, 1:-1] * (vel_y[0, -1, 1:-1] - vel_y[0, -2, 1:-1]) + var.size_ratio * vel_z[0, -1, 1:-1] * (vel_y[0, -1, 2::] - vel_y[0, -1, 0:-2]) + 2 * k * den_inv[0, -1, 1:-1] * (den[0, -1, 1:-1] - den[0, -2, 1:-1])
        vel_y_out[-1, 0, 1:-1] = vel_x[-1, 0, 1:-1] * (vel_y[-1, 0, 1:-1] - vel_y[-2, 0, 1:-1]) + vel_y[-1, 0, 1:-1] * (vel_y[-1, 1, 1:-1] - vel_y[-1, 0, 1:-1]) + var.size_ratio * vel_z[-1, 0, 1:-1] * (vel_y[-1, 0, 2::] - vel_y[-1, 0, 0:-2]) + 2 * k * den_inv[-1, 0, 1:-1] * (den[-1, 1, 1:-1] - den[-1, 0, 1:-1])
        vel_y_out[-1, -1, 1:-1] = vel_x[-1, -1, 1:-1] * (vel_y[-1, -1, 1:-1] - vel_y[-2, -1, 1:-1]) + vel_y[-1, -1, 1:-1] * (vel_y[-1, -1, 1:-1] - vel_y[-1, -2, 1:-1]) + var.size_ratio * vel_z[-1, -1, 1:-1] * (vel_y[-1, -1, 2::] - vel_y[-1, -1, 0:-2]) + 2 * k * den_inv[-1, -1, 1:-1] * (den[-1, -1, 1:-1] - den[-1, -2, 1:-1])
    
        vel_y_out[0, 1:-1, 0] = vel_x[0, 1:-1, 0] * (vel_y[1, 1:-1, 0] - vel_y[0, 1:-1, 0]) + vel_y[0, 1:-1, 0] * (vel_y[0, 2::, 0] - vel_y[0, 0:-2, 0]) + var.size_ratio * vel_z[0, 1:-1, 0] * (vel_y[0, 1:-1, 1] - vel_y[0, 1:-1, 0]) + 2 * k * den_inv[0, 1:-1, 0] * (den[0, 2::, 0] - den[0, 0:-2, 0])
        vel_y_out[0, 1:-1, -1] = vel_x[0, 1:-1, -1] * (vel_y[1, 1:-1, -1] - vel_y[0, 1:-1, -1]) + vel_y[0, 1:-1, -1] * (vel_y[0, 2::, -1] - vel_y[0, 0:-2, -1]) + var.size_ratio * vel_z[0, 1:-1, -1] * (vel_y[0, 1:-1, -1] - vel_y[0, 1:-1, -2]) + 2 * k * den_inv[0, 1:-1, -1] * (den[0, 2::, -1] - den[0, 0:-2, -1])
        vel_y_out[-1, 1:-1, 0] = vel_x[-1, 1:-1, 0] * (vel_y[-1, 1:-1, 0] - vel_y[-2, 1:-1, 0]) + vel_y[-1, 1:-1, 0] * (vel_y[-1, 2::, 0] - vel_y[-1, 0:-2, 0]) + var.size_ratio * vel_z[-1, 1:-1, 0] * (vel_y[-1, 1:-1, 1] - vel_y[-1, 1:-1, 0]) + 2 * k * den_inv[-1, 1:-1, 0] * (den[-1, 2::, 0] - den[-1, 0:-2, 0])
        vel_y_out[-1, 1:-1, -1] = vel_x[-1, 1:-1, -1] * (vel_y[-1, 1:-1, -1] - vel_y[-2, 1:-1, -1]) + vel_y[-1, 1:-1, -1] * (vel_y[-1, 2::, -1] - vel_y[-1, 0:-2, -1]) + var.size_ratio * vel_z[-1, 1:-1, -1] * (vel_y[-1, 1:-1, -1] - vel_y[-1, 1:-1, -2]) + 2 * k * den_inv[-1, 1:-1, -1] * (den[-1, 2::, -1] - den[-1, 0:-2, -1])
    
        # Corner Boundary vel_y_out
        vel_y_out[0,0,0] = vel_x[0,0,0] * (vel_y[1,0,0] - vel_y[0,0,0]) + vel_y[0,0,0] * (vel_y[0,1,0] - vel_y[0,0,0]) + var.size_ratio * vel_z[0,0,0] * (vel_y[0,0,1] - vel_y[0,0,0]) + 2 * k * den_inv[0,0,0] * (den[0,1,0] - den[0,0,0])
        vel_y_out[0,-1,0] = vel_x[0,-1,0] * (vel_y[1,-1,0] - vel_y[0,-1,0]) + vel_y[0,-1,0] * (vel_y[0,-2,0] - vel_y[0,-1,0]) + var.size_ratio * vel_z[0,-1,0] * (vel_y[0,-1,1] - vel_y[0,-1,0]) + 2 * k * den_inv[0,-1,0] * (den[0,-1,0] - den[0,-2,0])
        vel_y_out[-1,0,0] = vel_x[-1,0,0] * (vel_y[-1,0,0] - vel_y[-2,0,0]) + vel_y[-1,0,0] * (vel_y[-1,1,0] - vel_y[-1,0,0]) + var.size_ratio * vel_z[-1,0,0] * (vel_y[-1,0,1] - vel_y[-1,0,0]) + 2 * k * den_inv[-1,0,0] * (den[-1,1,0] - den[-1,0,0])
        vel_y_out[-1,-1,0] = vel_x[-1,-1,0] * (vel_y[-1,-1,0] - vel_y[-2,-1,0]) + vel_y[-1,-1,0] * (vel_y[-1,-2,0] - vel_y[-1,-1,0]) + var.size_ratio * vel_z[-1,-1,0] * (vel_y[-1,-1,1] - vel_y[-1,-1,0]) + 2 * k * den_inv[-1,-1,0] * (den[-1,-1,0] - den[-1,-2,0])
        
        vel_y_out[0,0,-1] = vel_x[0,0,-1] * (vel_y[1,0,-1] - vel_y[0,0,-1]) + vel_y[0,0,-1] * (vel_y[0,1,-1] - vel_y[0,0,-1]) + var.size_ratio * vel_z[0,0,-1] * (vel_y[0,0,-1] - vel_y[0,0,-2]) + 2 * k * den_inv[0,0,-1] * (den[0,1,-1] - den[0,0,-1])
        vel_y_out[0,-1,-1] = vel_x[0,-1,-1] * (vel_y[1,-1,-1] - vel_y[0,-1,-1]) + vel_y[0,-1,-1] * (vel_y[0,-2,-1] - vel_y[0,-1,-1]) + var.size_ratio * vel_z[0,-1,-1] * (vel_y[0,-1,-1] - vel_y[0,-1,-2]) + 2 * k * den_inv[0,-1,-1] * (den[0,-1,-1] - den[0,-2,-1])
        vel_y_out[-1,0,-1] = vel_x[-1,0,-1] * (vel_y[-1,0,-1] - vel_y[-2,0,-1]) + vel_y[-1,0,-1] * (vel_y[-1,1,-1] - vel_y[-1,0,-1]) + var.size_ratio * vel_z[-1,0,-1] * (vel_y[-1,0,-1] - vel_y[-1,0,-2]) + 2 * k * den_inv[-1,0,-1] * (den[-1,1,-1] - den[-1,0,-1])
        vel_y_out[-1,-1,-1] = vel_x[-1,-1,-1] * (vel_y[-1,-1,-1] - vel_y[-2,-1,-1]) + vel_y[-1,-1,-1] * (vel_y[-1,-2,-1] - vel_y[-1,-1,-1]) + var.size_ratio * vel_z[-1,-1,-1] * (vel_y[-1,-1,-1] - vel_y[-1,-1,-2]) + 2 * k * den_inv[-1,-1,-1] * (den[-1,-1,-1] - den[-1,-2,-1])
        
        # Out Put Calcuation
        vel_y_out = vel_y_out - hv    
        vel_y_out = vel_y - .5 * var.step_ratio * vel_y_out - var.dT * vel_sh

    return vel_y_out
    
def calculate_velocity_z(vel_x, vel_y, vel_z, den, den_inv, vel_sh, k, Zsize, periodic):
    """ Returns time derivative of fluid velocity in z direction.
    
    Parameters
    ----------
    vel_x : array
        fluid velocity in x direction    
    
    vel_y : array
        fluid velocity in y direction    

    vel_z : array
        fluid velocity in z direction    

    den : array
        fluid density    
    
    den_inv : array
        inverse fluid density   

    vel_sh : array
        source/sink terms in momentum equation
        
    k : float
        scalar for density gradient
    
    Zsize : int
        number of dimensions along axis=2
    """
    if periodic:
        # Define Data Size
        hv = np.zeros(vel_z.shape)
        vel_z_out = np.zeros(vel_z.shape)
        
        # Calculate Central Differences
        hv[1:-1, 1:-1, 1:-1] = var.hv_vel_r * ( (vel_z[2::, 1:-1, 1:-1] - 2*vel_z[1:-1, 1:-1, 1:-1] + vel_z[0:-2, 1:-1, 1:-1]) + (vel_z[1:-1, 2::, 1:-1] - 2*vel_z[1:-1, 1:-1, 1:-1] + vel_z[1:-1, 0:-2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_z[1:-1, 1:-1, 2::] - 2*vel_z[1:-1, 1:-1, 1:-1] + vel_z[1:-1, 1:-1, 0:-2]) 
        vel_z_out[1:-1, 1:-1, 1:-1] = vel_x[1:-1, 1:-1, 1:-1] * (vel_z[2::, 1:-1, 1:-1] - vel_z[0:-2, 1:-1, 1:-1]) + vel_y[1:-1, 1:-1, 1:-1] * (vel_z[1:-1, 2::, 1:-1] - vel_z[1:-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[1:-1, 1:-1, 1:-1] * (vel_z[1:-1, 1:-1, 2::] - vel_z[1:-1, 1:-1, 0:-2]) + var.size_ratio * 2 * k * den_inv[1:-1, 1:-1, 1:-1] * (den[1:-1, 1:-1, 2::] - den[1:-1, 1:-1, 0:-2])
        
        # Out Put Calulation
        vel_z_out[1:-1, 1:-1, 1:-1] = vel_z_out[1:-1, 1:-1, 1:-1] - hv[1:-1, 1:-1, 1:-1]
        vel_z_out[1:-1, 1:-1, 1:-1] = vel_z[1:-1, 1:-1, 1:-1] - .5 * var.step_ratio * vel_z_out[1:-1, 1:-1, 1:-1] - var.dT * vel_sh[1:-1, 1:-1, 1:-1]
        
        # Periodic Boundary Calculation
        vel_z_out = periodic_boundary_vel_z(vel_z_out) 
        
    else:
        # Define Data Size
        hv = np.empty(vel_z.shape)
        vel_z_out = np.empty(vel_z.shape)
        
        # Central hv
        hv[1:-1, 1:-1, 1:-1] = var.hv_vel_r * ( (vel_z[2::, 1:-1, 1:-1] - 2*vel_z[1:-1, 1:-1, 1:-1] + vel_z[0:-2, 1:-1, 1:-1]) + (vel_z[1:-1, 2::, 1:-1] - 2*vel_z[1:-1, 1:-1, 1:-1] + vel_z[1:-1, 0:-2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_z[1:-1, 1:-1, 2::] - 2*vel_z[1:-1, 1:-1, 1:-1] + vel_z[1:-1, 1:-1, 0:-2]) 
        
        # X Boundary hv
        hv[0, 1:-1, 1:-1] = var.hv_vel_r * ( var.hv_vel_bnd * (vel_z[0, 1:-1, 1:-1] - 2*vel_z[1, 1:-1, 1:-1] + vel_z[2, 1:-1, 1:-1]) + (vel_z[0, 2::, 1:-1] - 2*vel_z[0, 1:-1, 1:-1] + vel_z[0, 0:-2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_z[0, 1:-1, 2::] - 2*vel_z[0, 1:-1, 1:-1] + vel_z[0, 1:-1, 0:-2])
        hv[-1, 1:-1, 1:-1] = var.hv_vel_r * ( var.hv_vel_bnd * (vel_z[-1, 1:-1, 1:-1] - 2*vel_z[-2, 1:-1, 1:-1] + vel_z[-3, 1:-1, 1:-1]) + (vel_z[-1, 2::, 1:-1] - 2*vel_z[-1, 1:-1, 1:-1] + vel_z[-1, 0:-2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_z[-1, 1:-1, 2::] - 2*vel_z[-1, 1:-1, 1:-1] + vel_z[-1, 1:-1, 0:-2])
        
        # Y Boundary hv
        hv[1:-1, 0, 1:-1] = var.hv_vel_r * ( (vel_z[2::, 0, 1:-1] - 2*vel_z[1:-1, 0, 1:-1] + vel_z[0:-2, 0, 1:-1]) + var.hv_vel_bnd * (vel_z[1:-1, 0, 1:-1] - 2*vel_z[1:-1, 1, 1:-1] + vel_z[1:-1, 2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_z[1:-1, 0, 2::] - 2*vel_z[1:-1, 0, 1:-1] + vel_z[1:-1, 0, 0:-2])
        hv[1:-1, -1, 1:-1] = var.hv_vel_r * ( (vel_z[2::, -1, 1:-1] - 2*vel_z[1:-1, -1, 1:-1] + vel_z[0:-2, -1, 1:-1]) + var.hv_vel_bnd * (vel_z[1:-1, -1, 1:-1] - 2*vel_z[1:-1, -2, 1:-1] + vel_z[1:-1, -3, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_z[1:-1, -1, 2::] - 2*vel_z[1:-1, -1, 1:-1] + vel_z[1:-1, -1, 0:-2])
        
        # Z Boundary hv
        hv[1:-1, 1:-1, 0] = var.hv_vel_r * ( (vel_z[2::, 1:-1, 0] - 2*vel_z[1:-1, 1:-1, 0] + vel_z[0:-2, 1:-1, 0]) + (vel_z[1:-1, 2::, 0] - 2*vel_z[1:-1, 1:-1, 0] + vel_z[1:-1, 0:-2, 0]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_bnd * (vel_z[1:-1, 1:-1, 0] - 2*vel_z[1:-1, 1:-1, 1] + vel_z[1:-1, 1:-1, 2])
        hv[1:-1, 1:-1, -1] = var.hv_vel_r * ( (vel_z[2::, 1:-1, -1] - 2*vel_z[1:-1, 1:-1, -1] + vel_z[0:-2, 1:-1, -1]) + (vel_z[1:-1, 2::, -1] - 2*vel_z[1:-1, 1:-1, -1] + vel_z[1:-1, 0:-2, -1]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_bnd * (vel_z[1:-1, 1:-1, -1] - 2*vel_z[1:-1, 1:-1, -2] + vel_z[1:-1, 1:-1, -3])
        
        # Edge Boundaries hv
        hv[0, 0, 1:-1] = var.hv_vel_r * var.hv_vel_edg * ( (vel_z[0, 0, 1:-1] - 2*vel_z[1, 0, 1:-1] + vel_z[2, 0, 1:-1]) + (vel_z[0, 0, 1:-1] - 2*vel_z[0, 1, 1:-1] + vel_z[0, 2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_z[0, 0, 2::] - 2*vel_z[0, 0, 1:-1] + vel_z[0, 0, 0:-2])
        hv[0, -1, 1:-1] = var.hv_vel_r * var.hv_vel_edg * ( (vel_z[0, -1, 1:-1] - 2*vel_z[1, -1, 1:-1] + vel_z[2, -1, 1:-1]) + (vel_z[0, -1, 1:-1] - 2*vel_z[0, -2, 1:-1] + vel_z[0, -3, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_z[0, -1, 2::] - 2*vel_z[0, -1, 1:-1] + vel_z[0, -1, 0:-2])
        hv[-1, 0, 1:-1] = var.hv_vel_r * var.hv_vel_edg * ( (vel_z[-1, 0, 1:-1] - 2*vel_z[-2, 0, 1:-1] + vel_z[-3, 0, 1:-1]) + (vel_z[-1, 0, 1:-1] - 2*vel_z[-1, 1, 1:-1] + vel_z[-1, 2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_z[-1, 0, 2::] - 2*vel_z[-1, 0, 1:-1] + vel_z[-1, 0, 0:-2])
        hv[-1, -1, 1:-1] = var.hv_vel_r * var.hv_vel_edg * ( (vel_z[-1, -1, 1:-1] - 2*vel_z[-2, -1, 1:-1] + vel_z[-3, -1, 1:-1]) + (vel_z[-1, -1, 1:-1] - 2*vel_z[-1, -2, 1:-1] + vel_z[-1, -3, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_z[-1, -1, 2::] - 2*vel_z[-1, -1, 1:-1] + vel_z[-1, -1, 0:-2])
        
        hv[0, 1:-1, 0] = var.hv_vel_r * ( var.hv_vel_edg * (vel_z[0, 1:-1, 0] - 2*vel_z[1, 1:-1, 0] + vel_z[2, 1:-1, 0]) + (vel_z[0, 2::, 0] - 2*vel_z[0, 1:-1, 0] + vel_z[0, 0:-2, 0]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_z[0, 1:-1, 0] - 2*vel_z[0, 1:-1, 1] + vel_z[0, 1:-1, 2])
        hv[-1, 1:-1, 0] = var.hv_vel_r * ( var.hv_vel_edg * (vel_z[-1, 1:-1, 0] - 2*vel_z[-2, 1:-1, 0] + vel_z[-3, 1:-1, 0]) + (vel_z[-1, 2::, 0] - 2*vel_z[-1, 1:-1, 0] + vel_z[-1, 0:-2, 0]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_z[-1, 1:-1, 0] - 2*vel_z[-1, 1:-1, 1] + vel_z[-1, 1:-1, 2])
        hv[1:-1, 0, 0] = var.hv_vel_r * ( (vel_z[2::, 0, 0] - 2*vel_z[1:-1, 0, 0] + vel_z[0:-2, 0, 0]) + var.hv_vel_edg * (vel_z[1:-1, 0, 0] - 2*vel_z[1:-1, 1, 0] + vel_z[1:-1, 2, 0]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_z[1:-1, 0, 0] - 2*vel_z[1:-1, 0, 1] + vel_z[1:-1, 0, 2])
        hv[1:-1, -1, 0] = var.hv_vel_r * ( (vel_z[2::, -1, 0] - 2*vel_z[1:-1, -1, 0] + vel_z[0:-2, -1, 0]) + var.hv_vel_edg * (vel_z[1:-1, -1, 0] - 2*vel_z[1:-1, -2, 0] + vel_z[1:-1, -3, 0]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_z[1:-1, -1, 0] - 2*vel_z[1:-1, -1, 1] + vel_z[1:-1, -1, 2])
        
        hv[0, 1:-1, -1] = var.hv_vel_r * ( var.hv_vel_edg * (vel_z[0, 1:-1, -1] - 2*vel_z[1, 1:-1, -1] + vel_z[2, 1:-1, -1]) + (vel_z[0, 2::, -1] - 2*vel_z[0, 1:-1, -1] + vel_z[0, 0:-2, -1]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_z[0, 1:-1, -1] - 2*vel_z[0, 1:-1, -2] + vel_z[0, 1:-1, -3])
        hv[-1, 1:-1, -1] = var.hv_vel_r * ( var.hv_vel_edg * (vel_z[-1, 1:-1, -1] - 2*vel_z[-2, 1:-1, -1] + vel_z[-3, 1:-1, -1]) + (vel_z[-1, 2::, -1] - 2*vel_z[-1, 1:-1, -1] + vel_z[-1, 0:-2, -1]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_z[-1, 1:-1, -1] - 2*vel_z[-1, 1:-1, -2] + vel_z[-1, 1:-1, -3])
        hv[1:-1, 0, -1] = var.hv_vel_r * ( (vel_z[2::, 0, -1] - 2*vel_z[1:-1, 0, -1] + vel_z[0:-2, 0, -1]) + var.hv_vel_edg * (vel_z[1:-1, 0, -1] - 2*vel_z[1:-1, 1, -1] + vel_z[1:-1, 2, -1]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_z[1:-1, 0, -1] - 2*vel_z[1:-1, 0, -2] + vel_z[1:-1, 0, -3])
        hv[1:-1, -1, -1] = var.hv_vel_r * ( (vel_z[2::, -1, -1] - 2*vel_z[1:-1, -1, -1] + vel_z[0:-2, -1, -1]) + var.hv_vel_edg * (vel_z[1:-1, -1, -1] - 2*vel_z[1:-1, -2, -1] + vel_z[1:-1, -3, -1]) ) + var.hv_vel_z * var.size_ratio * var.hv_vel_edg * (vel_z[1:-1, -1, -1] - 2*vel_z[1:-1, -1, -2] + vel_z[1:-1, -1, -3])
        
        # Corner Boundaries hv
        hv[0, 0, 0] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_z[0,0,0] - 2*vel_z[1,0,0] + vel_z[2,0,0]) + (vel_z[0,0,0] - 2*vel_z[0,1,0] + vel_z[0,2,0]) ) + var.hv_vel_z * var.size_ratio * (vel_z[0,0,0] - 2*vel_z[0,0,1] + vel_z[0,0,2]) )
        hv[-1, 0, 0] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_z[-1,0,0] - 2*vel_z[-2,0,0] + vel_z[-3,0,0]) + (vel_z[-1,0,0] - 2*vel_z[-1,1,0] + vel_z[-1,2,0]) ) + var.hv_vel_z * var.size_ratio * (vel_z[-1,0,0] - 2*vel_z[-1,0,1] + vel_z[-1,0,2]) )
        hv[0, -1, 0] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_z[0,-1,0] - 2*vel_z[1,-1,0] + vel_z[2,-1,0]) + (vel_z[0,-1,0] - 2*vel_z[0,-2,0] + vel_z[0,-3,0]) ) + var.hv_vel_z * var.size_ratio * (vel_z[0,-1,0] - 2*vel_z[0,-1,1] + vel_z[0,-1,2]) )
        hv[-1, -1, 0] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_z[-1,-1,0] - 2*vel_z[-2,-1,0] + vel_z[-3,-1,0]) + (vel_z[-1,-1,0] - 2*vel_z[-1,-2,0] + vel_z[-1,-3,0]) ) + var.hv_vel_z * var.size_ratio * (vel_z[-1,-1,0] - 2*vel_z[-1,-1,1] + vel_z[-1,-1,2]) )
        
        hv[0, 0, -1] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_z[0,0,-1] - 2*vel_z[1,0,-1] + vel_z[2,0,-1]) + (vel_z[0,0,-1] - 2*vel_z[0,1,-1] + vel_z[0,2,-1]) ) + var.hv_vel_z * var.size_ratio * (vel_z[0,0,-1] - 2*vel_z[0,0,-2] + vel_z[0,0,-3]) )
        hv[-1, 0, -1] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_z[-1,0,-1] - 2*vel_z[-2,0,-1] + vel_z[-3,0,-1]) + (vel_z[-1,0,-1] - 2*vel_z[-1,1,-1] + vel_z[-1,2,-1]) ) + var.hv_vel_z * var.size_ratio * (vel_z[-1,0,-1] - 2*vel_z[-1,0,-2] + vel_z[-1,0,-3]) )
        hv[0, -1, -1] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_z[0,-1,-1] - 2*vel_z[1,-1,-1] + vel_z[2,-1,-1]) + (vel_z[0,-1,-1] - 2*vel_z[0,-2,-1] + vel_z[0,-3,-1]) ) + var.hv_vel_z * var.size_ratio * (vel_z[0,-1,-1] - 2*vel_z[0,-1,-2] + vel_z[0,-1,-3]) )
        hv[-1, -1, -1] = var.hv_vel_bnd * ( var.hv_vel_r * ( (vel_z[-1,-1,-1] - 2*vel_z[-2,-1,-1] + vel_z[-3,-1,-1]) + (vel_z[-1,-1,-1] - 2*vel_z[-1,-2,-1] + vel_z[-1,-3,-1]) ) + var.hv_vel_z * var.size_ratio * (vel_z[-1,-1,-1] - 2*vel_z[-1,-1,-2] + vel_z[-1,-1,-3]) )
        
        # Central vel_z_out        
        vel_z_out[1:-1, 1:-1, 1:-1] = vel_x[1:-1, 1:-1, 1:-1] * (vel_z[2::, 1:-1, 1:-1] - vel_z[0:-2, 1:-1, 1:-1]) + vel_y[1:-1, 1:-1, 1:-1] * (vel_z[1:-1, 2::, 1:-1] - vel_z[1:-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[1:-1, 1:-1, 1:-1] * (vel_z[1:-1, 1:-1, 2::] - vel_z[1:-1, 1:-1, 0:-2]) + var.size_ratio * 2 * k * den_inv[1:-1, 1:-1, 1:-1] * (den[1:-1, 1:-1, 2::] - den[1:-1, 1:-1, 0:-2])
        
        # X Boundary vel_z_out
        vel_z_out[0, 1:-1, 1:-1] = vel_x[0, 1:-1, 1:-1] * (vel_z[1, 1:-1, 1:-1] - vel_z[0, 1:-1, 1:-1]) + vel_y[0, 1:-1, 1:-1] * (vel_z[0, 2::, 1:-1] - vel_z[0, 0:-2, 1:-1]) + var.size_ratio * vel_z[0, 1:-1, 1:-1] * (vel_z[0, 1:-1, 2::] - vel_z[0, 1:-1, 0:-2]) + 2 * k * den_inv[0, 1:-1, 1:-1] * (den[0, 1:-1, 2::] - den[0, 1:-1, 0:-2])
        vel_z_out[-1, 1:-1, 1:-1] = vel_x[-1, 1:-1, 1:-1] * (vel_z[-1, 1:-1, 1:-1] - vel_z[-2, 1:-1, 1:-1]) + vel_y[-1, 1:-1, 1:-1] * (vel_z[-1, 2::, 1:-1] - vel_z[-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[-1, 1:-1, 1:-1] * (vel_z[-1, 1:-1, 2::] - vel_z[-1, 1:-1, 0:-2]) + 2 * k * den_inv[-1, 1:-1, 1:-1] * (den[-1, 1:-1, 2::] - den[-1, 1:-1, 0:-2])
        
        # Y Boundarty vel_z_out
        vel_z_out[1:-1, 0, 1:-1] = vel_x[1:-1, 0, 1:-1] * (vel_z[2::, 0, 1:-1] - vel_z[0:-2, 0, 1:-1]) + vel_y[1:-1, 0, 1:-1] * (vel_z[1:-1, 1, 1:-1] - vel_z[1:-1, 0, 1:-1]) + var.size_ratio * vel_z[1:-1, 0, 1:-1] * (vel_z[1:-1, 0, 2::] - vel_z[1:-1, 0, 0:-2]) + 2 * k * den_inv[1:-1, 0, 1:-1] * (den[1:-1, 0, 2::] - den[1:-1, 0, 0:-2])
        vel_z_out[1:-1, -1, 1:-1] = vel_x[1:-1, -1, 1:-1] * (vel_z[2::, -1, 1:-1] - vel_z[0:-2, -1, 1:-1]) + vel_y[1:-1, -1, 1:-1] * (vel_z[1:-1, -1, 1:-1] - vel_z[1:-1, -2, 1:-1]) + var.size_ratio * vel_z[1:-1, -1, 1:-1] * (vel_z[1:-1, -1, 2::] - vel_z[1:-1, -1, 0:-2]) + 2 * k * den_inv[1:-1, -1, 1:-1] * (den[1:-1, -1, 2::] - den[1:-1, -1, 0:-2])
        
        # Z Boundary vel_z_out
        vel_z_out[1:-1, 1:-1, 0] = vel_x[1:-1, 1:-1, 0] * (vel_z[2::, 1:-1, 0] - vel_z[0:-2, 1:-1, 0]) + vel_y[1:-1, 1:-1, 0] * (vel_z[1:-1, 2::, 0] - vel_z[1:-1, 0:-2, 0]) + var.size_ratio * vel_z[1:-1, 1:-1, 0] * (vel_z[1:-1, 1:-1, 1] - vel_z[1:-1, 1:-1, 0]) + 2 * k * den_inv[1:-1, 1:-1, 0] * (den[1:-1, 1:-1, 1] - den[1:-1, 1:-1, 0])
        vel_z_out[1:-1, 1:-1, -1] = vel_x[1:-1, 1:-1, -1] * (vel_z[2::, 1:-1, -1] - vel_z[0:-2, 1:-1, -1]) + vel_y[1:-1, 1:-1, -1] * (vel_z[1:-1, 2::, -1] - vel_z[1:-1, 0:-2, -1]) + var.size_ratio * vel_z[1:-1, 1:-1, -1] * (vel_z[1:-1, 1:-1, -1] - vel_z[1:-1, 1:-1, -2]) + 2 * k * den_inv[1:-1, 1:-1, -1] * (den[1:-1, 1:-1, -1] - den[1:-1, 1:-1, -2])
        
        # Edge Boundary vel_z_out
        vel_z_out[0, 0, 1:-1] = vel_x[0, 0, 1:-1] * (vel_z[1, 0, 1:-1] - vel_z[0, 0, 1:-1]) + vel_y[0, 0, 1:-1] * (vel_z[0, 1, 1:-1] - vel_z[0, 0, 1:-1]) + var.size_ratio * vel_z[0, 0, 1:-1] * (vel_z[0, 0, 2::] - vel_z[0, 0, 0:-2]) + 2 * k * den_inv[0, 0, 1:-1] * (den[0, 0, 2::] - den[0, 0, 0:-2])
        vel_z_out[0, -1, 1:-1] = vel_x[0, -1, 1:-1] * (vel_z[1, -1, 1:-1] - vel_z[0, -1, 1:-1]) + vel_y[0, -1, 1:-1] * (vel_z[0, -1, 1:-1] - vel_z[0, -2, 1:-1]) + var.size_ratio * vel_z[0, -1, 1:-1] * (vel_z[0, -1, 2::] - vel_z[0, -1, 0:-2]) + 2 * k * den_inv[0, -1, 1:-1] * (den[0, -1, 2::] - den[0, -1, 0:-2])
        vel_z_out[-1, 0, 1:-1] = vel_x[-1, 0, 1:-1] * (vel_z[-1, 0, 1:-1] - vel_z[-2, 0, 1:-1]) + vel_y[-1, 0, 1:-1] * (vel_z[-1, 1, 1:-1] - vel_z[-1, 0, 1:-1]) + var.size_ratio * vel_z[-1, 0, 1:-1] * (vel_z[-1, 0, 2::] - vel_z[-1, 0, 0:-2]) + 2 * k * den_inv[-1, 0, 1:-1] * (den[-1, 0, 2::] - den[-1, 0, 0:-2])
        vel_z_out[-1, -1, 1:-1] = vel_x[-1, -1, 1:-1] * (vel_z[-1, -1, 1:-1] - vel_z[-2, -1, 1:-1]) + vel_y[-1, -1, 1:-1] * (vel_z[-1, -1, 1:-1] - vel_z[-1, -2, 1:-1]) + var.size_ratio * vel_z[-1, -1, 1:-1] * (vel_z[-1, -1, 2::] - vel_z[-1, -1, 0:-2]) + 2 * k * den_inv[-1, -1, 1:-1] * (den[-1, -1, 2::] - den[-1, -1, 0:-2])
    
        vel_z_out[0, 1:-1, 0] = vel_x[0, 1:-1, 0] * (vel_z[1, 1:-1, 0] - vel_z[0, 1:-1, 0]) + vel_y[0, 1:-1, 0] * (vel_z[0, 2::, 0] - vel_z[0, 0:-2, 0]) + var.size_ratio * vel_z[0, 1:-1, 0] * (vel_z[0, 1:-1, 1] - vel_z[0, 1:-1, 0]) + 2 * k * den_inv[0, 1:-1, 0] * (den[0, 1:-1, 1] - den[0, 1:-1, 0])
        vel_z_out[0, 1:-1, -1] = vel_x[0, 1:-1, -1] * (vel_z[1, 1:-1, -1] - vel_z[0, 1:-1, -1]) + vel_y[0, 1:-1, -1] * (vel_z[0, 2::, -1] - vel_z[0, 0:-2, -1]) + var.size_ratio * vel_z[0, 1:-1, -1] * (vel_z[0, 1:-1, -1] - vel_z[0, 1:-1, -2]) + 2 * k * den_inv[0, 1:-1, -1] * (den[0, 1:-1, -1] - den[0, 1:-1, -2])
        vel_z_out[-1, 1:-1, 0] = vel_x[-1, 1:-1, 0] * (vel_z[-1, 1:-1, 0] - vel_z[-2, 1:-1, 0]) + vel_y[-1, 1:-1, 0] * (vel_z[-1, 2::, 0] - vel_z[-1, 0:-2, 0]) + var.size_ratio * vel_z[-1, 1:-1, 0] * (vel_z[-1, 1:-1, 1] - vel_z[-1, 1:-1, 0]) + 2 * k * den_inv[-1, 1:-1, 0] * (den[-1, 1:-1, 1] - den[-1, 1:-1, 0])
        vel_z_out[-1, 1:-1, -1] = vel_x[-1, 1:-1, -1] * (vel_z[-1, 1:-1, -1] - vel_z[-2, 1:-1, -1]) + vel_y[-1, 1:-1, -1] * (vel_z[-1, 2::, -1] - vel_z[-1, 0:-2, -1]) + var.size_ratio * vel_z[-1, 1:-1, -1] * (vel_z[-1, 1:-1, -1] - vel_z[-1, 1:-1, -2]) + 2 * k * den_inv[-1, 1:-1, -1] * (den[-1, 1:-1, -1] - den[-1, 1:-1, -2])
    
        # Corner Boundary vel_z_out
        vel_z_out[0,0,0] = vel_x[0,0,0] * (vel_z[1,0,0] - vel_z[0,0,0]) + vel_y[0,0,0] * (vel_z[0,1,0] - vel_z[0,0,0]) + var.size_ratio * vel_z[0,0,0] * (vel_z[0,0,1] - vel_z[0,0,0]) + 2 * k * den_inv[0,0,0] * (den[0,0,1] - den[0,0,0])
        vel_z_out[0,-1,0] = vel_x[0,-1,0] * (vel_z[1,-1,0] - vel_z[0,-1,0]) + vel_y[0,-1,0] * (vel_z[0,-2,0] - vel_z[0,-1,0]) + var.size_ratio * vel_z[0,-1,0] * (vel_z[0,-1,1] - vel_z[0,-1,0]) + 2 * k * den_inv[0,-1,0] * (den[0,-1,1] - den[0,-1,0])
        vel_z_out[-1,0,0] = vel_x[-1,0,0] * (vel_z[-1,0,0] - vel_z[-2,0,0]) + vel_y[-1,0,0] * (vel_z[-1,1,0] - vel_z[-1,0,0]) + var.size_ratio * vel_z[-1,0,0] * (vel_z[-1,0,1] - vel_z[-1,0,0]) + 2 * k * den_inv[-1,0,0] * (den[-1,0,1] - den[-1,0,0])
        vel_z_out[-1,-1,0] = vel_x[-1,-1,0] * (vel_z[-1,-1,0] - vel_z[-2,-1,0]) + vel_y[-1,-1,0] * (vel_z[-1,-2,0] - vel_z[-1,-1,0]) + var.size_ratio * vel_z[-1,-1,0] * (vel_z[-1,-1,1] - vel_z[-1,-1,0]) + 2 * k * den_inv[-1,-1,0] * (den[-1,-1,1] - den[-1,-1,0])
        
        vel_z_out[0,0,-1] = vel_x[0,0,-1] * (vel_z[1,0,-1] - vel_z[0,0,-1]) + vel_y[0,0,-1] * (vel_z[0,1,-1] - vel_z[0,0,-1]) + var.size_ratio * vel_z[0,0,-1] * (vel_z[0,0,-1] - vel_z[0,0,-2]) + 2 * k * den_inv[0,0,-1] * (den[0,0,-1] - den[0,0,-2])
        vel_z_out[0,-1,-1] = vel_x[0,-1,-1] * (vel_z[1,-1,-1] - vel_z[0,-1,-1]) + vel_y[0,-1,-1] * (vel_z[0,-2,-1] - vel_z[0,-1,-1]) + var.size_ratio * vel_z[0,-1,-1] * (vel_z[0,-1,-1] - vel_z[0,-1,-2]) + 2 * k * den_inv[0,-1,-1] * (den[0,-1,-1] - den[0,-1,-2])
        vel_z_out[-1,0,-1] = vel_x[-1,0,-1] * (vel_z[-1,0,-1] - vel_z[-2,0,-1]) + vel_y[-1,0,-1] * (vel_z[-1,1,-1] - vel_z[-1,0,-1]) + var.size_ratio * vel_z[-1,0,-1] * (vel_z[-1,0,-1] - vel_z[-1,0,-2]) + 2 * k * den_inv[-1,0,-1] * (den[-1,0,-1] - den[-1,0,-2])
        vel_z_out[-1,-1,-1] = vel_x[-1,-1,-1] * (vel_z[-1,-1,-1] - vel_z[-2,-1,-1]) + vel_y[-1,-1,-1] * (vel_z[-1,-2,-1] - vel_z[-1,-1,-1]) + var.size_ratio * vel_z[-1,-1,-1] * (vel_z[-1,-1,-1] - vel_z[-1,-1,-2]) + 2 * k * den_inv[-1,-1,-1] * (den[-1,-1,-1] - den[-1,-1,-2])
            
        vel_z_out = vel_z_out - hv    
        vel_z_out = vel_z - .5 * var.step_ratio * vel_z_out - var.dT * vel_sh

    return vel_z_out

def integrable_function(in_put, queue_in, queue_out, Zbeg, Zend, periodic, sect):
    """ Returns the time derivative of the plasma/neutral density equations 
    and plasma/neutral momentum equations.
    
    Parameters
    ----------
    queue_in : mp.queue 
        queue in which plasma and neutral gas state is placed from previous time step
        
    queue_out : mp.queue
        queue in which plasma and neutral gas state is placed for current time step
    """  
    cnt = 1
    Zsize = in_put.shape[3]
    for i in range(1, var.Tstps+1):  
        stich = queue_in.get()
        
        in_put[0::, 0::, 0::, 0] = stich[0]
        in_put[0::, 0::, 0::, -1] = stich[1]

        if sect == 'bot':            
            in_put[4, 0::, 0::, 0] = -in_put[4, 0::, 0::, 0]
            in_put[7, 0::, 0::, 0] = -in_put[7, 0::, 0::, 0]
            
        elif sect == 'top':
            in_put[4, 0::, 0::, -1] = -in_put[4, 0::, 0::, -1]
            in_put[7, 0::, 0::, -1] = -in_put[7, 0::, 0::, -1]
            
        plasma_den = in_put[0]
        neutral_den = in_put[1]
        
        plasma_vel_x = in_put[2]
        plasma_vel_y = in_put[3]
        plasma_vel_z = in_put[4]
        
        neutral_vel_x = in_put[5]
        neutral_vel_y = in_put[6]
        neutral_vel_z = in_put[7]
        
        # Density threshold inversion
        plasma_den[plasma_den < var.thresh] = var.thresh
        neutral_den[neutral_den < var.thresh] = var.thresh     

        plasma_den_inv = 1. / plasma_den
        neutral_den_inv = 1. / neutral_den
        
        # Shared Variables
        den_shared = var.dT * var.alpha * (var.eps * neutral_den - plasma_den) * plasma_den
    
        rel_vel_x = neutral_vel_x - plasma_vel_x
        rel_vel_y = neutral_vel_y - plasma_vel_y
        rel_vel_z = neutral_vel_z - plasma_vel_z
        
        rel_vel_x_with_den = rel_vel_x * neutral_den
        rel_vel_y_with_den = rel_vel_y * neutral_den
        rel_vel_z_with_den = rel_vel_z * neutral_den
        
        sigma, psi_x, psi_y, psi_z = calculate_sigma_psi(rel_vel_x, rel_vel_y, rel_vel_z, neutral_den)
        
        vel_x_shared = sigma * (rel_vel_x_with_den - 2 * psi_x)
        vel_y_shared = sigma * (rel_vel_y_with_den - 2 * psi_y)
        vel_z_shared = sigma * (rel_vel_z_with_den - 2 * psi_z)
        
        den_ratio = plasma_den * neutral_den_inv
            
        # Calculate Densities
        plasma_den_out = calculate_density(plasma_den, plasma_vel_x, plasma_vel_y, plasma_vel_z, -den_shared, Zsize, periodic)
        neutral_den_out = calculate_density(neutral_den, neutral_vel_x, neutral_vel_y, neutral_vel_z, den_shared, Zsize, periodic)
        
        # Calculate Velocities
        plasma_vel_x_shared = - rel_vel_x_with_den - vel_x_shared
        plasma_vel_x_out = calculate_velocity_x(plasma_vel_x, plasma_vel_y, plasma_vel_z, plasma_den, plasma_den_inv, plasma_vel_x_shared, var.kappa, Zsize, periodic)
        
        plasma_vel_y_shared = - rel_vel_y_with_den - vel_y_shared
        plasma_vel_y_out = calculate_velocity_y(plasma_vel_x, plasma_vel_y, plasma_vel_z, plasma_den, plasma_den_inv, plasma_vel_y_shared, var.kappa, Zsize, periodic)
        
        plasma_vel_z_shared = - rel_vel_z_with_den - vel_z_shared
        plasma_vel_z_out = calculate_velocity_z(plasma_vel_x, plasma_vel_y, plasma_vel_z, plasma_den, plasma_den_inv, plasma_vel_z_shared, var.kappa, Zsize, periodic)
        
        neutral_vel_x_shared = den_ratio * (var.eta * rel_vel_x + vel_x_shared)
        neutral_vel_x_out = calculate_velocity_x(neutral_vel_x, neutral_vel_y, neutral_vel_z, neutral_den, neutral_den_inv, neutral_vel_x_shared, var.kappa_n, Zsize, periodic)
        
        neutral_vel_y_shared = den_ratio * (var.eta * rel_vel_y + vel_y_shared)
        neutral_vel_y_out = calculate_velocity_y(neutral_vel_x, neutral_vel_y, neutral_vel_z, neutral_den, neutral_den_inv, neutral_vel_y_shared, var.kappa_n, Zsize, periodic)
        
        neutral_vel_z_shared = den_ratio * (var.eta * rel_vel_z + vel_z_shared)
        neutral_vel_z_out = calculate_velocity_z(neutral_vel_x, neutral_vel_y, neutral_vel_z, neutral_den, neutral_den_inv, neutral_vel_z_shared, var.kappa_n, Zsize, periodic)
        
        # Compile Out Put
        in_put = np.array([plasma_den_out,
                           neutral_den_out,
                           plasma_vel_x_out,
                           plasma_vel_y_out,
                           plasma_vel_z_out,
                           neutral_vel_x_out,
                           neutral_vel_y_out,
                           neutral_vel_z_out])
        
        new_stich = np.empty((2, 8, var.Rpts+2, var.Rpts+2))
        new_stich = np.array([in_put[0::, 0::, 0::, 1],
                              in_put[0::, 0::, 0::, -2]])
        
        queue_out.put(new_stich)
        '''
        lst = np.argwhere(np.isnan(in_put[0::, 1:-1, 1:-1, 1:-1]))     
        if lst.any():
            time_key = 'time_%s' % str(cnt-1)
            
            while True:
                try:
                    hf = h5.File(var.data_file, 'r')    
                    
                    plas_den = hf['Plasma/Density/'+time_key][:]
                    neut_den = hf['Neutral/Density/'+time_key][:]
                    
                    plas_vel_x = hf['Plasma/Velocity/'+time_key][0::, 0::, 0::, 0]
                    plas_vel_y = hf['Plasma/Velocity/'+time_key][0::, 0::, 0::, 1]
                    plas_vel_z = hf['Plasma/Velocity/'+time_key][0::, 0::, 0::, 2]
            
                    neut_vel_x = hf['Neutral/Velocity/'+time_key][0::, 0::, 0::, 0]
                    neut_vel_y = hf['Neutral/Velocity/'+time_key][0::, 0::, 0::, 1]
                    neut_vel_z = hf['Neutral/Velocity/'+time_key][0::, 0::, 0::, 2]
                    
                    y_pst = np.array([plas_den,
                                      neut_den,
                                      plas_vel_x,
                                      plas_vel_y,
                                      plas_vel_z,
                                      neut_vel_x,
                                      neut_vel_y,
                                      neut_vel_z])
                                      
                    hf.close()
                    
                    break
                
                except:
                    continue
            
            for l in lst:
                print('{0} : {1}'.format(l, 1e-3 * var.v * y_pst[l[0], l[1], l[2], l[3]]))
            print('Step {0}: {1} of {2}'.format(cnt, i, var.Tstps))
            break
        '''
        if i == var.save_steps[cnt]:
            key = 'time_{}'.format(cnt)

            plasma_vel_out = np.stack((plasma_vel_x_out[1:-1, 1:-1, 1:-1], plasma_vel_y_out[1:-1, 1:-1, 1:-1], plasma_vel_z_out[1:-1, 1:-1, 1:-1]), axis=3)
            neutral_vel_out = np.stack((neutral_vel_x_out[1:-1, 1:-1, 1:-1], neutral_vel_y_out[1:-1, 1:-1, 1:-1], neutral_vel_z_out[1:-1, 1:-1, 1:-1]), axis=3)
            
            while True:
                try:
                    hf = h5.File(var.data_file, 'a')      
                    
                    hf['Plasma/Density/'+key][0::, 0::, Zbeg:Zend] = plasma_den_out[1:-1, 1:-1, 1:-1]
                    hf['Neutral/Density/'+key][0::, 0::, Zbeg:Zend] = neutral_den_out[1:-1, 1:-1, 1:-1]
                    
                    hf['Plasma/Velocity/'+key][0::, 0::, Zbeg:Zend, 0::] = plasma_vel_out
                    hf['Neutral/Velocity/'+key][0::, 0::, Zbeg:Zend, 0::] = neutral_vel_out
                    
                    hf.close()
                    
                    break
                
                except:
                    continue
            
            cnt+=1

def write_meta_file(fileName, desc):
    """ Writes .txt meta file for simulation.
    
    Parameters
    ----------
    fileName : str
        File name of meta file.
        
    desc : str
        Description of data file.
    """
    myFile = open(fileName, 'w')
    myFile.write(var.sim_type+' Simulation\n'
                 '   Imported: '+var.importName+'\n'
                 '   '+var.source+' Plasma ('+str(var.temp)+' eV)\n'
                 '   '+desc+'\n\n'
                 'Grid Parameters \n'
                 '   XY domain: ['+str(var.s * var.R_beg)+', '+str(var.s * var.R_end)+'] (m) \n'
                 '   Z domain: ['+str(0)+', '+str(var.s * var.Z_end)+'] (m) \n'
                 '   grid dimensions (X, Y, Z): ('+str(var.Rpts)+', '+str(var.Rpts)+', '+str(var.Zpts)+') \n'
                 '   dx, dy, dz: ('+str(var.dR)+', '+str(var.dR)+', '+str(var.dZ)+')\n'
                 '   velocity hv: '+str(var.visc_vel)+'\n'
                 '   density hv: '+str(var.visc_den)+'\n'
                 '   sigularity threshold: '+str(var.thresh)+'\n'
                 '   integration steps: '+str(var.Tstps)+'\n'
                 '   save dumps: '+str(var.saves)+'\n\n'
                 'Simulation Time (s)\n'
                 '   time domain: 0 to '+str(var.dT * var.Tstps)+'\n'
                 '   time step: '+str(var.dT)+'\n\n'
                 'Real Time (s)\n'
                 '   time domain: 0 to '+str(var.t * var.dT * var.Tstps)+'\n'
                 '   time step: '+str(var.t * var.dT)+'\n\n'
                 'Physical Parameters\n'
                 '   ion mass (kg): '+str(var.mass_i)+'\n'
                 '   kT_{i} (J): '+str(var.C_i)+'\n'
                 '   kT_{n} (J): '+str(var.C_n)+'\n'
                 '   kT_{e} (J): '+str(var.C_e)+'\n\n'
                 'Fluid Reaction Parameters\n'
                 '   g_ion (m^{3}/s): '+str(var.g_ion)+'\n'
                 '   g_rec (m^{3}/s): '+str(var.g_rec)+'\n'
                 '   g_cx (m^{2}): '+str(var.g_cx)+'\n\n'
                 'Simulation Variable Scalars\n'
                 '   n (m^{-3}): '+str(var.n)+'\n'
                 '   v (m/s): '+str(var.v)+'\n'
                 '   t (s): '+str(var.t)+'\n'
                 '   s (m): '+str(var.s)+'\n')
    
    myFile.close()