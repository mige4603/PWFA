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

def velX_domain_shift(in_put):
    """ Returns fluid velocity in x direction as witnessed by density cells.
    
    Parameters
    ----------
    in_put : array
        velocity in x direction
    """
    out_put = 0.125 * (in_put[0:-1, 0:-1, 0:-1] + in_put[0:-1, 0:-1, 1::] + in_put[0:-1, 1::, 0:-1] + in_put[1::, 0:-1, 0:-1] + in_put[0:-1, 1::, 1::] + in_put[1::, 0:-1, 1::] + in_put[1::, 1::, 0:-1] + in_put[1::, 1::, 1::])
    
    out_put_zBound = 0.25 * (in_put[0:-1, 0:-1, -1] + in_put[0:-1, 1::, -1] + in_put[1::, 0:-1, -1] + in_put[1::, 1::, -1])
    out_put = np.insert(out_put, 0, out_put_zBound, axis=2)
        
    out_put_yBound = 0.5 * (in_put[0:-1, -1, 0::] + in_put[1::, -1, 0::])
    out_put = np.insert(out_put, 0, out_put_yBound, axis=1)
    
    out_put_xBound = - in_put[-1, 0::, 0::]
    out_put = np.insert(out_put, 0, out_put_xBound, axis=0)
  
    return out_put

def velY_domain_shift(in_put):
    """ Returns fluid velocity in y direction as witnessed by density cells.
    
    Parameters
    ----------
    in_put : array
        velocity in y direction
    """
    out_put = 0.125 * (in_put[0:-1, 0:-1, 0:-1] + in_put[0:-1, 0:-1, 1::] + in_put[0:-1, 1::, 0:-1] + in_put[1::, 0:-1, 0:-1] + in_put[0:-1, 1::, 1::] + in_put[1::, 0:-1, 1::] + in_put[1::, 1::, 0:-1] + in_put[1::, 1::, 1::])
    
    out_put_zBound = 0.25 * (in_put[0:-1, 0:-1, -1] + in_put[0:-1, 1::, -1] + in_put[1::, 0:-1, -1] + in_put[1::, 1::, -1])
    out_put = np.insert(out_put, 0, out_put_zBound, axis=2)
        
    out_put_yBound = - 0.5 * (in_put[0:-1, -1, 0::] + in_put[1::, -1, 0::])
    out_put = np.insert(out_put, 0, out_put_yBound, axis=1)
    
    out_put_xBound = in_put[-1, 0::, 0::]
    out_put = np.insert(out_put, 0, out_put_xBound, axis=0)
  
    return out_put

def velZ_domain_shift(in_put):
    """ Returns fluid velocity in z direction as witnessed by density cells.
    
    Parameters
    ----------
    in_put : array
        velocity in z direction
    """
    out_put = 0.125 * (in_put[0:-1, 0:-1, 0:-1] + in_put[0:-1, 0:-1, 1::] + in_put[0:-1, 1::, 0:-1] + in_put[1::, 0:-1, 0:-1] + in_put[0:-1, 1::, 1::] + in_put[1::, 0:-1, 1::] + in_put[1::, 1::, 0:-1] + in_put[1::, 1::, 1::])
    
    out_put_zBound = - 0.25 * (in_put[0:-1, 0:-1, -1] + in_put[0:-1, 1::, -1] + in_put[1::, 0:-1, -1] + in_put[1::, 1::, -1])
    out_put = np.insert(out_put, 0, out_put_zBound, axis=2)
        
    out_put_yBound = 0.5 * (in_put[0:-1, -1, 0::] + in_put[1::, -1, 0::])
    out_put = np.insert(out_put, 0, out_put_yBound, axis=1)
    
    out_put_xBound = in_put[-1, 0::, 0::]
    out_put = np.insert(out_put, 0, out_put_xBound, axis=0)
  
    return out_put

def den_domain_shift(in_put, Zsize):
    """ Returns fluid density as witnessed by velocity cells.
    
    Parameters
    ----------
    in_put : array
        plasma/neutral gas density
    """
    out_put = 0.125 * (in_put[0:-1, 0:-1, 0:-1] + in_put[0:-1, 0:-1, 1::] + in_put[0:-1, 1::, 0:-1] + in_put[1::, 0:-1, 0:-1] + in_put[0:-1, 1::, 1::] + in_put[1::, 0:-1, 1::] + in_put[1::, 1::, 0:-1] + in_put[1::, 1::, 1::])

    out_put_zBnd = 2 * out_put[0::, 0::, -1] - out_put[0::, 0::, -2]
    out_put = np.append(out_put, out_put_zBnd.reshape(var.Rpts+1, var.Rpts+1, 1), axis=2)
    
    out_put_yBnd = 2 * out_put[0::, -1, 0::] - out_put[0::, -2, 0::]
    out_put = np.append(out_put, out_put_yBnd.reshape(var.Rpts+1, 1, Zsize), axis=1)
    
    out_put_xBnd = 2 * out_put[-1, 0::, 0::] - out_put[-2, 0::, 0::]
    out_put = np.append(out_put, out_put_xBnd.reshape(1, var.Rpts+2, Zsize), axis=0)
    
    return out_put

def periodic_boundary_den(in_put, Zsize):    
    """ Returns spliced density domain for periodic boundary conditions.
    
    Parameters
    ----------
    in_put : array
        plasma/neutral gas density
        
    Zsize : int
        number of cells along axis=2
    """
    out_put = np.insert(in_put, 0, in_put[0::, -1, 0::], axis=1)
    out_put = np.append(out_put, out_put[0::, 1, 0::].reshape(var.Rpts, 1, Zsize), axis=1)
    
    out_put = np.insert(out_put, 0, out_put[-1, 0::, 0::], axis=0)
    out_put = np.append(out_put, out_put[1, 0::, 0::].reshape(1, var.Rpts+2, Zsize), axis=0)
    
    return out_put
    
def periodic_boundary_vel_x(in_put, Zsize):  
    """ Returns spliced x velocity domain for periodic boundary conditions.
    
    Parameters
    ----------
    in_put : array
        fluid velocity in x direction
        
    Zsize : int
        number of cells along axis=2
    """
    out_put = np.insert(in_put, 0, in_put[0::, -1, 0::], axis=1)
    out_put = np.append(out_put, out_put[0::, 1, 0::].reshape(var.Rpts, 1, Zsize), axis=1)
    
    out_put = np.insert(out_put, 0, -out_put[-1, 0::, 0::], axis=0)
    out_put = np.append(out_put, -out_put[1, 0::, 0::].reshape(1, var.Rpts+2, Zsize), axis=0)
    
    return out_put

def periodic_boundary_vel_y(in_put, Zsize):
    """ Returns spliced y velocity domain for periodic boundary conditions.
    
    Parameters
    ----------
    in_put : array
        fluid velocity in y direction
        
    Zsize : int
        number of cells along axis=2
    """
    out_put = np.insert(in_put, 0, -in_put[0::, -1, 0::], axis=1)
    out_put = np.append(out_put, -out_put[0::, 1, 0::].reshape(var.Rpts, 1, Zsize), axis=1)
    
    out_put = np.insert(out_put, 0, out_put[-1, 0::, 0::], axis=0)
    out_put = np.append(out_put, out_put[1, 0::, 0::].reshape(1, var.Rpts+2, Zsize), axis=0)
    
    return out_put
    
def periodic_boundary_vel_z(in_put, Zsize):
    """ Returns spliced z velocity domain for periodic boundary conditions.
    
    Parameters
    ----------
    in_put : array
        fluid velocity in z direction
        
    Zsize : int
        number of cells along axis=2
    """
    out_put = np.insert(in_put, 0, in_put[0::, -1, 0::], axis=1)
    out_put = np.append(out_put, out_put[0::, 1, 0::].reshape(var.Rpts, 1, Zsize), axis=1)
    
    out_put = np.insert(out_put, 0, out_put[-1, 0::, 0::], axis=0)
    out_put = np.append(out_put, out_put[1, 0::, 0::].reshape(1, var.Rpts+2, Zsize), axis=0)
    
    return out_put

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
    sigma = np.log(var.v*vel_cx) - 14.0708
    
    psi_base = - (var.v_thermal_sq * neut_den) / np.sqrt( 7.06858 * var.v_thermal_sq + 2 * (vel_cx*vel_cx + vel_in_2) )
    
    psi_x = rel_vel_x * psi_base
    psi_y = rel_vel_y * psi_base
    psi_z = rel_vel_z * psi_base
    
    return sigma, psi_x, psi_y, psi_z

def calculate_density(den, vel_x, vel_y, vel_z, den_sh, Zsize):
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
    hv = var.hv_den_r * ( (den[2::, 1:-1, 1:-1] - 2*den[1:-1, 1:-1, 1:-1] + den[0:-2, 1:-1, 1:-1]) + (den[1:-1, 2::, 1:-1] - 2*den[1:-1, 1:-1, 1:-1] + den[1:-1, 0:-2, 1:-1]) ) + var.hv_den_z * var.size_ratio * (den[1:-1, 1:-1, 2::] - 2*den[1:-1, 1:-1, 1:-1] + den[1:-1, 1:-1, 0:-2]) 
    den_out = vel_x[1:-1, 1:-1, 1:-1] * (den[2::, 1:-1, 1:-1] - den[0:-2, 1:-1, 1:-1]) + vel_y[1:-1, 1:-1, 1:-1] * (den[1:-1, 2::, 1:-1] - den[1:-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[1:-1, 1:-1, 1:-1] * (den[1:-1, 1:-1, 2::] - den[1:-1, 1:-1, 0:-2]) + den[1:-1, 1:-1, 1:-1] * (vel_x[2::, 1:-1, 1:-1] - vel_x[0:-2, 1:-1, 1:-1] + vel_y[1:-1, 2::, 1:-1] - vel_y[1:-1, 0:-2, 1:-1] + var.size_ratio * (vel_z[1:-1, 1:-1, 2::] - vel_z[1:-1, 1:-1, 0:-2]) )
    
    den_out = den[1:-1, 1:-1, 1:-1] - 0.5 * var.step_ratio * (den_out - hv) - den_sh
    return den_out

def calculate_velocity_x(vel_x, vel_y, vel_z, den, den_inv, vel_sh, k, Zsize):
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
    hv = var.hv_vel_r * ( (vel_x[2::, 1:-1, 1:-1] - 2*vel_x[1:-1, 1:-1, 1:-1] + vel_x[0:-2, 1:-1, 1:-1]) + (vel_x[1:-1, 2::, 1:-1] - 2*vel_x[1:-1, 1:-1, 1:-1] + vel_x[1:-1, 0:-2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_x[1:-1, 1:-1, 2::] - 2*vel_x[1:-1, 1:-1, 1:-1] + vel_x[1:-1, 1:-1, 0:-2]) 
    vel_x_out = vel_x[1:-1, 1:-1, 1:-1] * (vel_x[2::, 1:-1, 1:-1] - vel_x[0:-2, 1:-1, 1:-1]) + vel_y[1:-1, 1:-1, 1:-1] * (vel_x[1:-1, 2::, 1:-1] - vel_x[1:-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[1:-1, 1:-1, 1:-1] * (vel_x[1:-1, 1:-1, 2::] - vel_x[1:-1, 1:-1, 0:-2]) + 2 * k * den_inv[1:-1, 1:-1, 1:-1] * (den[2::, 1:-1, 1:-1] - den[0:-2, 1:-1, 1:-1])
    
    vel_x_out = vel_x[1:-1, 1:-1, 1:-1] - .5 * var.step_ratio * (vel_x_out - hv) - var.dT * vel_sh
    return vel_x_out
    
def calculate_velocity_y(vel_x, vel_y, vel_z, den, den_inv, vel_sh, k, Zsize):
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
    hv = var.hv_vel_r * ( (vel_y[2::, 1:-1, 1:-1] - 2*vel_y[1:-1, 1:-1, 1:-1] + vel_y[0:-2, 1:-1, 1:-1]) + (vel_y[1:-1, 2::, 1:-1] - 2*vel_y[1:-1, 1:-1, 1:-1] + vel_y[1:-1, 0:-2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_y[1:-1, 1:-1, 2::] - 2*vel_y[1:-1, 1:-1, 1:-1] + vel_y[1:-1, 1:-1, 0:-2]) 
    vel_y_out = vel_x[1:-1, 1:-1, 1:-1] * (vel_y[2::, 1:-1, 1:-1] - vel_y[0:-2, 1:-1, 1:-1]) + vel_y[1:-1, 1:-1, 1:-1] * (vel_y[1:-1, 2::, 1:-1] - vel_y[1:-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[1:-1, 1:-1, 1:-1] * (vel_y[1:-1, 1:-1, 2::] - vel_y[1:-1, 1:-1, 0:-2]) + 2 * k * den_inv[1:-1, 1:-1, 1:-1] * (den[1:-1, 2::, 1:-1] - den[1:-1, 0:-2, 1:-1])
    
    vel_y_out = vel_y[1:-1, 1:-1, 1:-1] - .5 * var.step_ratio * (vel_y_out - hv) - var.dT * vel_sh
    return vel_y_out
    
def calculate_velocity_z(vel_x, vel_y, vel_z, den, den_inv, vel_sh, k, Zsize):
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
    hv = var.hv_vel_r * ( (vel_z[2::, 1:-1, 1:-1] - 2*vel_z[1:-1, 1:-1, 1:-1] + vel_z[0:-2, 1:-1, 1:-1]) + (vel_z[1:-1, 2::, 1:-1] - 2*vel_z[1:-1, 1:-1, 1:-1] + vel_z[1:-1, 0:-2, 1:-1]) ) + var.hv_vel_z * var.size_ratio * (vel_z[1:-1, 1:-1, 2::] - 2*vel_z[1:-1, 1:-1, 1:-1] + vel_z[1:-1, 1:-1, 0:-2]) 
    vel_z_out = vel_x[1:-1, 1:-1, 1:-1] * (vel_z[2::, 1:-1, 1:-1] - vel_z[0:-2, 1:-1, 1:-1]) + vel_y[1:-1, 1:-1, 1:-1] * (vel_z[1:-1, 2::, 1:-1] - vel_z[1:-1, 0:-2, 1:-1]) + var.size_ratio * vel_z[1:-1, 1:-1, 1:-1] * (vel_z[1:-1, 1:-1, 2::] - vel_z[1:-1, 1:-1, 0:-2]) + var.size_ratio * 2 * k * den_inv[1:-1, 1:-1, 1:-1] * (den[1:-1, 1:-1, 2::] - den[1:-1, 1:-1, 0:-2])
    
    vel_z_out = vel_z[1:-1, 1:-1, 1:-1] - .5 * var.step_ratio * (vel_z_out - hv) - var.dT * vel_sh
    return vel_z_out

def integrable_function(queue_in, queue_out, Zsize):
    """ Returns the time derivative of the plasma/neutral density equations 
    and plasma/neutral momentum equations.
    
    Parameters
    ----------
    queue_in : mp.queue 
        queue in which plasma and neutral gas state is placed from previous time step
        
    queue_out : mp.queue
        queue in which plasma and neutral gas state is placed for current time step
    """  
    for i in range(var.Tstps):
        in_put = queue_in.get()         
        
        plasma_den = in_put[0]
        neutral_den = in_put[1]
        
        plasma_vel_x = in_put[2]
        plasma_vel_y = in_put[3]
        plasma_vel_z = in_put[4]
        
        neutral_vel_x = in_put[5]
        neutral_vel_y = in_put[6]
        neutral_vel_z = in_put[7]
        
        # Periodic Boundary Edition
        plasma_den = periodic_boundary_den(plasma_den, Zsize)
        neutral_den = periodic_boundary_den(neutral_den, Zsize)
        
        plasma_vel_x = periodic_boundary_vel_x(plasma_vel_x, Zsize)
        plasma_vel_y = periodic_boundary_vel_y(plasma_vel_y, Zsize)
        plasma_vel_z = periodic_boundary_vel_z(plasma_vel_z, Zsize)
        
        neutral_vel_x = periodic_boundary_vel_x(neutral_vel_x, Zsize)
        neutral_vel_y = periodic_boundary_vel_y(neutral_vel_y, Zsize)
        neutral_vel_z = periodic_boundary_vel_z(neutral_vel_z, Zsize)
        
        # Shift variables to where values are caluclated
        plasma_velSH_x = velX_domain_shift(plasma_vel_x)
        plasma_velSH_y = velY_domain_shift(plasma_vel_y)
        plasma_velSH_z = velZ_domain_shift(plasma_vel_z)
        
        neutral_velSH_x = velX_domain_shift(neutral_vel_x)
        neutral_velSH_y = velY_domain_shift(neutral_vel_y)
        neutral_velSH_z = velZ_domain_shift(neutral_vel_z)
        
        plasma_denSH = den_domain_shift(plasma_den, Zsize)   
        neutral_denSH = den_domain_shift(neutral_den, Zsize)
            
        # Density threshold inversion
        plasma_denSH[plasma_denSH < var.thresh] = var.thresh
        neutral_denSH[neutral_denSH < var.thresh] = var.thresh     
        
        plasma_denSH_inv = 1. / plasma_denSH
        neutral_denSH_inv = 1. / neutral_denSH
        
        # Shared Variables
        den_shared = var.dT * var.alpha * (var.eps * neutral_den - plasma_den) * plasma_den
    
        rel_vel_x = neutral_vel_x - plasma_vel_x
        rel_vel_y = neutral_vel_y - plasma_vel_y
        rel_vel_z = neutral_vel_z - plasma_vel_z
        
        rel_vel_x_with_den = rel_vel_x * neutral_denSH
        rel_vel_y_with_den = rel_vel_y * neutral_denSH
        rel_vel_z_with_den = rel_vel_z * neutral_denSH
        
        sigma, psi_x, psi_y, psi_z = calculate_sigma_psi(rel_vel_x, rel_vel_y, rel_vel_z, neutral_denSH)
        
        vel_x_shared = sigma * (rel_vel_x_with_den - 2 * psi_x)
        vel_y_shared = sigma * (rel_vel_y_with_den - 2 * psi_y)
        vel_z_shared = sigma * (rel_vel_z_with_den - 2 * psi_z)
        
        den_ratio = plasma_denSH * neutral_denSH_inv
            
        # Calculate Densities
        plasma_den_out = calculate_density(plasma_den, plasma_velSH_x, plasma_velSH_y, plasma_velSH_z, -den_shared[1:-1, 1:-1, 1:-1], Zsize)
        neutral_den_out = calculate_density(neutral_den, neutral_velSH_x, neutral_velSH_y, neutral_velSH_z, den_shared[1:-1, 1:-1, 1:-1], Zsize)
        
        # Calculate Velocities
        plasma_vel_x_shared = - rel_vel_x_with_den - vel_x_shared
        plasma_vel_x_out = calculate_velocity_x(plasma_vel_x, plasma_vel_y, plasma_vel_z, plasma_denSH, plasma_denSH_inv, plasma_vel_x_shared[1:-1, 1:-1, 1:-1], var.kappa, Zsize)
        
        plasma_vel_y_shared = - rel_vel_y_with_den - vel_y_shared
        plasma_vel_y_out = calculate_velocity_y(plasma_vel_x, plasma_vel_y, plasma_vel_z, plasma_denSH, plasma_denSH_inv, plasma_vel_y_shared[1:-1, 1:-1, 1:-1], var.kappa, Zsize)
        
        plasma_vel_z_shared = - rel_vel_z_with_den - vel_z_shared
        plasma_vel_z_out = calculate_velocity_z(plasma_vel_x, plasma_vel_y, plasma_vel_z, plasma_denSH, plasma_denSH_inv, plasma_vel_z_shared[1:-1, 1:-1, 1:-1], var.kappa, Zsize)
        
        neutral_vel_x_shared = den_ratio * (var.eta * rel_vel_x + vel_x_shared)
        neutral_vel_x_out = calculate_velocity_x(neutral_vel_x, neutral_vel_y, neutral_vel_z, neutral_denSH, neutral_denSH_inv, neutral_vel_x_shared[1:-1, 1:-1, 1:-1], var.kappa_n, Zsize)
        
        neutral_vel_y_shared = den_ratio * (var.eta * rel_vel_y + vel_y_shared)
        neutral_vel_y_out = calculate_velocity_y(neutral_vel_x, neutral_vel_y, neutral_vel_z, neutral_denSH, neutral_denSH_inv, neutral_vel_y_shared[1:-1, 1:-1, 1:-1], var.kappa_n, Zsize)
        
        neutral_vel_z_shared = den_ratio * (var.eta * rel_vel_z + vel_z_shared)
        neutral_vel_z_out = calculate_velocity_z(neutral_vel_x, neutral_vel_y, neutral_vel_z, neutral_denSH, neutral_denSH_inv, neutral_vel_z_shared[1:-1, 1:-1, 1:-1], var.kappa_n, Zsize)
        
        # Compile Out Put
        out_put = np.array([plasma_den_out,
                            neutral_den_out,
                            plasma_vel_x_out,
                            plasma_vel_y_out,
                            plasma_vel_z_out,
                            neutral_vel_x_out,
                            neutral_vel_y_out,
                            neutral_vel_z_out])

        queue_out.put(out_put)
                
def saver(in_queue):    
    cnt = 1
    sub_divide = var.save_steps
    sets = int(var.Tstps / sub_divide)
    
    write_meta_file(var.meta_file, var.desc)
    for i in var.Tstps_range:
        i, y = in_queue.get()
        
        y_sum = np.sum(y)
        if np.isnan(y_sum):
            break
        
        if i == (sub_divide * cnt):
            time_key = 'time_%s' % str(cnt)
            print( '\nStep: '+str(cnt)+' of '+str(sets) )
            
            plasma_vel = np.dstack((y[2], y[3], y[4]))
            neutral_vel = np.dstack((y[5], y[6], y[7]))
        
            myFile = h5.File(var.data_file, 'a')
            
            myFile.create_dataset('Plasma/Density/'+time_key, data=y[0])
            myFile.create_dataset('Plasma/Velocity/'+time_key, data=plasma_vel)
            
            myFile.create_dataset('Neutral/Density/'+time_key, data=y[1])
            myFile.create_dataset('Neutral/Velocity/'+time_key, data=neutral_vel)
            
            myFile.close()
            
            cnt+=1
            
        else:
            continue
    
    if not np.isnan(y_sum):
        time_key = 'time_%s' % str(cnt)
    
        myFile = h5.File(var.data_file, 'a')
                
        plasma_vel = np.dstack((y[2], y[3], y[4]))
        neutral_vel = np.dstack((y[5], y[6], y[7]))
        
        myFile.create_dataset('Plasma/Density/'+time_key, data=y[0])
        myFile.create_dataset('Neutral/Density/'+time_key, data=y[1])
        
        myFile.create_dataset('Plasma/Velocity/'+time_key, data=plasma_vel)
        myFile.create_dataset('Neutral/Velocity/'+time_key, data=neutral_vel)
            
        myFile.close()
        
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
                 '   '+desc+'\n\n'
                 'Grid Parameters \n'
                 '   XY domain: ['+str(var.s * var.R_beg)+', '+str(var.s * var.R_end)+'] (m) \n'
                 '   Z domain: ['+str(0)+', '+str(var.s * var.Z_end)+'] (m) \n'
                 '   grid dimensions (X, Y, Z): ('+str(var.Rpts)+', '+str(var.Rpts)+', '+str(var.Zpts)+') \n'
                 '   grid spacing: '+str(var.dR)+'\n'
                 '   velocity hv: '+str(var.visc_vel)+'\n'
                 '   density hv: '+str(var.visc_den)+'\n'
                 '   sigularity threshold: '+str(var.thresh)+'\n\n'
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