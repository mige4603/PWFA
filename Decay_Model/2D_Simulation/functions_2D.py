#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:34:04 2019

@author: michael
"""
import sys
sys.path.append('/home/michael/Documents/PWFA/Decay_Model/')

import h5py as h5
import numpy as np
import global_variables as var

def initial_condition(x, y):
    r2 = x**2 + y**2
    
    n = np.exp(-.5 * var.delta * r2)
    
    plasma_den = n
    neutral_den = 1. - plasma_den
    
    return plasma_den, neutral_den

def velX_domain_shift(in_put):
    out_put = 0.25 * (in_put[0:-1, 0:-1] + in_put[1::, 1::] + in_put[0:-1, 1::] + in_put[1::, 0:-1])
    
    out_put_xBound = 0.25 * (in_put[0, 0:-1] + in_put[0, 1::] - in_put[-1, 0:-1] - in_put[-1, 1::])
    
    out_put_yBound = 0.25 * (in_put[0:-1, 0] + in_put[1::, 0] + in_put[0:-1, -1] + in_put[1::, -1])
    out_put_cBound = 0.25 * (in_put[0,0] + in_put[0,-1] - in_put[-1,0] - in_put[-1,-1])
    out_put_yBound = np.insert(out_put_yBound, 0, out_put_cBound)
    
    out_put = np.insert(out_put, 0, out_put_xBound, axis=0)
    out_put = np.insert(out_put, 0, out_put_yBound, axis=1)
    
    return out_put

def velY_domain_shift(in_put):
    out_put = 0.25 * (in_put[0:-1, 0:-1] + in_put[1::, 1::] + in_put[0:-1, 1::] + in_put[1::, 0:-1])
    
    out_put_xBound = 0.25 * (in_put[0, 0:-1] + in_put[0, 1::] + in_put[-1, 0:-1] + in_put[-1, 1::])
    out_put_cBound = 0.25 * (in_put[0,0] - in_put[0,-1] + in_put[-1,0] - in_put[-1,-1])
    out_put_xBound = np.insert(out_put_xBound, 0, out_put_cBound)

    out_put_yBound = 0.25 * (in_put[0:-1, 0] + in_put[1::, 0] - in_put[0:-1, -1] - in_put[1::, -1])
    
    out_put = np.insert(out_put, 0, out_put_yBound, axis=1)
    out_put = np.insert(out_put, 0, out_put_xBound, axis=0)
    
    return out_put

def den_domain_shift(in_put):
    out_put = 0.25 * (in_put[0:-1, 0:-1] + in_put[1::, 1::] + in_put[0:-1, 1::] + in_put[1::, 0:-1])
    
    out_put_xBound = 0.25 * (in_put[0, 0:-1] + in_put[0, 1::] + in_put[-1, 0:-1] + in_put[-1, 1::])
    out_put_cBound = 0.25 * (in_put[0,0] + in_put[0,-1] + in_put[-1,0] + in_put[-1,-1])
    out_put_xBound = np.append(out_put_xBound, out_put_cBound)

    out_put_yBound = 0.25 * (in_put[0:-1, 0] + in_put[1::, 0] + in_put[0:-1, -1] + in_put[1::, -1])
    
    out_put = np.insert(out_put, -1, out_put_yBound, axis=1)
    out_put = np.insert(out_put, -1, out_put_xBound, axis=0)
    
    return out_put    

def calculate_sigma_psi(rel_vel_x, rel_vel_y, neut_den):
    """ Calculate the sigma and psi values need at each step of the 
    integrable function.
    
    Parameters
    ----------
    rel_vel : array
        relative fluid velocities (V_n - V_p) in each grid point
        
    neut_den : array
        neutral gas density at each grid point
    """
    vel_in = - np.dstack((rel_vel_x, rel_vel_y))
    vel_in_2 = np.linalg.norm(vel_in)
    
    vel_cx = 2.54648 * var.v_thermal_sq + (vel_in_2)
    sigma = np.log(var.v*vel_cx) - 14.0708
    
    psi_base =- (var.v_thermal_sq * neut_den) / np.sqrt( 7.06858 * var.v_thermal_sq + 2 * (vel_cx*vel_cx + vel_in_2) )
    psi_x = rel_vel_x * psi_base
    psi_y = rel_vel_y * psi_base
    
    return sigma, psi_x, psi_y

def calculate_hyper_viscosity(array, hv_scalar):
    hv_cent = (array[2::, 1:-1] - 2*array[1:-1, 1:-1] + array[0:-2, 1:-1]) + (array[1:-1, 2::] - 2*array[1:-1, 1:-1] + array[1:-1, 0:-2]) 

    hv_xBnd_for = (array[2, 1:-1] - 2*array[1, 1:-1] + array[0, 1:-1]) + (array[0, 2::] - 2*array[0, 1:-1] + array[0, 0:-2])
    hv_xBnd_back = (array[-1, 1:-1] - 2*array[-2, 1:-1] + array[-3, 1:-1]) + (array[-1, 2::] - 2*array[-1, 1:-1] + array[-1, 0:-2]) 
    
    hv_yBnd_for = (array[2::, 0] - 2*array[1:-1, 0] + array[0:-2, 0]) + (array[1:-1, 2] - 2*array[1:-1, 1] + array[1:-1, 0])
    hv_yBnd_back = (array[2::, -1] - 2*array[1:-1, -1] + array[0:-2, -1]) + (array[1:-1, -1] - 2*array[1:-1, -2] + array[1:-1, -3])
    
    hv_TL_Corner = (array[2,0] - 2*array[1,0] + array[0,0]) + (array[0,2] - 2*array[0,1] + array[0,0])
    hv_TR_Corner = (array[2,-1] - 2*array[1,-1] + array[0,-1]) + (array[0,-1] - 2*array[0,-2] + array[0,-3])
    hv_BL_Corner = (array[-1,0] - 2*array[-2,0] + array[-3,0]) + (array[-1,2] - 2*array[-1,1] + array[-1,0])
    hv_BR_Corner = (array[-1,-1] - 2*array[-2,-1] + array[-3,-1]) + (array[-1,-1] - 2*array[-1,-2] + array[-1,-3])
     
    hv_yBnd_for = np.insert(hv_yBnd_for, 0, hv_TL_Corner)
    hv_yBnd_for = np.insert(hv_yBnd_for, -1, hv_BL_Corner)
    
    hv_yBnd_back = np.insert(hv_yBnd_back, 0, hv_TR_Corner)
    hv_yBnd_back = np.insert(hv_yBnd_back, -1, hv_BR_Corner)
    
    hv = np.insert(hv_cent, 0, hv_xBnd_for, axis=0)
    hv = np.insert(hv, -1, hv_xBnd_back, axis=0)
    hv = np.insert(hv, 0, hv_yBnd_for, axis=1)
    hv = np.insert(hv, -1, hv_yBnd_back, axis=1)
    
    return hv_scalar * hv

def calculate_density(den, vel_x, vel_y, den_sh):
    den_cent = vel_x[1:-1, 1:-1] * (den[2::, 1:-1] - den[0:-2, 1:-1]) + vel_y[1:-1, 1:-1] * (den[1:-1, 2::] - den[1:-1, 0:-2]) + den[1:-1, 1:-1] * (vel_x[2::, 1:-1] - vel_x[0:-2, 1:-1] + vel_y[1:-1, 2::] - vel_y[1:-1, 0:-2])
    
    den_xBnd_for = vel_x[0, 1:-1] * (den[1, 1:-1] - den[0, 1:-1]) + vel_y[0, 1:-1] * (den[0, 2::] - den[0, 0:-2]) + den[0, 1:-1] * (vel_x[1, 1:-1] - vel_x[0, 1:-1] + vel_y[0, 2::] - vel_y[0, 0:-2])
    den_xBnd_back = vel_x[-1, 1:-1] * (den[-1, 1:-1] - den[-2, 1:-1]) + vel_y[-1, 1:-1] * (den[-1, 2::] - den[-1, 0:-2]) + den[-1, 1:-1] * (vel_x[-1, 1:-1] - vel_x[-2, 1:-1] + vel_y[-1, 2::] - vel_y[-1, 0:-2])

    den_yBnd_for = vel_x[1:-1, 0] * (den[2::, 0] - den[0:-2, 0]) + vel_y[1:-1, 0] * (den[1:-1, 1] - den[1:-1, 0]) + den[1:-1, 0] * (vel_x[2::, 0] - vel_x[0:-2, 0] + vel_y[1:-1, 1] - vel_y[1:-1, 0])
    den_yBnd_back = vel_x[1:-1, -1] * (den[2::, -1] - den[0:-2, -1]) + vel_y[1:-1, -1] * (den[1:-1, -1] - den[1:-1, -2]) + den[1:-1, -1] * (vel_x[2::, -1] - vel_x[0:-2, -1] + vel_y[1:-1, -1] - vel_y[1:-1, -2])
    
    den_TL_Corner = vel_x[0,0] * (den[1,0] - den[0,0]) + vel_y[0,0] * (den[0,1] - den[0,0]) + den[0,0] * (vel_x[1,0] - vel_x[0,0] + vel_y[0,1] - vel_y[0,0])
    den_TR_Corner = vel_x[0,-1] * (den[1,-1] - den[0,-1]) + vel_y[0,-1] * (den[0,-1] - den[0,-2]) + den[0,-1] * (vel_x[1,-1] - vel_x[0,-1] + vel_y[0,-1] - vel_y[0,-2])
    den_BL_Corner = vel_x[-1,0] * (den[-1,0] - den[-2,0]) + vel_y[-1,0] * (den[-1,1] - den[-1,0]) + den[-1,0] * (vel_x[-1,0] - vel_x[-2,0] + vel_y[-1,1] - vel_y[-1,0])
    den_BR_Corner = vel_x[-1,-1] * (den[-1,-1] - den[-2,-1]) + vel_y[-1,-1] * (den[-1,-1] - den[-1,-2]) + den[-1,-1] * (vel_x[-1,-1] - vel_x[-2,-1] + vel_y[-1,-1] - vel_y[-1,-2])
    
    den_yBnd_for = np.insert(den_yBnd_for, 0, den_TL_Corner)
    den_yBnd_for = np.insert(den_yBnd_for, -1, den_BL_Corner)
    
    den_yBnd_back = np.insert(den_yBnd_back, 0, den_TR_Corner)
    den_yBnd_back = np.insert(den_yBnd_back, -1, den_BR_Corner)
        
    den_out = np.insert(den_cent, 0, den_xBnd_for, axis=0)
    den_out = np.insert(den_out, -1, den_xBnd_back, axis=0)
    den_out = np.insert(den_out, 0, den_yBnd_for, axis=1)
    den_out = np.insert(den_out, -1, den_yBnd_back, axis=1)
    
    hv = calculate_hyper_viscosity(den, var.hv_den)
    
    den_out = den - 0.5 * var.step_ratio * (den_out - hv) - den_sh
    return den_out

def calculate_velocity_x(vel_x, vel_y, den, den_inv, vel_sh, k):
    vel_x_cent = vel_x[1:-1, 1:-1] * (vel_x[2::, 1:-1] - vel_x[0:-2, 1:-1]) + vel_y[1:-1, 1:-1] * (vel_x[1:-1, 2::] - vel_x[1:-1, 0:-2]) + 2 * k * den_inv[1:-1, 1:-1] * (den[2::, 1:-1] - den[0:-2, 1:-1])
        
    vel_x_xBnd_for = vel_x[0, 1:-1] * (vel_x[1, 1:-1] - vel_x[0, 1:-1]) + vel_y[0, 1:-1] * (vel_x[0, 2::] - vel_x[0, 0:-2]) + 2 * k * den_inv[0, 1:-1] * (den[1, 1:-1] - den[0, 1:-1])
    vel_x_xBnd_back = vel_x[-1, 1:-1] * (vel_x[-1, 1:-1] - vel_x[-2, 1:-1]) + vel_y[-1, 1:-1] * (vel_x[-1, 2::] - vel_x[-1, 0:-2]) + 2 * k * den_inv[-1, 1:-1] * (den[-1, 1:-1] - den[-2, 1:-1])
    
    vel_x_yBnd_for = vel_x[1:-1, 0] * (vel_x[2::, 0] - vel_x[0:-2, 0]) + vel_y[1:-1, 0] * (vel_x[1:-1, 1] - vel_x[1:-1, 0]) + 2 * k * den_inv[1:-1, 0] * (den[2::, 0] - den[0:-2, 0])
    vel_x_yBnd_back = vel_x[1:-1, -1] * (vel_x[2::, -1] - vel_x[0:-2, -1]) + vel_y[1:-1, -1] * (vel_x[1:-1, -1] - vel_x[1:-1, -2]) + 2 * k * den_inv[1:-1, -1] * (den[2::, -1] - den[0:-2, -1])
    
    vel_x_TL_Corner = vel_x[0,0] * (vel_x[1,0] - vel_x[0,0]) + vel_y[0,0] * (vel_x[0,1] - vel_x[0,0]) + 2 * k * den_inv[0,0] * (den[1,0] - den[0,0])
    vel_x_TR_Corner = vel_x[0,-1] * (vel_x[1,-1] - vel_x[0,-1]) + vel_y[0,-1] * (vel_x[0,-1] - vel_x[0,-2]) + 2 * k * den_inv[0,-1] * (den[1,-1] - den[0,-1])
    vel_x_BL_Corner = vel_x[-1,0] * (vel_x[-1,0] - vel_x[-2,0]) + vel_y[-1,0] * (vel_x[-1,1] - vel_x[-1,0]) + 2 * k * den_inv[-1,0] * (den[-1,0] - den[-2,0])
    vel_x_BR_Conrer = vel_x[-1][-1] * (vel_x[-1][-1] - vel_x[-2][-1]) + vel_y[-1][-1] * (vel_x[-1][-1] - vel_x[-1][-2]) + 2 * k * den_inv[-1][-1] * (den[-1][-1] - den[-2][-1])
        
    vel_x_yBnd_for = np.insert(vel_x_yBnd_for, 0, vel_x_TL_Corner)
    vel_x_yBnd_for = np.append(vel_x_yBnd_for, vel_x_BL_Corner)
    
    vel_x_yBnd_back = np.insert(vel_x_yBnd_back, 0, vel_x_TR_Corner)
    vel_x_yBnd_back = np.append(vel_x_yBnd_back, vel_x_BR_Conrer)
    
    vel_x_out = np.insert(vel_x_cent, 0, vel_x_xBnd_for, axis=0)
    vel_x_out = np.insert(vel_x_out, -1, vel_x_xBnd_back, axis=0)
    vel_x_out = np.insert(vel_x_out, 0, vel_x_yBnd_for, axis=1)
    vel_x_out = np.insert(vel_x_out, -1, vel_x_yBnd_back, axis=1)
    
    hv = calculate_hyper_viscosity(vel_x, var.hv_vel)
    
    vel_x_out = vel_x - .5 * var.step_ratio * (vel_x_out - hv) - var.dT * vel_sh
    return vel_x_out
    
def calculate_velocity_y(vel_x, vel_y, den, den_inv, vel_sh, k):
    vel_y_cent = vel_x[1:-1, 1:-1] * (vel_y[2::, 1:-1] - vel_y[0:-2, 1:-1]) + vel_y[1:-1, 1:-1] * (vel_y[1:-1, 2::] - vel_y[1:-1, 0:-2]) + 2 * k * den_inv[1:-1, 1:-1] * (den[1:-1, 2::] - den[1:-1, 0:-2])
        
    vel_y_xBnd_for = vel_x[0, 1:-1] * (vel_y[1, 1:-1] - vel_y[0, 1:-1]) + vel_y[0, 1:-1] * (vel_y[0, 2::] - vel_y[0, 0:-2]) + 2 * k * den_inv[0, 1:-1] * (den[0, 2::] - den[0, 0:-2])
    vel_y_xBnd_back = vel_x[-1, 1:-1] * (vel_y[-1, 1:-1] - vel_y[-2, 1:-1]) + vel_y[-1, 1:-1] * (vel_y[-1, 2::] - vel_y[-1, 0:-2]) + 2 * k * den_inv[-1, 1:-1] * (den[-1, 2::] - den[-1, 0:-2])
    
    vel_y_yBnd_for = vel_x[1:-1, 0] * (vel_y[2::, 0] - vel_y[0:-2, 0]) + vel_y[1:-1, 0] * (vel_y[1:-1, 1] - vel_y[1:-1, 0]) + 2 * k * den_inv[1:-1, 0] * (den[1:-1, 1] - den[1:-1, 0])
    vel_y_yBnd_back = vel_x[1:-1, -1] * (vel_y[2::, -1] - vel_y[0:-2, -1]) + vel_y[1:-1, -1] * (vel_y[1:-1, -1] - vel_y[1:-1, -2]) + 2 * k * den_inv[1:-1, -1] * (den[1:-1, -1] - den[1:-1, -2])
    
    vel_y_TL_Corner = vel_x[0,0] * (vel_y[1,0] - vel_y[0,0]) + vel_y[0,0] * (vel_y[0,1] - vel_y[0,0]) + 2 * k * den_inv[0,0] * (den[0,1] - den[0,0])
    vel_y_TR_Corner = vel_x[0,-1] * (vel_y[1,-1] - vel_y[0,-1]) + vel_y[0,-1] * (vel_y[0,-1] - vel_y[0,-2]) + 2 * k * den_inv[0,-1] * (den[0,-1] - den[0,-2])
    vel_y_BL_Corner = vel_x[-1,0] * (vel_y[-1,0] - vel_y[-2,0]) + vel_y[-1,0] * (vel_y[-1,1] - vel_y[-1,0]) + 2 * k * den_inv[-1,0] * (den[-1,1] - den[-1,0])
    vel_y_BR_Conrer = vel_x[-1,-1] * (vel_y[-1,-1] - vel_y[-2,-1]) + vel_y[-1,-1] * (vel_y[-1,-1] - vel_y[-1,-2]) + 2 * k * den_inv[-1,-1] * (den[-1,-1] - den[-1,-2])
        
    vel_y_yBnd_for = np.insert(vel_y_yBnd_for, 0, vel_y_TL_Corner)
    vel_y_yBnd_for = np.append(vel_y_yBnd_for, vel_y_BL_Corner)
    
    vel_y_yBnd_back = np.insert(vel_y_yBnd_back, 0, vel_y_TR_Corner)
    vel_y_yBnd_back = np.append(vel_y_yBnd_back, vel_y_BR_Conrer)
    
    vel_y_out = np.insert(vel_y_cent, 0, vel_y_xBnd_for, axis=0)
    vel_y_out = np.insert(vel_y_out, -1, vel_y_xBnd_back, axis=0)
    vel_y_out = np.insert(vel_y_out, 0, vel_y_yBnd_for, axis=1)
    vel_y_out = np.insert(vel_y_out, -1, vel_y_yBnd_back, axis=1)
    
    hv = calculate_hyper_viscosity(vel_y, var.hv_vel)
    
    vel_y_out = vel_y - .5 * var.step_ratio * (vel_y_out - hv) - var.dT * vel_sh
    return vel_y_out
    
def integrable_function(in_put):
    """ Returns the time derivative of the plasma/neutral density equation 
    and plasma/neutral momentum equation.
    
    Parameters
    ----------
    t : float
        time step
        
    y : array
        density and velocity values of previous time step
    """            
    plasma_density = in_put[0]
    neutral_density = in_put[1]
    
    plasma_velocity_x = in_put[2]
    plasma_velocity_y = in_put[3]
    
    neutral_velocity_x = in_put[4]
    neutral_velocity_y = in_put[5]
    
    # Shift variables to where values are caluclated
    plasma_density_vel_x = velX_domain_shift(plasma_velocity_x)
    plasma_density_vel_y = velY_domain_shift(plasma_velocity_y)
    
    neutral_density_vel_x = velX_domain_shift(neutral_velocity_x)
    neutral_density_vel_y = velY_domain_shift(neutral_velocity_y)
    
    plasma_velocity_den = den_domain_shift(plasma_density)   
    neutral_velocity_den = den_domain_shift(neutral_density)
        
    # Density threshold inversion
    plasma_velocity_den[plasma_velocity_den < var.thresh] = var.thresh
    neutral_velocity_den[neutral_velocity_den < var.thresh] = var.thresh     
    
    plasma_velocity_den_inv = 1. / plasma_velocity_den
    neutral_velocity_den_inv = 1. / neutral_velocity_den
    
    # Shared Variables
    density_shared = var.dT * var.alpha * (var.eps * neutral_density - plasma_density) * plasma_density

    rel_velocity_x = neutral_velocity_x - plasma_velocity_x
    rel_velocity_y = neutral_velocity_y - plasma_velocity_y
    
    rel_velocity_x_with_den = rel_velocity_x * neutral_velocity_den
    rel_velocity_y_with_den = rel_velocity_y * neutral_velocity_den
    
    sigma, psi_x, psi_y = calculate_sigma_psi(rel_velocity_x, rel_velocity_y, neutral_velocity_den)
    
    velocity_x_shared = sigma * (rel_velocity_x_with_den - 2 * psi_x)
    velocity_y_shared = sigma * (rel_velocity_y_with_den - 2 * psi_y)
    
    density_ratio = plasma_velocity_den * neutral_velocity_den_inv
        
    # Calculate Densities
    plasma_density_out = calculate_density(plasma_density, plasma_density_vel_x, plasma_density_vel_y, density_shared)
    neutral_density_out = calculate_density(neutral_density, neutral_density_vel_x, neutral_density_vel_y, density_shared)
    
    # Calculate Velocities
    plasma_velocity_x_shared = - rel_velocity_x_with_den - velocity_x_shared
    plasma_velocity_x_out = calculate_velocity_x(plasma_velocity_x, plasma_velocity_y, plasma_velocity_den, plasma_velocity_den_inv, plasma_velocity_x_shared, var.kappa)
    
    plasma_velocity_y_shared = - rel_velocity_y_with_den - velocity_y_shared
    plasma_velocity_y_out = calculate_velocity_y(plasma_velocity_x, plasma_velocity_y, plasma_velocity_den, plasma_velocity_den_inv, plasma_velocity_y_shared, var.kappa)
    
    neutral_velocity_x_shared = density_ratio * (var.eta * rel_velocity_x + velocity_x_shared)
    neutral_velocity_x_out = calculate_velocity_x(neutral_velocity_x, neutral_velocity_y, neutral_velocity_den, neutral_velocity_den_inv, neutral_velocity_x_shared, var.kappa_n)
    
    neutral_velocity_y_shared = density_ratio * (var.eta * rel_velocity_y + velocity_y_shared)
    neutral_velocity_y_out = calculate_velocity_y(neutral_velocity_x, neutral_velocity_y, neutral_velocity_den, neutral_velocity_den_inv, neutral_velocity_y_shared, var.kappa_n)
    
    # Compile Out Put
    out_put = np.array([plasma_density_out,
                        neutral_density_out,
                        plasma_velocity_x_out,
                        plasma_velocity_y_out,
                        neutral_velocity_x_out,
                        neutral_velocity_y_out])

    return out_put
                
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
            print '\nStep: '+str(cnt)+' of '+str(sets)  
            
            plasma_vel = np.dstack((y[2], y[3]))
            neutral_vel = np.dstack((y[4], y[5]))
        
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
                
        plasma_vel = np.dstack((y[2], y[3]))
        neutral_vel = np.dstack((y[4], y[5]))
        
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
    myFile.write(var.data_type+' Simulation\n'
                 '   '+desc+'\n\n'
                 'Grid Parameters \n'
                 '   grid domain: ['+str(var.s * var.R_beg)+', '+str(var.s * var.R_end)+'] (m) \n'
                 '   grid dimensions: '+str(var.Rpts)+' X '+str(var.Rpts)+'\n'
                 '   grid spacing: '+str(var.dR)+'\n'
                 '   velocity hv: '+str(var.visc)+'\n'
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
    