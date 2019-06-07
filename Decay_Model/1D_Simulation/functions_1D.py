#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:44:02 2019

@author: michael
"""
import sys
sys.path.append('/home/michael/Documents/PWFA/Decay_Model/')

import numpy as np
import global_variables as var

def initial_condition(r):
    """ Calculates initial plasma/neutral gass density in cylindrical coordinants.
    
    Parameters
    ----------
    r : float
        radial value
    """
    
    # Gaussian Density
    n = np.exp(-.5 * var.delta * r * r)
    
    plasma_den = 0.99*n
    neutral_den = 1. - plasma_den
    
    '''
    # Linear Density 
    m = -1./var.R_end
    b = 1
    n = m * r + b
    plasma_den = 0.99*n
    neutral_den = 1. - n
    
    plasma_den = np.array([.5] * len(r))
    neutral_den = 1 - plasma_den
    
    # Flat Top Density
    thresh = var.half_max / var.s
    
    plasma_den = np.empty(len(r))
    neutral_den = np.empty(len(r))
    for ind, val in enumerate(r):
        if abs(val) < thresh:
            plasma_den[ind] = 1.
            neutral_den[ind] = 0.
        else:
            n = np.exp(-.5 * var.delta * val * val)
            plasma_den[ind] = n
            neutral_den[ind] = 1. - n
    '''
    return plasma_den, neutral_den
    
def density_domain_shift(in_put):
    out_put = (in_put[0:-1] + in_put[1::]) / 2
    out_put_boundary = (in_put[0] + in_put[-1]) / 2
    
    return np.append(out_put, out_put_boundary)

def velocity_domain_shift(in_put):
    out_put = (in_put[0:-1] + in_put[1::]) / 2
    #out_put_boundary = (in_put[0] + in_put[-1]) / 2
    out_put_boundary = 2 * in_put[0] - in_put[1]
    
    return np.insert(out_put, 0, out_put_boundary)

def add_boundary(array, left, right):
    array = np.insert(array, 0, left)
    array = np.append(array, right)
    
    return array

def calculate_sigma_psi(vel_in, neut_den):
    """ Calculate the sigma and psi values need at each step of the 
    integrable function.
    
    Parameters
    ----------
    rel_vel : array
        relative fluid velocities (V_n - V_p) in each grid point
        
    neut_den : array
        neutral gas density at each grid point
    """
    vel_in_2 = vel_in * vel_in
    
    vel_cx = 2.54648 * var.v_thermal_sq + (vel_in_2)
    sigma = np.log(var.v*vel_cx) - 14.0708

    psi = (var.v_thermal_sq * neut_den * vel_in) / np.sqrt( 7.06858 * var.v_thermal_sq + 2 * (vel_cx*vel_cx + vel_in_2) )
    
    return sigma, psi

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
    
    plasma_velocity = in_put[2]
    neutral_velocity = in_put[3]
    
    # Shift variables to where values are caluclated
    plasma_density_vel = velocity_domain_shift(plasma_velocity)
    neutral_density_vel = velocity_domain_shift(neutral_velocity)
    
    # Density threshold inversion
    plasma_velocity_den = density_domain_shift(plasma_density)   
    neutral_velocity_den = density_domain_shift(neutral_density)
    
    plasma_velocity_den[plasma_velocity_den < var.thresh] = var.thresh
    neutral_velocity_den[neutral_velocity_den < var.thresh] = var.thresh     
    
    plasma_density_inv = 1. / plasma_velocity_den
    neutral_density_inv = 1. / neutral_velocity_den
    
    rel_velocity = neutral_velocity - plasma_velocity
    rel_velocity_with_den = rel_velocity * neutral_velocity_den
    
    sigma, psi = calculate_sigma_psi(rel_velocity, neutral_velocity_den)
    velocity_shared = sigma * (rel_velocity_with_den - 2*psi)
    
    density_shared = (var.eps * neutral_density - plasma_density) * plasma_density
    
    # Plasma Density
    plasma_hyper_viscosity = var.hv_den * (plasma_density[2::] - 2*plasma_density[1:-1] + plasma_density[0:-2])
    left_bound_hv = var.hv_den * (plasma_density[2] - 2*plasma_density[1] + plasma_density[0])
    right_bound_hv = var.hv_den * (plasma_density[-1] - 2*plasma_density[-2] + plasma_density[-3])

    plasma_density_out = plasma_density_vel[1:-1] * (plasma_density[2::] - plasma_density[0:-2]) + plasma_density[1:-1] * (plasma_density_vel[2::] - plasma_density_vel[0:-2]) - plasma_hyper_viscosity
    left_bound = plasma_density_vel[0] * (plasma_density[1] - plasma_density[0]) + plasma_density[0] * (plasma_density_vel[1] - plasma_density_vel[0]) - left_bound_hv
    right_bound = plasma_density_vel[-1] * (plasma_density[-1] - plasma_density[-2]) + plasma_density[-1] * (plasma_density_vel[-1] - plasma_density_vel[-2]) - right_bound_hv
    plasma_density_out = add_boundary(plasma_density_out, left_bound, right_bound)

    plasma_density_out = plasma_density - .5 * var.step_ratio * plasma_density_out + var.dT * var.alpha * density_shared
      
    # Neutral Density
    neutral_hyper_viscosity = var.hv_den * (neutral_density[2::] - 2*neutral_density[1:-1] + neutral_density[0:-2])
    left_bound_hv = var.hv_den * (neutral_density[2] - 2*neutral_density[1] + neutral_density[0])
    right_bound_hv = var.hv_den * (neutral_density[-1] - 2*neutral_density[-2] + neutral_density[-3])
    
    neutral_density_out = neutral_density_vel[1:-1] * (neutral_density[2::] - neutral_density[0:-2]) + neutral_density[1:-1] * (neutral_density_vel[2::] - neutral_density_vel[0:-2]) - neutral_hyper_viscosity
    left_bound = neutral_density_vel[0] * (neutral_density[1] - neutral_density[0]) + neutral_density[0] * (neutral_density_vel[1] - neutral_density_vel[0]) - left_bound_hv
    right_bound = neutral_density_vel[-1] * (neutral_density[-1] - neutral_density[-2]) + neutral_density[-1] * (neutral_density_vel[-1] - neutral_density_vel[-2]) - right_bound_hv
    neutral_density_out = add_boundary(neutral_density_out, left_bound, right_bound)

    neutral_density_out = neutral_density - .5 * var.step_ratio * neutral_density_out - var.dT * var.alpha * density_shared
    
    # Plasma Velocity
    plasmaVEL_hyper_viscosity = var.hv_vel * (plasma_velocity[2::] - 2*plasma_velocity[1:-1] + plasma_velocity[0:-2])
    left_bound_hv = var.hv_vel * (plasma_velocity[2] - 2*plasma_velocity[1] + plasma_velocity[0])
    right_bound_hv = var.hv_vel * (plasma_velocity[-1] - 2*plasma_velocity[-2] + plasma_velocity[-3])
    
    plasma_velocity_out_1 = plasma_velocity[1:-1] * (plasma_velocity[2::] - plasma_velocity[0:-2]) + 2 * var.kappa * plasma_density_inv[1:-1] * (plasma_velocity_den[2::] - plasma_velocity_den[0:-2]) - plasmaVEL_hyper_viscosity
    left_bound_1 = plasma_velocity[0] * (plasma_velocity[1] - plasma_velocity[0]) + 2 * var.kappa * plasma_density_inv[0] * (plasma_velocity_den[1] - plasma_velocity_den[0]) - left_bound_hv
    right_bound_1 = plasma_velocity[-1] * (plasma_velocity[-1] - plasma_velocity[-2]) + 2 * var.kappa * plasma_density_inv[-1] * (plasma_velocity_den[-1] - plasma_velocity_den[-2]) - right_bound_hv
    plasma_velocity_out_1 = add_boundary(plasma_velocity_out_1, left_bound_1, right_bound_1)

    plasma_velocity_out_2 = rel_velocity_with_den - velocity_shared
    
    plasma_velocity_out = plasma_velocity - .5 * var.step_ratio * plasma_velocity_out_1 + var.dT * plasma_velocity_den * neutral_density_inv * plasma_velocity_out_2
    
    # Neutral Velocity
    neutralVEL_hyper_viscosity = var.hv_vel * (neutral_velocity[2::] - 2*neutral_velocity[1:-1] + neutral_velocity[0:-2])
    left_bound_hv = var.hv_vel * (neutral_velocity[2] - 2*neutral_velocity[1] + neutral_velocity[0])
    right_bound_hv = var.hv_vel * (neutral_velocity[-1] - 2*neutral_velocity[-2] + neutral_velocity[-3])
    
    neutral_velocity_out_1 = neutral_velocity[1:-1] * (neutral_velocity[2::] - neutral_velocity[0:-2]) + var.kappa_n * neutral_density_inv[1:-1] * (neutral_velocity_den[2::] - neutral_velocity_den[0:-2]) - neutralVEL_hyper_viscosity
    left_bound_1 = neutral_velocity[0] * (neutral_velocity[1] - neutral_velocity[0]) + 2 * var.kappa * neutral_density_inv[0] * (neutral_velocity_den[1] - neutral_velocity_den[0]) - left_bound_hv
    right_bound_1 = neutral_velocity[-1] * (neutral_velocity[-1] - neutral_velocity[-2]) + 2 * var.kappa * neutral_density_inv[-1] * (neutral_velocity_den[-1] - neutral_velocity_den[-2]) - right_bound_hv
    neutral_velocity_out_1 = add_boundary(neutral_velocity_out_1, left_bound_1, right_bound_1)

    neutral_velocity_out_2 = var.eta * rel_velocity - velocity_shared
    neutral_velocity_out = neutral_velocity - .5 * var.step_ratio * neutral_velocity_out_1 - var.dT * plasma_velocity_den * neutral_density_inv * neutral_velocity_out_2
    
    out_put = np.array([plasma_density_out,
                        neutral_density_out,
                        plasma_velocity_out,
                        neutral_velocity_out])
    
    return out_put

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
                 '   grid domain (m): ['+str(var.R_beg*var.s)+', '+str(var.R_end*var.s)+'] \n'
                 '   number of points: '+str(var.Rpts)+'\n'
                 '   grid spacing: '+str(var.dR)+'\n'
                 '   hyperviscosity velocity: '+str(var.visc_vel)+'\n'
                 '   hyperviscosity density: '+str(var.visc_den)+'\n'
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
    
