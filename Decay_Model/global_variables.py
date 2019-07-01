#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 11:38:40 2019

@author: michael
"""
import sys
sys.path.append('/home/michael/Documents/PWFA/Decay_Model/')

import numpy as np
import h5py as h5

def ion_cross_section(A, P, X, Phi, T, K):
    phi_t = Phi / T
    return A * ( (1. + P * np.sqrt(phi_t)) / (X + (phi_t))) * ( (phi_t)**K ) * np.exp(-phi_t) * 1e-6
    
def rec_cross_section(T, Z=1):
    return 2.6e-19 * ( (Z*Z) / np.sqrt(T) )
    
def cx_cross_section(source):
    if source == 'Argon':
        return -5.65e-20
    elif source == 'Helium':
        return -7.15e-20
        
def C_ion(T):
    return T * 11604 * 1.3807e-23
    
def sigma_func(vel, source):
    if source == 'Argon':
        sig = np.log(v*vel) - 14.0708
    elif source == 'Helium':
        sig = np.log(v*vel) - 15.2448
    
    return sig

# Plasma source and temp (eV)
shared_Name = 'imported_ionization_008'
desc = 'Standard Ionization'
importName = 'plasma_density_reduced.h5'
source = 'Argon'
temp = 1
t_end = 10e-9
saves = 2
procs = 3

if source == 'Argon':
    A = .599e-7
    P = 1
    X = 0.136
    Phi = 15.8
    K = .26
elif source == 'Helium':
    A = .175e-7
    P = 0
    X = .18
    Phi = 24.6
    K = .35

# Physical Parameters
g_ion = ion_cross_section(A, P, X, Phi, temp, K)
g_rec = rec_cross_section(temp)
g_cx = cx_cross_section(source)

gam_i = 1
gam_i_minus = gam_i - 1

gam_e = 1
gam_e_minus = gam_e - 1

gam_n = 1
gam_n_minus = gam_n - 1

mass_i = 6.6335209 * 10**-26
C_i = C_ion(temp)
C_e = C_i
C_n = C_i

half_max = 0.00015

# Variable Scalars
n = 10**22
t = 1. / (n * g_ion)
s = -1. / (n * g_cx)
v = s / t

# Gaussian Density Parameter
delta = ( (s * 1.17741) / half_max )**2

# Non Dimensionlized Parameters
eps = g_ion / g_rec
eta = 1. / eps
alpha = t * n * g_rec
kappa = ((gam_i * C_i) / mass_i) * (g_cx / g_ion)**2
kappa_n = ((gam_n * C_n) / mass_i) * (g_cx / g_ion)**2

v_thermal = (1./v) * np.sqrt( (8*C_i) / (mass_i) )
v_thermal_sq = v_thermal * v_thermal

thresh = 1e-8

Rpts = 101
Zpts = 51

init_path = '/home/michael/Documents/PWFA/Decay_Model/3D_Simulation/Init_Density/'+importName

myFile = h5.File(init_path, 'r')
X_dom_1D = myFile['X Domain'][:] / s
Z_dom_1D = myFile['Z Domain'][:] / s
myFile.close()

R_beg = X_dom_1D[0]
R_end = X_dom_1D[-1]
Z_end = Z_dom_1D[-1]

dR = X_dom_1D[1] - X_dom_1D[0]
dZ = Z_dom_1D[1] - Z_dom_1D[0]

dR_inv = 1. / dR
dZ_inv = 1. / dZ

dT = 1e-6 * dR
step_ratio = dT *dR_inv
size_ratio = dR * dZ_inv

Tstps = int(t_end / (t * dT))
save_steps = np.linspace(0, Tstps, saves, dtype=int)

sub_steps = 1
sub_steps_inv = 1. / sub_steps

vis_scalar = 0.00001

visc_vel = 1000
visc_den = 10

hv_vel_r = 2 * visc_vel * dR_inv
hv_vel_z = 2 * visc_vel * dZ_inv

hv_den_r = 2 * visc_den * dR_inv
hv_den_z = 2 * visc_den * dZ_inv

# Visualization Parameters
Bin = 150
N_Bin = Bin * Bin
Spc = dR / s
Org = X_dom_1D[0] / s

# Processes
#Z_edge = np.linspace(0, Zpts+3, procs+1, dtype=int)
Z_edge = np.linspace(0, Zpts+1, procs+1, dtype=int)

sim_type = '3D_Simulation'

parent_dirc = '/home/michael/Documents/PWFA/Decay_Model/'+sim_type+'/'
viz_dirc = parent_dirc+'Visuals/'
viz_plasma_dirc = viz_dirc+'Visuals/Plasma_Density/'
viz_neutral_dirc = viz_dirc+'Visuals/Neutral_Density/'
viz_total_dirc = viz_dirc+'Visuals/Total_Density/'

data_file = parent_dirc+'Sim_Data/Data/'+shared_Name+'.h5'
meta_file = parent_dirc+'Meta_Data/'+shared_Name+'.txt'
