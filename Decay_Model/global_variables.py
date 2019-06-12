#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 11:38:40 2019

@author: michael
"""
import sys
sys.path.append('/home/michael/Documents/PWFA/PWFA/Decay_Model/')

import numpy as np
import h5py as h5

# Physical Parameters
g_ion = 5.26783 * 10**-21
g_rec = 2.6 * 10**-19
g_cx = -5.65 * 10**-20

gam_i = 1
gam_i_minus = gam_i - 1

gam_e = 1
gam_e_minus = gam_e - 1

gam_n = 1
gam_n_minus = gam_n - 1

mass_i = 6.6335209 * 10**-26
C_i = 1.60216 * 10**-19
C_e = C_i
C_n = C_i

half_max = 0.0005

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

R_beg = -3 * (half_max / s)
R_end = 3 * (half_max / s)
Z_end = 500e-6 / s

Rpts = 101#int( round( ((R_end - R_beg) / 0.01) + 1 ) )
Zpts = 51#int( round(Z_end / ((R_end - R_beg) / Rpts)) )

'''
X_dom_1D = np.linspace(R_beg, R_end, Rpts)
X_dom_2D = np.repeat(X_dom_1D.reshape(Rpts, 1), Rpts, axis=1)
X_dom_3D = np.repeat(X_dom_2D.reshape(Rpts, Rpts, 1), Zpts, axis=2)

Y_dom_2D = np.transpose(X_dom_2D)
Y_dom_3D = np.repeat(Y_dom_2D.reshape(Rpts, Rpts, 1), Zpts, axis=2)

Z_dom_1D = np.linspace(0, Z_end, Zpts)
Z_dom_3D = np.tile(Z_dom_1D, (Rpts, Rpts, 1))
'''
init_path = '3D_Simulation/Init_Density/plasma_density_reduced.h5'

myFile = h5.File(init_path, 'r')
X_dom_1D = myFile['X Domain'][:]
Z_dom_1D = myFile['Z Domain'][:]
myFile.close()

dR = X_dom_1D[1] - X_dom_1D[0]
dZ = Z_dom_1D[1] - Z_dom_1D[0]

dR_inv = 1. / dR
dZ_inv = 1. / dZ

dT = 1e-7 * dR
step_ratio = dT *dR_inv
size_ratio = dR * dZ_inv

t_end = 1e-9
Tstps = int(t_end / (t * dT))
Tstps_range = range(Tstps)

save_steps = 10

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
nProcs = 20
Z_edge = np.linspace(0, Zpts, nProcs, dtype=int)

sim_type = '3D_Simulation'

parent_dirc = '/home/michael/Documents/PWFA/Decay_Model/'+sim_type+'/'
viz_dirc = parent_dirc+'Visuals/'
viz_plasma_dirc = viz_dirc+'Visuals/Plasma_Density/'
viz_neutral_dirc = viz_dirc+'Visuals/Neutral_Density/'
viz_total_dirc = viz_dirc+'Visuals/Total_Density/'

shared_Name = 'test_3D_005'
desc = 'Test Parallelization'

data_file = parent_dirc+'Sim_Data/'+shared_Name+'.h5'
meta_file = parent_dirc+'Meta_Data/'+shared_Name+'.txt'
