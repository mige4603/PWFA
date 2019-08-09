# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:24:37 2019

@author: michael
"""

import sys
sys.path.append('/home/michael/Documents/PWFA/Decay_Model/')

import global_variables as var

import numpy as np
import os

vec = 'Vel_y'

fluid = 'Plasma'
name = var.shared_Name

dx = var.dR
dz = var.dZ
dt = var.dT

v_scl = 1e-3 * var.v
s_scl = var.s 
t_scl = 1e9 * var.t
n_scl = var.n

steps = var.Tstps
dumps = var.saves

dump = 6

save_steps = np.linspace(0, steps, dumps, dtype=int)
t_end = t_scl * dt * steps

surf_area = 2 * s_scl * (dx * var.Rpts + dz * var.Zpts) * 1e4
dom_volum = s_scl * ( (dx * var.Rpts) * (dx * var.Rpts) * (dz * var.Zpts) ) * 1e6

if not os.path.isdir('Visuals/'+name+'/'):
    os.mkdir('Visuals/'+name+'/')
    os.mkdir('Visuals/'+name+'/'+fluid+'/')
    
    os.mkdir('Visuals/'+name+'/'+fluid+'/velocity/')
    os.mkdir('Visuals/'+name+'/'+fluid+'/velocity/movies/')
    
    os.mkdir('Visuals/'+name+'/'+fluid+'/density/')
    os.mkdir('Visuals/'+name+'/'+fluid+'/density/movies/')

elif not os.path.isdir('Visuals/'+name+'/'+fluid+'/'):
    os.mkdir('Visuals/'+name+'/'+fluid+'/')
    
    os.mkdir('Visuals/'+name+'/'+fluid+'/velocity/')
    os.mkdir('Visuals/'+name+'/'+fluid+'/velocity/movies/')
    
    os.mkdir('Visuals/'+name+'/'+fluid+'/density/')
    os.mkdir('Visuals/'+name+'/'+fluid+'/density/movies/')
    
elif not os.path.isdir('Visuals/'+name+'/'+fluid+'/velocity/'):
    os.mkdir('Visuals/'+name+'/'+fluid+'/velocity/')
    os.mkdir('Visuals/'+name+'/'+fluid+'/velocity/movies/')

elif not os.path.isdir('Visuals/'+name+'/'+fluid+'/density/'):    
    os.mkdir('Visuals/'+name+'/'+fluid+'/density/')
    os.mkdir('Visuals/'+name+'/'+fluid+'/density/movies/')

int_path = 'Visuals/'+name+'/'
den_path = 'Visuals/'+name+'/'+fluid+'/density/'
vel_path = 'Visuals/'+name+'/'+fluid+'/velocity/'
