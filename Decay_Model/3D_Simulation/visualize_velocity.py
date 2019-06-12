#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:05:13 2019

@author: michael
"""
import sys
sys.path.append('/home/michael/Documents/PWFA/Decay_Model/')

import global_variables as var

import os
import h5py as h5
import numpy as np

dirc = var.parent_dirc+'Sim_Data/'
dirc_viz = var.parent_dirc+'Visuals/Animate/'
name = var.shared_Name

if not os.path.isdir(dirc_viz+'Plasma/'+name+'/'):
    os.mkdir(dirc_viz+'Plasma/'+name+'/')
    os.mkdir(dirc_viz+'Plasma/'+name+'/velocity/')
elif not os.path.isdir(dirc_viz+'Plasma/'+name+'/velocity/'):
    os.mkdir(dirc_viz+'Plasma/'+name+'/velocity/')    
    
if not os.path.isdir(dirc_viz+'Neutral/'+name+'/'):
    os.mkdir(dirc_viz+'Neutral/'+name+'/')
    os.mkdir(dirc_viz+'Neutral/'+name+'/velocity/')
elif not os.path.isdir(dirc_viz+'Neutral/'+name+'/velocity/'):
    os.mkdir(dirc_viz+'Neutral/'+name+'/velocity/')

Rpts = var.Rpts
Zpts = var.Zpts
Npts = Rpts * Rpts * Zpts

dr = var.s * var.dR
dz = 1e-3 * var.s * var.dZ

x_org = var.s * var.X_dom_1D[0]
z_org = 0.

t_end = var.t_end
t_steps = var.Tstps
myFile = h5.File(dirc+name+'.h5', 'r')
rng = len(myFile['Plasma/Velocity'])

for i in range(rng):
    time_key = 'time_{}'.format(i)
    print time_key + ' of ' + str(rng)
        
    plasma_velocity = myFile['Plasma/Velocity/'+time_key][:]
    neutral_velocity = myFile['Neutral/Velocity/'+time_key][:]
    
    file = open(dirc_viz+'Plasma/'+name+'/velocity/step_'+str(i)+'.vtk', 'w')
    file.write('# vtk DataFile Version 1.0\n'
               'B Field from Parsek\nASCII\n'
               'DATASET STRUCTURED_POINTS\n'
               'DIMENSIONS {0} {1} {1}\n'
               'ORIGIN {2} {3} {3}\n'
               'SPACING {4} {5} {5}\n'
               'POINT_DATA {6}\n'
               'VECTORS B float\n'.format(Zpts, Rpts, z_org, x_org, dz, dr, Npts))
    
    for x in range(Rpts):
        for y in range(Rpts):
            for z in range(Zpts):
                vel = plasma_velocity[x,y,z] * 10**-3
                file.write(str(vel[0])+' '+str(vel[1])+' '+str(vel[2])+'\n')
                
    file.close()

    file = open(dirc_viz+'Neutral/'+name+'/velocity/step_'+str(i)+'.vtk', 'w')
    file.write('# vtk DataFile Version 1.0\n'
               'B Field from Parsek\nASCII\n'
               'DATASET STRUCTURED_POINTS\n'
               'DIMENSIONS {0} {1} {1}\n'
               'ORIGIN {2} {3} {3}\n'
               'SPACING {4} {5} {5}\n'
               'POINT_DATA {6}\n'
               'VECTORS B float\n'.format(Zpts, Rpts, z_org, x_org, dz, dr, Npts))
    
    for x in range(Rpts):
        for y in range(Rpts):
            for z in range(Zpts):
                vel = neutral_velocity[x,y,z] * 10**-3
                file.write(str(vel[0])+' '+str(vel[1])+' '+str(vel[2])+'\n')
       
    file.close()
