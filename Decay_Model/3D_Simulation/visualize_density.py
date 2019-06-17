#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:41:12 2019

@author: michael
"""
import sys
sys.path.append('/home/cu-pwfa/Documents/Michael/PWFA/Decay_Model/')

import global_variables as var

import os
import h5py as h5

dirc = var.parent_dirc+'Sim_Data/'
dirc_viz = var.parent_dirc+'Visuals/Animate/'
name = var.shared_Name

if not os.path.isdir(dirc_viz+'Plasma/'+name+'/'):
    os.mkdir(dirc_viz+'Plasma/'+name+'/')
    os.mkdir(dirc_viz+'Plasma/'+name+'/density/')
elif not os.path.isdir(dirc_viz+'Plasma/'+name+'/density/'):
    os.mkdir(dirc_viz+'Plasma/'+name+'/density/')    
    
if not os.path.isdir(dirc_viz+'Neutral/'+name+'/'):
    os.mkdir(dirc_viz+'Neutral/'+name+'/')
    os.mkdir(dirc_viz+'Neutral/'+name+'/density/')
elif not os.path.isdir(dirc_viz+'Neutral/'+name+'/density/'):
    os.mkdir(dirc_viz+'Neutral/'+name+'/density/')
    
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
rng = len(myFile['Plasma/Density'])

for i in range(rng):
    time_key = 'time_{}'.format(i)
    print( time_key + ' of ' + str(rng-1) )
    
    plasma_density = myFile['Plasma/Density/'+time_key][:]
    neutral_density = myFile['Neutral/Density/'+time_key][:]
    
    file = open(dirc_viz+'Plasma/'+name+'/density/step_'+str(i)+'.vtk', 'w')
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
                den = plasma_density[x,y,z]
                file.write(str(den)+' '+str(0)+' '+str(0)+'\n')
                
    file.close()
    
    file = open(dirc_viz+'Neutral/'+name+'/density/step_'+str(i)+'.vtk', 'w')
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
                den = neutral_density[x,y,z]
                file.write(str(den)+' '+str(0)+' '+str(0)+'\n')
                
    file.close()