#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 08:24:37 2019

@author: michael
"""
import sys
sys.path.append('/home/michael/Documents/PWFA/Decay_Model/')

import global_variables as var

import os
import h5py as h5
import scipy as sp

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
    
pts = var.Rpts
Npts = pts * pts
dr = var.s * var.dR
orig = var.s * var.R_beg

t_end = var.t_end
t_steps = var.Tstps

myFile = h5.File(dirc+name+'.h5', 'r')
rng = len(myFile['Plasma/Density'])

for i in range(rng):
    time_key = 'time_{}'.format(i)
    print time_key + ' of ' + str(rng)
    
    plasma_density = myFile['Plasma/Density/'+time_key][:]
    neutral_density = myFile['Neutral/Density/'+time_key][:]
    
    file = open(dirc_viz+'Plasma/'+name+'/density/step_'+str(i)+'.vtk', 'w')
    file.write('# vtk DataFile Version 1.0\n'
               'B Field from Parsek\nASCII\n'
               'DATASET STRUCTURED_POINTS\n'
               'DIMENSIONS {0} {0} 1\n'
               'ORIGIN {1} {1} 0\n'
               'SPACING {2} {2} 0\n'
               'POINT_DATA {3}\n'
               'VECTORS B float\n'.format(pts, orig, dr, Npts))
    
    for x in range(pts):
        for y in range(pts):
            p_den = plasma_density[x,y]
            file.write(str(p_den)+' '+str(0)+' '+str(0)+'\n')
            
    file.close()
    
    file = open(dirc_viz+'Neutral/'+name+'/density/step_'+str(i)+'.vtk', 'w')
    file.write('# vtk DataFile Version 1.0\n'
               'B Field from Parsek\nASCII\n'
               'DATASET STRUCTURED_POINTS\n'
               'DIMENSIONS {0} {0} 1\n'
               'ORIGIN {1} {1} 0\n'
               'SPACING {2} {2} 0\n'
               'POINT_DATA {3}\n'
               'VECTORS B float\n'.format(pts, orig, dr, Npts))
    
    for x in range(pts):
        for y in range(pts):
            n_den = neutral_density[x,y]
            file.write(str(n_den)+' '+str(0)+' '+str(0)+'\n')
    
    file.close()