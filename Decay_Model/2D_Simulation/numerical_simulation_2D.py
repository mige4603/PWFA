#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:25:47 2019

@author: michael
"""

import sys
sys.path.append('/home/michael/Documents/PWFA/Decay_Model/')

import numpy as np
import h5py as h5

import global_variables as var
import functions_2D as fun

# Set Initial Conditions
plasma_den, neutral_den = fun.initial_condition(var.X_dom_2D, var.Y_dom_2D)

plasma_vel_x = np.zeros((var.Rpts, var.Rpts))
plasma_vel_y = np.zeros((var.Rpts, var.Rpts))
plasma_vel = np.dstack((plasma_vel_x, plasma_vel_y))

neutral_vel_x = np.zeros((var.Rpts, var.Rpts))
neutral_vel_y = np.zeros((var.Rpts, var.Rpts))
neutral_vel = np.dstack((neutral_vel_x, neutral_vel_y))

y = np.array([plasma_den,
              neutral_den,
              plasma_vel_x,
              plasma_vel_y,
              neutral_vel_x,
              neutral_vel_y])

# Save Initial Conditions
myFile = h5.File(var.data_file, 'w')

myFile.create_dataset('Plasma/Density/time_0', data=plasma_den)
myFile.create_dataset('Neutral/Density/time_0', data=neutral_den)

myFile.create_dataset('Plasma/Velocity/time_0', data=plasma_vel)
myFile.create_dataset('Neutral/Velocity/time_0', data=neutral_vel)

# Evolve Initial Condition
cnt = 1
sets = int(var.Tstps / var.save_steps)

for i in range(var.Tstps):   
    y = fun.integrable_function(y)
    
    if i == var.save_steps * cnt:
        time_key = 'time_%s' % str(cnt)
        
        print '\nStep: '+str(cnt)+' of '+str(sets)   
        
        plasma_vel = np.dstack((y[2], y[3]))
        neutral_vel = np.dstack((y[4], y[5]))
        
        myFile.create_dataset('Plasma/Density/'+time_key, data=y[0])
        myFile.create_dataset('Neutral/Density/'+time_key, data=y[1])
        
        myFile.create_dataset('Plasma/Velocity/'+time_key, data=plasma_vel)
        myFile.create_dataset('Neutral/Velocity/'+time_key, data=neutral_vel)
        
        cnt+=1
    
myFile.close()