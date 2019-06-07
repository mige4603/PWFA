#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 12:32:10 2019

@author: michael
"""
import sys
sys.path.append('/home/michael/Documents/PWFA/Decay_Model/')

import numpy as np
import h5py as h5

import global_variables as var
import functions_1D as fun

fun.write_meta_file(var.meta_file, var.desc)

myFile = h5.File(var.data_file, 'w')

# Set Initial Conditions
plasma_den, neutral_den = fun.initial_condition(var.X_dom_1D)

plasma_vel = np.zeros(var.Rpts)
neutral_vel = np.zeros(var.Rpts)

# Arange Initial Conditions
y = np.array([plasma_den,
              neutral_den,
              plasma_vel,
              neutral_vel])

# Save Initial Conditions

myFile.create_dataset('Plasma/Density/time_0', data=plasma_den)
myFile.create_dataset('Plasma/Velocity/time_0', data=plasma_vel)

myFile.create_dataset('Neutral/Density/time_0', data=neutral_den)
myFile.create_dataset('Neutral/Velocity/time_0', data=neutral_vel)

den_avg = np.sum(y[0] + y[1]) / var.Rpts
    
# Evolve Initial Conditions
cnt = 1
sets = int(var.Tstps / var.save_steps)

for i in range(var.Tstps):    
    y = fun.integrable_function(y)
    
    if np.isnan(np.sum(y)):
        break
    
    if i == var.save_steps * cnt:
        time_key = 'time_%s' % str(cnt)
        
        print '\nStep: '+str(cnt)+' of '+str(sets)   
        
        myFile.create_dataset('Plasma/Density/'+time_key, data=y[0])
        myFile.create_dataset('Plasma/Velocity/'+time_key, data=y[2])
        
        myFile.create_dataset('Neutral/Density/'+time_key, data=y[1])
        myFile.create_dataset('Neutral/Velocity/'+time_key, data=y[3])
        
        cnt+=1
    
myFile.close()