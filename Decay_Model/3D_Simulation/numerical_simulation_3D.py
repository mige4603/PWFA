#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:58:09 2019

@author: michael
"""


import sys
sys.path.append('/home/michael/Documents/PWFA/Decay_Model/')

import numpy as np
import h5py as h5
import time 

import multiprocessing as mp

import global_variables as var
import functions_P3D as fun

start_time = time.time()

# Set Initial Conditions
plasma_den, neutral_den = fun.initial_condition(var.X_dom_3D, var.Y_dom_3D, var.Z_dom_3D)

plasma_vel_x = np.zeros((var.Rpts, var.Rpts, var.Zpts))
plasma_vel_y = np.zeros((var.Rpts, var.Rpts, var.Zpts))
plasma_vel_z = np.zeros((var.Rpts, var.Rpts, var.Zpts))
plasma_vel = np.stack((plasma_vel_x, plasma_vel_y, plasma_vel_z), axis=3)

neutral_vel_x = np.zeros((var.Rpts, var.Rpts, var.Zpts))
neutral_vel_y = np.zeros((var.Rpts, var.Rpts, var.Zpts))
neutral_vel_z = np.zeros((var.Rpts, var.Rpts, var.Zpts))
neutral_vel = np.stack((neutral_vel_x, neutral_vel_y, neutral_vel_z), axis=3)

y = np.array([plasma_den,
              neutral_den,
              plasma_vel_x,
              plasma_vel_y,
              plasma_vel_z,
              neutral_vel_x,
              neutral_vel_y,
              neutral_vel_z])

y_split = {}

z_bot = var.Z_edge[1] + 1
z_top = var.Z_edge[-2] - 1

y_split['bottom'] = y[0::, 0::, 0::, 0:z_bot]
y_split['top'] = y[0::, 0::, 0::, z_top::]

for z_ind, z in enumerate(var.Z_edge[1:-1], 1):
    z_beg = z-1
    z_end = var.Z_edge[z_ind] + 1
    
    key = 'central {}'.format(z_ind)
    y_split[key] = y[0::, 0::, 0::, z_beg:z_end]    

# Create Queues and Processes
queues_in = {}
queues_out = {}

queues_in['bottom'] = mp.JoinableQueue()
queues_out['bottom'] = mp.JoinableQueue()

queues_in['top'] = mp.JoinableQueue()
queues_out['top'] = mp.JoinableQueue()

for i in range(1, var.procs-1):
    key = 'central {}'.format(i)
    
    queues_in[key] = mp.JoinableQueue()
    queues_out[key] = mp.JoinableQueue()

procs = {}

procs['bottom'] = mp.Process(target=fun.integrable_function, args=(queues_in['bottom'], queues_out['bottom'], z_bot))
procs['top'] = mp.Process(target=fun.integrable_function, args=(queues_in['top'], queues_out['top'], var.Zpts - z_top))

for i in range(1, var.procs-1):
    key = 'central {}'.format(i)
    
    procs[key] = mp.Process(target=fun.integrable_function, args=(queues_in[key], queues_out[key], y_split[key].shape[3],))

# Save Initial Conditions
'''
fun.write_meta_file(var.meta_file, var.desc)

myFile = h5.File(var.data_file, 'w')

myFile.create_dataset('Plasma/Density/time_0', data=plasma_den)
myFile.create_dataset('Neutral/Density/time_0', data=neutral_den)

myFile.create_dataset('Plasma/Velocity/time_0', data=plasma_vel)
myFile.create_dataset('Neutral/Velocity/time_0', data=neutral_vel)
'''
# Evolve Initial Condition
for key in procs:
    procs[key].start()

cnt = 1
sets = int(var.Tstps / var.save_steps)

for i in range(var.Tstps):   
    for key in y_split:
        queues_in[key].put( y_split[key] )
    
    for key in y_split:
        y_split[key] = queues_out[key].get()
    
    y_split['bottom'][0::, 0::, 0::, -1] = y_split['central 1'][0::, 0::, 0::, 1]
    y_split['central 1'][0::, 0::, 0::, 0] = y_split['bottom'][0::, 0::, 0::, -2]
    
    for z_ind in range(1, var.Z_edge.shape[0]-3):
        key_1 = 'central {}'.format(z_ind)
        key_2 = 'central {}'.format(z_ind+1)
        
        y_split[key_1][0::, 0::, 0::, -1] = y_split[key_2][0::, 0::, 0::, 1]
        y_split[key_2][0::, 0::, 0::, 0] = y_split[key_1][0::, 0::, 0::, -2]
        
    y_split[key_2][0::, 0::, 0::, -1] = y_split['top'][0::, 0::, 0::, 1]
    y_split['top'][0::, 0::, 0::, 0] = y_split[key_2][0::, 0::, 0::, -2]
    
    if i == var.save_steps * cnt:
        time_key = 'time_%s' % str(cnt)
        print '\nStep: '+str(cnt)+' of '+str(sets)   
        
        y = np.append(y_split['bottom'][0::, 0::, 0::, 0:-1], y_split['central 1'][0::, 0::, 0::, 1:-1], axis=3)
        for z_ind in range(1, var.Z_edge.shape[0]-2):
            key = 'central {}'.format(z_ind)
            y = np.append(y, y_split[key][0::, 0::, 0::, 1:-1], axis=3)
        y = np.append(y, y_split['top'][0::, 0::, 0::, 1::], axis=3)
        
        plasma_vel = np.stack((y[2], y[3], y[4]), axis=3)
        neutral_vel = np.stack((y[5], y[6], y[7]), axis=3)
        '''
        myFile.create_dataset('Plasma/Density/'+time_key, data=y[0])
        myFile.create_dataset('Neutral/Density/'+time_key, data=y[1])
        
        myFile.create_dataset('Plasma/Velocity/'+time_key, data=plasma_vel)
        myFile.create_dataset('Neutral/Velocity/'+time_key, data=neutral_vel)
        '''
        cnt+=1

#myFile.close()

end_time = time.time() - start_time

print '\nParallelized Code: '+str(end_time)