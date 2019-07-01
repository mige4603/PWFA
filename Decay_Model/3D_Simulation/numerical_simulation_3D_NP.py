# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 18:09:46 2019

@author: michael
"""

import sys
sys.path.append('/home/michael/Documents/PWFA/Decay_Model/')

import numpy as np
import h5py as h5
import time

import sharedmem as shm
import multiprocessing as mp

import global_variables as var
import functions_3D as fun

start_time = time.time()

# Set Initial Conditions
init_path = 'Init_Density/plasma_density_reduced.h5'

plasma_den = np.empty((var.Rpts+2, var.Rpts+2, var.Zpts+2))
neutral_den = np.empty((var.Rpts+2, var.Rpts+2, var.Zpts+2))
plasma_den[1:-1, 1:-1, 1:-1], neutral_den[1:-1, 1:-1, 1:-1]  = fun.import_initial_conditions(init_path)

plasma_vel_x = np.zeros((var.Rpts+2, var.Rpts+2, var.Zpts+2))
plasma_vel_y = np.zeros((var.Rpts+2, var.Rpts+2, var.Zpts+2))
plasma_vel_z = np.zeros((var.Rpts+2, var.Rpts+2, var.Zpts+2))
plasma_vel = np.stack((plasma_vel_x[1:-1, 1:-1, 1:-1], plasma_vel_y[1:-1, 1:-1, 1:-1], plasma_vel_z[1:-1, 1:-1, 1:-1]), axis=3)

neutral_vel_x = np.zeros((var.Rpts+2, var.Rpts+2, var.Zpts+2))
neutral_vel_y = np.zeros((var.Rpts+2, var.Rpts+2, var.Zpts+2))
neutral_vel_z = np.zeros((var.Rpts+2, var.Rpts+2, var.Zpts+2))
neutral_vel = np.stack((neutral_vel_x[1:-1, 1:-1, 1:-1], neutral_vel_y[1:-1, 1:-1, 1:-1], neutral_vel_z[1:-1, 1:-1, 1:-1]), axis=3)

# Splice Domains
y = np.array([plasma_den,
              neutral_den,
              plasma_vel_x,
              plasma_vel_y,
              plasma_vel_z,
              neutral_vel_x,
              neutral_vel_y,
              neutral_vel_z])
              
# Save Initial Conditions
fun.write_meta_file(var.meta_file, var.desc)

myFile = h5.File(var.data_file, 'w')

myFile.create_dataset('Plasma/Density/time_0', data=plasma_den[1:-1, 1:-1, 1:-1])
myFile.create_dataset('Neutral/Density/time_0', data=neutral_den[1:-1, 1:-1, 1:-1])

myFile.create_dataset('Plasma/Velocity/time_0', data=plasma_vel)
myFile.create_dataset('Neutral/Velocity/time_0', data=neutral_vel)

for i in range(1, var.saves):
    key = 'time_{}'.format(i)
    
    myFile.create_dataset('Plasma/Density/'+key, data=np.empty((var.Rpts, var.Rpts, var.Zpts)))
    myFile.create_dataset('Neutral/Density/'+key, data=np.empty((var.Rpts, var.Rpts, var.Zpts)))
    
    myFile.create_dataset('Plasma/Velocity/'+key, data=np.empty((var.Rpts, var.Rpts, var.Zpts, 3)))
    myFile.create_dataset('Neutral/Velocity/'+key, data=np.empty((var.Rpts, var.Rpts, var.Zpts, 3)))

myFile.close()

print( '\nStep {0} of {1} saved'.format(1, var.saves) )

queues = []
bounds = []
procs = []

for i in range(var.procs):    
    z_bot = var.Z_edge[i]
    z_top = var.Z_edge[i+1]
    
    y_split = y[0::, 0::, 0::, z_bot:z_top+2] 
    
    queue = mp.JoinableQueue() 
    queues.append(queue)
    
    bound = shm.empty((2, 8, var.Rpts+2, var.Rpts+2))
    bound[0] = y_split[0::, 0::, 0::, 1]
    bound[1] = y_split[0::, 0::, 0::, -2]
    bounds.append(bound)
    
    procs.append( mp.Process(target=fun.integrable_function_3P, args=(y_split, queue, bound, z_bot, z_top,)) )

for i in range(var.procs):
    procs[i].start()

cnt = 1
for i in range(1, var.Tstps+1):
    queues[0].put(bound)
    
    for j in range(1, var.procs-1):
        bound = np.array([bounds[j-1][1],
                          bounds[j+1][0]])
        
        queues[j].put(bound)
        
    bound = np.array([bounds[1][1],
                      bounds[0][0]])

    queues[-1].put(bound)
    
    if i == var.save_steps[cnt]:
        print( '\nStep {0} of {1} saved'.format(cnt+1, var.saves) )
        cnt+=1
    
    for j in range(var.procs):
        queues[j].join()

for j in range(var.procs):
    procs[j].join()

end_time = time.time() - start_time
print(end_time / 60)