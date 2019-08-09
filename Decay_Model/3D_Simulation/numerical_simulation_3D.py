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

import threading
import multiprocessing as mp

import global_variables as var
import functions_3D as fun

start_time = time.time()

# Set Initial Conditions
init_path = 'Init_Density/'+var.importName

if var.periodic:
    # Initial Conditions
    plasma_den = np.empty((var.Rpts+2, var.Rpts+2, var.Zpts+2))
    neutral_den = np.empty((var.Rpts+2, var.Rpts+2, var.Zpts+2))
    
    plasma_den[1:-1, 1:-1, 1:-1], neutral_den[1:-1, 1:-1, 1:-1]  = fun.import_initial_conditions(var.init_path) #fun.initial_condition(var.X_dom_1D, var.X_dom_1D, var.Z_dom_1D)
    
    plasma_den[0::, 0::, 0] = plasma_den[0::, 0::, -2]
    plasma_den[0::, 0::, -1] = plasma_den[0::, 0::, 1]
    
    neutral_den[0::, 0::, 0] = neutral_den[0::, 0::, -2]
    neutral_den[0::, 0::, -1] = neutral_den[0::, 0::, 1]
    
    plasma_vel_x = np.zeros((var.Rpts+2, var.Rpts+2, var.Zpts+2))
    plasma_vel_y = np.zeros((var.Rpts+2, var.Rpts+2, var.Zpts+2))
    plasma_vel_z = np.zeros((var.Rpts+2, var.Rpts+2, var.Zpts+2))
    plasma_vel = np.stack((plasma_vel_x[1:-1, 1:-1, 1:-1], plasma_vel_y[1:-1, 1:-1, 1:-1], plasma_vel_z[1:-1, 1:-1, 1:-1]), axis=3)
    
    neutral_vel_x = np.zeros((var.Rpts+2, var.Rpts+2, var.Zpts+2))
    neutral_vel_y = np.zeros((var.Rpts+2, var.Rpts+2, var.Zpts+2))
    neutral_vel_z = np.zeros((var.Rpts+2, var.Rpts+2, var.Zpts+2))
    neutral_vel = np.stack((neutral_vel_x[1:-1, 1:-1, 1:-1], neutral_vel_y[1:-1, 1:-1, 1:-1], neutral_vel_z[1:-1, 1:-1, 1:-1]), axis=3)
    
    # Periodic Boundary
    plasma_den = fun.periodic_boundary_den(plasma_den)
    neutral_den = fun.periodic_boundary_den(neutral_den)
    
    plasma_vel_x = fun.periodic_boundary_vel_x(plasma_vel_x)
    plasma_vel_y = fun.periodic_boundary_vel_y(plasma_vel_y)
    plasma_vel_z = fun.periodic_boundary_vel_z(plasma_vel_z)
    
    neutral_vel_x = fun.periodic_boundary_vel_x(neutral_vel_x)
    neutral_vel_y = fun.periodic_boundary_vel_y(neutral_vel_y)
    neutral_vel_z = fun.periodic_boundary_vel_z(neutral_vel_z)

else:
    plasma_den, neutral_den  = fun.import_initial_conditions(var.init_path) #fun.initial_condition(var.X_dom_1D, var.X_dom_1D, var.Z_dom_1D)

    plasma_vel_x = np.zeros((var.Rpts, var.Rpts, var.Zpts))
    plasma_vel_y = np.zeros((var.Rpts, var.Rpts, var.Zpts))
    plasma_vel_z = np.zeros((var.Rpts, var.Rpts, var.Zpts))
    plasma_vel = np.stack((plasma_vel_x, plasma_vel_y, plasma_vel_z), axis=3)
    
    neutral_vel_x = np.zeros((var.Rpts, var.Rpts, var.Zpts))
    neutral_vel_y = np.zeros((var.Rpts, var.Rpts, var.Zpts))
    neutral_vel_z = np.zeros((var.Rpts, var.Rpts, var.Zpts))
    neutral_vel = np.stack((neutral_vel_x, neutral_vel_y, neutral_vel_z), axis=3)

# Density threshold inversion
plasma_den[plasma_den < var.thresh] = var.thresh
neutral_den[neutral_den < var.thresh] = var.thresh     

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

if var.periodic:
    myFile.create_dataset('Plasma/Density/time_0', data=plasma_den[1:-1, 1:-1, 1:-1])
    myFile.create_dataset('Neutral/Density/time_0', data=neutral_den[1:-1, 1:-1, 1:-1])    
    
else:
    myFile.create_dataset('Plasma/Density/time_0', data=plasma_den)
    myFile.create_dataset('Neutral/Density/time_0', data=neutral_den)
    
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

y_split = [y[0::, 0::, 0::, var.Z_edge[0]:var.Z_edge[1]+2],
           y[0::, 0::, 0::, var.Z_edge[1]:var.Z_edge[2]+2],
           y[0::, 0::, 0::, var.Z_edge[2]::]]
           
stiches = np.empty((3, 2, y.shape[0], y.shape[1], y.shape[2]))

stiches[0] = np.array([y_split[2][0::, 0::, 0::, -2],
                       y_split[1][0::, 0::, 0::, 1]])                      

stiches[1] = np.array([y_split[0][0::, 0::, 0::, -2],
                       y_split[2][0::, 0::, 0::, 1]])

stiches[2] = np.array([y_split[1][0::, 0::, 0::, -2], 
                       y_split[0][0::, 0::, 0::, 1]])
      
z_beg = [var.Z_edge[0], 
         var.Z_edge[1],
         var.Z_edge[2]]

z_end = [var.Z_edge[1], 
         var.Z_edge[2],
         var.Z_edge[3]-1]
         
sect = ['bot',
        'mid',
        'top']

queues_in = []
queues_out = []
threads = []

for i in range(var.procs):      
    queue_in = mp.Queue() 
    queue_out = mp.Queue()
    
    queues_in.append(queue_in)
    queues_out.append(queue_out)
    
    threads.append( threading.Thread(target=fun.integrable_function, args=(y_split[i], queue_in, queue_out, z_beg[i], z_end[i], var.periodic, sect[i],)) )

for i in range(var.procs):
    threads[i].start()

cnt = 1
for i in range(1, var.Tstps+1):                     
    queues_in[0].put( stiches[0] )
    queues_in[1].put( stiches[1] )
    queues_in[2].put( stiches[2] )
    
    if i == var.save_steps[cnt]:
        print( '\nStep {0} of {1} saved'.format(cnt+1, var.saves) )
        cnt+=1

    bnds_bot = queues_out[0].get()    
    bnds_mid = queues_out[1].get()
    bnds_top = queues_out[2].get()
    
    stiches[0] = np.array([bnds_top[1],
                           bnds_mid[0]])

    stiches[1] = np.array([bnds_bot[1],
                           bnds_top[0]])

    stiches[2] = np.array([bnds_mid[1],
                           bnds_bot[0]])                           
                           
    
end_time = time.time() - start_time
print(end_time / 60)