# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:40:29 2019

@author: michael
"""

import sys
sys.path.append('/home/michael/Documents/PWFA/Decay_Model/')

import numpy as np
import h5py as h5
import time

import global_variables as var
import functions_3D as fun

start_time = time.time()

# Set Initial Conditions
init_path = 'Init_Density/'+var.importName

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

print( '\nStep {0} of {1} saved'.format(1, var.saves) )

# Evolve Initial Condition
cnt = 1
for i in range(1, var.Tstps+1): 
    y = fun.integrable_function_1P(y, y.shape[3])
    
    if i == var.save_steps[cnt]:
        time_key = 'time_%s' % str(cnt)
        
        plasma_vel = np.stack((y[2][1:-1, 1:-1, 1:-1], y[3][1:-1, 1:-1, 1:-1], y[4][1:-1, 1:-1, 1:-1]), axis=3)
        neutral_vel = np.stack((y[5][1:-1, 1:-1, 1:-1], y[6][1:-1, 1:-1, 1:-1], y[7][1:-1, 1:-1, 1:-1]), axis=3)
        
        myFile.create_dataset('Plasma/Density/'+time_key, data=y[0][1:-1, 1:-1, 1:-1])
        myFile.create_dataset('Neutral/Density/'+time_key, data=y[1][1:-1, 1:-1, 1:-1])
        
        myFile.create_dataset('Plasma/Velocity/'+time_key, data=plasma_vel)
        myFile.create_dataset('Neutral/Velocity/'+time_key, data=neutral_vel)
        
        print( '\nStep {0} of {1} saved'.format(cnt+1, var.saves) )
        
        cnt+=1

myFile.close()

end_time = time.time() - start_time

print(end_time / 60)