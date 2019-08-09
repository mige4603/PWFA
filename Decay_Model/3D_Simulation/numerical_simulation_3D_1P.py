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
import functions_3D_1P as fun

start_time = time.time()

# Set Initial Conditions
init_path = 'Init_Density/'+var.importName

if var.periodic:
    # Initial Conditions
    plasma_den = np.empty((var.Rpts+2, var.Rpts+2, var.Zpts+2))
    neutral_den = np.empty((var.Rpts+2, var.Rpts+2, var.Zpts+2))
    
    plasma_den[1:-1, 1:-1, 1:-1], neutral_den[1:-1, 1:-1, 1:-1]  = fun.import_initial_conditions(var.init_path) #fun.initial_condition(var.X_dom_1D, var.X_dom_1D, var.Z_dom_1D)

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
    plasma_den = np.empty((var.Rpts, var.Rpts, var.Zpts))
    neutral_den = np.empty((var.Rpts, var.Rpts, var.Zpts))
    
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

print( '\nStep {0} of {1} saved'.format(1, var.saves) )

# Evolve Initial Condition
cnt = 1
for i in range(1, var.Tstps+1): 
    y = fun.integrable_function(y, i, var.periodic)
    
    lst = np.argwhere(np.isnan(y))     
    if lst.any():
        time_key = 'time_%s' % str(cnt-1)
        
        plas_den = myFile['Plasma/Density/'+time_key][:]
        neut_den = myFile['Neutral/Density/'+time_key][:]
        
        plas_vel_x = myFile['Plasma/Velocity/'+time_key][0::, 0::, 0::, 0]
        plas_vel_y = myFile['Plasma/Velocity/'+time_key][0::, 0::, 0::, 1]
        plas_vel_z = myFile['Plasma/Velocity/'+time_key][0::, 0::, 0::, 2]

        neut_vel_x = myFile['Neutral/Velocity/'+time_key][0::, 0::, 0::, 0]
        neut_vel_y = myFile['Neutral/Velocity/'+time_key][0::, 0::, 0::, 1]
        neut_vel_z = myFile['Neutral/Velocity/'+time_key][0::, 0::, 0::, 2]
        
        y_pst = np.array([plas_den,
                          neut_den,
                          plas_vel_x,
                          plas_vel_y,
                          plas_vel_z,
                          neut_vel_x,
                          neut_vel_y,
                          neut_vel_z])
        for l in lst:
            print('{0} : {1}'.format(l, 1e-3 * var.v * y_pst[l[0], l[1], l[2], l[3]]))
        print('Step {0}: {1} of {2}'.format(cnt, i, var.Tstps))
        break
    else:
        if i == var.save_steps[cnt]:
            time_key = 'time_%s' % str(cnt)
            
            if var.periodic:
                plasma_vel = np.stack((y[2][1:-1, 1:-1, 1:-1], y[3][1:-1, 1:-1, 1:-1], y[4][1:-1, 1:-1, 1:-1]), axis=3)
                neutral_vel = np.stack((y[5][1:-1, 1:-1, 1:-1], y[6][1:-1, 1:-1, 1:-1], y[7][1:-1, 1:-1, 1:-1]), axis=3)
                
                myFile.create_dataset('Plasma/Density/'+time_key, data=y[0][1:-1, 1:-1, 1:-1])
                myFile.create_dataset('Neutral/Density/'+time_key, data=y[1][1:-1, 1:-1, 1:-1])
                
            else:
                plasma_vel = np.stack((y[2], y[3], y[4]), axis=3)
                neutral_vel = np.stack((y[5], y[6], y[7]), axis=3)
                
                myFile.create_dataset('Plasma/Density/'+time_key, data=y[0])
                myFile.create_dataset('Neutral/Density/'+time_key, data=y[1])
                
            myFile.create_dataset('Plasma/Velocity/'+time_key, data=plasma_vel)
            myFile.create_dataset('Neutral/Velocity/'+time_key, data=neutral_vel)
            
            print( '\nStep {0} of {1} saved'.format(cnt+1, var.saves) )
            cnt+=1            

myFile.close()

end_time = time.time() - start_time

print(end_time / 60)