# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:08:04 2019

@author: michael
"""

import sys
sys.path.append('/home/michael/Documents/PWFA/Decay_Model/')

import global_variables as var
import shared_var as svar

import h5py as h5
import numpy as np
import scipy.interpolate as sci
from matplotlib import pyplot as plt

def radial_vel(vel, x, y):
    vel_r = np.linalg.norm(vel)
    if (x >= 0) and (y >= 0):            
        if vel[0] < 0:
            vel_r = - vel_r    
    elif (x < 0) and (y >= 0):
        if vel[0] >= 0:
            vel_r = - vel_r    
    elif (x < 0) and (y < 0):
        if vel[0] >= 0:
           vel_r = - vel_r
    else:
        if vel[0] < 0:
            vel_r = - vel_r
 
    return vel_r 
  
myFile = h5.File('Data/'+svar.name+'.h5', 'r')

vel = myFile[svar.fluid+'/Velocity/time_0']

vel_z_ind = int( .5 * vel.shape[2] )
x_dom_og = var.X_dom_1D * svar.s_scl * 1e3

pts = 501
half_pts = int(.5*pts)

x_dom = np.linspace(x_dom_og[0], x_dom_og[-1], pts) 

vel_long_rad = np.empty((svar.dumps, pts, pts))
for d in range(svar.dumps):
    vel = myFile['{0}/Velocity/time_{1}'.format(svar.fluid, d)]
    vel_long = vel[0::, 0::, vel_z_ind] * svar.v_scl
    vel_long_spline_x = sci.RectBivariateSpline(x_dom_og, x_dom_og, vel_long[0::, 0::, 0])
    vel_long_spline_y = sci.RectBivariateSpline(x_dom_og, x_dom_og, vel_long[0::, 0::, 1])
    for x_ind, x in enumerate(x_dom):
        for y_ind, y in enumerate(x_dom):
            vel_x = vel_long_spline_x(x, y)
            vel_y = vel_long_spline_y(x, y)
            vel = np.array([vel_x, vel_y])
            
            vel_long_rad[d, x_ind, y_ind] = radial_vel(vel, x, y)

vel_min = np.min( vel_long_rad )
vel_max = np.max( vel_long_rad )
vel_hlf = np.linspace(vel_min, 3*vel_max, pts)[half_pts]

ln = vel_long_rad[svar.dump, half_pts, 0::]
im = vel_long_rad[svar.dump]
    
fnt = 18
fig, ax1 = plt.subplots()

if svar.fluid == 'Plasma':
    color = 'r'
elif svar.fluid == 'Neutral':
    color = 'b'

t_mark =  round( svar.t_scl * svar.dt * svar.save_steps[svar.dump])
ax1.text(0.215, 0.9, 'time: {} ns'.format(t_mark), bbox={'facecolor':'white', 'alpha':1, 'pad':5}, 
     transform=ax1.transAxes, ha='center', fontsize=fnt, animated=True)

ax1.imshow(im, extent=(x_dom[0], x_dom[-1], x_dom[0], x_dom[-1]), vmin=vel_min, vmax=vel_max, animated=True)
ax1.set_xlabel('X (mm)', fontsize=fnt)
ax1.set_xlim(x_dom[0], x_dom[-1])
ax1.set_ylabel('Y (mm)', fontsize=fnt)
ax1.set_ylim(x_dom[0], x_dom[-1])
ax1.tick_params(axis='y', labelcolor='k')

ax2 = ax1.twinx()
ax2.plot(x_dom, ln, color=color, label='line out')
ax2.plot([x_dom[0], x_dom[-1]], [vel_hlf, vel_hlf], color=color, ls='--',  label='line out path')
ax2.set_ylabel('{} Velocity'.format(svar.fluid), color=color, fontsize=fnt)
ax2.set_ylim(vel_min, 3*vel_max)
ax2.set_xlim(x_dom[0], x_dom[-1])
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.savefig(svar.vel_path+'trans_frame_{}.png'.format(svar.dump))

plt.close()